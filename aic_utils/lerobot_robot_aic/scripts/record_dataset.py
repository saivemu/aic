#!/usr/bin/env python3
"""Record a LeRobot dataset by snooping /observations and /aic_controller/pose_commands
during a CheatCode-driven eval rollout.

Architecture: this script runs as a ROS2 node alongside a normal eval +
aic_model+CheatCode pipeline. CheatCode publishes MotionUpdates to
/aic_controller/pose_commands; the AIC adapter publishes synchronized
Observations to /observations. We snoop both, build (obs, action) frames at
camera rate, and write to a LeRobot dataset.

Episode boundaries are detected via an idle-gap heuristic on
/aic_controller/pose_commands: when no command has arrived for >= --episode-idle-timeout
seconds, we close the current episode. Trial-to-trial reset has a natural
quiet period (~5-8s) which exceeds the default 2s timeout.

Usage (from the AIC repo root):
    pixi run python aic_utils/lerobot_robot_aic/scripts/record_dataset.py \\
        --repo-id ${HF_USER}/aic_act_v1 --num-episodes 300

Run in parallel with the eval entrypoint (gazebo + controller + adapter +
engine + aic_model running CheatCode). Both must use ground_truth=true so
CheatCode can do its TF-based trajectory generation.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_aic.recording_utils import pose_targets_to_action, stamp_to_nanoseconds
from lerobot_robot_aic.task_encoding import TASK_DIM, encode_task


# Plan D schema (aic_act_v2). Differs from v1 in three ways:
# 1. State adds wrist_wrench (6-D) and task one-hot (11-D) — see task_encoding.py.
# 2. Images record at 0.5x scale (576x512) instead of 0.25x (288x256).
# 3. Both SFP ports are exercised (gen_random_trials.py patch).
# RunACT.py must mirror this schema at inference (state ordering + image scale).
IMAGE_SCALE = 0.5        # 1152x1024 -> 576x512
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 576
STATE_BASE_DIM = 26      # TCP pose 7 + lin vel 3 + ang vel 3 + tcp_error 6 + joints 7
WRENCH_DIM = 6           # force xyz + torque xyz
STATE_DIM = STATE_BASE_DIM + WRENCH_DIM + TASK_DIM  # 26 + 6 + 11 = 43
ACTION_DIM = 7           # linear xyz + angular xyz + 1 unused (gripper, kept 0)
DEFAULT_TASK_DESCRIPTION = "Insert cable plug into target port"
DEFAULT_MIN_EPISODE_FRAMES = 30  # discard sub-1.5s episodes (likely scene-reset glitches)


def features_spec() -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": None,
        },
        "observation.images.left_camera": {
            "dtype": "video",
            "shape": (3, IMAGE_HEIGHT, IMAGE_WIDTH),
            "names": ["channels", "height", "width"],
        },
        "observation.images.center_camera": {
            "dtype": "video",
            "shape": (3, IMAGE_HEIGHT, IMAGE_WIDTH),
            "names": ["channels", "height", "width"],
        },
        "observation.images.right_camera": {
            "dtype": "video",
            "shape": (3, IMAGE_HEIGHT, IMAGE_WIDTH),
            "names": ["channels", "height", "width"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": None,
        },
    }


def _load_trial_task_list(trials_config_path: Path) -> list[tuple[str, str, str]]:
    """Read a random_trials.yaml and extract per-trial (target_module, port, plug)
    in the same iteration order the aic_engine uses. The engine iterates the
    ``trials:`` dict insertion-order, so this needs to be a Python 3.7+ dict —
    PyYAML safe_load preserves insertion order from the YAML source."""
    with trials_config_path.open() as f:
        cfg = yaml.safe_load(f)
    trials = cfg.get("trials", {})
    if not trials:
        raise ValueError(f"No trials in {trials_config_path}")
    out: list[tuple[str, str, str]] = []
    for trial_name, trial in trials.items():
        tasks = trial.get("tasks", {})
        if not tasks:
            raise ValueError(f"{trial_name} has no tasks")
        # We assume one task per trial (true for all sample/random configs).
        task = next(iter(tasks.values()))
        out.append(
            (
                task["target_module_name"],
                task["port_name"],
                task["plug_type"],
            )
        )
    return out


class DatasetRecorder(Node):
    def __init__(
        self,
        repo_id: str,
        root: str,
        fps: int,
        episode_idle_timeout: float,
        max_action_age: float,
        min_episode_frames: int,
        task_description: str,
        trial_task_list: list[tuple[str, str, str]],
        perturbing_topic: str,
        resume: bool = False,
    ):
        super().__init__("dataset_recorder")

        self.fps = fps
        self.episode_idle_timeout_ns = int(episode_idle_timeout * 1e9)
        self.max_action_age_ns = int(max_action_age * 1e9)
        self.min_episode_frames = min_episode_frames
        self.task_description = task_description
        self.last_observation: Observation | None = None
        self.last_action: np.ndarray | None = None
        self.last_action_time = None
        self.episode_in_progress = False
        self.frames_in_episode = 0
        self.frames_skipped_stale_action = 0
        self.frames_skipped_perturbing = 0
        self.unsupported_command_count = 0
        self.episodes_saved = 0
        self.episodes_dropped = 0
        self._consecutive_save_failures = 0
        self._max_consecutive_save_failures = 2  # exit & finalize after N in a row
        self.perturbing = False
        self.last_action_is_perturbing = False

        # Trial-task indexing. The recorder snoops topics, not the action goal,
        # so we sync the task identity off the trial config + an attempt counter.
        # Each call to _start_episode advances the index by one. This is robust to
        # dropped episodes (drop still bumps the attempt index) but NOT robust to
        # engine-side trial skips (rare; engine always runs every trial in order).
        self.trial_task_list = trial_task_list
        self.episode_attempt_idx = 0  # 0-based; next attempt gets this index
        # Pre-built task vector for the currently-running episode.
        self.current_task_vec = np.zeros(TASK_DIM, dtype=np.float32)

        # Pose-delta state: when MotionUpdate is in MODE_POSITION (CheatCode does
        # this), msg.velocity is zero. We synthesize the action by differencing
        # consecutive pose targets: action = (pose[t] - pose[t-1]) / dt.
        # Reset on episode boundaries so we don't bleed velocity across trials.
        self.prev_pose_target = None  # tuple(pos_xyz, quat_wxyz, ros_time_ns)

        # Build the dataset. use_videos=True triggers AV1 encoding of image features.
        # metadata_buffer_size=1 flushes per-episode metadata immediately so the
        # episodes parquet is durable on every save_episode (instead of every 10).
        # The data parquet still requires finalize() to write its footer; combined
        # with the disk-fill watcher and exit-on-write-failure logic below, that
        # finalize will reliably run.
        if resume:
            self.dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=root,
            )
            self.episodes_saved = self.dataset.meta.total_episodes
            self.get_logger().info(
                f"Resuming dataset {repo_id} at {self.dataset.root} "
                f"(existing episodes={self.episodes_saved})"
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features_spec(),
                root=root,
                robot_type="aic_controller",
                use_videos=True,
                metadata_buffer_size=1,
            )
        if not resume:
            self.get_logger().info(
                f"LeRobotDataset created at {self.dataset.root} (repo_id={repo_id})"
            )

        # Subscribe with sensor-data QoS (best effort) for the high-rate observation
        # topic, matched RELIABLE for the action topic.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        action_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Observation, "/observations", self._on_observation, sensor_qos)
        self.create_subscription(
            MotionUpdate, "/aic_controller/pose_commands", self._on_pose_command, action_qos
        )
        if perturbing_topic:
            self.create_subscription(Bool, perturbing_topic, self._on_perturbing, action_qos)
            self.get_logger().info(
                f"Skipping frames while {perturbing_topic} publishes true."
            )

        # Periodic episode-end check
        self.create_timer(0.5, self._check_episode_end)

    # ------------------------------------------------------------------ subscribers

    def _on_perturbing(self, msg: Bool) -> None:
        self.perturbing = bool(msg.data)

    def _on_pose_command(self, msg: MotionUpdate) -> None:
        # MotionUpdate has both velocity and pose fields. CheatCode uses
        # MODE_POSITION (publishes pose, leaves velocity=0). RunACT-style velocity
        # mode uses MODE_VELOCITY (publishes velocity, leaves pose=0). Detect mode
        # and synthesize a 6-D velocity action accordingly.
        mode = msg.trajectory_generation_mode.mode
        now = self.get_clock().now()
        action = None

        if mode == TrajectoryGenerationMode.MODE_VELOCITY:
            v = msg.velocity
            action = np.array(
                [
                    v.linear.x,
                    v.linear.y,
                    v.linear.z,
                    v.angular.x,
                    v.angular.y,
                    v.angular.z,
                    0.0,
                ],
                dtype=np.float32,
            )
        elif mode == TrajectoryGenerationMode.MODE_POSITION:
            # CheatCode publishes position targets and leaves msg.velocity at zero,
            # so synthesize the action by differencing consecutive pose targets.
            p = msg.pose.position
            q = msg.pose.orientation
            cur_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
            cur_quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
            cur_time_ns = stamp_to_nanoseconds(msg.header.stamp) or now.nanoseconds

            if self.prev_pose_target is not None:
                prev_pos, prev_quat_wxyz, prev_time_ns = self.prev_pose_target
                action = pose_targets_to_action(
                    prev_pos,
                    prev_quat_wxyz,
                    prev_time_ns,
                    cur_pos,
                    cur_quat_wxyz,
                    cur_time_ns,
                )
            # Update for next delta computation
            self.prev_pose_target = (cur_pos, cur_quat_wxyz, cur_time_ns)
        else:
            self.unsupported_command_count += 1
            if self.unsupported_command_count <= 5:
                self.get_logger().warn(
                    f"Ignoring MotionUpdate with unsupported trajectory mode {mode}"
                )

        if action is not None:
            self.last_action = action
            self.last_action_is_perturbing = self.perturbing
            self.last_action_time = now
            if not self.episode_in_progress:
                self._start_episode()

    def _on_observation(self, msg: Observation) -> None:
        self.last_observation = msg
        if self.episode_in_progress and self.last_action is not None:
            self._try_add_frame()

    # ------------------------------------------------------------------- episode

    def _start_episode(self) -> None:
        self.episode_in_progress = True
        self.frames_in_episode = 0
        # Pull the next trial's task identity. If --resume bumped episodes_saved,
        # we still want to align to (saved+dropped) attempts; for a fresh run
        # that's just the running attempt counter.
        idx = self.episode_attempt_idx
        if idx < len(self.trial_task_list):
            target_module, port_name, plug_type = self.trial_task_list[idx]
            self.current_task_vec = encode_task(target_module, port_name, plug_type)
            task_label = f"{target_module}/{port_name}/{plug_type}"
        else:
            self.get_logger().warn(
                f"Episode attempt {idx} exceeds trial list of {len(self.trial_task_list)}; "
                "task vector will be all zeros."
            )
            self.current_task_vec = np.zeros(TASK_DIM, dtype=np.float32)
            task_label = "UNKNOWN"
        self.episode_attempt_idx += 1
        self.get_logger().info(
            f"Episode {self.episodes_saved + 1} starting "
            f"(saved={self.episodes_saved}, dropped={self.episodes_dropped}, "
            f"attempt_idx={idx}, task={task_label})"
        )

    def _end_episode(self) -> None:
        if not self.episode_in_progress:
            return
        if self.frames_in_episode < self.min_episode_frames:
            self.get_logger().warn(
                f"Episode too short ({self.frames_in_episode} frames < {self.min_episode_frames}), "
                f"clearing buffer."
            )
            self.dataset.clear_episode_buffer()
            self.episodes_dropped += 1
        else:
            try:
                self.dataset.save_episode()
                self.episodes_saved += 1
                self._consecutive_save_failures = 0
                self.get_logger().info(
                    f"Saved episode {self.episodes_saved} with {self.frames_in_episode} frames."
                )
            except Exception as e:
                self.get_logger().error(f"save_episode failed: {e}")
                self.episodes_dropped += 1
                self._consecutive_save_failures += 1
                # Bail out cleanly so finalize() runs and writes parquet footers
                # for what was successfully saved up to this point. Better to lose
                # the trailing in-progress data than to keep spinning and never
                # close the parquet files.
                if self._consecutive_save_failures >= self._max_consecutive_save_failures:
                    self.get_logger().error(
                        f"{self._consecutive_save_failures} consecutive save_episode "
                        f"failures; shutting down so finalize() can write parquet "
                        f"footers for the {self.episodes_saved} saved episodes."
                    )
                    rclpy.shutdown()
        self.episode_in_progress = False
        self.frames_in_episode = 0
        # Drop pose-delta history so the first pose of the next episode initializes
        # cleanly without a spurious huge velocity from the inter-trial pose jump.
        self.prev_pose_target = None

    def _check_episode_end(self) -> None:
        if not self.episode_in_progress or self.last_action_time is None:
            return
        now = self.get_clock().now()
        gap = (now - self.last_action_time).nanoseconds
        if gap > self.episode_idle_timeout_ns:
            self._end_episode()

    # ------------------------------------------------------------------- frame add

    def _img_to_chw_uint8(self, img_msg) -> np.ndarray:
        """ROS sensor_msgs/Image -> resized CHW uint8 numpy."""
        arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )
        if IMAGE_SCALE != 1.0:
            arr = cv2.resize(
                arr, None, fx=IMAGE_SCALE, fy=IMAGE_SCALE, interpolation=cv2.INTER_AREA
            )
        # ROS images are BGR by default in some pipelines; the AIC eval publishes RGB
        # already (verified via RunACT.py treating buffer as 3-channel without conversion).
        # If video colours look swapped post-encode, add cv2.cvtColor here.
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    def _try_add_frame(self) -> None:
        obs = self.last_observation
        action = self.last_action
        if obs is None or action is None:
            return
        if self.perturbing or self.last_action_is_perturbing:
            self.frames_skipped_perturbing += 1
            return
        if self.last_action_time is not None and self.max_action_age_ns > 0:
            action_age = (self.get_clock().now() - self.last_action_time).nanoseconds
            if action_age > self.max_action_age_ns:
                self.frames_skipped_stale_action += 1
                return
        try:
            tcp = obs.controller_state.tcp_pose
            tcp_vel = obs.controller_state.tcp_velocity
            wrench = obs.wrist_wrench.wrench
            state = np.array(
                [
                    # Base 26-D (Plan A/B/C compatible) ------------------------
                    tcp.position.x,
                    tcp.position.y,
                    tcp.position.z,
                    tcp.orientation.x,
                    tcp.orientation.y,
                    tcp.orientation.z,
                    tcp.orientation.w,
                    tcp_vel.linear.x,
                    tcp_vel.linear.y,
                    tcp_vel.linear.z,
                    tcp_vel.angular.x,
                    tcp_vel.angular.y,
                    tcp_vel.angular.z,
                    *list(obs.controller_state.tcp_error),
                    *list(obs.joint_states.position[:7]),
                    # Wrist wrench 6-D — contact-stage discriminator ----------
                    wrench.force.x,
                    wrench.force.y,
                    wrench.force.z,
                    wrench.torque.x,
                    wrench.torque.y,
                    wrench.torque.z,
                    # Task identity 11-D — see task_encoding.py ---------------
                    *list(self.current_task_vec),
                ],
                dtype=np.float32,
            )
            if state.shape[0] != STATE_DIM:
                self.get_logger().warn(
                    f"State dim {state.shape[0]} != expected {STATE_DIM}, skipping frame."
                )
                return
            frame = {
                "observation.state": state,
                "observation.images.left_camera": self._img_to_chw_uint8(obs.left_image),
                "observation.images.center_camera": self._img_to_chw_uint8(obs.center_image),
                "observation.images.right_camera": self._img_to_chw_uint8(obs.right_image),
                "action": action,
                "task": self.task_description,
            }
            self.dataset.add_frame(frame)
            self.frames_in_episode += 1
        except Exception as e:
            self.get_logger().error(f"add_frame error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--root",
        default=None,
        help="Dataset root (default: ~/.cache/huggingface/lerobot/<repo_id>)",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--episode-idle-timeout",
        type=float,
        default=2.0,
        help="If no /aic_controller/pose_commands received for this many wall-seconds, "
        "the current episode is closed.",
    )
    parser.add_argument(
        "--max-action-age",
        type=float,
        default=0.25,
        help="Skip observation frames when the latest action command is older than this "
        "many seconds. Set <=0 to disable.",
    )
    parser.add_argument(
        "--min-episode-frames",
        type=int,
        default=DEFAULT_MIN_EPISODE_FRAMES,
        help="Drop episodes with fewer recorded frames than this threshold.",
    )
    parser.add_argument(
        "--task-description",
        default=DEFAULT_TASK_DESCRIPTION,
        help="LeRobot task string stored with each frame.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10**9,
        help="Stop after this many saved episodes (in this run; for --resume, in addition to existing). "
        "Default: run until killed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing dataset at --root. Required when adding episodes to a dataset "
        "produced by an earlier run.",
    )
    parser.add_argument(
        "--trials-config",
        type=Path,
        required=True,
        help="Path to the random_trials.yaml the aic_engine is running. Used to map "
        "attempt-index -> task identity (target_module / port_name / plug_type) so the "
        "state vector carries a Plan-D one-hot task encoding. MUST be the same file the "
        "engine is loading.",
    )
    parser.add_argument(
        "--perturbing-topic",
        default="/aic/cheatcode/perturbing",
        help="Bool topic used to skip artificial perturbation frames. Set to '' to disable.",
    )
    args = parser.parse_args()

    if args.root is None:
        args.root = str(Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id)
    Path(args.root).parent.mkdir(parents=True, exist_ok=True)

    trial_task_list = _load_trial_task_list(args.trials_config)
    print(
        f"Loaded {len(trial_task_list)} trial task identities from "
        f"{args.trials_config}. First 3: {trial_task_list[:3]}"
    )

    rclpy.init()
    recorder = DatasetRecorder(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        episode_idle_timeout=args.episode_idle_timeout,
        max_action_age=args.max_action_age,
        min_episode_frames=args.min_episode_frames,
        task_description=args.task_description,
        trial_task_list=trial_task_list,
        perturbing_topic=args.perturbing_topic,
        resume=args.resume,
    )
    # On --resume, restart the attempt counter at the saved-episodes mark.
    # This assumes the user is resuming from where collection died and the
    # engine is going to play the next trial from the YAML in order.
    if args.resume:
        recorder.episode_attempt_idx = recorder.episodes_saved
    target_total = (recorder.episodes_saved + args.num_episodes) if args.resume else args.num_episodes
    try:
        while rclpy.ok() and recorder.episodes_saved < target_total:
            rclpy.spin_once(recorder, timeout_sec=0.1)
    except KeyboardInterrupt:
        recorder.get_logger().info("Interrupted by user.")
    finally:
        if recorder.episode_in_progress:
            recorder._end_episode()
        try:
            recorder.dataset.finalize()
        except Exception as e:
            recorder.get_logger().error(f"dataset.finalize() error: {e}")
        recorder.get_logger().info(
            f"Recorder shutting down. saved={recorder.episodes_saved} "
            f"dropped={recorder.episodes_dropped} "
            f"stale_action_frames={recorder.frames_skipped_stale_action} "
            f"perturbing_frames={recorder.frames_skipped_perturbing} "
            f"unsupported_commands={recorder.unsupported_command_count}"
        )
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
