#!/usr/bin/env python3
"""Record training-only visual-servo labels from legal observation images.

This node is meant to run beside the normal AIC eval/model stack with
``ground_truth:=true``. It subscribes to the official ``/observations`` stream
and controller commands, then uses training-time TF to project the target port
and plug tip into the camera images. The resulting dataset is legal to use for
training a detector or visual-servo policy, but the TF labels must not be used
at runtime/scoring.

Output layout:

    <root>/
      labels.jsonl
      manifest.json
      images/ep000000/frame000160_center.jpg

Each JSONL row contains the saved image path(s), task identity, raw 43-D state,
latest action, and per-camera projected port/plug pixels.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import yaml
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from lerobot_robot_aic.recording_utils import pose_targets_to_action, stamp_to_nanoseconds
from lerobot_robot_aic.task_encoding import TASK_DIM, encode_task


IMAGE_SCALE = 0.5
STATE_BASE_DIM = 26
WRENCH_DIM = 6
STATE_DIM = STATE_BASE_DIM + WRENCH_DIM + TASK_DIM
ACTION_DIM = 7


@dataclass(frozen=True)
class TaskSpec:
    target_module_name: str
    port_name: str
    plug_type: str
    cable_name: str
    plug_name: str

    @property
    def port_frame(self) -> str:
        return f"task_board/{self.target_module_name}/{self.port_name}_link"

    @property
    def plug_frame(self) -> str:
        return f"{self.cable_name}/{self.plug_name}_link"


def _load_trial_task_specs(trials_config_path: Path) -> list[TaskSpec]:
    with trials_config_path.open() as f:
        cfg = yaml.safe_load(f)
    trials = cfg.get("trials", {})
    if not trials:
        raise ValueError(f"No trials in {trials_config_path}")

    out: list[TaskSpec] = []
    for trial_name, trial in trials.items():
        tasks = trial.get("tasks", {})
        if not tasks:
            raise ValueError(f"{trial_name} has no tasks")
        task = next(iter(tasks.values()))
        out.append(
            TaskSpec(
                target_module_name=task["target_module_name"],
                port_name=task["port_name"],
                plug_type=task["plug_type"],
                cable_name=task["cable_name"],
                plug_name=task["plug_name"],
            )
        )
    return out


def _stamp_ns(msg_with_header: Any) -> int | None:
    header = getattr(msg_with_header, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return stamp_to_nanoseconds(stamp)


class VisualServoLabelRecorder(Node):
    def __init__(
        self,
        root: Path,
        task_specs: list[TaskSpec],
        num_episodes: int,
        episode_idle_timeout: float,
        max_action_age: float,
        min_frame_index: int,
        sample_every: int,
        image_scale: float,
        cameras: list[str],
        require_visible_cameras: set[str],
        jpeg_quality: int,
        max_labels_per_episode: int,
    ) -> None:
        super().__init__("visual_servo_label_recorder")

        self.root = root
        self.images_root = root / "images"
        self.labels_path = root / "labels.jsonl"
        self.manifest_path = root / "manifest.json"
        self.task_specs = task_specs
        self.num_episodes = num_episodes
        self.episode_idle_timeout_ns = int(episode_idle_timeout * 1e9)
        self.max_action_age_ns = int(max_action_age * 1e9)
        self.min_frame_index = min_frame_index
        self.sample_every = max(1, sample_every)
        self.image_scale = image_scale
        self.cameras = cameras
        self.require_visible_cameras = require_visible_cameras
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.max_labels_per_episode = max_labels_per_episode

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_observation: Observation | None = None
        self.last_action: np.ndarray | None = None
        self.last_action_time = None
        self.prev_pose_target = None

        self.episode_in_progress = False
        self.episode_attempt_idx = 0
        self.current_task_spec: TaskSpec | None = None
        self.current_task_vec = np.zeros(TASK_DIM, dtype=np.float32)
        self.observed_frames_in_episode = 0
        self.labels_in_episode = 0

        self.episodes_completed = 0
        self.episodes_without_labels = 0
        self.total_labels = 0
        self.skipped_stale_action = 0
        self.skipped_unlabeled = 0
        self.skipped_visibility = 0
        self.unsupported_command_count = 0

        self._labels_f = self.labels_path.open("a", buffering=1)

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
        self.create_timer(0.5, self._check_episode_end)

        self._write_manifest(status="running")
        self.get_logger().info(
            "Visual label recorder ready: "
            f"root={self.root} cameras={self.cameras} "
            f"min_frame_index={self.min_frame_index} sample_every={self.sample_every} "
            f"require_visible={sorted(self.require_visible_cameras)}"
        )

    # ------------------------------------------------------------------ callbacks

    def _on_pose_command(self, msg: MotionUpdate) -> None:
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
            self.prev_pose_target = (cur_pos, cur_quat_wxyz, cur_time_ns)
        else:
            self.unsupported_command_count += 1
            if self.unsupported_command_count <= 5:
                self.get_logger().warn(
                    f"Ignoring MotionUpdate with unsupported trajectory mode {mode}"
                )

        if action is not None:
            self.last_action = action
            self.last_action_time = now
            if not self.episode_in_progress:
                self._start_episode()

    def _on_observation(self, msg: Observation) -> None:
        self.last_observation = msg
        if not self.episode_in_progress or self.last_action is None:
            return

        frame_index = self.observed_frames_in_episode
        self.observed_frames_in_episode += 1

        if frame_index < self.min_frame_index:
            return
        if (frame_index - self.min_frame_index) % self.sample_every != 0:
            return
        if (
            self.max_labels_per_episode > 0
            and self.labels_in_episode >= self.max_labels_per_episode
        ):
            return
        self._try_add_label(msg, frame_index)

    # ------------------------------------------------------------------- episode

    def _start_episode(self) -> None:
        self.episode_in_progress = True
        self.observed_frames_in_episode = 0
        self.labels_in_episode = 0

        idx = self.episode_attempt_idx
        if idx < len(self.task_specs):
            self.current_task_spec = self.task_specs[idx]
            self.current_task_vec = encode_task(
                self.current_task_spec.target_module_name,
                self.current_task_spec.port_name,
                self.current_task_spec.plug_type,
            )
            task_label = (
                f"{self.current_task_spec.target_module_name}/"
                f"{self.current_task_spec.port_name}/"
                f"{self.current_task_spec.plug_type}"
            )
        else:
            self.current_task_spec = None
            self.current_task_vec = np.zeros(TASK_DIM, dtype=np.float32)
            task_label = "UNKNOWN"
            self.get_logger().warn(
                f"Episode attempt {idx} exceeds trial list of {len(self.task_specs)}"
            )
        self.episode_attempt_idx += 1

        self.get_logger().info(
            f"Episode attempt {idx} starting "
            f"(completed={self.episodes_completed}, task={task_label})"
        )

    def _end_episode(self) -> None:
        if not self.episode_in_progress:
            return

        if self.labels_in_episode == 0:
            self.episodes_without_labels += 1
            self.get_logger().warn(
                f"Episode attempt {self.episode_attempt_idx - 1} ended with no labels "
                f"after {self.observed_frames_in_episode} observed frames."
            )
        else:
            self.get_logger().info(
                f"Episode attempt {self.episode_attempt_idx - 1} ended: "
                f"labels={self.labels_in_episode} observed_frames={self.observed_frames_in_episode}"
            )

        self.episodes_completed += 1
        self.episode_in_progress = False
        self.current_task_spec = None
        self.current_task_vec = np.zeros(TASK_DIM, dtype=np.float32)
        self.observed_frames_in_episode = 0
        self.labels_in_episode = 0
        self.prev_pose_target = None
        self._write_manifest(status="running")

    def _check_episode_end(self) -> None:
        if not self.episode_in_progress or self.last_action_time is None:
            return
        gap = (self.get_clock().now() - self.last_action_time).nanoseconds
        if gap > self.episode_idle_timeout_ns:
            self._end_episode()

    # ------------------------------------------------------------------- labeling

    def _try_add_label(self, obs: Observation, frame_index: int) -> None:
        if self.current_task_spec is None or self.last_action is None:
            return
        if self.last_action_time is not None and self.max_action_age_ns > 0:
            action_age = (self.get_clock().now() - self.last_action_time).nanoseconds
            if action_age > self.max_action_age_ns:
                self.skipped_stale_action += 1
                return

        state = self._state_from_observation(obs)
        if state is None:
            self.skipped_unlabeled += 1
            return

        camera_rows: dict[str, Any] = {}
        camera_images: dict[str, np.ndarray] = {}
        for camera_name in self.cameras:
            img_msg = getattr(obs, f"{camera_name}_image")
            info_msg = getattr(obs, f"{camera_name}_camera_info")
            rgb = self._image_to_rgb(img_msg)
            if self.image_scale != 1.0:
                rgb = cv2.resize(
                    rgb,
                    None,
                    fx=self.image_scale,
                    fy=self.image_scale,
                    interpolation=cv2.INTER_AREA,
                )

            projection = self._project_camera_labels(
                info_msg=info_msg,
                image_width=rgb.shape[1],
                image_height=rgb.shape[0],
                task_spec=self.current_task_spec,
            )
            if projection is None:
                camera_rows[camera_name] = None
                continue
            camera_rows[camera_name] = projection
            camera_images[camera_name] = rgb

        for camera_name in self.require_visible_cameras:
            labels = camera_rows.get(camera_name)
            if (
                labels is None
                or not labels["port"]["visible"]
                or not labels["plug"]["visible"]
            ):
                self.skipped_visibility += 1
                return

        image_paths: dict[str, str] = {}
        for camera_name, rgb in camera_images.items():
            if camera_rows.get(camera_name) is None:
                continue

            episode_dir = self.images_root / f"ep{self.episode_attempt_idx - 1:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            rel_path = Path("images") / episode_dir.name / f"frame{frame_index:06d}_{camera_name}.jpg"
            abs_path = self.root / rel_path
            ok = cv2.imwrite(
                str(abs_path),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                self.get_logger().warn(f"Failed to write {abs_path}")
                camera_rows[camera_name] = None
                continue
            image_paths[camera_name] = rel_path.as_posix()

        for camera_name in self.require_visible_cameras:
            if camera_name not in image_paths:
                self.skipped_unlabeled += 1
                return

        base_labels = self._base_labels(self.current_task_spec)
        if base_labels is None:
            self.skipped_unlabeled += 1
            return

        row = {
            "episode_index": self.episode_attempt_idx - 1,
            "frame_index": frame_index,
            "timestamp_ns": _stamp_ns(obs.center_image),
            "task": asdict(self.current_task_spec),
            "task_vec": self.current_task_vec.astype(float).tolist(),
            "state": state.astype(float).tolist(),
            "action": self.last_action.astype(float).tolist(),
            "image_scale": self.image_scale,
            "images": image_paths,
            "cameras": camera_rows,
            "base": base_labels,
        }
        self._labels_f.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.labels_in_episode += 1
        self.total_labels += 1
        if self.total_labels % 100 == 0:
            self.get_logger().info(
                f"Recorded {self.total_labels} labels "
                f"(episode labels={self.labels_in_episode})"
            )

    def _state_from_observation(self, obs: Observation) -> np.ndarray | None:
        try:
            tcp = obs.controller_state.tcp_pose
            tcp_vel = obs.controller_state.tcp_velocity
            wrench = obs.wrist_wrench.wrench
            state = np.array(
                [
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
                    wrench.force.x,
                    wrench.force.y,
                    wrench.force.z,
                    wrench.torque.x,
                    wrench.torque.y,
                    wrench.torque.z,
                    *list(self.current_task_vec),
                ],
                dtype=np.float32,
            )
            if state.shape[0] != STATE_DIM:
                self.get_logger().warn(
                    f"State dim {state.shape[0]} != expected {STATE_DIM}; skipping."
                )
                return None
            return state
        except Exception as exc:
            self.get_logger().warn(f"Could not build state: {exc}")
            return None

    def _project_camera_labels(
        self,
        info_msg: Any,
        image_width: int,
        image_height: int,
        task_spec: TaskSpec,
    ) -> dict[str, Any] | None:
        camera_frame = info_msg.header.frame_id
        if not camera_frame:
            return None

        port = self._project_frame(
            camera_frame,
            task_spec.port_frame,
            info_msg.k,
            image_width,
            image_height,
        )
        plug = self._project_frame(
            camera_frame,
            task_spec.plug_frame,
            info_msg.k,
            image_width,
            image_height,
        )
        if port is None or plug is None:
            return None

        delta_uv = None
        if port["uv_px"] is not None and plug["uv_px"] is not None:
            delta_uv = [
                port["uv_px"][0] - plug["uv_px"][0],
                port["uv_px"][1] - plug["uv_px"][1],
            ]
        return {
            "camera_frame": camera_frame,
            "width": image_width,
            "height": image_height,
            "port": port,
            "plug": plug,
            "delta_port_minus_plug_px": delta_uv,
        }

    def _project_frame(
        self,
        camera_frame: str,
        object_frame: str,
        k: list[float],
        image_width: int,
        image_height: int,
    ) -> dict[str, Any] | None:
        try:
            tf = self.tf_buffer.lookup_transform(camera_frame, object_frame, Time())
        except TransformException as exc:
            self.get_logger().debug(
                f"TF unavailable for {object_frame} -> {camera_frame}: {exc}"
            )
            return None

        t = tf.transform.translation
        x, y, z = float(t.x), float(t.y), float(t.z)
        in_front = z > 1e-5
        uv_px = None
        uv_norm = None
        visible = False
        if in_front:
            fx = float(k[0]) * self.image_scale
            fy = float(k[4]) * self.image_scale
            cx = float(k[2]) * self.image_scale
            cy = float(k[5]) * self.image_scale
            u = fx * x / z + cx
            v = fy * y / z + cy
            uv_px = [float(u), float(v)]
            uv_norm = [
                float(u / max(image_width - 1, 1)),
                float(v / max(image_height - 1, 1)),
            ]
            visible = 0.0 <= u < image_width and 0.0 <= v < image_height

        return {
            "frame": object_frame,
            "xyz_camera_m": [x, y, z],
            "in_front": bool(in_front),
            "visible": bool(visible),
            "uv_px": uv_px,
            "uv_norm": uv_norm,
        }

    def _base_labels(self, task_spec: TaskSpec) -> dict[str, Any] | None:
        try:
            port_tf = self.tf_buffer.lookup_transform("base_link", task_spec.port_frame, Time())
            plug_tf = self.tf_buffer.lookup_transform("base_link", task_spec.plug_frame, Time())
        except TransformException as exc:
            self.get_logger().debug(f"Base TF unavailable: {exc}")
            return None

        p = port_tf.transform.translation
        q = plug_tf.transform.translation
        port_xyz = [float(p.x), float(p.y), float(p.z)]
        plug_xyz = [float(q.x), float(q.y), float(q.z)]
        return {
            "port_xyz_m": port_xyz,
            "plug_xyz_m": plug_xyz,
            "delta_port_minus_plug_m": [
                port_xyz[0] - plug_xyz[0],
                port_xyz[1] - plug_xyz[1],
                port_xyz[2] - plug_xyz[2],
            ],
        }

    def _image_to_rgb(self, img_msg: Any) -> np.ndarray:
        channels = 3
        row_bytes = img_msg.width * channels
        raw = np.frombuffer(img_msg.data, dtype=np.uint8)
        if img_msg.step and img_msg.step != row_bytes:
            arr = raw.reshape(img_msg.height, img_msg.step)[:, :row_bytes]
            arr = arr.reshape(img_msg.height, img_msg.width, channels)
        else:
            arr = raw.reshape(img_msg.height, img_msg.width, channels)
        return np.ascontiguousarray(arr)

    # ------------------------------------------------------------------- manifest

    def _write_manifest(self, status: str) -> None:
        manifest = {
            "status": status,
            "schema": "aic_visual_servo_labels_v1",
            "root": str(self.root),
            "image_scale": self.image_scale,
            "cameras": self.cameras,
            "require_visible_cameras": sorted(self.require_visible_cameras),
            "min_frame_index": self.min_frame_index,
            "sample_every": self.sample_every,
            "num_episodes_target": self.num_episodes,
            "episodes_completed": self.episodes_completed,
            "episodes_without_labels": self.episodes_without_labels,
            "total_labels": self.total_labels,
            "skipped_stale_action": self.skipped_stale_action,
            "skipped_unlabeled": self.skipped_unlabeled,
            "skipped_visibility": self.skipped_visibility,
            "unsupported_command_count": self.unsupported_command_count,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "note": (
                "TF-derived labels are training-only. Runtime/scoring may use "
                "only official observations and learned weights."
            ),
        }
        tmp_path = self.manifest_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(self.manifest_path)

    def close(self) -> None:
        self._write_manifest(status="complete")
        self._labels_f.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trials-config", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--episode-idle-timeout", type=float, default=2.0)
    parser.add_argument("--max-action-age", type=float, default=0.25)
    parser.add_argument(
        "--min-frame-index",
        type=int,
        default=120,
        help="Only label frames at or after this per-episode observation index.",
    )
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--image-scale", type=float, default=IMAGE_SCALE)
    parser.add_argument(
        "--cameras",
        default="center",
        help="Comma-separated camera names from: left,center,right.",
    )
    parser.add_argument(
        "--require-visible",
        default="center",
        help=(
            "Comma-separated subset of cameras that must have both port and plug "
            "projected inside the image. Empty string disables this filter."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--max-labels-per-episode",
        type=int,
        default=0,
        help="Optional cap; 0 means unlimited.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output root before recording.",
    )
    args = parser.parse_args()

    valid_cameras = {"left", "center", "right"}
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    require_visible = {c.strip() for c in args.require_visible.split(",") if c.strip()}
    unknown = (set(cameras) | require_visible) - valid_cameras
    if unknown:
        raise ValueError(f"Unsupported camera name(s): {sorted(unknown)}")
    if not cameras:
        raise ValueError("At least one camera must be selected.")
    if not require_visible.issubset(set(cameras)):
        raise ValueError("--require-visible must be a subset of --cameras")

    if args.root.exists() and args.overwrite:
        shutil.rmtree(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "images").mkdir(parents=True, exist_ok=True)
    if (args.root / "labels.jsonl").exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.root / 'labels.jsonl'} already exists. Use --overwrite or a new --root."
        )

    task_specs = _load_trial_task_specs(args.trials_config)
    print(
        f"Loaded {len(task_specs)} task specs from {args.trials_config}. "
        f"First 3: {task_specs[:3]}"
    )

    rclpy.init()
    recorder = VisualServoLabelRecorder(
        root=args.root,
        task_specs=task_specs,
        num_episodes=args.num_episodes,
        episode_idle_timeout=args.episode_idle_timeout,
        max_action_age=args.max_action_age,
        min_frame_index=args.min_frame_index,
        sample_every=args.sample_every,
        image_scale=args.image_scale,
        cameras=cameras,
        require_visible_cameras=require_visible,
        jpeg_quality=args.jpeg_quality,
        max_labels_per_episode=args.max_labels_per_episode,
    )
    try:
        while rclpy.ok() and recorder.episodes_completed < args.num_episodes:
            rclpy.spin_once(recorder, timeout_sec=0.1)
    except KeyboardInterrupt:
        recorder.get_logger().info("Interrupted by user.")
    finally:
        if recorder.episode_in_progress:
            recorder._end_episode()
        recorder.get_logger().info(
            f"Recorder shutting down. episodes_completed={recorder.episodes_completed} "
            f"total_labels={recorder.total_labels} "
            f"skipped_visibility={recorder.skipped_visibility} "
            f"skipped_unlabeled={recorder.skipped_unlabeled}"
        )
        recorder.close()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
