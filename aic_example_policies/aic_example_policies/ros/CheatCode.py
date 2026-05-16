#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


import os

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

QuaternionTuple = tuple[float, float, float, float]


class CheatCode(Policy):
    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = float(
            os.environ.get("AIC_CHEATCODE_MAX_INTEGRATOR_WINDUP_M", "0.05")
        )
        self._xy_i_gain = float(os.environ.get("AIC_CHEATCODE_XY_I_GAIN", "0.15"))
        self._xy_dither_amp_m = float(
            os.environ.get("AIC_CHEATCODE_XY_DITHER_AMP_M", "0.0")
        )
        self._xy_dither_period_s = float(
            os.environ.get("AIC_CHEATCODE_XY_DITHER_PERIOD_S", "2.5")
        )
        self._xy_offset_max_m = float(
            os.environ.get("AIC_CHEATCODE_XY_OFFSET_MAX_M", "0.0")
        )
        self._xy_offset_decay_start_m = float(
            os.environ.get("AIC_CHEATCODE_XY_OFFSET_DECAY_START_M", "0.08")
        )
        self._xy_offset_decay_end_m = float(
            os.environ.get("AIC_CHEATCODE_XY_OFFSET_DECAY_END_M", "0.01")
        )
        self._descent_step_m = float(
            os.environ.get("AIC_CHEATCODE_DESCENT_STEP_M", "0.0005")
        )
        seed_raw = os.environ.get("AIC_CHEATCODE_XY_OFFSET_SEED", "").strip()
        self._offset_rng = (
            np.random.default_rng(int(seed_raw))
            if seed_raw
            else np.random.default_rng()
        )
        self._perturb_mode = os.environ.get(
            "AIC_CHEATCODE_PERTURB_MODE", "none"
        ).strip().lower()
        self._perturb_prob = float(
            os.environ.get("AIC_CHEATCODE_PERTURB_PROB", "1.0")
        )
        self._perturb_xy_min_m = float(
            os.environ.get("AIC_CHEATCODE_PERTURB_XY_MIN_M", "0.005")
        )
        self._perturb_xy_max_m = float(
            os.environ.get("AIC_CHEATCODE_PERTURB_XY_MAX_M", "0.025")
        )
        self._perturb_z_max_m = float(
            os.environ.get("AIC_CHEATCODE_PERTURB_Z_MAX_M", "0.0")
        )
        self._perturb_duration_s = float(
            os.environ.get("AIC_CHEATCODE_PERTURB_DURATION_S", "1.0")
        )
        self._final_perturb_trigger_z_m = float(
            os.environ.get("AIC_CHEATCODE_FINAL_PERTURB_TRIGGER_Z_M", "0.08")
        )
        perturb_seed_raw = os.environ.get("AIC_CHEATCODE_PERTURB_SEED", "").strip()
        self._perturb_rng = (
            np.random.default_rng(int(perturb_seed_raw))
            if perturb_seed_raw
            else np.random.default_rng()
        )
        self._perturbing = False
        self._perturb_pub = None
        self._task = None
        super().__init__(parent_node)
        self._perturb_pub = self._parent_node.create_publisher(
            Bool, "/aic/cheatcode/perturbing", 10
        )
        self._set_perturbing(False)
        if (
            self._xy_i_gain != 0.15
            or self._xy_dither_amp_m > 0.0
            or self._xy_offset_max_m > 0.0
            or self._descent_step_m != 0.0005
        ):
            self.get_logger().warn(
                "Training-only CheatCode alignment enrichment enabled: "
                f"xy_i_gain={self._xy_i_gain:.3f} "
                f"windup={self._max_integrator_windup:.3f}m "
                f"dither={self._xy_dither_amp_m * 1000:.1f}mm "
                f"offset_max={self._xy_offset_max_m * 1000:.1f}mm "
                f"descent_step={self._descent_step_m * 1000:.2f}mm"
            )
        if self._perturb_mode != "none":
            self.get_logger().warn(
                "Training-only CheatCode perturbation recovery enabled: "
                f"mode={self._perturb_mode} prob={self._perturb_prob:.2f} "
                f"xy=[{self._perturb_xy_min_m * 1000:.1f}, "
                f"{self._perturb_xy_max_m * 1000:.1f}]mm "
                f"z_max={self._perturb_z_max_m * 1000:.1f}mm "
                f"duration={self._perturb_duration_s:.2f}s"
            )

    def _set_perturbing(self, value: bool) -> None:
        self._perturbing = value
        if self._perturb_pub is None:
            return
        msg = Bool()
        msg.data = value
        self._perturb_pub.publish(msg)

    def _choose_perturb_stage(self) -> str | None:
        if self._perturb_mode in ("", "none", "off", "false", "0"):
            return None
        if float(self._perturb_rng.uniform(0.0, 1.0)) > self._perturb_prob:
            return None
        if self._perturb_mode in ("midcourse", "final"):
            return self._perturb_mode
        if self._perturb_mode == "mixed":
            return str(self._perturb_rng.choice(["midcourse", "final"]))
        self.get_logger().warn(
            f"Unknown AIC_CHEATCODE_PERTURB_MODE={self._perturb_mode!r}; "
            "perturbation disabled for this episode."
        )
        return None

    def _sample_perturbation(self) -> tuple[float, float, float]:
        lo = min(self._perturb_xy_min_m, self._perturb_xy_max_m)
        hi = max(self._perturb_xy_min_m, self._perturb_xy_max_m)
        radius = float(self._perturb_rng.uniform(lo, hi))
        angle = float(self._perturb_rng.uniform(0.0, 2.0 * np.pi))
        z = float(self._perturb_rng.uniform(0.0, max(self._perturb_z_max_m, 0.0)))
        return radius * np.cos(angle), radius * np.sin(angle), z

    def _combine_xy_offsets(
        self, lhs: tuple[float, float], rhs: tuple[float, float]
    ) -> tuple[float, float]:
        return lhs[0] + rhs[0], lhs[1] + rhs[1]

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        """Wait for a TF frame to become available."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... -- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
        xy_offset: tuple[float, float] = (0.0, 0.0),
    ) -> Pose:
        """Find the gripper pose that results in plug alignment."""
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )

        self.get_logger().info(
            f"pfrac: {position_fraction:.3} xy_error: {tip_x_error:0.3} {tip_y_error:0.3}   integrators: {self._tip_x_error_integrator:.3} , {self._tip_y_error_integrator:.3}"
        )

        i_gain = self._xy_i_gain

        target_x = port_xy[0] + xy_offset[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + xy_offset[1] + i_gain * self._tip_y_error_integrator
        if self._xy_dither_amp_m > 0.0 and not reset_xy_integrator:
            t = self.time_now().nanoseconds * 1e-9
            theta = 2.0 * np.pi * t / max(self._xy_dither_period_s, 1e-3)
            target_x += self._xy_dither_amp_m * np.cos(theta)
            target_y += self._xy_dither_amp_m * np.sin(theta)
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(
                x=blend_xyz[0],
                y=blend_xyz[1],
                z=blend_xyz[2],
            ),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"CheatCode.insert_cable() task: {task}")
        self._task = task

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        # Wait for both the port and cable tip TFs to become available.
        # These come via ground_truth and may not be immediate.
        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        z_offset = 0.2
        xy_offset = (0.0, 0.0)
        if self._xy_offset_max_m > 0.0:
            # Training-only exploration: start final insertion with a deliberate
            # lateral error, then decay it to zero while descending. This creates
            # directionally useful alignment labels without using ground truth at
            # runtime.
            angle = float(self._offset_rng.uniform(0.0, 2.0 * np.pi))
            radius = float(self._offset_rng.uniform(0.35, 1.0) * self._xy_offset_max_m)
            xy_offset = (radius * np.cos(angle), radius * np.sin(angle))
            self.get_logger().warn(
                "CheatCode xy offset episode: "
                f"dx={xy_offset[0] * 1000:.1f}mm dy={xy_offset[1] * 1000:.1f}mm"
            )

        perturb_stage = self._choose_perturb_stage()
        perturb_xyz = (0.0, 0.0, 0.0)
        perturb_steps = max(1, int(self._perturb_duration_s / 0.05))
        if perturb_stage is not None:
            perturb_xyz = self._sample_perturbation()
            self.get_logger().warn(
                "CheatCode perturbation episode: "
                f"stage={perturb_stage} "
                f"dx={perturb_xyz[0] * 1000:.1f}mm "
                f"dy={perturb_xyz[1] * 1000:.1f}mm "
                f"dz={perturb_xyz[2] * 1000:.1f}mm "
                f"steps={perturb_steps}"
            )

        # Over five seconds, smoothly interpolate from the current position to
        # a position above the port.
        midcourse_start_step = 50
        midcourse_end_step = midcourse_start_step + perturb_steps
        for t in range(0, 100):
            interp_fraction = t / 100.0
            midcourse_active = (
                perturb_stage == "midcourse"
                and midcourse_start_step <= t < midcourse_end_step
            )
            if midcourse_active and not self._perturbing:
                self._set_perturbing(True)
            elif not midcourse_active and self._perturbing:
                self._set_perturbing(False)
            pose_xy_offset = xy_offset
            pose_z_offset = z_offset
            if midcourse_active:
                pose_xy_offset = self._combine_xy_offsets(
                    pose_xy_offset, (perturb_xyz[0], perturb_xyz[1])
                )
                pose_z_offset += perturb_xyz[2]
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=pose_z_offset,
                        reset_xy_integrator=True,
                        xy_offset=pose_xy_offset,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(0.05)
        if self._perturbing:
            self._set_perturbing(False)

        # Descend until the cable is inserted into the port.
        final_perturb_started = False
        final_perturb_steps_remaining = 0
        while True:
            if z_offset < -0.015:
                break

            z_offset -= self._descent_step_m
            self.get_logger().info(f"z_offset: {z_offset:0.5}")
            try:
                if final_perturb_started and final_perturb_steps_remaining == 0:
                    if self._perturbing:
                        self._set_perturbing(False)

                if (
                    perturb_stage == "final"
                    and not final_perturb_started
                    and z_offset <= self._final_perturb_trigger_z_m
                ):
                    final_perturb_started = True
                    final_perturb_steps_remaining = perturb_steps
                    self._set_perturbing(True)

                decay_span = max(
                    self._xy_offset_decay_start_m - self._xy_offset_decay_end_m,
                    1e-6,
                )
                offset_fraction = np.clip(
                    (z_offset - self._xy_offset_decay_end_m) / decay_span,
                    0.0,
                    1.0,
                )
                decayed_xy_offset = (
                    xy_offset[0] * offset_fraction,
                    xy_offset[1] * offset_fraction,
                )
                pose_xy_offset = decayed_xy_offset
                pose_z_offset = z_offset
                if final_perturb_steps_remaining > 0:
                    pose_xy_offset = self._combine_xy_offsets(
                        pose_xy_offset, (perturb_xyz[0], perturb_xyz[1])
                    )
                    pose_z_offset += perturb_xyz[2]
                    final_perturb_steps_remaining -= 1
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        z_offset=pose_z_offset,
                        xy_offset=pose_xy_offset,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
            self.sleep_for(0.05)
        if self._perturbing:
            self._set_perturbing(False)

        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(5.0)

        self.get_logger().info("CheatCode.insert_cable() exiting...")
        return True
