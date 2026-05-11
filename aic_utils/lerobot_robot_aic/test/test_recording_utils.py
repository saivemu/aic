import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parents[1] / "lerobot_robot_aic" / "recording_utils.py"
_SPEC = importlib.util.spec_from_file_location("recording_utils", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
recording_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recording_utils)

pose_targets_to_action = recording_utils.pose_targets_to_action
quaternion_delta_to_angular_velocity = recording_utils.quaternion_delta_to_angular_velocity
stamp_to_nanoseconds = recording_utils.stamp_to_nanoseconds


def yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    half = yaw_rad / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def test_pose_targets_to_action_differentiates_linear_and_angular_motion():
    action = pose_targets_to_action(
        prev_pos_xyz=np.array([0.0, 0.0, 0.0]),
        prev_quat_wxyz=yaw_quat_wxyz(0.0),
        prev_time_ns=0,
        cur_pos_xyz=np.array([0.1, -0.2, 0.04]),
        cur_quat_wxyz=yaw_quat_wxyz(0.2),
        cur_time_ns=500_000_000,
    )

    assert action.shape == (7,)
    np.testing.assert_allclose(action[0:3], [0.2, -0.4, 0.08], rtol=1e-6)
    np.testing.assert_allclose(action[3:6], [0.0, 0.0, 0.4], atol=1e-6)
    assert action[6] == 0.0


def test_quaternion_delta_uses_shortest_path_for_equivalent_signs():
    omega = quaternion_delta_to_angular_velocity(
        prev_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        cur_quat_wxyz=np.array([-1.0, 0.0, 0.0, 0.0]),
        dt_s=0.05,
    )

    np.testing.assert_allclose(omega, [0.0, 0.0, 0.0], atol=1e-8)


def test_pose_targets_to_action_clamps_tiny_timestamp_delta():
    action = pose_targets_to_action(
        prev_pos_xyz=np.array([0.0, 0.0, 0.0]),
        prev_quat_wxyz=yaw_quat_wxyz(0.0),
        prev_time_ns=100,
        cur_pos_xyz=np.array([0.001, 0.0, 0.0]),
        cur_quat_wxyz=yaw_quat_wxyz(0.0),
        cur_time_ns=100,
    )

    np.testing.assert_allclose(action[0:3], [1.0, 0.0, 0.0], rtol=1e-6)


def test_quaternion_delta_rejects_bad_inputs():
    with pytest.raises(ValueError):
        quaternion_delta_to_angular_velocity(
            prev_quat_wxyz=np.array([0.0, 0.0, 0.0, 0.0]),
            cur_quat_wxyz=yaw_quat_wxyz(0.0),
            dt_s=1.0,
        )

    with pytest.raises(ValueError):
        quaternion_delta_to_angular_velocity(
            prev_quat_wxyz=yaw_quat_wxyz(0.0),
            cur_quat_wxyz=yaw_quat_wxyz(0.0),
            dt_s=0.0,
        )


def test_stamp_to_nanoseconds_returns_none_for_unset_stamp():
    assert stamp_to_nanoseconds(SimpleNamespace(sec=12, nanosec=34)) == 12_000_000_034
    assert stamp_to_nanoseconds(SimpleNamespace(sec=0, nanosec=0)) is None
