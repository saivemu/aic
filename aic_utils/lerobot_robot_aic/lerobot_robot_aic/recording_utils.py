"""Helpers for converting recorded AIC controller targets into LeRobot actions."""

from __future__ import annotations

from typing import Any

import numpy as np

NS_PER_SECOND = 1_000_000_000
ACTION_DIM = 7


def stamp_to_nanoseconds(stamp: Any) -> int | None:
    """Return a ROS stamp as nanoseconds, or None for an unset stamp."""
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    ns = sec * NS_PER_SECOND + nanosec
    return ns if ns > 0 else None


def _normalized_quaternion_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too close to zero")
    return quat / norm


def quaternion_delta_to_angular_velocity(
    prev_quat_wxyz: np.ndarray,
    cur_quat_wxyz: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Return angular velocity from two wxyz quaternions over dt_s.

    The delta uses the shortest quaternion path so q and -q represent the same
    pose and do not create a spurious full-turn action.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")

    prev = _normalized_quaternion_wxyz(prev_quat_wxyz)
    cur = _normalized_quaternion_wxyz(cur_quat_wxyz)

    pw, px, py, pz = prev
    cw, cx, cy, cz = cur

    # q_delta = q_cur * q_prev^-1, Hamilton convention.
    dw = cw * pw + cx * px + cy * py + cz * pz
    dx = -cw * px + cx * pw - cy * pz + cz * py
    dy = -cw * py + cx * pz + cy * pw - cz * px
    dz = -cw * pz - cx * py + cy * px + cz * pw

    if dw < 0.0:
        dw, dx, dy, dz = -dw, -dx, -dy, -dz

    xyz_norm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if xyz_norm < 1e-9:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * float(np.arctan2(xyz_norm, dw))
    axis = np.array([dx, dy, dz], dtype=np.float64) / xyz_norm
    return (axis * angle / dt_s).astype(np.float32)


def pose_targets_to_action(
    prev_pos_xyz: np.ndarray,
    prev_quat_wxyz: np.ndarray,
    prev_time_ns: int,
    cur_pos_xyz: np.ndarray,
    cur_quat_wxyz: np.ndarray,
    cur_time_ns: int,
    min_dt_s: float = 1e-3,
) -> np.ndarray:
    """Convert two position-mode MotionUpdate pose targets into a 7-D action."""
    raw_dt_s = (int(cur_time_ns) - int(prev_time_ns)) / NS_PER_SECOND
    dt_s = max(min_dt_s, raw_dt_s)

    prev_pos = np.asarray(prev_pos_xyz, dtype=np.float64)
    cur_pos = np.asarray(cur_pos_xyz, dtype=np.float64)
    lin_v = (cur_pos - prev_pos) / dt_s
    omega = quaternion_delta_to_angular_velocity(prev_quat_wxyz, cur_quat_wxyz, dt_s)

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0:3] = lin_v.astype(np.float32)
    action[3:6] = omega
    return action
