#!/usr/bin/env python3
"""Audit an AIC LeRobot dataset before training.

This script checks the parts of the dataset that matter most for ACT training:
episode lengths, zero-action fraction, high-speed action spikes, and simple TCP
path statistics. It reads table columns directly when available, so it does not
decode camera videos unless it has to fall back to item-by-item loading.

Usage:
  pixi run python aic_utils/lerobot_robot_aic/scripts/audit_dataset.py \\
      --dataset-repo-id ${HF_USER}/aic_act_v1 \\
      --dataset-root ~/.cache/huggingface/lerobot/${HF_USER}/aic_act_v1 \\
      --output-json outputs/aic_act_v1_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ACTION_NAMES = ("lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z", "pad")


def to_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype) if dtype is not None else array


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": 0.0, "p05": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def get_table_column(dataset: LeRobotDataset, key: str) -> np.ndarray | None:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None or key not in getattr(hf_dataset, "column_names", []):
        return None
    return to_numpy(hf_dataset[key])


def get_column(dataset: LeRobotDataset, key: str) -> np.ndarray:
    column = get_table_column(dataset, key)
    if column is not None:
        return column

    values = []
    for index in range(dataset.num_frames):
        sample = dataset[index]
        values.append(to_numpy(sample[key]))
    return np.stack(values)


def get_episode_indices(dataset: LeRobotDataset) -> np.ndarray:
    column = get_table_column(dataset, "episode_index")
    if column is not None:
        return column.astype(np.int64)

    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is None:
        return np.zeros(dataset.num_frames, dtype=np.int64)

    starts = to_numpy(episode_data_index["from"], dtype=np.int64)
    ends = to_numpy(episode_data_index["to"], dtype=np.int64)
    out = np.zeros(dataset.num_frames, dtype=np.int64)
    for episode, (start, end) in enumerate(zip(starts, ends, strict=True)):
        out[start:end] = episode
    return out


def summarize_dataset(
    dataset: LeRobotDataset,
    max_linear_speed: float,
    max_angular_speed: float,
    zero_action_threshold: float,
) -> dict[str, Any]:
    actions = get_column(dataset, "action").astype(np.float32)
    states = get_column(dataset, "observation.state").astype(np.float32)
    episode_indices = get_episode_indices(dataset)

    linear_speed = np.linalg.norm(actions[:, 0:3], axis=1)
    angular_speed = np.linalg.norm(actions[:, 3:6], axis=1)
    action_norm = np.linalg.norm(actions[:, 0:6], axis=1)

    episode_ids = np.unique(episode_indices)
    episode_lengths = np.array(
        [int(np.sum(episode_indices == episode)) for episode in episode_ids],
        dtype=np.int64,
    )

    tcp_path_lengths = []
    tcp_final_displacements = []
    tcp_positions = states[:, 0:3]
    for episode in episode_ids:
        positions = tcp_positions[episode_indices == episode]
        if len(positions) < 2:
            tcp_path_lengths.append(0.0)
            tcp_final_displacements.append(0.0)
            continue
        deltas = np.diff(positions, axis=0)
        tcp_path_lengths.append(float(np.linalg.norm(deltas, axis=1).sum()))
        tcp_final_displacements.append(float(np.linalg.norm(positions[-1] - positions[0])))

    action_stats = {}
    for idx, name in enumerate(ACTION_NAMES):
        action_stats[name] = percentile_summary(actions[:, idx])

    return {
        "num_episodes": int(len(episode_ids)),
        "num_frames": int(actions.shape[0]),
        "episode_lengths": percentile_summary(episode_lengths),
        "action": action_stats,
        "linear_speed_norm": percentile_summary(linear_speed),
        "angular_speed_norm": percentile_summary(angular_speed),
        "zero_action_fraction": float(np.mean(action_norm <= zero_action_threshold)),
        "linear_speed_spike_fraction": float(np.mean(linear_speed > max_linear_speed)),
        "angular_speed_spike_fraction": float(np.mean(angular_speed > max_angular_speed)),
        "tcp_path_length_m": percentile_summary(np.array(tcp_path_lengths)),
        "tcp_final_displacement_m": percentile_summary(np.array(tcp_final_displacements)),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Episodes: {summary['num_episodes']}  Frames: {summary['num_frames']}")
    print(f"Episode length: {summary['episode_lengths']}")
    print(f"Zero-action fraction: {summary['zero_action_fraction']:.4f}")
    print(
        "Speed spike fractions: "
        f"linear={summary['linear_speed_spike_fraction']:.4f}, "
        f"angular={summary['angular_speed_spike_fraction']:.4f}"
    )
    print(f"Linear speed norm: {summary['linear_speed_norm']}")
    print(f"Angular speed norm: {summary['angular_speed_norm']}")
    print(f"TCP path length (m): {summary['tcp_path_length_m']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--max-linear-speed", type=float, default=1.0)
    parser.add_argument("--max-angular-speed", type=float, default=4.0)
    parser.add_argument("--zero-action-threshold", type=float, default=1e-5)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
    )
    summary = summarize_dataset(
        dataset,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        zero_action_threshold=args.zero_action_threshold,
    )

    print_summary(summary)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote audit summary to {args.output_json}")


if __name__ == "__main__":
    main()
