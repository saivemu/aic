#!/usr/bin/env python3
"""Generate exact-config-centered AIC engine trials for score-targeted collection.

This generator keeps the same three task identities as ``sample_config.yaml`` and
only jitters board pose, mounted component poses, and grasp pose by small amounts.
It is intended for CheatCode recovery-data collection, not broad randomization.

Usage:
    pixi run python aic_engine/config/gen_exact_jitter_trials.py \\
        --n 60 --seed 7 \\
        --out aic_engine/config/exact_jitter_smoke60.yaml
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SAMPLE_CONFIG = Path(__file__).with_name("sample_config.yaml")


def _jitter(rng: random.Random, value: float, half_range: float) -> float:
    if half_range <= 0.0:
        return value
    return value + rng.uniform(-half_range, half_range)


def _jitter_pose_value(
    rng: random.Random,
    node: dict[str, Any],
    key: str,
    half_range: float,
) -> None:
    if key in node:
        node[key] = _jitter(rng, float(node[key]), half_range)


def _jitter_trial(
    rng: random.Random,
    trial: dict[str, Any],
    board_xy_jitter_m: float,
    board_yaw_jitter_rad: float,
    rail_translation_jitter_m: float,
    rail_yaw_jitter_rad: float,
    grasp_xyz_jitter_m: float,
    grasp_rpy_jitter_rad: float,
) -> dict[str, Any]:
    out = copy.deepcopy(trial)

    board = out["scene"]["task_board"]
    board_pose = board["pose"]
    _jitter_pose_value(rng, board_pose, "x", board_xy_jitter_m)
    _jitter_pose_value(rng, board_pose, "y", board_xy_jitter_m)
    _jitter_pose_value(rng, board_pose, "yaw", board_yaw_jitter_rad)

    for rail_name, rail in board.items():
        if not rail_name.endswith("_rail_0") and not rail_name.endswith("_rail_1"):
            if not rail_name.startswith("nic_rail_"):
                continue
        if not isinstance(rail, dict) or not rail.get("entity_present"):
            continue
        pose = rail.get("entity_pose")
        if not isinstance(pose, dict):
            continue
        _jitter_pose_value(rng, pose, "translation", rail_translation_jitter_m)
        _jitter_pose_value(rng, pose, "yaw", rail_yaw_jitter_rad)

    for cable in out["scene"]["cables"].values():
        pose = cable["pose"]
        offset = pose["gripper_offset"]
        _jitter_pose_value(rng, offset, "x", grasp_xyz_jitter_m)
        _jitter_pose_value(rng, offset, "y", grasp_xyz_jitter_m)
        _jitter_pose_value(rng, offset, "z", grasp_xyz_jitter_m)
        _jitter_pose_value(rng, pose, "roll", grasp_rpy_jitter_rad)
        _jitter_pose_value(rng, pose, "pitch", grasp_rpy_jitter_rad)
        _jitter_pose_value(rng, pose, "yaw", grasp_rpy_jitter_rad)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60, help="Number of trials to emit.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-config", type=Path, default=DEFAULT_SAMPLE_CONFIG)
    parser.add_argument("--out", type=Path, default=Path("exact_jitter_trials.yaml"))
    parser.add_argument(
        "--include-exact-prefix",
        action="store_true",
        help="Emit one exact copy of sample trial_1..trial_3 before jittered trials.",
    )
    parser.add_argument("--board-xy-jitter-m", type=float, default=0.0075)
    parser.add_argument("--board-yaw-jitter-rad", type=float, default=0.04)
    parser.add_argument("--rail-translation-jitter-m", type=float, default=0.005)
    parser.add_argument("--rail-yaw-jitter-rad", type=float, default=0.03)
    parser.add_argument("--grasp-xyz-jitter-m", type=float, default=0.002)
    parser.add_argument("--grasp-rpy-jitter-rad", type=float, default=0.04)
    args = parser.parse_args()

    with args.sample_config.open() as f:
        sample = yaml.safe_load(f)

    sample_trials = sample["trials"]
    anchor_names = ["trial_1", "trial_2", "trial_3"]
    anchors = [sample_trials[name] for name in anchor_names]

    rng = random.Random(args.seed)
    trials: dict[str, Any] = {}
    idx = 1
    if args.include_exact_prefix:
        for anchor in anchors:
            trials[f"trial_{idx}"] = copy.deepcopy(anchor)
            idx += 1

    while idx <= args.n:
        anchor = anchors[(idx - 1) % len(anchors)]
        trials[f"trial_{idx}"] = _jitter_trial(
            rng,
            anchor,
            board_xy_jitter_m=args.board_xy_jitter_m,
            board_yaw_jitter_rad=args.board_yaw_jitter_rad,
            rail_translation_jitter_m=args.rail_translation_jitter_m,
            rail_yaw_jitter_rad=args.rail_yaw_jitter_rad,
            grasp_xyz_jitter_m=args.grasp_xyz_jitter_m,
            grasp_rpy_jitter_rad=args.grasp_rpy_jitter_rad,
        )
        idx += 1

    cfg = {
        "scoring": sample["scoring"],
        "task_board_limits": sample["task_board_limits"],
        "trials": trials,
        "robot": sample["robot"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {len(trials)} exact-jitter trials to {args.out}")


if __name__ == "__main__":
    main()
