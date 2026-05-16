#!/usr/bin/env python3
"""Build a kept-only LeRobot dataset from scored source episodes.

The source recorder stores one LeRobot episode per AIC trial attempt. This tool
uses the scoring YAML as the authority, keeps only clean successful trials, and
replays those episodes into a new local LeRobot dataset with contiguous episode
indices.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _trial_index(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return 10**9


def _category_score(trial: dict[str, Any], category: str) -> float | None:
    node = trial.get("tier_2", {}).get("categories", {}).get(category)
    if not isinstance(node, dict) or "score" not in node:
        return None
    return float(node["score"])


def _load_keep_episodes(args: argparse.Namespace) -> tuple[list[int], list[dict[str, Any]]]:
    with args.scoring_yaml.open() as f:
        scoring = yaml.safe_load(f)

    trials = {
        name: value
        for name, value in scoring.items()
        if isinstance(name, str) and name.startswith("trial_")
    }
    keep: list[int] = []
    manifest_trials: list[dict[str, Any]] = []
    explicit = set(args.episodes) if args.episodes is not None else None

    for name, trial in sorted(trials.items(), key=lambda item: _trial_index(item[0])):
        ep_idx = _trial_index(name) - 1
        if explicit is not None and ep_idx not in explicit:
            continue

        tier_1 = float(trial.get("tier_1", {}).get("score", 0.0))
        tier_2 = float(trial.get("tier_2", {}).get("score", 0.0))
        tier_3 = float(trial.get("tier_3", {}).get("score", 0.0))
        total = tier_1 + tier_2 + tier_3
        contacts = _category_score(trial, "contacts")
        insertion_force = _category_score(trial, "insertion force")

        reasons = []
        if tier_3 < args.min_tier3:
            reasons.append("tier3_below_threshold")
        if total < args.min_total:
            reasons.append("total_below_threshold")
        if args.require_no_contacts and contacts not in (0.0, None):
            reasons.append("contact_penalty")
        if args.require_no_force_penalty and insertion_force not in (0.0, None):
            reasons.append("force_penalty")

        accepted = not reasons
        if accepted:
            keep.append(ep_idx)

        manifest_trials.append(
            {
                "trial": name,
                "source_episode": ep_idx,
                "accepted": accepted,
                "reasons": reasons,
                "total": total,
                "tier_3": tier_3,
                "contacts": contacts,
                "insertion_force": insertion_force,
                "tier_3_message": trial.get("tier_3", {}).get("message", ""),
            }
        )

    if args.max_episodes is not None:
        keep = keep[: args.max_episodes]
    return keep, manifest_trials


def _frame_for_writer(item: dict[str, Any], features: dict[str, dict]) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for key in features:
        if key in DEFAULT_FEATURES:
            continue
        frame[key] = item[key]
    frame["task"] = item["task"]
    return frame


def extract(args: argparse.Namespace) -> dict[str, Any]:
    keep, manifest_trials = _load_keep_episodes(args)
    if not keep:
        raise RuntimeError("No clean episodes passed the configured score gates.")

    if args.out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {args.out_root}")
        shutil.rmtree(args.out_root)

    src = LeRobotDataset(
        args.source_repo_id,
        root=args.source_root,
        episodes=keep,
        video_backend=args.video_backend,
    )
    dst = LeRobotDataset.create(
        args.out_repo_id,
        root=args.out_root,
        fps=src.meta.fps,
        robot_type=src.meta.robot_type,
        features=src.features,
        use_videos=True,
        video_backend=args.video_backend,
        vcodec=args.vcodec,
        metadata_buffer_size=1,
        image_writer_threads=args.image_writer_threads,
    )

    frames_written = 0
    episodes_written = 0
    current_source_episode: int | None = None
    frames_in_episode = 0

    for idx in range(len(src)):
        item = src[idx]
        source_episode = int(item["episode_index"])
        if current_source_episode is None:
            current_source_episode = source_episode
        elif source_episode != current_source_episode:
            dst.save_episode()
            episodes_written += 1
            if frames_in_episode < args.min_episode_frames:
                raise RuntimeError(
                    f"Copied source episode {current_source_episode} has only "
                    f"{frames_in_episode} frames."
                )
            current_source_episode = source_episode
            frames_in_episode = 0

        dst.add_frame(_frame_for_writer(item, src.features))
        frames_written += 1
        frames_in_episode += 1
        if args.progress_every and frames_written % args.progress_every == 0:
            print(
                f"copied {frames_written} frames across {episodes_written} saved "
                f"episodes; current source episode {source_episode}",
                flush=True,
            )

    if frames_in_episode:
        dst.save_episode()
        episodes_written += 1
        if frames_in_episode < args.min_episode_frames:
            raise RuntimeError(
                f"Copied source episode {current_source_episode} has only "
                f"{frames_in_episode} frames."
            )
    dst.finalize()

    report = {
        "source_repo_id": args.source_repo_id,
        "source_root": str(args.source_root),
        "scoring_yaml": str(args.scoring_yaml),
        "out_repo_id": args.out_repo_id,
        "out_root": str(args.out_root),
        "keep_episodes": keep,
        "num_kept": len(keep),
        "episodes_written": episodes_written,
        "frames_written": frames_written,
        "trials": manifest_trials,
    }
    if args.manifest_json is not None:
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_json.open("w") as f:
            json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scoring-yaml", type=Path, required=True)
    parser.add_argument("--out-repo-id", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--min-total", type=float, default=90.0)
    parser.add_argument("--min-tier3", type=float, default=75.0)
    parser.add_argument("--min-episode-frames", type=int, default=30)
    parser.add_argument("--require-no-contacts", action="store_true")
    parser.add_argument("--require-no-force-penalty", action="store_true")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = extract(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
