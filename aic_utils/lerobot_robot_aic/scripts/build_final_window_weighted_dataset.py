#!/usr/bin/env python3
"""Build a LeRobot dataset with extra final-window recovery clips.

This is a targeted weighting tool for the AIC insertion task. It preserves the
successful full demonstrations, then appends short suffix clips from each
episode as additional episodes. The goal is to make final alignment and
insertion frames a much larger share of the imitation loss without changing the
runtime policy interface.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _frame_for_writer(item: dict[str, Any], features: dict[str, dict]) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for key in features:
        if key in DEFAULT_FEATURES:
            continue
        frame[key] = item[key]
    frame["task"] = item["task"]
    return frame


def _episode_indices(src: LeRobotDataset) -> dict[int, list[int]]:
    by_episode: dict[int, list[int]] = defaultdict(list)
    for idx, episode_index in enumerate(src.hf_dataset["episode_index"]):
        by_episode[int(episode_index)].append(idx)
    return dict(sorted(by_episode.items()))


def _copy_episode(
    *,
    src: LeRobotDataset,
    dst: LeRobotDataset,
    indices: list[int],
    min_frames: int,
    progress_every: int,
    progress: dict[str, int],
) -> None:
    if len(indices) < min_frames:
        raise RuntimeError(f"Refusing to write {len(indices)}-frame episode.")

    for src_idx in indices:
        dst.add_frame(_frame_for_writer(src[src_idx], src.features))
        progress["frames"] += 1
        if progress_every and progress["frames"] % progress_every == 0:
            print(
                f"wrote {progress['frames']} frames across "
                f"{progress['episodes']} saved episodes",
                flush=True,
            )
    dst.save_episode()
    progress["episodes"] += 1


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {args.out_root}")
        shutil.rmtree(args.out_root)

    src = LeRobotDataset(
        args.source_repo_id,
        root=args.source_root,
        episodes=args.episodes,
        video_backend=args.video_backend,
    )
    grouped = _episode_indices(src)
    if not grouped:
        raise RuntimeError("Source dataset has no episodes.")

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

    manifest: dict[str, Any] = {
        "source_repo_id": args.source_repo_id,
        "source_root": str(args.source_root),
        "out_repo_id": args.out_repo_id,
        "out_root": str(args.out_root),
        "source_episode_count": len(grouped),
        "source_frame_count": len(src),
        "full_repeats": args.full_repeats,
        "final_window_frames": args.final_window_frames,
        "final_window_repeats": args.final_window_repeats,
        "clips": [],
    }
    progress = {"frames": 0, "episodes": 0}

    for repeat in range(args.full_repeats):
        for source_episode, indices in grouped.items():
            _copy_episode(
                src=src,
                dst=dst,
                indices=indices,
                min_frames=args.min_episode_frames,
                progress_every=args.progress_every,
                progress=progress,
            )
            manifest["clips"].append(
                {
                    "kind": "full",
                    "repeat": repeat,
                    "source_episode": source_episode,
                    "source_start_index": indices[0],
                    "source_end_index": indices[-1],
                    "frames": len(indices),
                }
            )

    for repeat in range(args.final_window_repeats):
        for source_episode, indices in grouped.items():
            window = indices[-min(args.final_window_frames, len(indices)) :]
            _copy_episode(
                src=src,
                dst=dst,
                indices=window,
                min_frames=args.min_episode_frames,
                progress_every=args.progress_every,
                progress=progress,
            )
            manifest["clips"].append(
                {
                    "kind": "final_window",
                    "repeat": repeat,
                    "source_episode": source_episode,
                    "source_start_index": window[0],
                    "source_end_index": window[-1],
                    "frames": len(window),
                }
            )

    dst.finalize()
    manifest["episodes_written"] = progress["episodes"]
    manifest["frames_written"] = progress["frames"]

    if args.manifest_json is not None:
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_json.open("w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-repo-id", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--full-repeats", type=int, default=1)
    parser.add_argument("--final-window-frames", type=int, default=180)
    parser.add_argument("--final-window-repeats", type=int, default=5)
    parser.add_argument("--min-episode-frames", type=int, default=30)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.full_repeats < 0 or args.final_window_repeats < 0:
        raise ValueError("Repeat counts must be non-negative.")
    if args.full_repeats == 0 and args.final_window_repeats == 0:
        raise ValueError("At least one repeat count must be positive.")
    if args.final_window_frames <= 0:
        raise ValueError("--final-window-frames must be positive.")

    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
