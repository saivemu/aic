#!/usr/bin/env python3
"""Train a compact rectified-flow final-stage policy on AIC LeRobot data."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from itertools import count
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_aic.flow_policy import (
    FlowPolicyConfig,
    RectifiedFlowActionModel,
    normalize_action,
    normalize_images,
    normalize_state,
    save_checkpoint,
)


CAMERA_KEYS = (
    "observation.images.left_camera",
    "observation.images.center_camera",
    "observation.images.right_camera",
)


def parse_weights(raw: str, dim: int) -> torch.Tensor:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) != dim:
        raise ValueError(f"Expected {dim} comma-separated weights, got {len(values)}: {raw!r}")
    return torch.tensor(values, dtype=torch.float32)


def stack_column(dataset: LeRobotDataset, key: str) -> torch.Tensor:
    return torch.stack(
        [torch.as_tensor(value, dtype=torch.float32) for value in dataset.hf_dataset[key]],
        dim=0,
    )


def compute_stats(dataset: LeRobotDataset) -> dict[str, torch.Tensor]:
    state = stack_column(dataset, "observation.state")
    action = stack_column(dataset, "action")
    return {
        "state_mean": state.mean(dim=0),
        "state_std": state.std(dim=0, unbiased=False),
        "action_mean": action.mean(dim=0),
        "action_std": action.std(dim=0, unbiased=False),
    }


def episode_groups(dataset: LeRobotDataset) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, episode_index in enumerate(dataset.hf_dataset["episode_index"]):
        grouped[int(episode_index)].append(idx)
    return dict(sorted(grouped.items()))


class FlowChunkDataset(Dataset):
    def __init__(self, dataset: LeRobotDataset, chunk_len: int):
        self.dataset = dataset
        self.chunk_len = chunk_len
        self.actions = stack_column(dataset, "action")
        self.valid_starts: list[int] = []
        grouped = episode_groups(dataset)
        for indices in grouped.values():
            if len(indices) < chunk_len:
                continue
            for offset in range(0, len(indices) - chunk_len + 1):
                self.valid_starts.append(indices[offset])
        if not self.valid_starts:
            raise RuntimeError(f"No valid {chunk_len}-frame chunks found.")

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.valid_starts[index]
        frame = self.dataset[start]
        images = torch.stack(
            [
                torch.as_tensor(frame[key], dtype=torch.float32)
                for key in CAMERA_KEYS
            ],
            dim=0,
        )
        return {
            "images": images,
            "state": torch.as_tensor(frame["observation.state"], dtype=torch.float32),
            "action_chunk": self.actions[start : start + self.chunk_len],
        }


def infinite_loader(loader: DataLoader):
    for _ in count():
        yield from loader


def train(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir already exists: {args.out_dir}")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=args.episodes,
        video_backend=args.video_backend,
    )
    chunk_dataset = FlowChunkDataset(dataset, args.chunk_len)
    stats = compute_stats(dataset)

    state_dim = int(stats["state_mean"].numel())
    action_dim = int(stats["action_mean"].numel())
    cfg = FlowPolicyConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_len=args.chunk_len,
        image_height=args.image_height,
        image_width=args.image_width,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        flow_steps=args.flow_steps,
        replan_every=args.replan_every,
        zero_start=not args.random_start_inference,
    )
    model = RectifiedFlowActionModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    action_weights = parse_weights(args.action_weights, action_dim).to(device).view(1, 1, -1)
    stats_device = {k: v.to(device).view(1, -1) for k, v in stats.items()}

    loader = DataLoader(
        chunk_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if len(loader) == 0:
        raise RuntimeError("Training loader is empty; lower --batch-size or --chunk-len.")
    batches = infinite_loader(loader)

    train_config: dict[str, Any] = {
        **vars(args),
        "dataset_frames": int(dataset.num_frames),
        "dataset_episodes": int(dataset.num_episodes),
        "valid_chunks": int(len(chunk_dataset)),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "device": str(device),
    }
    with (args.out_dir / "train_config.json").open("w") as f:
        json.dump(train_config, f, indent=2, default=str)

    print(
        f"Training flow policy on {dataset.num_frames} frames, "
        f"{dataset.num_episodes} episodes, {len(chunk_dataset)} chunks "
        f"device={device} batch={args.batch_size} steps={args.steps}",
        flush=True,
    )
    ema_loss = None
    model.train()
    for step in range(1, args.steps + 1):
        batch = next(batches)
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        action_chunk = batch["action_chunk"].to(device, non_blocking=True)

        images = normalize_images(images, cfg)
        state_norm = normalize_state(state, stats_device)
        x1 = normalize_action(action_chunk, stats_device)

        if args.noise_start_prob > 0.0:
            noise_mask = (
                torch.rand((x1.shape[0], 1, 1), device=device) < args.noise_start_prob
            )
            x0 = torch.where(noise_mask, torch.randn_like(x1), torch.zeros_like(x1))
        else:
            x0 = torch.zeros_like(x1)
        t = torch.rand((x1.shape[0],), device=device)
        view_shape = (x1.shape[0],) + (1,) * (x1.dim() - 1)
        x_t = (1.0 - t.view(view_shape)) * x0 + t.view(view_shape) * x1
        target_v = x1 - x0

        pred_v = model(x_t, t, images, state_norm)
        loss = ((pred_v - target_v).square() * action_weights).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        loss_value = float(loss.detach().cpu())
        ema_loss = loss_value if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_value
        if step == 1 or step % args.log_freq == 0:
            print(
                f"step={step:06d} loss={loss_value:.6f} ema={ema_loss:.6f} "
                f"lr={opt.param_groups[0]['lr']:.2e}",
                flush=True,
            )
        if step % args.save_freq == 0 or step == args.steps:
            ckpt_dir = args.out_dir / "checkpoints" / f"{step:06d}"
            save_checkpoint(
                ckpt_dir,
                model,
                cfg,
                stats,
                step=step,
                extra={
                    "loss": loss_value,
                    "ema_loss": float(ema_loss),
                    "action_weights": args.action_weights,
                    "noise_start_prob": args.noise_start_prob,
                },
            )
            print(f"saved {ckpt_dir / 'flow_policy.pt'}", flush=True)

    if math.isfinite(float(ema_loss)):
        (args.out_dir / "last_loss.txt").write_text(f"{ema_loss:.8f}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-repo-id", required=True)
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--episodes", type=int, nargs="+", default=None)
    p.add_argument("--video-backend", default="pyav")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--save-freq", type=int, default=500)
    p.add_argument("--log-freq", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--chunk-len", type=int, default=16)
    p.add_argument("--image-height", type=int, default=128)
    p.add_argument("--image-width", type=int, default=144)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--cond-dim", type=int, default=512)
    p.add_argument("--flow-steps", type=int, default=4)
    p.add_argument("--replan-every", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--noise-start-prob", type=float, default=0.25)
    p.add_argument("--random-start-inference", action="store_true")
    p.add_argument(
        "--action-weights",
        default="1.0,1.0,1.0,0.25,0.25,0.25,0.05",
        help="Comma-separated MSE weights for the normalized action dimensions.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
