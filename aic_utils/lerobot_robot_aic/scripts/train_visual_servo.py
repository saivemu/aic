#!/usr/bin/env python3
"""Train a small visual-servo model from TF-labeled image data.

The input dataset is produced by ``record_visual_servo_dataset.py``. The model
uses the legal center camera image plus the 43-D observation state. It can train
either a regressor or a discrete xy direction classifier. At runtime, the
controller can use the output as a bounded final correction without accessing TF
or simulator state.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


STATE_DIM = 43
TARGET_SCALE_MM = 1000.0
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _load_rows(
    root: Path,
    camera: str,
    max_samples: int | None,
    max_xy_target_m: float,
    min_xy_target_m: float,
    min_frame_index: int,
    target_mode: str,
) -> list[dict[str, Any]]:
    labels_path = root / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    rows: list[dict[str, Any]] = []
    with labels_path.open() as f:
        for line in f:
            row = json.loads(line)
            image_rel = row.get("images", {}).get(camera)
            labels = row.get("cameras", {}).get(camera)
            if image_rel is None or labels is None:
                continue
            if not labels["port"]["visible"] or not labels["plug"]["visible"]:
                continue
            image_path = root / image_rel
            if not image_path.exists():
                continue
            state = row.get("state", [])
            if len(state) != STATE_DIM:
                continue
            delta = row.get("base", {}).get("delta_port_minus_plug_m")
            if delta is None or len(delta) != 3:
                continue
            action = row.get("action")
            if target_mode == "action_linear" and (action is None or len(action) < 3):
                continue
            if row.get("frame_index", 0) < min_frame_index:
                continue
            xy_norm = float((delta[0] ** 2 + delta[1] ** 2) ** 0.5)
            if max_xy_target_m > 0.0:
                if xy_norm > max_xy_target_m:
                    continue
            if min_xy_target_m > 0.0 and xy_norm < min_xy_target_m:
                continue
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No usable rows found in {labels_path}")
    return rows


class VisualServoDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, Any]],
        camera: str,
        image_width: int,
        image_height: int,
        augment: bool,
        target_mode: str,
        target_scale: float,
        direction_deadband_m: float,
    ) -> None:
        self.root = root
        self.rows = rows
        self.camera = camera
        self.image_width = image_width
        self.image_height = image_height
        self.augment = augment
        self.target_mode = target_mode
        self.target_scale = target_scale
        self.direction_deadband_m = direction_deadband_m

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        image = cv2.imread(str(self.root / row["images"][self.camera]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.root / row["images"][self.camera])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_AREA,
        )

        if self.augment:
            # Photometric jitter only. Geometric flips would change base-frame
            # target signs, so keep geometry fixed.
            image_f = image.astype(np.float32)
            contrast = np.random.uniform(0.85, 1.15)
            brightness = np.random.uniform(-12.0, 12.0)
            image = np.clip(image_f * contrast + brightness, 0, 255).astype(np.uint8)

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - IMAGE_MEAN) / IMAGE_STD
        state_t = torch.tensor(row["state"], dtype=torch.float32)
        if self.target_mode == "delta":
            target = row["base"]["delta_port_minus_plug_m"]
            target_t = torch.tensor(target, dtype=torch.float32)
            target_t = target_t * self.target_scale
        elif self.target_mode == "action_linear":
            target = row["action"][:3]
            target_t = torch.tensor(target, dtype=torch.float32)
            target_t = target_t * self.target_scale
        elif self.target_mode == "pixel_delta":
            target = row["cameras"][self.camera]["delta_port_minus_plug_px"]
            target_t = torch.tensor(target[:2], dtype=torch.float32)
            target_t = target_t * self.target_scale
        elif self.target_mode == "xy_direction":
            delta = row["base"]["delta_port_minus_plug_m"]
            target_t = torch.tensor(
                [
                    self._direction_class(float(delta[0])),
                    self._direction_class(float(delta[1])),
                ],
                dtype=torch.long,
            )
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode!r}")
        return {"image": image_t, "state": state_t, "target": target_t}

    def _direction_class(self, value_m: float) -> int:
        if value_m < -self.direction_deadband_m:
            return 0
        if value_m > self.direction_deadband_m:
            return 2
        return 1


class VisualServoNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, output_dim: int = 3) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        state_features = self.state_encoder(state)
        return self.head(torch.cat([image_features, state_features], dim=1))


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "state": torch.stack([b["state"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
    }


def _split_rows(
    rows: list[dict[str, Any]],
    val_fraction: float,
    split_by_episode: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not split_by_episode:
        random.shuffle(rows)
        val_count = max(1, int(len(rows) * val_fraction))
        return rows[val_count:], rows[:val_count]

    episode_ids = sorted({int(row["episode_index"]) for row in rows})
    random.shuffle(episode_ids)
    val_episode_count = max(1, int(len(episode_ids) * val_fraction))
    val_episodes = set(episode_ids[:val_episode_count])
    val_rows = [row for row in rows if int(row["episode_index"]) in val_episodes]
    train_rows = [row for row in rows if int(row["episode_index"]) not in val_episodes]
    return train_rows, val_rows


def _direction_class_counts(
    rows: list[dict[str, Any]],
    direction_deadband_m: float,
) -> torch.Tensor:
    counts = torch.zeros((2, 3), dtype=torch.float32)
    for row in rows:
        delta = row["base"]["delta_port_minus_plug_m"]
        for axis in (0, 1):
            value_m = float(delta[axis])
            if value_m < -direction_deadband_m:
                cls = 0
            elif value_m > direction_deadband_m:
                cls = 2
            else:
                cls = 1
            counts[axis, cls] += 1.0
    return counts


def _direction_class_weights(
    rows: list[dict[str, Any]],
    direction_deadband_m: float,
    device: torch.device,
) -> torch.Tensor:
    counts = _direction_class_counts(rows, direction_deadband_m)
    weights = counts.sum(dim=1, keepdim=True) / (counts.clamp_min(1.0) * 3.0)
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return weights.to(device)


def _fit_pixel_to_base_xy(
    rows: list[dict[str, Any]],
    camera: str,
) -> list[list[float]]:
    features = []
    targets = []
    for row in rows:
        delta_px = row["cameras"][camera]["delta_port_minus_plug_px"]
        delta_base = row["base"]["delta_port_minus_plug_m"]
        features.append([float(delta_px[0]), float(delta_px[1]), 1.0])
        targets.append([float(delta_base[0]), float(delta_base[1])])
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    # Shape: base_xy = matrix @ [delta_px_x, delta_px_y, 1].
    return coef.T.astype(float).tolist()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mode: str,
) -> dict[str, float]:
    model.eval()
    losses = []
    if target_mode == "xy_direction":
        axis_correct = []
        pair_correct = []
        nonzero_axis_correct = []
        nonzero_axis_count = 0
        ce_loss = nn.CrossEntropyLoss(reduction="none")
        for batch in loader:
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            target = batch["target"].to(device)
            logits = model(image, state).view(-1, 2, 3)
            losses.append(
                0.5
                * (
                    ce_loss(logits[:, 0, :], target[:, 0])
                    + ce_loss(logits[:, 1, :], target[:, 1])
                ).cpu()
            )
            pred_cls = torch.argmax(logits, dim=2)
            correct = pred_cls.eq(target)
            axis_correct.append(correct.float().cpu().reshape(-1))
            pair_correct.append(correct.all(dim=1).float().cpu())
            nonzero_mask = target.ne(1)
            if bool(nonzero_mask.any()):
                nonzero_axis_correct.append(correct[nonzero_mask].float().cpu())
                nonzero_axis_count += int(nonzero_mask.sum().item())
        loss = torch.cat(losses, dim=0)
        axis = torch.cat(axis_correct, dim=0)
        pair = torch.cat(pair_correct, dim=0)
        if nonzero_axis_correct:
            nonzero_axis = torch.cat(nonzero_axis_correct, dim=0)
            nonzero_acc = float(nonzero_axis.mean().item())
        else:
            nonzero_acc = float("nan")
        return {
            "loss": float(loss.mean().item()),
            "axis_acc": float(axis.mean().item()),
            "pair_acc": float(pair.mean().item()),
            "nonzero_axis_acc": nonzero_acc,
            "nonzero_axis_count": float(nonzero_axis_count),
        }

    abs_errors = []
    loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="none")
    for batch in loader:
        image = batch["image"].to(device)
        state = batch["state"].to(device)
        target = batch["target"].to(device)
        pred = model(image, state)
        losses.append(loss_fn(pred, target).mean(dim=1).cpu())
        abs_errors.append((pred - target).abs().cpu())

    err = torch.cat(abs_errors, dim=0)
    loss = torch.cat(losses, dim=0)
    xy_norm = torch.linalg.norm(err[:, :2], dim=1)
    if target_mode == "pixel_delta":
        return {
            "loss": float(loss.mean().item()),
            "mae_x_px": float(err[:, 0].mean().item()),
            "mae_y_px": float(err[:, 1].mean().item()),
            "mae_xy_norm_px": float(xy_norm.mean().item()),
            "p50_xy_norm_px": float(torch.quantile(xy_norm, 0.50).item()),
            "p90_xy_norm_px": float(torch.quantile(xy_norm, 0.90).item()),
            "within_5px": float((xy_norm < 5.0).float().mean().item()),
            "within_10px": float((xy_norm < 10.0).float().mean().item()),
        }
    return {
        "loss": float(loss.mean().item()),
        "mae_x_mm": float(err[:, 0].mean().item()),
        "mae_y_mm": float(err[:, 1].mean().item()),
        "mae_z_mm": float(err[:, 2].mean().item()),
        "mae_xy_norm_mm": float(xy_norm.mean().item()),
        "p50_xy_norm_mm": float(torch.quantile(xy_norm, 0.50).item()),
        "p90_xy_norm_mm": float(torch.quantile(xy_norm, 0.90).item()),
        "within_2mm": float((xy_norm < 2.0).float().mean().item()),
        "within_5mm": float((xy_norm < 5.0).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", default="center")
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--target",
        choices=("delta", "action_linear", "xy_direction", "pixel_delta"),
        default="delta",
        help=(
            "delta predicts base.delta_port_minus_plug_m; action_linear predicts "
            "the recorded linear xyz action in m/s; xy_direction classifies "
            "base-frame x/y correction signs; pixel_delta predicts center-image "
            "port-minus-plug pixel offset."
        ),
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=TARGET_SCALE_MM,
        help="Scale applied to target values. 1000 makes m or m/s into mm or mm/s.",
    )
    parser.add_argument("--xy-weight", type=float, default=3.0)
    parser.add_argument("--z-weight", type=float, default=1.0)
    parser.add_argument(
        "--direction-deadband-m",
        type=float,
        default=0.0015,
        help="Deadband for xy_direction classes. Values inside this are hold.",
    )
    parser.add_argument(
        "--direction-speed-mps",
        type=float,
        default=0.006,
        help="Runtime fixed xy speed saved into xy_direction checkpoints.",
    )
    parser.add_argument(
        "--no-direction-class-balance",
        action="store_true",
        help="Disable inverse-frequency class weights for xy_direction.",
    )
    parser.add_argument(
        "--split-by-episode",
        action="store_true",
        help="Hold out whole episodes for validation instead of random frames.",
    )
    parser.add_argument(
        "--max-xy-target-m",
        type=float,
        default=0.0,
        help="Optional filter on |base xy port-minus-plug|. 0 disables.",
    )
    parser.add_argument(
        "--min-xy-target-m",
        type=float,
        default=0.0,
        help="Optional lower filter on |base xy port-minus-plug|. 0 disables.",
    )
    parser.add_argument(
        "--min-frame-index",
        type=int,
        default=0,
        help="Optional per-episode frame-index filter.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = _load_rows(
        args.root,
        camera=args.camera,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        max_xy_target_m=args.max_xy_target_m,
        min_xy_target_m=args.min_xy_target_m,
        min_frame_index=args.min_frame_index,
        target_mode=args.target,
    )
    train_rows, val_rows = _split_rows(
        rows,
        val_fraction=args.val_fraction,
        split_by_episode=args.split_by_episode,
    )
    if not train_rows:
        raise ValueError("Not enough rows for a train split.")
    if not val_rows:
        raise ValueError("Not enough rows for a validation split.")

    train_ds = VisualServoDataset(
        args.root,
        train_rows,
        args.camera,
        args.image_width,
        args.image_height,
        augment=True,
        target_mode=args.target,
        target_scale=args.target_scale,
        direction_deadband_m=args.direction_deadband_m,
    )
    val_ds = VisualServoDataset(
        args.root,
        val_rows,
        args.camera,
        args.image_width,
        args.image_height,
        augment=False,
        target_mode=args.target,
        target_scale=args.target_scale,
        direction_deadband_m=args.direction_deadband_m,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.target == "xy_direction":
        output_dim = 6
    elif args.target == "pixel_delta":
        output_dim = 2
    else:
        output_dim = 3
    model = VisualServoNet(output_dim=output_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.target == "xy_direction":
        class_weight = (
            None
            if args.no_direction_class_balance
            else _direction_class_weights(train_rows, args.direction_deadband_m, device)
        )
    else:
        loss_fn = nn.SmoothL1Loss(beta=1.0, reduction="none")
        reg_dim = 2 if args.target == "pixel_delta" else 3
        weight_values = (
            [args.xy_weight, args.xy_weight]
            if args.target == "pixel_delta"
            else [args.xy_weight, args.xy_weight, args.z_weight]
        )
        loss_weight = torch.tensor(
            weight_values,
            device=device,
        ).view(1, reg_dim)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf")
    best_path = args.output_dir / "best_visual_servo.pt"
    history = []

    print(
        f"Training visual servo model on {len(train_rows)} train / {len(val_rows)} val rows "
        f"from {args.root} using {device}."
        f" target={args.target} target_scale={args.target_scale}"
        f" filters: min_xy_target_m={args.min_xy_target_m} "
        f"max_xy_target_m={args.max_xy_target_m} min_frame_index={args.min_frame_index}"
        f" split_by_episode={args.split_by_episode}",
        flush=True,
    )
    if args.target == "xy_direction":
        counts = _direction_class_counts(train_rows, args.direction_deadband_m)
        print(
            "xy_direction train class counts "
            f"x={counts[0].int().tolist()} y={counts[1].int().tolist()} "
            f"deadband={args.direction_deadband_m * 1000.0:.1f}mm",
            flush=True,
        )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            pred = model(image, state)
            if args.target == "xy_direction":
                logits = pred.view(-1, 2, 3)
                if class_weight is None:
                    loss_x = nn.functional.cross_entropy(logits[:, 0, :], target[:, 0])
                    loss_y = nn.functional.cross_entropy(logits[:, 1, :], target[:, 1])
                else:
                    loss_x = nn.functional.cross_entropy(
                        logits[:, 0, :],
                        target[:, 0],
                        weight=class_weight[0],
                    )
                    loss_y = nn.functional.cross_entropy(
                        logits[:, 1, :],
                        target[:, 1],
                        weight=class_weight[1],
                    )
                loss = 0.5 * (loss_x + loss_y)
            else:
                loss = (loss_fn(pred, target) * loss_weight).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            bs = image.shape[0]
            running_loss += float(loss.item()) * bs
            seen += bs

        metrics = evaluate(model, val_loader, device, args.target)
        train_loss = running_loss / max(seen, 1)
        record = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(record)
        if args.target == "xy_direction":
            print(
                f"epoch={epoch:03d} train={train_loss:.4f} val={metrics['loss']:.4f} "
                f"axis_acc={metrics['axis_acc']:.3f} "
                f"pair_acc={metrics['pair_acc']:.3f} "
                f"nonzero_acc={metrics['nonzero_axis_acc']:.3f}",
                flush=True,
            )
        elif args.target == "pixel_delta":
            print(
                f"epoch={epoch:03d} train={train_loss:.4f} val={metrics['loss']:.4f} "
                f"px_mae={metrics['mae_xy_norm_px']:.2f}px "
                f"p90={metrics['p90_xy_norm_px']:.2f}px "
                f"<10px={metrics['within_10px']:.3f}",
                flush=True,
            )
        else:
            print(
                f"epoch={epoch:03d} train={train_loss:.4f} val={metrics['loss']:.4f} "
                f"xy_mae={metrics['mae_xy_norm_mm']:.2f}mm "
                f"p90={metrics['p90_xy_norm_mm']:.2f}mm "
                f"<5mm={metrics['within_5mm']:.3f}",
                flush=True,
            )
        (args.output_dir / "train_history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )

        checkpoint_metric = (
            1.0 - 0.5 * (metrics["pair_acc"] + metrics["nonzero_axis_acc"])
            if args.target == "xy_direction"
            else metrics["mae_xy_norm_px"]
            if args.target == "pixel_delta"
            else metrics["mae_xy_norm_mm"]
        )
        if checkpoint_metric < best_metric:
            best_metric = checkpoint_metric
            if args.target == "delta":
                target_name = "base.delta_port_minus_plug_m"
            elif args.target == "action_linear":
                target_name = "action.linear_xyz_mps"
            elif args.target == "pixel_delta":
                target_name = f"image.{args.camera}.delta_port_minus_plug_px"
            else:
                target_name = "base.xy_direction_class"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "schema": "aic_visual_servo_model_v1",
                        "camera": args.camera,
                        "image_width": args.image_width,
                        "image_height": args.image_height,
                        "state_dim": STATE_DIM,
                        "output_dim": output_dim,
                        "target": target_name,
                        "target_mode": args.target,
                        "target_scale": args.target_scale,
                        "direction_deadband_m": args.direction_deadband_m,
                        "direction_speed_mps": args.direction_speed_mps,
                        "pixel_to_base_xy": _fit_pixel_to_base_xy(train_rows, args.camera)
                        if args.target == "pixel_delta"
                        else None,
                        "image_mean": IMAGE_MEAN.view(3).tolist(),
                        "image_std": IMAGE_STD.view(3).tolist(),
                    },
                    "metrics": metrics,
                    "epoch": epoch,
                },
                best_path,
            )

    if args.target == "xy_direction":
        print(f"Best checkpoint: {best_path} (1-balanced_direction_score={best_metric:.4f})")
    elif args.target == "pixel_delta":
        print(f"Best checkpoint: {best_path} (px_mae={best_metric:.2f}px)")
    else:
        print(f"Best checkpoint: {best_path} (xy_mae={best_metric:.2f}mm)")


if __name__ == "__main__":
    main()
