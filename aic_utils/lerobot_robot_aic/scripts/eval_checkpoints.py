#!/usr/bin/env python3
"""Per-dimension MAE evaluation across saved LeRobot ACT checkpoints.

Loads each checkpoint in --checkpoint-dir, runs `predict_action_chunk` on a
held-out (or arbitrary) episode subset of a LeRobotDataset, computes per-dim
MAE in *physical units* (mm/s for linear, deg/s for angular), and logs to W&B
along with best-step-per-dim tracking.

Use this on a held-out episode split to choose checkpoints before running the
full scoring pipeline. If a model was trained without a validation split, the
same command still runs but the result is only a train-set imitation proxy.

Usage:
  pixi run python aic_utils/lerobot_robot_aic/scripts/eval_checkpoints.py \\
      --checkpoint-dir outputs/train/act_aic_v1/checkpoints \\
      --dataset-repo-id ${HF_USER}/aic_act_v1 \\
      --val-episodes 0 10 20 30 40 50 60 70 80 90 \\
      --max-frames 600 \\
      --wandb-project aic-act-checkpoint-eval \\
      --wandb-run-name act_aic_v1_eval
"""

import argparse
import json
from pathlib import Path

import draccus
import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

ACTION_DIM_NAMES = ["lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z", "pad"]
ACTION_DIM_UNITS = ["mm_s", "mm_s", "mm_s", "deg_s", "deg_s", "deg_s", "raw"]
ACTION_DIM_SCALE = [1000.0, 1000.0, 1000.0, 180.0 / np.pi, 180.0 / np.pi, 180.0 / np.pi, 1.0]
NORM_EPS = 1e-6


def safe_denominator(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t.abs() < NORM_EPS, torch.full_like(t, NORM_EPS), t)


def find_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, /path/to/.../pretrained_model), ...] sorted by step."""
    out = []
    for p in ckpt_dir.iterdir():
        if not p.is_dir() or p.name in {"last"}:
            continue
        try:
            step = int(p.name)
        except ValueError:
            continue
        pretrained = p / "pretrained_model"
        if (pretrained / "config.json").exists():
            out.append((step, pretrained))
    out.sort(key=lambda x: x[0])
    return out


def load_policy_and_stats(pretrained: Path, device: torch.device):
    """Load either an ACT or Diffusion policy depending on config.json's `type` field.

    Returns (policy, cfg, norm) where `norm` is a unified dict with everything needed
    to manually normalize obs and unnormalize predicted actions. The `norm` dict
    also carries `policy_type`, `n_obs_steps`, and the action+state norm modes.
    """
    with open(pretrained / "config.json") as f:
        cfg_dict = json.load(f)
    policy_type = cfg_dict.pop("type", "act")
    norm_map = cfg_dict.get("normalization_mapping", {})

    if policy_type == "act":
        cfg = draccus.decode(ACTConfig, cfg_dict)
        policy = ACTPolicy(cfg)
    elif policy_type == "diffusion":
        cfg = draccus.decode(DiffusionConfig, cfg_dict)
        policy = DiffusionPolicy(cfg)
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")
    policy.load_state_dict(load_file(pretrained / "model.safetensors"))
    policy.eval().to(device)

    stats = load_file(pretrained / "policy_preprocessor_step_3_normalizer_processor.safetensors")
    norm = {
        "policy_type": policy_type,
        "n_obs_steps": getattr(cfg, "n_obs_steps", 1),
        "state_norm_mode": norm_map.get("STATE", "MEAN_STD"),
        "action_norm_mode": norm_map.get("ACTION", "MEAN_STD"),
        # Visual is MEAN_STD for both ACT and diffusion in our configs
        "img_left_mean": stats["observation.images.left_camera.mean"].to(device).view(1, 3, 1, 1),
        "img_left_std": stats["observation.images.left_camera.std"].to(device).view(1, 3, 1, 1),
        "img_center_mean": stats["observation.images.center_camera.mean"].to(device).view(1, 3, 1, 1),
        "img_center_std": stats["observation.images.center_camera.std"].to(device).view(1, 3, 1, 1),
        "img_right_mean": stats["observation.images.right_camera.mean"].to(device).view(1, 3, 1, 1),
        "img_right_std": stats["observation.images.right_camera.std"].to(device).view(1, 3, 1, 1),
    }
    if norm["state_norm_mode"] == "MEAN_STD":
        norm["state_mean"] = stats["observation.state.mean"].to(device).view(1, -1)
        norm["state_std"] = safe_denominator(stats["observation.state.std"].to(device).view(1, -1))
    elif norm["state_norm_mode"] == "MIN_MAX":
        norm["state_min"] = stats["observation.state.min"].to(device).view(1, -1)
        norm["state_max"] = stats["observation.state.max"].to(device).view(1, -1)
    if norm["action_norm_mode"] == "MEAN_STD":
        norm["action_mean"] = stats["action.mean"].to(device).view(1, -1)
        norm["action_std"] = stats["action.std"].to(device).view(1, -1)
    elif norm["action_norm_mode"] == "MIN_MAX":
        norm["action_min"] = stats["action.min"].to(device).view(1, -1)
        norm["action_max"] = stats["action.max"].to(device).view(1, -1)
    return policy, cfg, norm


def normalize_batch(batch: dict, norm: dict, device: torch.device) -> dict:
    """Normalize obs to the policy's expected input space.

    State: MEAN_STD (z-score) or MIN_MAX (to [-1,1]) per `norm['state_norm_mode']`.
    Images: always MEAN_STD on float [0,1] inputs.
    """
    out = {}
    state = batch["observation.state"].to(device).float()
    if norm["state_norm_mode"] == "MEAN_STD":
        out["observation.state"] = (state - norm["state_mean"]) / norm["state_std"]
    elif norm["state_norm_mode"] == "MIN_MAX":
        out["observation.state"] = 2 * (state - norm["state_min"]) / safe_denominator(
            norm["state_max"] - norm["state_min"]
        ) - 1
    else:
        out["observation.state"] = state

    for cam_key, m_key, s_key in [
        ("observation.images.left_camera", "img_left_mean", "img_left_std"),
        ("observation.images.center_camera", "img_center_mean", "img_center_std"),
        ("observation.images.right_camera", "img_right_mean", "img_right_std"),
    ]:
        img = batch[cam_key].to(device).float()
        # LeRobotDataset returns uint8 [0..255] for video features; normalize to [0,1] first.
        if img.max() > 1.5:
            img = img / 255.0
        out[cam_key] = (img - norm[m_key]) / safe_denominator(norm[s_key])
    return out


def unnormalize_action(pred_norm: torch.Tensor, norm: dict) -> torch.Tensor:
    if norm["action_norm_mode"] == "MEAN_STD":
        return pred_norm * norm["action_std"] + norm["action_mean"]
    elif norm["action_norm_mode"] == "MIN_MAX":
        return (pred_norm + 1) / 2 * (norm["action_max"] - norm["action_min"]) + norm["action_min"]
    return pred_norm


@torch.no_grad()
def eval_checkpoint(
    pretrained: Path,
    loader: DataLoader,
    device: torch.device,
    max_frames: int,
) -> dict:
    policy, cfg, norm = load_policy_and_stats(pretrained, device)

    abs_err_sum = torch.zeros(7, device=device)
    n_frames = 0

    for batch in loader:
        if n_frames >= max_frames:
            break
        gt_action_raw = batch["action"].to(device).float()  # (B, action_dim)
        # Dataloader sometimes yields shape (B, T, action_dim) when delta_timestamps used; we don't.
        if gt_action_raw.dim() == 3:
            gt_action_raw = gt_action_raw[:, 0]
        normed = normalize_batch(batch, norm, device)

        if norm["policy_type"] == "act":
            # ACT predict_action_chunk is stateless — returns (B, chunk_size, action_dim).
            chunk = policy.predict_action_chunk(normed)
        else:
            # DiffusionPolicy.predict_action_chunk reads from internal queues that
            # are populated by select_action(). For an offline batched eval where
            # we just want "given current obs, what action does the model predict",
            # we manually stack the single observation `n_obs_steps` times along
            # dim=1 and call diffusion.generate_actions directly. This is a proxy
            # (the policy was trained with 2 distinct timesteps of observation),
            # but it's a reasonable apples-to-apples comparison metric since both
            # ACT and diffusion get the same input here.
            n_obs = norm["n_obs_steps"]
            stacked = {}
            stacked["observation.state"] = normed["observation.state"].unsqueeze(1).repeat(1, n_obs, 1)
            from lerobot.utils.constants import OBS_IMAGES
            img_keys = list(cfg.image_features.keys())
            stacked_imgs = torch.stack(
                [normed[k].unsqueeze(1).repeat(1, n_obs, 1, 1, 1) for k in img_keys], dim=2
            )  # (B, n_obs, n_cams, C, H, W)
            stacked[OBS_IMAGES] = stacked_imgs
            chunk = policy.diffusion.generate_actions(stacked)

        pred_action_norm = chunk[:, 0]  # first slot = the action that would be commanded
        pred_action = unnormalize_action(pred_action_norm, norm)

        abs_err_sum += (pred_action - gt_action_raw).abs().sum(dim=0)
        n_frames += gt_action_raw.shape[0]

    abs_err_sum = abs_err_sum.cpu().numpy()
    mae_raw = abs_err_sum / max(n_frames, 1)
    return {
        "n_frames": int(n_frames),
        "mae_raw_per_dim": mae_raw.tolist(),  # in m/s, rad/s, raw
        "mae_display_per_dim": (mae_raw * np.array(ACTION_DIM_SCALE)).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True, type=Path)
    p.add_argument("--dataset-repo-id", required=True)
    p.add_argument("--dataset-root", default=None)
    p.add_argument(
        "--val-episodes",
        type=int,
        nargs="+",
        default=None,
        help="Episode indices to use as val. None = use all.",
    )
    p.add_argument("--video-backend", default="pyav")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=600, help="Cap eval frames per checkpoint.")
    p.add_argument(
        "--tail-frames",
        type=int,
        default=None,
        help="Evaluate only the last N frames of each selected episode.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-id", default=None, help="Resume an existing run to log val/* into it.")
    p.add_argument("--wandb-run-name", default=None, help="Used when starting a new analysis run.")
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset {args.dataset_repo_id} (val_episodes={args.val_episodes})...")
    ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=args.val_episodes,
        video_backend=args.video_backend,
    )
    print(f"  {ds.num_episodes} episodes, {ds.num_frames} frames")

    if args.tail_frames is not None:
        if args.tail_frames <= 0:
            raise ValueError("--tail-frames must be positive")
        episode_indices = (
            list(args.val_episodes)
            if args.val_episodes is not None
            else list(range(ds.meta.total_episodes))
        )
        tail_indices = []
        compact_start = 0
        for ep_idx in episode_indices:
            length = int(ds.meta.episodes["length"][ep_idx])
            compact_end = compact_start + length
            tail_start = max(compact_start, compact_end - args.tail_frames)
            tail_indices.extend(range(tail_start, compact_end))
            compact_start = compact_end
        ds = Subset(ds, tail_indices)
        print(f"  tail eval: last {args.tail_frames} frames/episode -> {len(ds)} frames")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    ckpts = find_checkpoints(args.checkpoint_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found under {args.checkpoint_dir}")
    print(f"Found {len(ckpts)} checkpoints: {[s for s, _ in ckpts]}")

    wandb_run = None
    if args.wandb_project:
        import wandb

        if args.wandb_run_id:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=args.wandb_run_id,
                resume="must",
            )
            print(f"Resumed wandb run {args.wandb_run_id}")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or f"eval_{args.checkpoint_dir.name}",
            )
            print(f"Started new wandb run {wandb_run.id}")

    results = {"checkpoints": []}
    best_per_dim = {f"val/best_mae_{ACTION_DIM_NAMES[d]}_{ACTION_DIM_UNITS[d]}": (float("inf"), -1) for d in range(7)}

    for step, pretrained in ckpts:
        print(f"\n=== checkpoint step {step} ===  ({pretrained})")
        m = eval_checkpoint(pretrained, loader, device, args.max_frames)
        m["step"] = step
        results["checkpoints"].append(m)

        log_dict = {}
        for d in range(7):
            key = f"val/mae_{ACTION_DIM_NAMES[d]}_{ACTION_DIM_UNITS[d]}"
            val = m["mae_display_per_dim"][d]
            log_dict[key] = val
            best_key = f"val/best_mae_{ACTION_DIM_NAMES[d]}_{ACTION_DIM_UNITS[d]}"
            if val < best_per_dim[best_key][0]:
                best_per_dim[best_key] = (val, step)

        # Compact summary (mean translational, mean angular)
        log_dict["val/mae_lin_mean_mm_s"] = float(np.mean(m["mae_display_per_dim"][0:3]))
        log_dict["val/mae_ang_mean_deg_s"] = float(np.mean(m["mae_display_per_dim"][3:6]))

        for k, v in log_dict.items():
            print(f"  {k}: {v:.4f}")

        if wandb_run is not None:
            wandb_run.log(log_dict, step=step)

    print("\n=== best per dim ===")
    for k, (v, step) in best_per_dim.items():
        print(f"  {k}: {v:.4f}  @ step {step}")
        if wandb_run is not None:
            wandb_run.summary[k] = v
            wandb_run.summary[f"{k}_step"] = step

    if wandb_run is not None:
        wandb_run.finish()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"results": results, "best": {k: list(v) for k, v in best_per_dim.items()}}, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
