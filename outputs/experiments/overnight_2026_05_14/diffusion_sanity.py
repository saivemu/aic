"""Diffusion checkpoint sanity check — load, run a few inferences, measure latency
and predicted action statistics vs Plan D ACT on the same frames."""

import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

REPO = Path("/home/saivemu/code/aic")
sys.path.insert(0, str(REPO / ".pixi/envs/default/lib/python3.12/site-packages"))

import draccus
from safetensors.torch import load_file
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def load_policy(pretrained_path: Path, device: str = "cuda"):
    """Mirror eval_checkpoints.load_policy_and_stats (bypasses make_policy)."""
    with open(pretrained_path / "config.json") as f:
        cfg_dict = json.load(f)
    policy_type = cfg_dict.pop("type", "act")
    if policy_type == "act":
        cfg = draccus.decode(ACTConfig, cfg_dict)
        policy = ACTPolicy(cfg)
    elif policy_type == "diffusion":
        cfg = draccus.decode(DiffusionConfig, cfg_dict)
        policy = DiffusionPolicy(cfg)
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")
    policy.load_state_dict(load_file(pretrained_path / "model.safetensors"))
    policy.eval().to(device)
    return policy, cfg


def bench(policy, name, sample, device, n_warmup=2, n_iter=10):
    """Measure inference latency in select_action mode (matches runtime)."""
    if hasattr(policy, "reset"):
        policy.reset()
    # Warmup
    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = policy.select_action(sample)
    if device == "cuda":
        torch.cuda.synchronize()
    # Bench
    times = []
    for _ in range(n_iter):
        if hasattr(policy, "reset"):
            policy.reset()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(8):  # simulate 8 ticks of control loop
                action = policy.select_action(sample)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
    times = np.array(times)
    per_tick_ms = times.mean() / 8 * 1000
    print(f"  {name}: 8-tick total mean {times.mean()*1000:.1f}ms "
          f"(median {np.median(times)*1000:.1f}, min {times.min()*1000:.1f}, max {times.max()*1000:.1f})")
    print(f"  {name}: per-tick equivalent {per_tick_ms:.1f}ms (50ms budget @ 20Hz)")
    return per_tick_ms


def main():
    device = "cuda"
    plan_d = REPO / "outputs/plan_d/pretrained_model"
    diffusion = REPO / "outputs/train/diffusion_plan_g_v1/checkpoints/last/pretrained_model"

    # Synthetic input shaped like Plan D / Plan G expect.
    H, W = 512, 576
    sample = {
        "observation.state": torch.randn(1, 43, device=device).float(),
        "observation.images.left_camera": torch.rand(1, 3, H, W, device=device).float(),
        "observation.images.center_camera": torch.rand(1, 3, H, W, device=device).float(),
        "observation.images.right_camera": torch.rand(1, 3, H, W, device=device).float(),
    }

    print(f"GPU: {torch.cuda.get_device_name()}, free mem before: "
          f"{torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    print(f"\n--- Plan D ACT ({plan_d}) ---")
    act_policy, act_cfg = load_policy(plan_d, device)
    print(f"  type={act_cfg.type} n_obs_steps={getattr(act_cfg, 'n_obs_steps', '?')} "
          f"n_action_steps={getattr(act_cfg, 'n_action_steps', '?')} "
          f"chunk_size={getattr(act_cfg, 'chunk_size', '?')}")
    act_per_tick = bench(act_policy, "ACT", sample, device)
    del act_policy
    torch.cuda.empty_cache()

    print(f"\n--- Diffusion Plan G v1 ({diffusion}) ---")
    diff_policy, diff_cfg = load_policy(diffusion, device)
    print(f"  type={diff_cfg.type} n_obs_steps={getattr(diff_cfg, 'n_obs_steps', '?')} "
          f"horizon={getattr(diff_cfg, 'horizon', '?')} "
          f"n_action_steps={getattr(diff_cfg, 'n_action_steps', '?')} "
          f"num_inference_steps={getattr(diff_cfg, 'num_inference_steps', '?')}")
    diff_per_tick = bench(diff_policy, "Diffusion", sample, device)

    print(f"\n--- Summary ---")
    print(f"  20 Hz budget: 50.0 ms/tick")
    print(f"  ACT per-tick:       {act_per_tick:.1f} ms  ({'OK' if act_per_tick < 50 else 'OVER BUDGET'})")
    print(f"  Diffusion per-tick: {diff_per_tick:.1f} ms  ({'OK' if diff_per_tick < 50 else 'OVER BUDGET'})")

    # Compare predicted actions on identical synthetic input
    if hasattr(diff_policy, "reset"):
        diff_policy.reset()
    with torch.inference_mode():
        diff_actions = []
        for _ in range(16):
            a = diff_policy.select_action(sample)
            diff_actions.append(a[0].cpu().numpy())
        diff_actions = np.array(diff_actions)
        print(f"\n  Diffusion 16-action sequence (xyz mm/s):")
        for i, a in enumerate(diff_actions[:8]):
            print(f"    t={i}: lin=[{a[0]*1000:.2f}, {a[1]*1000:.2f}, {a[2]*1000:.2f}] mm/s")
        # check for any obvious pathology
        lin_norm = np.linalg.norm(diff_actions[:, :3], axis=1)
        print(f"  lin-vel-norm stats: mean={lin_norm.mean()*1000:.2f} mm/s, "
              f"max={lin_norm.max()*1000:.2f}, min={lin_norm.min()*1000:.2f}")


if __name__ == "__main__":
    main()
