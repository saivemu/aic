"""Compact rectified-flow action chunker for AIC final-stage experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


NORM_EPS = 1e-6


@dataclass
class FlowPolicyConfig:
    state_dim: int = 43
    action_dim: int = 7
    num_cameras: int = 3
    chunk_len: int = 16
    image_height: int = 128
    image_width: int = 144
    image_channels: int = 3
    image_mean: float = 0.5
    image_std: float = 0.5
    hidden_dim: int = 512
    cond_dim: int = 512
    flow_steps: int = 4
    replan_every: int = 4
    zero_start: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowPolicyConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in allowed})


class TinyCameraEncoder(nn.Module):
    def __init__(self, out_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def timestep_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0, -4.0, half, device=t.device, dtype=t.dtype)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class RectifiedFlowActionModel(nn.Module):
    def __init__(self, cfg: FlowPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.camera_encoder = TinyCameraEncoder(out_dim=96)
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 128),
            nn.SiLU(inplace=True),
        )
        self.cond = nn.Sequential(
            nn.Linear((96 * cfg.num_cameras) + 128, cfg.cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
            nn.SiLU(inplace=True),
        )
        action_flat = cfg.chunk_len * cfg.action_dim
        self.flow = nn.Sequential(
            nn.Linear(action_flat + cfg.cond_dim + 64, cfg.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.hidden_dim, action_flat),
        )

    def encode_condition(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = images.shape
        img_features = self.camera_encoder(images.reshape(b * n, c, h, w)).reshape(b, n, -1)
        state_features = self.state_encoder(state)
        return self.cond(torch.cat([img_features.flatten(1), state_features], dim=1))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        images: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.encode_condition(images, state)
        x_flat = x_t.flatten(1)
        t_emb = timestep_embedding(t, dim=64)
        v = self.flow(torch.cat([x_flat, cond, t_emb], dim=1))
        return v.reshape_as(x_t)

    @torch.no_grad()
    def sample(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        *,
        steps: int | None = None,
        zero_start: bool | None = None,
    ) -> torch.Tensor:
        steps = int(steps or self.cfg.flow_steps)
        zero_start = self.cfg.zero_start if zero_start is None else zero_start
        shape = (images.shape[0], self.cfg.chunk_len, self.cfg.action_dim)
        if zero_start:
            x = torch.zeros(shape, device=images.device, dtype=images.dtype)
        else:
            x = torch.randn(shape, device=images.device, dtype=images.dtype)
        dt = 1.0 / max(steps, 1)
        for i in range(steps):
            t = torch.full((images.shape[0],), i * dt, device=images.device, dtype=images.dtype)
            x = x + dt * self.forward(x, t, images, state)
        return x


def safe_denominator(tensor: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    return torch.where(
        tensor.abs() < eps,
        torch.full_like(tensor, eps),
        tensor,
    )


def normalize_images(images: torch.Tensor, cfg: FlowPolicyConfig) -> torch.Tensor:
    if images.max() > 1.5:
        images = images.float() / 255.0
    else:
        images = images.float()
    if images.shape[-2:] != (cfg.image_height, cfg.image_width):
        b, n, c, h, w = images.shape
        images = F.interpolate(
            images.reshape(b * n, c, h, w),
            size=(cfg.image_height, cfg.image_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, n, c, cfg.image_height, cfg.image_width)
    return (images - cfg.image_mean) / max(cfg.image_std, NORM_EPS)


def normalize_state(state: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (state - stats["state_mean"]) / safe_denominator(stats["state_std"])


def normalize_action(action: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (action - stats["action_mean"]) / safe_denominator(stats["action_std"])


def unnormalize_action(action: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return action * stats["action_std"] + stats["action_mean"]


def save_checkpoint(
    out_dir: Path,
    model: RectifiedFlowActionModel,
    cfg: FlowPolicyConfig,
    stats: dict[str, torch.Tensor],
    *,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "stats": {k: v.detach().cpu() for k, v in stats.items()},
        "step": step,
        "extra": extra or {},
    }
    torch.save(payload, out_dir / "flow_policy.pt")
    with (out_dir / "config.json").open("w") as f:
        json.dump({"type": "aic_rectified_flow", **asdict(cfg), "step": step}, f, indent=2)


class FlowPolicyRunner:
    def __init__(
        self,
        model: RectifiedFlowActionModel,
        cfg: FlowPolicyConfig,
        stats: dict[str, torch.Tensor],
        device: torch.device,
        *,
        steps: int | None = None,
        replan_every: int | None = None,
    ):
        self.model = model.eval().to(device)
        self.cfg = cfg
        self.stats = {k: v.to(device).view(1, -1) for k, v in stats.items()}
        self.device = device
        self.steps = int(steps or cfg.flow_steps)
        self.replan_every = int(replan_every or cfg.replan_every)
        self.queue: list[torch.Tensor] = []
        self.ticks_since_plan = 0

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device,
        *,
        steps: int | None = None,
        replan_every: int | None = None,
    ) -> "FlowPolicyRunner":
        path = Path(path)
        ckpt_path = path / "flow_policy.pt" if path.is_dir() else path
        payload = torch.load(ckpt_path, map_location=device)
        cfg = FlowPolicyConfig.from_dict(payload["config"])
        model = RectifiedFlowActionModel(cfg)
        model.load_state_dict(payload["model_state_dict"])
        return cls(
            model,
            cfg,
            payload["stats"],
            device,
            steps=steps,
            replan_every=replan_every,
        )

    def reset(self) -> None:
        self.queue.clear()
        self.ticks_since_plan = 0

    @torch.no_grad()
    def select_action(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if (
            not self.queue
            or self.ticks_since_plan >= self.replan_every
        ):
            images = images.to(self.device).unsqueeze(0)
            state = state.to(self.device).float().view(1, -1)
            images = normalize_images(images, self.cfg)
            state = normalize_state(state, self.stats)
            pred_norm = self.model.sample(images, state, steps=self.steps)[0]
            pred = unnormalize_action(pred_norm, self.stats)
            self.queue = [row.detach().clone() for row in pred]
            self.ticks_since_plan = 0
        self.ticks_since_plan += 1
        return self.queue.pop(0)
