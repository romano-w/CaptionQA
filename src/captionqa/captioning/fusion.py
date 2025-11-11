"""Feature fusion for multimodal conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FusionConfig:
    hidden_size: int = 512
    dropout: float = 0.1
    device: Optional[str] = None


class FusionHead:
    """Lightweight fusion of visual/audio features into a fixed vector.

    Strategy: mean-pool along time for each modality, concatenate, and pass
    through a small MLP to produce a single conditioning vector.
    """

    def __init__(self, config: FusionConfig):
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = None

    def _ensure_net(self, in_dim: int) -> None:
        if self.net is None:
            h = max(int(self.config.hidden_size or 256), 32)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, h),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.config.dropout),
                torch.nn.Linear(h, h),
            ).to(self.device)

    def fuse(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        def _pool(x: torch.Tensor) -> torch.Tensor:
            if x is None or x.numel() == 0:
                return torch.zeros(1, 1, device=self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return x.to(self.device).float().mean(dim=0, keepdim=True)

        v = _pool(visual)
        a = _pool(audio)
        fused = torch.cat([v, a], dim=-1)  # [1, Dv+Da]

        self._ensure_net(fused.shape[-1])
        with torch.no_grad():
            out = self.net(fused)
        return out  # [1, hidden_size]

