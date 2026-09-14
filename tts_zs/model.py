import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tts_zs.config import AUDIO_DIM, N_AUDIO_LAYERS


class ProjectionMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        style: str = "mlp",
        normalize: bool = True,
    ):
        super().__init__()
        if style == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        elif style == "linear2":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Linear(hidden_dim, out_dim),
            )
        elif style == "linear1":
            self.net = nn.Linear(in_dim, out_dim)
        elif style == "identity":
            if in_dim != out_dim:
                raise ValueError(
                    f"identity requires in_dim==out_dim, got {in_dim} vs {out_dim}"
                )
            self.net = nn.Identity()
        else:
            raise ValueError(f"unknown proj style: {style!r}")
        self._normalize = normalize

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1) if self._normalize else x


class WeightedLayerSum(nn.Module):
    def __init__(self, n_layers: int = N_AUDIO_LAYERS):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.ones(n_layers) / n_layers)

    def weights(self) -> torch.Tensor:
        return self.raw_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.raw_weights
        return (x * w.view(1, -1, 1)).sum(dim=1)


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        proj_dim: int,
        hidden_dim: int,
        dropout: float,
        text_dim: int,
        proj_style: str = "mlp",
        text_proj_style: str | None = None,
        scale_max: float = 30.0,
        normalize: bool = True,
    ):
        super().__init__()
        if text_proj_style is None:
            text_proj_style = proj_style
        self.layer_sum = WeightedLayerSum(N_AUDIO_LAYERS)
        self.audio_proj = ProjectionMLP(
            AUDIO_DIM, hidden_dim, proj_dim, dropout, proj_style, normalize=normalize
        )
        self.text_proj = ProjectionMLP(
            text_dim,
            hidden_dim,
            proj_dim,
            dropout,
            text_proj_style,
            normalize=normalize,
        )
        self.scale_max = scale_max
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07)))

    def forward(self, audio, text):
        a = self.audio_proj(self.layer_sum(audio))
        t = self.text_proj(text)
        return a, t, self.logit_scale.clamp(max=np.log(self.scale_max)).exp()
