"""
Standalone CNN-Mamba v2 model class — MATCHES fold_10_best.pt checkpoint exactly.

Architecture (from weight shapes):
  - CNN: 3x Conv1d(kernel=5) + residual + projection to d_model
  - Mamba: 3 blocks with d_model=96, d_inner=192, d_state=32, dt_rank=6
  - Head: LayerNorm → Linear(96,96) → GELU → Linear(96,3)

DO NOT IMPORT FROM train_cnn_mamba_v2.py — that file has a DIFFERENT architecture
that causes silent model loading failures with strict=False.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class SelectiveSSM(nn.Module):
    """Mamba SSM block — matches checkpoint key prefix 'blocks.N.ssm.*'"""

    def __init__(self, d_model: int = 96, d_state: int = 32, dt_rank: int = 6, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        d_inner = d_model * 2  # 192

        # Projections
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)   # -> (384,)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)  # -> (70,)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)  # -> (192,)

        # SSM parameters
        self.A_log = nn.Parameter(torch.zeros(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape
        d_inner = self.d_model * 2

        # in_proj -> split into x_branch and gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.split(d_inner, dim=-1)

        # Causal conv1d
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_branch = F.silu(x_conv)

        # SSM parameters from input
        x_dbl = self.x_proj(x_branch)  # (B, L, dt_rank + 2*d_state)
        dt, B_param, C_param = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan
        y = torch.zeros_like(x_branch)
        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(L):
            dt_t = dt[:, t]               # (B, d_inner)
            B_t = B_param[:, t]           # (B, d_state)
            C_t = C_param[:, t]           # (B, d_state)
            x_t = x_branch[:, t]         # (B, d_inner)

            dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)           # (B, d_inner, d_state)
            h = h * dA + dB * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * x_t
            y[:, t] = y_t

        # Gate and project out
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba block — matches 'blocks.N.norm.*' + 'blocks.N.ssm.*'"""

    def __init__(self, d_model=96, d_state=32, dt_rank=6, d_conv=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dt_rank, d_conv)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.ssm(self.norm(x)))


class CNNMambaV2(nn.Module):
    """
    CNN-Mamba v2 — architecture matching fold_10_best.pt

    Weight key mapping:
      cnn.{0,3,6}.weight/bias  -> 3x Conv1d layers
      cnn_residual.weight      -> 1x1 residual conv
      cnn_to_model.0.*         -> Linear(64, 96)
      cnn_to_model.1.*         -> LayerNorm(96)
      blocks.{0,1,2}.norm.*    -> LayerNorm per block
      blocks.{0,1,2}.ssm.*     -> Mamba SSM per block
      final_norm.*             -> LayerNorm(96)
      head.{0,3}.*             -> MLP head
    """

    def __init__(self, input_dim=25, d_model=96, d_state=32, n_layers=3,
                 dt_rank=6, d_conv=4, dropout=0.1, cnn_channels=64, cnn_kernel=5):
        super().__init__()
        self.d_model = d_model

        # CNN front-end: 3 layers of Conv1d
        # Keys: cnn.0, cnn.3, cnn.6 (positions 1,2,4,5,7,8 are param-free acts)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, cnn_kernel, padding=cnn_kernel // 2),  # 0
            nn.GELU(),  # 1
            nn.Dropout(dropout),  # 2
            nn.Conv1d(cnn_channels, cnn_channels, cnn_kernel, padding=cnn_kernel // 2),  # 3
            nn.GELU(),  # 4
            nn.Dropout(dropout),  # 5
            nn.Conv1d(cnn_channels, cnn_channels, cnn_kernel, padding=cnn_kernel // 2),  # 6
            nn.GELU(),  # 7
            nn.Dropout(dropout),  # 8
        )

        # Residual projection from input to CNN output dim
        self.cnn_residual = nn.Conv1d(input_dim, cnn_channels, 1, bias=False)

        # Project CNN output to Mamba d_model
        self.cnn_to_model = nn.Sequential(
            nn.Linear(cnn_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # Mamba backbone
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, dt_rank, d_conv, dropout)
            for _ in range(n_layers)
        ])

        # Final norm + head
        # Keys: head.0 = Linear(96,96), head.3 = Linear(96,3)
        # So positions 1,2 are param-free (GELU + Dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),    # 0
            nn.GELU(),                       # 1
            nn.Dropout(dropout),             # 2
            nn.Linear(d_model, 3),           # 3
        )

    def forward(self, x, return_embedding=False):
        """
        x: (B, L, input_dim) — sequence of MBO events
        Returns: (B, 3) predictions for 1s, 5s, 10s horizons
        """
        B, L, D = x.shape

        # CNN pathway (operates on time dimension)
        x_t = x.transpose(1, 2)  # (B, input_dim, L)
        cnn_out = self.cnn(x_t)  # (B, cnn_channels, L)
        residual = self.cnn_residual(x_t)  # (B, cnn_channels, L)
        cnn_out = cnn_out + residual
        cnn_out = cnn_out.transpose(1, 2)  # (B, L, cnn_channels)

        # Project to model dim
        h = self.cnn_to_model(cnn_out)  # (B, L, d_model)

        # Mamba blocks
        for block in self.blocks:
            h = block(h)

        # Take last position, normalize
        h = self.final_norm(h[:, -1, :])  # (B, d_model)

        if return_embedding:
            return self.head(h), h
        return self.head(h)


def load_model(checkpoint_path: str, device: str = 'cuda') -> CNNMambaV2:
    """Load CNN-Mamba v2 model from checkpoint with correct architecture."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model_state']

    # Infer architecture from weight shapes
    input_dim = state['cnn.0.weight'].shape[1]  # First conv input channels
    cnn_channels = state['cnn.0.weight'].shape[0]  # CNN output channels
    d_model = state['final_norm.weight'].shape[0]
    d_state = state['blocks.0.ssm.A_log'].shape[1]
    d_inner = state['blocks.0.ssm.A_log'].shape[0]
    x_proj_out = state['blocks.0.ssm.x_proj.weight'].shape[0]
    dt_rank = x_proj_out - 2 * d_state
    n_layers = sum(1 for k in state if k.endswith('.ssm.A_log'))
    cnn_kernel = state['cnn.0.weight'].shape[2]

    print(f"[cnn_mamba_v2_model] Architecture: input={input_dim}, cnn_ch={cnn_channels}, "
          f"d_model={d_model}, d_state={d_state}, dt_rank={dt_rank}, n_layers={n_layers}")

    model = CNNMambaV2(
        input_dim=input_dim, d_model=d_model, d_state=d_state,
        n_layers=n_layers, dt_rank=dt_rank, cnn_channels=cnn_channels,
        cnn_kernel=cnn_kernel
    )

    # Load weights — should be strict since architecture matches
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[cnn_mamba_v2_model] WARNING missing keys: {missing}")
    if unexpected:
        print(f"[cnn_mamba_v2_model] WARNING unexpected keys: {unexpected}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cnn_mamba_v2_model] Loaded: {n_params:,} params, fold={ckpt.get('fold')}, "
          f"val_ic_10s={ckpt.get('val_ic_10s', '?'):.4f}")

    model.to(device).eval()
    return model
