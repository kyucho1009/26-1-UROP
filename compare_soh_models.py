"""Run SOH model comparison experiments.

Compared models:
- persistence: previous known cycle SOH
- cpmlp: CyclePatch-style MLP with per-cycle and inter-cycle MLP blocks
- cpmlp_gru_fusion: CPMLP cycle embeddings plus GRU branch, Figure-4-style fusion
- cpgru: CyclePatch-style GRU with intra-cycle and inter-cycle GRU blocks
- cpmlp_cpgru_fusion: parallel CPMLP and CPGRU branches with dense fusion
- cpmlp_dsconv_fusion: parallel CPMLP and original GRU-DSConv branches with dense fusion
- cpmlp_dsconv_nogru: parallel CPMLP and pure DSConv branches, no GRU
- cpdsconv: CyclePatch-style DSConv with intra-cycle and inter-cycle DSConv blocks
- cpmlp_cpdsconv_fusion: parallel CPMLP and CPDSConv branches with dense fusion
- cpmlp_gru_residual: CPMLP base prediction plus GRU residual correction
- flatten_mlp: flatten (n_cycles, fixed_len, num_var) and regress SOH
- curve_cnn: convolution only along each cycle curve, then MLP over cycle embeddings
- gru_only: intra-cycle GRU plus inter-cycle GRU, no DSConv
- gru_dsconv: current GRU + DSConv model from soh_gru_dsconv_pipeline
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import soh_gru_dsconv_pipeline as pipe

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is required for neural model comparison. "
        "Install torch in the Python environment used to run this script."
    ) from exc


MODEL_ORDER = [
    "persistence",
    "cpmlp",
    "cpmlp_gru_fusion",
    "cpgru",
    "cpmlp_cpgru_fusion",
    "cpmlp_dsconv_fusion",
    "cpmlp_dsconv_nogru",
    "cpdsconv",
    "cpmlp_cpdsconv_fusion",
    "cpmlp_gru_residual",
    "flatten_mlp",
    "curve_cnn",
    "gru_only",
    "gru_dsconv",
]

DATASET_LABEL_PREFIXES = [
    ("ISU-ILCC", "ISU_ILCC"),
    ("UL-PUR", "UL_PUR"),
    ("NA-ion", "NA-ion"),
    ("ZN-coin", "ZN-coin"),
    ("Stanford_Nova_Regular_Ref", "Stanford"),
    ("Stanford_", "Stanford_2"),
    ("MICH_MCForm", "MICH"),
    ("MICH_", "MICH_EXP"),
    ("Tongji", "Tongji"),
    ("CALB", "CALB"),
    ("CALCE", "CALCE"),
    ("HNEI", "HNEI"),
    ("HUST", "HUST"),
    ("MATR", "MATR"),
    ("RWTH", "RWTH"),
    ("SDU", "SDU"),
    ("SNL", "SNL"),
    ("XJTU", "XJTU"),
]


def infer_dataset_label(filename: str | Path) -> str:
    name = Path(filename).name
    for prefix, dataset in DATASET_LABEL_PREFIXES:
        if name.startswith(prefix):
            return dataset
    return pipe.infer_battery_domain(name)


def train_model_compat(model, train_loader, val_loader, **kwargs):
    """Call pipe.train_model while tolerating older pipeline files in Colab."""
    signature = inspect.signature(pipe.train_model)
    supported = set(signature.parameters)
    train_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    ignored = sorted(set(kwargs) - set(train_kwargs))
    if ignored:
        print(
            "[warning] pipe.train_model does not support these options and they will be ignored: "
            + ", ".join(ignored)
        )
        print("[warning] Upload the latest soh_gru_dsconv_pipeline.py to use all tuning options.")
    return pipe.train_model(model, train_loader, val_loader, **train_kwargs)


class FlattenMLPSOH(nn.Module):
    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        fixed_len: int = pipe.FIXED_LEN,
        num_var: int = pipe.NUM_VAR,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = n_cycles * fixed_len * num_var
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CycleMLPEncoder(nn.Module):
    """Encode each cycle patch independently with an MLP."""

    def __init__(
        self,
        fixed_len: int = pipe.FIXED_LEN,
        num_var: int = pipe.NUM_VAR,
        embed_dim: int = 64,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fixed_len = fixed_len
        self.num_var = num_var
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fixed_len * num_var, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, cycles, length, num_var = x.shape
        z = x.reshape(batch * cycles, length, num_var)
        z = self.net(z)
        return z.reshape(batch, cycles, self.embed_dim)


class CPMLPSOH(nn.Module):
    """CyclePatch-style MLP baseline: intra-cycle MLP + inter-cycle MLP."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        fixed_len: int = pipe.FIXED_LEN,
        num_var: int = pipe.NUM_VAR,
        embed_dim: int = 64,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_cycles = n_cycles
        self.embed_dim = embed_dim
        self.cycle_encoder = CycleMLPEncoder(
            fixed_len=fixed_len,
            num_var=num_var,
            embed_dim=embed_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.inter_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_cycles * embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def encode_cycles(self, x: torch.Tensor) -> torch.Tensor:
        return self.cycle_encoder(x)

    def predict_from_embeddings(self, cycle_embeddings: torch.Tensor) -> torch.Tensor:
        return self.inter_mlp(cycle_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cycle_embeddings = self.encode_cycles(x)
        return self.predict_from_embeddings(cycle_embeddings)


class CPMLPGRUResidualSOH(nn.Module):
    """CPMLP prediction corrected by a GRU residual branch."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        fixed_len: int = pipe.FIXED_LEN,
        num_var: int = pipe.NUM_VAR,
        embed_dim: int = 64,
        hidden: int = 256,
        gru_hidden: int = 64,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
        cpmlp: CPMLPSOH | None = None,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.cpmlp = cpmlp or CPMLPSOH(
            n_cycles=n_cycles,
            fixed_len=fixed_len,
            num_var=num_var,
            embed_dim=embed_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.freeze_base = freeze_base
        if self.freeze_base:
            for param in self.cpmlp.parameters():
                param.requires_grad = False
        self.residual_scale = residual_scale
        self.residual_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1),
        )
        final_layer = self.residual_head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.cpmlp.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_base:
            with torch.no_grad():
                cycle_embeddings = self.cpmlp.encode_cycles(x)
                base_pred = self.cpmlp.predict_from_embeddings(cycle_embeddings)
            cycle_embeddings = cycle_embeddings.detach()
        else:
            cycle_embeddings = self.cpmlp.encode_cycles(x)
            base_pred = self.cpmlp.predict_from_embeddings(cycle_embeddings)
        _, hidden = self.residual_gru(cycle_embeddings)
        correction = self.residual_head(hidden[-1])
        return base_pred + self.residual_scale * correction


class CPMLPGRUFusionSOH(nn.Module):
    """Figure-4-style hybrid: CPMLP branch feeds a GRU branch, then both are fused."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        fixed_len: int = pipe.FIXED_LEN,
        num_var: int = pipe.NUM_VAR,
        embed_dim: int = 64,
        hidden: int = 256,
        gru_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cycle_encoder = CycleMLPEncoder(
            fixed_len=fixed_len,
            num_var=num_var,
            embed_dim=embed_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.cpmlp_context = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_cycles * embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(hidden + gru_hidden),
            nn.Linear(hidden + gru_hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cycle_embeddings = self.cycle_encoder(x)
        cpmlp_state = self.cpmlp_context(cycle_embeddings)
        _, gru_state = self.gru(cycle_embeddings)
        fused = torch.cat([cpmlp_state, gru_state[-1]], dim=-1)
        return self.fusion_head(fused)


class CycleGRUEncoder(nn.Module):
    """Encode each cycle patch with a GRU over points inside the cycle."""

    def __init__(
        self,
        num_var: int = pipe.NUM_VAR,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.gru = nn.GRU(
            input_size=num_var,
            hidden_size=embed_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, cycles, length, num_var = x.shape
        z = x.reshape(batch * cycles, length, num_var)
        _, hidden = self.gru(z)
        cycle_embeddings = self.dropout(self.norm(hidden[-1]))
        return cycle_embeddings.reshape(batch, cycles, self.embed_dim)


class CPGRUSOH(nn.Module):
    """CyclePatch-style GRU baseline: intra-cycle GRU + inter-cycle GRU."""

    def __init__(
        self,
        num_var: int = pipe.NUM_VAR,
        embed_dim: int = 64,
        cycle_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cycle_encoder = CycleGRUEncoder(
            num_var=num_var,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.inter_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=cycle_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cycle_hidden),
            nn.Linear(cycle_hidden, cycle_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cycle_hidden // 2, 1),
        )

    def encode_cycles(self, x: torch.Tensor) -> torch.Tensor:
        return self.cycle_encoder(x)

    def encode_history(self, cycle_embeddings: torch.Tensor) -> torch.Tensor:
        _, hidden = self.inter_gru(cycle_embeddings)
        return hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cycle_embeddings = self.encode_cycles(x)
        history = self.encode_history(cycle_embeddings)
        return self.head(history)


class CPMLPCPGRUFusionSOH(nn.Module):
    """Parallel CPMLP and CPGRU branches, inspired by the Figure-4 hybrid."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        fixed_len: int = pipe.FIXED_LEN,
        num_var: int = pipe.NUM_VAR,
        mlp_embed_dim: int = 64,
        gru_embed_dim: int = 64,
        hidden: int = 256,
        gru_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cpmlp_encoder = CycleMLPEncoder(
            fixed_len=fixed_len,
            num_var=num_var,
            embed_dim=mlp_embed_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.cpmlp_context = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_cycles * mlp_embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cpgru_encoder = CycleGRUEncoder(
            num_var=num_var,
            embed_dim=gru_embed_dim,
            dropout=dropout,
        )
        self.cpgru_history = nn.GRU(
            input_size=gru_embed_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(hidden + gru_hidden),
            nn.Linear(hidden + gru_hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_embeddings = self.cpmlp_encoder(x)
        mlp_state = self.cpmlp_context(mlp_embeddings)

        gru_embeddings = self.cpgru_encoder(x)
        _, gru_state = self.cpgru_history(gru_embeddings)

        fused = torch.cat([mlp_state, gru_state[-1]], dim=-1)
        return self.fusion_head(fused)


class GRUDSConvEncoder(nn.Module):
    """Feature extractor from the original GRU-DSConv model, without the final FC."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        num_var: int = pipe.NUM_VAR,
        fixed_len: int = pipe.FIXED_LEN,
        gru_hidden: int = 64,
        channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_var = num_var
        self.fixed_len = fixed_len
        self.n_cycles = n_cycles
        self.channels = channels
        self.gru = nn.GRU(
            input_size=num_var,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.project = nn.Linear(gru_hidden, channels)
        self.pos = pipe.SinusoidalEncoding(channels, max_len=max(n_cycles + 32, 128))
        self.dsconv = pipe.DepthwiseSeparableConv1d(channels, dropout=dropout)
        self.dilated = pipe.MultiScaleDilatedStack(channels, dropout=dropout)
        self.memory = pipe.MemoryAugmentedModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, cycles, length, num_var = x.shape
        z = x.reshape(batch * cycles, length, num_var)
        _, hidden = self.gru(z)
        cycle_embeddings = self.project(hidden[-1]).reshape(batch, cycles, self.channels)
        cycle_embeddings = self.pos(cycle_embeddings)

        z = cycle_embeddings.transpose(1, 2)
        z = self.dsconv(z)
        z = self.dilated(z).transpose(1, 2)
        return self.memory(z)


class CPMLPDSConvFusionSOH(nn.Module):
    """Parallel CPMLP and original GRU-DSConv branches with dense fusion."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        fixed_len: int = pipe.FIXED_LEN,
        num_var: int = pipe.NUM_VAR,
        mlp_embed_dim: int = 64,
        hidden: int = 256,
        dsconv_channels: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cpmlp_encoder = CycleMLPEncoder(
            fixed_len=fixed_len,
            num_var=num_var,
            embed_dim=mlp_embed_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.cpmlp_context = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_cycles * mlp_embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dsconv_context = GRUDSConvEncoder(
            n_cycles=n_cycles,
            num_var=num_var,
            fixed_len=fixed_len,
            gru_hidden=gru_hidden,
            channels=dsconv_channels,
            dropout=dropout,
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(hidden + dsconv_channels),
            nn.Linear(hidden + dsconv_channels, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_embeddings = self.cpmlp_encoder(x)
        mlp_state = self.cpmlp_context(mlp_embeddings)
        dsconv_state = self.dsconv_context(x)
        fused = torch.cat([mlp_state, dsconv_state], dim=-1)
        return self.fusion_head(fused)


class CycleDSConvEncoder(nn.Module):
    """Encode each cycle patch with depthwise-separable convolution over time."""

    def __init__(
        self,
        num_var: int = pipe.NUM_VAR,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Conv1d(
                num_var,
                num_var,
                kernel_size=7,
                padding=3,
                groups=num_var,
            ),
            nn.Conv1d(num_var, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            pipe.DepthwiseSeparableConv1d(embed_dim, kernel_size=5, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, cycles, length, num_var = x.shape
        z = x.reshape(batch * cycles, length, num_var).transpose(1, 2)
        z = self.net(z).squeeze(-1)
        return z.reshape(batch, cycles, self.embed_dim)


class CPDSConvContextEncoder(nn.Module):
    """Encode a sequence of cycle embeddings with DSConv blocks across cycles."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos = pipe.SinusoidalEncoding(channels, max_len=max(n_cycles + 32, 128))
        self.dsconv = pipe.DepthwiseSeparableConv1d(channels, dropout=dropout)
        self.dilated = pipe.MultiScaleDilatedStack(channels, dropout=dropout)
        self.memory = pipe.MemoryAugmentedModule(channels)

    def forward(self, cycle_embeddings: torch.Tensor) -> torch.Tensor:
        z = self.pos(cycle_embeddings)
        z = self.dsconv(z.transpose(1, 2))
        z = self.dilated(z).transpose(1, 2)
        return self.memory(z)


class CPDSConvSOH(nn.Module):
    """CyclePatch-style DSConv baseline: intra-cycle DSConv + inter-cycle DSConv."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        num_var: int = pipe.NUM_VAR,
        channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cycle_encoder = CycleDSConvEncoder(
            num_var=num_var,
            embed_dim=channels,
            dropout=dropout,
        )
        self.context_encoder = CPDSConvContextEncoder(
            n_cycles=n_cycles,
            channels=channels,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cycle_embeddings = self.cycle_encoder(x)
        context = self.context_encoder(cycle_embeddings)
        return self.head(context)


class CPMLPCPDSConvFusionSOH(nn.Module):
    """Parallel CPMLP and CPDSConv branches with dense fusion."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        fixed_len: int = pipe.FIXED_LEN,
        num_var: int = pipe.NUM_VAR,
        mlp_embed_dim: int = 64,
        hidden: int = 256,
        dsconv_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cpmlp_encoder = CycleMLPEncoder(
            fixed_len=fixed_len,
            num_var=num_var,
            embed_dim=mlp_embed_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.cpmlp_context = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_cycles * mlp_embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cpdsconv_encoder = CycleDSConvEncoder(
            num_var=num_var,
            embed_dim=dsconv_channels,
            dropout=dropout,
        )
        self.cpdsconv_context = CPDSConvContextEncoder(
            n_cycles=n_cycles,
            channels=dsconv_channels,
            dropout=dropout,
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(hidden + dsconv_channels),
            nn.Linear(hidden + dsconv_channels, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_embeddings = self.cpmlp_encoder(x)
        mlp_state = self.cpmlp_context(mlp_embeddings)

        dsconv_embeddings = self.cpdsconv_encoder(x)
        dsconv_state = self.cpdsconv_context(dsconv_embeddings)

        fused = torch.cat([mlp_state, dsconv_state], dim=-1)
        return self.fusion_head(fused)


class CPMLPDSConvNoGRUSOH(CPMLPCPDSConvFusionSOH):
    """Parallel CPMLP and pure DSConv branches without any GRU layers."""


class CurveCNNSOH(nn.Module):
    """Convolve only within each cycle curve, not across cycle history."""

    def __init__(
        self,
        n_cycles: int = pipe.EARLY_CYCLE,
        num_var: int = pipe.NUM_VAR,
        channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_cycles = n_cycles
        self.encoder = nn.Sequential(
            nn.Conv1d(num_var, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(n_cycles * channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, cycles, length, num_var = x.shape
        z = x.reshape(batch * cycles, length, num_var).transpose(1, 2)
        z = self.encoder(z).squeeze(-1)
        z = z.reshape(batch, cycles * z.shape[-1])
        return self.head(z)


class GRUOnlySOH(nn.Module):
    """GRU-only baseline: no convolutional blocks."""

    def __init__(
        self,
        num_var: int = pipe.NUM_VAR,
        gru_hidden: int = 64,
        cycle_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.intra_gru = nn.GRU(
            input_size=num_var,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.inter_gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=cycle_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(cycle_hidden),
            nn.Linear(cycle_hidden, cycle_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cycle_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, cycles, length, num_var = x.shape
        z = x.reshape(batch * cycles, length, num_var)
        _, h = self.intra_gru(z)
        cycle_emb = h[-1].reshape(batch, cycles, -1)
        _, hist = self.inter_gru(cycle_emb)
        return self.fc(hist[-1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, result_df: pd.DataFrame) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100.0)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denom <= 0 else float(1.0 - np.sum(err**2) / denom)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_percent": mape,
        "R2": r2,
        "EOL_Error_cycles": pipe.eol_cycle_error(result_df.loc[finite].copy()),
    }


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_metrics(metrics_rows: list[dict[str, float]], out_dir: Path) -> pd.DataFrame:
    metric_cols = [
        "model",
        "val_RMSE",
        "val_MAE",
        "val_MAPE_percent",
        "val_R2",
        "val_EOL_Error_cycles",
        "RMSE",
        "MAE",
        "MAPE_percent",
        "R2",
        "EOL_Error_cycles",
    ]
    metrics = pd.DataFrame(metrics_rows)
    metric_cols = [column for column in metric_cols if column in metrics.columns]
    extra_cols = [column for column in metrics.columns if column not in metric_cols]
    sort_columns = [column for column in ["val_RMSE", "val_MAE", "RMSE", "MAE"] if column in metrics.columns]
    metrics = metrics[metric_cols + extra_cols].sort_values(sort_columns)
    metrics.to_csv(out_dir / "model_comparison_metrics.csv", index=False)
    return metrics


def add_report_group_columns(result_df: pd.DataFrame) -> pd.DataFrame:
    """Attach dataset, condition, and SOH-band labels for subgroup reporting."""
    df = result_df.copy()
    if "dataset" not in df.columns:
        df["dataset"] = df["file"].map(infer_dataset_label)
    if "condition_group" not in df.columns:
        if "experiment_condition" in df.columns:
            df["condition_group"] = df["experiment_condition"]
        else:
            df["condition_group"] = df["file"].map(pipe.infer_experiment_condition)

    soh = pd.to_numeric(df["actual_soh"], errors="coerce")
    df["soh_band"] = np.select(
        [soh >= 0.90, soh >= 0.80],
        ["early_ge_0.90", "mid_0.80_0.90"],
        default="late_lt_0.80",
    )
    return df


def collect_group_metric_rows(
    model_name: str,
    split_name: str,
    result_df: pd.DataFrame,
) -> list[dict[str, object]]:
    df = add_report_group_columns(result_df)
    rows: list[dict[str, object]] = []
    for group_by in ["dataset", "condition_group", "soh_band"]:
        for group_value, group in df.groupby(group_by, dropna=False):
            metrics = compute_metrics(group["actual_soh"].to_numpy(), group["pred_soh"].to_numpy(), group)
            if "target_cycle" in group.columns:
                target_cycle = pd.to_numeric(group["target_cycle"], errors="coerce")
                target_cycle_min = int(target_cycle.min()) if target_cycle.notna().any() else None
                target_cycle_max = int(target_cycle.max()) if target_cycle.notna().any() else None
            else:
                target_cycle_min = None
                target_cycle_max = None
            row: dict[str, object] = {
                "model": model_name,
                "split": split_name,
                "group_by": group_by,
                "group_value": str(group_value),
                "n_samples": int(len(group)),
                "n_cells": int(group["cell_id"].nunique()) if "cell_id" in group.columns else 0,
                "target_cycle_min": target_cycle_min,
                "target_cycle_max": target_cycle_max,
            }
            row.update(metrics)
            rows.append(row)
    return rows


def write_group_metrics(group_metric_rows: list[dict[str, object]], out_dir: Path) -> pd.DataFrame:
    if not group_metric_rows:
        return pd.DataFrame()
    group_metrics = pd.DataFrame(group_metric_rows)
    sort_columns = [
        column
        for column in ["split", "group_by", "group_value", "RMSE", "MAE", "model"]
        if column in group_metrics.columns
    ]
    group_metrics = group_metrics.sort_values(sort_columns)
    group_metrics.to_csv(out_dir / "model_group_metrics.csv", index=False)
    return group_metrics


def build_previous_soh_index(files: Iterable[Path], fixed_len: int) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for fp in files:
        cycles, soh, meta = pipe.load_cell_file(fp, fixed_len=fixed_len)
        del cycles
        file_key = Path(fp).name
        cycle_numbers = [int(row["cycle_number"]) for row in meta]
        soh_by_cycle = {cycle: float(value) for cycle, value in zip(cycle_numbers, soh)}
        prev_by_cycle: dict[int, float] = {}
        for pos, cycle in enumerate(cycle_numbers):
            if pos > 0:
                prev_by_cycle[cycle] = float(soh[pos - 1])
        index[file_key] = {
            "soh_by_cycle": soh_by_cycle,
            "prev_by_cycle": prev_by_cycle,
        }
    return index


def persistence_predict(split: pipe.SplitData, files: Iterable[Path], fixed_len: int) -> np.ndarray:
    index = build_previous_soh_index(files, fixed_len=fixed_len)
    preds = []
    for row in split.meta:
        file_index = index[row["file"]]
        horizon = int(row.get("horizon", 0))
        if horizon > 0:
            cycle = int(row["input_end_cycle"])
            pred = file_index["soh_by_cycle"].get(cycle, np.nan)
        else:
            cycle = int(row["target_cycle"])
            pred = file_index["prev_by_cycle"].get(cycle, np.nan)
        preds.append(pred)
    return np.asarray(preds, dtype=np.float32)


def input_end_soh(split: pipe.SplitData, files: Iterable[Path], fixed_len: int) -> np.ndarray:
    """Return the known SOH at the last input cycle for each sample."""
    index = build_previous_soh_index(files, fixed_len=fixed_len)
    values = []
    for row in split.meta:
        file_index = index[row["file"]]
        cycle = int(row["input_end_cycle"])
        values.append(file_index["soh_by_cycle"].get(cycle, np.nan))
    return np.asarray(values, dtype=np.float32)


def make_training_targets(
    split: pipe.SplitData,
    baseline_soh: np.ndarray,
    target_mode: str,
    target_scale: float = 1.0,
) -> np.ndarray:
    if target_scale <= 0:
        raise ValueError("--target-scale must be positive")
    if target_mode == "absolute":
        return (split.y * target_scale).astype(np.float32)
    if target_mode == "delta":
        return ((split.y - baseline_soh) * target_scale).astype(np.float32)
    raise ValueError(f"unknown target mode: {target_mode}")


def restore_soh_prediction(
    model_output: np.ndarray,
    baseline_soh: np.ndarray,
    target_mode: str,
    target_scale: float = 1.0,
) -> np.ndarray:
    if target_scale <= 0:
        raise ValueError("--target-scale must be positive")
    unscaled_output = model_output / target_scale
    if target_mode == "absolute":
        return unscaled_output.astype(np.float32)
    if target_mode == "delta":
        return (baseline_soh + unscaled_output).astype(np.float32)
    raise ValueError(f"unknown target mode: {target_mode}")


def zero_last_linear(module: nn.Module) -> None:
    """Make the model's initial scalar output exactly zero."""
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, nn.Linear):
            nn.init.zeros_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
            return
    raise ValueError(f"could not find a final Linear layer in {type(module).__name__}")


def make_model(
    name: str,
    early_cycle: int,
    fixed_len: int,
    residual_scale: float = 0.1,
    mlp_embed_dim: int = 64,
    gru_embed_dim: int = 64,
    model_hidden: int = 256,
    gru_hidden: int = 64,
    dsconv_channels: int = 64,
    dropout: float = 0.1,
):
    if name == "cpmlp":
        return CPMLPSOH(
            n_cycles=early_cycle,
            fixed_len=fixed_len,
            embed_dim=mlp_embed_dim,
            hidden=model_hidden,
            dropout=dropout,
        )
    if name == "cpmlp_gru_fusion":
        return CPMLPGRUFusionSOH(
            n_cycles=early_cycle,
            fixed_len=fixed_len,
            embed_dim=mlp_embed_dim,
            hidden=model_hidden,
            gru_hidden=gru_hidden,
            dropout=dropout,
        )
    if name == "cpgru":
        return CPGRUSOH(
            embed_dim=gru_embed_dim,
            cycle_hidden=gru_hidden,
            dropout=dropout,
        )
    if name == "cpmlp_cpgru_fusion":
        return CPMLPCPGRUFusionSOH(
            n_cycles=early_cycle,
            fixed_len=fixed_len,
            mlp_embed_dim=mlp_embed_dim,
            gru_embed_dim=gru_embed_dim,
            hidden=model_hidden,
            gru_hidden=gru_hidden,
            dropout=dropout,
        )
    if name == "cpmlp_dsconv_fusion":
        return CPMLPDSConvFusionSOH(
            n_cycles=early_cycle,
            fixed_len=fixed_len,
            mlp_embed_dim=mlp_embed_dim,
            hidden=model_hidden,
            dsconv_channels=dsconv_channels,
            gru_hidden=gru_hidden,
            dropout=dropout,
        )
    if name == "cpmlp_dsconv_nogru":
        return CPMLPDSConvNoGRUSOH(
            n_cycles=early_cycle,
            fixed_len=fixed_len,
            mlp_embed_dim=mlp_embed_dim,
            hidden=model_hidden,
            dsconv_channels=dsconv_channels,
            dropout=dropout,
        )
    if name == "cpdsconv":
        return CPDSConvSOH(
            n_cycles=early_cycle,
            channels=dsconv_channels,
            dropout=dropout,
        )
    if name == "cpmlp_cpdsconv_fusion":
        return CPMLPCPDSConvFusionSOH(
            n_cycles=early_cycle,
            fixed_len=fixed_len,
            mlp_embed_dim=mlp_embed_dim,
            hidden=model_hidden,
            dsconv_channels=dsconv_channels,
            dropout=dropout,
        )
    if name == "cpmlp_gru_residual":
        return CPMLPGRUResidualSOH(
            n_cycles=early_cycle,
            fixed_len=fixed_len,
            embed_dim=mlp_embed_dim,
            hidden=model_hidden,
            gru_hidden=gru_hidden,
            dropout=dropout,
            residual_scale=residual_scale,
        )
    if name == "flatten_mlp":
        return FlattenMLPSOH(
            n_cycles=early_cycle,
            fixed_len=fixed_len,
            hidden=model_hidden,
            dropout=dropout,
        )
    if name == "curve_cnn":
        return CurveCNNSOH(
            n_cycles=early_cycle,
            channels=dsconv_channels,
            dropout=dropout,
        )
    if name == "gru_only":
        return GRUOnlySOH(
            gru_hidden=gru_embed_dim,
            cycle_hidden=gru_hidden,
            dropout=dropout,
        )
    if name == "gru_dsconv":
        return pipe.GRUDSConvSOH(
            num_var=pipe.NUM_VAR,
            fixed_len=fixed_len,
            n_cycles=early_cycle,
            gru_hidden=gru_hidden,
            channels=dsconv_channels,
            dropout=dropout,
        )
    raise ValueError(f"unknown model: {name}")


def parse_models(value: str) -> list[str]:
    if value == "all":
        return MODEL_ORDER.copy()
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    return names


def run(args: argparse.Namespace) -> None:
    model_seed = args.seed
    split_seed = args.seed if args.split_seed < 0 else args.split_seed
    pipe.set_seed(model_seed)
    pipe.set_feature_mode(args.feature_mode)
    files = pipe.find_pkl_files(args.data_dir)
    if args.include_regex:
        include_re = re.compile(args.include_regex)
        files = [fp for fp in files if include_re.search(fp.name)]
    if args.exclude_regex:
        exclude_re = re.compile(args.exclude_regex)
        files = [fp for fp in files if not exclude_re.search(fp.name)]
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise ValueError("no pkl files found")

    selected_eval_domain = ""
    sample_split_details = []
    if args.split_mode == "battery":
        if len(files) < 3:
            raise ValueError("need at least 3 pkl files for train/val/test split")
        train_files, val_files, test_files = pipe.split_files(files, seed=split_seed)
        train = pipe.build_dataset(train_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
        val = pipe.build_dataset(val_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
        test = pipe.build_dataset(test_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
    elif args.split_mode == "condition-group":
        if len(files) < 3:
            raise ValueError("need at least 3 pkl files for train/val/test split")
        train_files, val_files, test_files = pipe.split_files_by_experiment_condition(files, seed=split_seed)
        train = pipe.build_dataset(train_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
        val = pipe.build_dataset(val_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
        test = pipe.build_dataset(test_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
    elif args.split_mode == "same-domain-eval":
        if len(files) < 3:
            raise ValueError("need at least 3 pkl files for train/val/test split")
        train_files, val_files, test_files, selected_eval_domain = pipe.split_files_same_domain_eval(
            files,
            seed=split_seed,
            eval_domain=args.eval_domain or None,
        )
        train = pipe.build_dataset(train_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
        val = pipe.build_dataset(val_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
        test = pipe.build_dataset(test_files, early_cycle=args.early_cycle, horizon=args.horizon, fixed_len=args.fixed_len)
    elif args.split_mode == "chronological-within-file":
        train_files = val_files = test_files = files
        train, val, test, sample_split_details = pipe.build_chronological_splits_within_files(
            files,
            early_cycle=args.early_cycle,
            horizon=args.horizon,
            fixed_len=args.fixed_len,
        )
    elif args.split_mode == "condition-gap-within-file":
        train, val, test, sample_split_details = pipe.build_condition_gap_splits_within_files(
            files,
            early_cycle=args.early_cycle,
            horizon=args.horizon,
            fixed_len=args.fixed_len,
            gap_samples=args.split_gap,
        )
        used_file_names = {item["file"] for item in sample_split_details if item.get("status") == "used"}
        used_files = [fp for fp in files if fp.name in used_file_names]
        train_files = val_files = test_files = used_files
    else:
        raise ValueError(f"unknown split mode: {args.split_mode}")

    X_train, X_val, X_test, X_mean, X_std = pipe.normalize_by_train(train.X, val.X, test.X)
    train_baseline = input_end_soh(train, train_files, fixed_len=args.fixed_len)
    val_baseline = input_end_soh(val, val_files, fixed_len=args.fixed_len)
    test_baseline = None if args.skip_test_eval else input_end_soh(test, test_files, fixed_len=args.fixed_len)
    y_train_model = make_training_targets(train, train_baseline, args.target_mode, args.target_scale)
    y_val_model = make_training_targets(val, val_baseline, args.target_mode, args.target_scale)
    y_test_model = None if args.skip_test_eval else make_training_targets(test, test_baseline, args.target_mode, args.target_scale)

    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predictions"
    val_pred_dir = out_dir / "validation_predictions"
    history_dir = out_dir / "histories"
    checkpoint_dir = out_dir / "checkpoints"
    pred_dir.mkdir(parents=True, exist_ok=True)
    val_pred_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    split_info = {
        "seed": args.seed,
        "model_seed": model_seed,
        "split_seed": split_seed,
        "split_mode": args.split_mode,
        "eval_domain": selected_eval_domain,
        "split_gap": args.split_gap,
        "early_cycle": args.early_cycle,
        "horizon": args.horizon,
        "fixed_len": args.fixed_len,
        "feature_mode": args.feature_mode,
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
        "test_files": [p.name for p in test_files],
        "file_domains": {p.name: pipe.infer_battery_domain(p) for p in files},
        "file_experiment_conditions": {p.name: pipe.infer_experiment_condition(p) for p in files},
        "train_domains": sorted({pipe.infer_battery_domain(p) for p in train_files}),
        "val_domains": sorted({pipe.infer_battery_domain(p) for p in val_files}),
        "test_domains": sorted({pipe.infer_battery_domain(p) for p in test_files}),
        "train_experiment_conditions": sorted({pipe.infer_experiment_condition(p) for p in train_files}),
        "val_experiment_conditions": sorted({pipe.infer_experiment_condition(p) for p in val_files}),
        "test_experiment_conditions": sorted({pipe.infer_experiment_condition(p) for p in test_files}),
        "sample_split_details": sample_split_details,
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "feature_names": pipe.FEATURE_NAMES,
        "target_mode": args.target_mode,
        "target_scale": args.target_scale,
        "normalization_mean": X_mean.reshape(-1).astype(float).tolist(),
        "normalization_std": X_std.reshape(-1).astype(float).tolist(),
        "train_baseline_soh_mean": float(np.nanmean(train_baseline)),
        "val_baseline_soh_mean": float(np.nanmean(val_baseline)),
        "test_baseline_soh_mean": None if args.skip_test_eval else float(np.nanmean(test_baseline)),
        "train_target_mean": float(np.nanmean(y_train_model)),
        "val_target_mean": float(np.nanmean(y_val_model)),
        "test_target_mean": None if args.skip_test_eval else float(np.nanmean(y_test_model)),
        "skip_test_eval": args.skip_test_eval,
        "residual_base_epochs": args.residual_base_epochs,
        "residual_finetune_base": args.residual_finetune_base,
        "residual_scale": args.residual_scale,
        "mlp_embed_dim": args.mlp_embed_dim,
        "gru_embed_dim": args.gru_embed_dim,
        "model_hidden": args.model_hidden,
        "gru_hidden": args.gru_hidden,
        "dsconv_channels": args.dsconv_channels,
        "dropout": args.dropout,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "huber_delta": args.huber_delta,
        "clip_grad_norm": args.clip_grad_norm,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "zero_output_init": args.zero_output_init,
    }
    (out_dir / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    test_loader = None if args.skip_test_eval else pipe.make_loader(X_test, y_test_model, batch_size=args.batch_size, shuffle=False)
    train_loader = pipe.make_loader(X_train, y_train_model, batch_size=args.batch_size, shuffle=True)
    val_loader = pipe.make_loader(X_val, y_val_model, batch_size=args.batch_size, shuffle=False)

    metrics_rows = []
    group_metric_rows = []
    selected_models = parse_models(args.models)

    if "persistence" in selected_models:
        val_pred = persistence_predict(val, val_files, fixed_len=args.fixed_len)
        val_result_df = pipe.make_result_df(val.meta, val.y, val_pred)
        val_result_df["baseline_soh"] = val_baseline
        val_result_df.to_csv(val_pred_dir / "persistence_val_predictions.csv", index=False)
        group_metric_rows.extend(collect_group_metric_rows("persistence", "val", val_result_df))

        row = {
            "model": "persistence",
            **prefix_metrics(compute_metrics(val.y, val_pred, val_result_df), "val"),
        }
        if not args.skip_test_eval:
            pred = persistence_predict(test, test_files, fixed_len=args.fixed_len)
            result_df = pipe.make_result_df(test.meta, test.y, pred)
            result_df["baseline_soh"] = test_baseline
            row.update(compute_metrics(test.y, pred, result_df))
            result_df.to_csv(pred_dir / "persistence_test_predictions.csv", index=False)
            group_metric_rows.extend(collect_group_metric_rows("persistence", "test", result_df))
        metrics_rows.append(row)
        write_metrics(metrics_rows, out_dir)
        write_group_metrics(group_metric_rows, out_dir)
        print(pd.Series(row).to_string())

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    for model_name in selected_models:
        if model_name == "persistence":
            continue

        pipe.set_seed(model_seed)
        base_history = None
        if model_name == "cpmlp_gru_residual" and args.residual_base_epochs != 0:
            base_epochs = args.epochs if args.residual_base_epochs < 0 else args.residual_base_epochs
            print(f"[cpmlp_gru_residual] pretrain CPMLP base for {base_epochs} epochs")
            base_model = CPMLPSOH(
                n_cycles=args.early_cycle,
                fixed_len=args.fixed_len,
                embed_dim=args.mlp_embed_dim,
                hidden=args.model_hidden,
                dropout=args.dropout,
            )
            base_model, base_history = train_model_compat(
                base_model,
                train_loader,
                val_loader,
                epochs=base_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                patience=args.patience,
                min_delta=args.min_delta,
                huber_delta=args.huber_delta,
                clip_grad_norm=args.clip_grad_norm,
                lr_scheduler_patience=args.lr_scheduler_patience,
                lr_scheduler_factor=args.lr_scheduler_factor,
            )
            model = CPMLPGRUResidualSOH(
                n_cycles=args.early_cycle,
                fixed_len=args.fixed_len,
                embed_dim=args.mlp_embed_dim,
                hidden=args.model_hidden,
                gru_hidden=args.gru_hidden,
                dropout=args.dropout,
                residual_scale=args.residual_scale,
                cpmlp=base_model,
                freeze_base=not args.residual_finetune_base,
            )
            freeze_label = "frozen" if not args.residual_finetune_base else "trainable"
            print(f"[cpmlp_gru_residual] train GRU residual branch with CPMLP base={freeze_label}")
        else:
            model = make_model(
                model_name,
                early_cycle=args.early_cycle,
                fixed_len=args.fixed_len,
                residual_scale=args.residual_scale,
                mlp_embed_dim=args.mlp_embed_dim,
                gru_embed_dim=args.gru_embed_dim,
                model_hidden=args.model_hidden,
                gru_hidden=args.gru_hidden,
                dsconv_channels=args.dsconv_channels,
                dropout=args.dropout,
            )
        if args.zero_output_init:
            zero_last_linear(model)
        model, history = train_model_compat(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            patience=args.patience,
            min_delta=args.min_delta,
            huber_delta=args.huber_delta,
            clip_grad_norm=args.clip_grad_norm,
            lr_scheduler_patience=args.lr_scheduler_patience,
            lr_scheduler_factor=args.lr_scheduler_factor,
        )
        val_raw_pred, val_raw_true = pipe.predict_loader(model, val_loader, device=device)
        val_pred = restore_soh_prediction(val_raw_pred, val_baseline, args.target_mode, args.target_scale)
        val_result_df = pipe.make_result_df(val.meta, val.y, val_pred)
        val_result_df["baseline_soh"] = val_baseline
        val_result_df["model_output"] = val_raw_pred
        val_result_df["model_target"] = val_raw_true
        if args.target_mode == "delta":
            val_result_df["actual_delta_soh"] = val_raw_true / args.target_scale
            val_result_df["pred_delta_soh"] = val_raw_pred / args.target_scale

        val_metrics = prefix_metrics(compute_metrics(val.y, val_pred, val_result_df), "val")
        group_metric_rows.extend(collect_group_metric_rows(model_name, "val", val_result_df))
        row = {"model": model_name, **val_metrics}
        if not args.skip_test_eval:
            raw_pred, raw_true = pipe.predict_loader(model, test_loader, device=device)
            pred = restore_soh_prediction(raw_pred, test_baseline, args.target_mode, args.target_scale)
            true = test.y
            result_df = pipe.make_result_df(test.meta, true, pred)
            result_df["baseline_soh"] = test_baseline
            result_df["model_output"] = raw_pred
            result_df["model_target"] = raw_true
            if args.target_mode == "delta":
                result_df["actual_delta_soh"] = raw_true / args.target_scale
                result_df["pred_delta_soh"] = raw_pred / args.target_scale
            row.update(compute_metrics(true, pred, result_df))
            group_metric_rows.extend(collect_group_metric_rows(model_name, "test", result_df))
        checkpoint_path = checkpoint_dir / f"{model_name}.pt"
        row["checkpoint_path"] = str(checkpoint_path)
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "split_info": split_info,
                "input_shape": list(X_train.shape[1:]),
                "feature_names": pipe.FEATURE_NAMES,
                "target_mode": args.target_mode,
                "normalization_mean": X_mean.astype(np.float32),
                "normalization_std": X_std.astype(np.float32),
                "metrics": row,
            },
            checkpoint_path,
        )
        metrics_rows.append(row)
        if base_history is not None:
            base_history = base_history.copy()
            base_history["stage"] = "cpmlp_base"
            history = history.copy()
            history["stage"] = "gru_residual"
            history = pd.concat([base_history, history], ignore_index=True)
        history.to_csv(history_dir / f"{model_name}_train_history.csv", index=False)
        val_result_df.to_csv(val_pred_dir / f"{model_name}_val_predictions.csv", index=False)
        if not args.skip_test_eval:
            result_df.to_csv(pred_dir / f"{model_name}_test_predictions.csv", index=False)
        write_metrics(metrics_rows, out_dir)
        write_group_metrics(group_metric_rows, out_dir)
        print(pd.Series(row).to_string())

    metrics = write_metrics(metrics_rows, out_dir)
    write_group_metrics(group_metric_rows, out_dir)
    print("\n=== model comparison ===")
    print(metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument("--output-dir", default="comparison_outputs")
    parser.add_argument("--fixed-len", type=int, default=pipe.FIXED_LEN)
    parser.add_argument("--early-cycle", type=int, default=pipe.EARLY_CYCLE)
    parser.add_argument("--horizon", type=int, default=pipe.HORIZON)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mlp-embed-dim", type=int, default=64)
    parser.add_argument("--gru-embed-dim", type=int, default=64)
    parser.add_argument("--model-hidden", type=int, default=256)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dsconv-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience on validation loss. 0 disables early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early stopping patience.",
    )
    parser.add_argument("--huber-delta", type=float, default=0.02)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=0,
        help="ReduceLROnPlateau patience. 0 disables the scheduler.",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument(
        "--zero-output-init",
        action="store_true",
        help=(
            "Zero-initialize the last Linear layer so delta-target models start from "
            "persistence-style zero delta predictions."
        ),
    )
    parser.add_argument("--seed", type=int, default=pipe.DEFAULT_SEED)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=-1,
        help="Battery-level split seed. -1 reuses --seed for backward-compatible behavior.",
    )
    parser.add_argument(
        "--split-mode",
        choices=[
            "battery",
            "condition-group",
            "same-domain-eval",
            "chronological-within-file",
            "condition-gap-within-file",
        ],
        default="battery",
        help=(
            "battery keeps the original file-level random split. "
            "condition-group keeps all files from the same inferred experiment condition in one split. "
            "same-domain-eval holds out one inferred domain and splits that same domain into validation/test. "
            "chronological-within-file splits sliding-window samples inside each pkl by target-cycle order. "
            "condition-gap-within-file does the same with experiment-condition labels and unused gap windows."
        ),
    )
    parser.add_argument(
        "--split-gap",
        type=int,
        default=5,
        help=(
            "Number of sliding-window samples to leave unused between train/val and val/test "
            "when --split-mode condition-gap-within-file is used."
        ),
    )
    parser.add_argument(
        "--eval-domain",
        default="",
        help=(
            "Domain label to hold out when --split-mode same-domain-eval is used. "
            "Leave blank to select an eligible domain from the split seed."
        ),
    )
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--models", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-mode", default=pipe.FEATURE_MODE, choices=sorted(pipe.FEATURE_NAMES_BY_MODE))
    parser.add_argument("--target-mode", default="absolute", choices=["absolute", "delta"])
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="Multiply training targets by this value and divide model outputs by it during SOH restoration.",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Do not compute or save test metrics/predictions. Use this during hyperparameter tuning.",
    )
    parser.add_argument("--include-regex", default="")
    parser.add_argument("--exclude-regex", default="")
    parser.add_argument(
        "--residual-base-epochs",
        type=int,
        default=-1,
        help=(
            "For cpmlp_gru_residual, pretrain the CPMLP base before residual training. "
            "-1 uses --epochs, 0 disables pretraining."
        ),
    )
    parser.add_argument(
        "--residual-finetune-base",
        action="store_true",
        help="Keep the pretrained CPMLP base trainable while learning the GRU residual branch.",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.1,
        help="Scale factor for the GRU correction added to the CPMLP prediction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
