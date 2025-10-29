"""Neural network models for temporal data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding used by Transformers."""

    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return self.pe[:seq_len].unsqueeze(0)


class TransformerBlock(nn.Module):
    """A single causal Transformer block with pre-norm layout."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        return x


class SSMEncoder(nn.Module):
    """State space model encoder for compressing long sequences."""

    def __init__(self, input_dim: int, state_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.rnn = nn.GRU(input_dim, state_dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(state_dim, state_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Return a compressed representation suitable for cross-attention."""

        states, _ = self.rnn(x)
        states = self.norm(states)
        states = self.dropout(states)
        return self.proj(states)


@dataclass
class TemporalTransformerConfig:
    input_dim: int
    model_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    num_classes: int
    regression_dim: int
    dropout: float = 0.1
    max_seq_len: int = 4096
    long_context_threshold: int = 1024


class TemporalTransformer(nn.Module):
    """Transformer tailored for temporal data with optional long-context fusion."""

    def __init__(
        self,
        config: TemporalTransformerConfig,
        ssm_encoder: Optional[SSMEncoder] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.model_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(config.model_dim, max_len=config.max_seq_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config.model_dim, config.num_heads, config.ff_dim, dropout=config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.cls_head = nn.Linear(config.model_dim, config.num_classes)
        self.reg_head = nn.Linear(config.model_dim, config.regression_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.long_context_threshold = config.long_context_threshold
        self.ssm_encoder = ssm_encoder
        if ssm_encoder is not None:
            self.cross_attn_norm = nn.LayerNorm(config.model_dim)
            self.cross_attn = nn.MultiheadAttention(
                config.model_dim, config.num_heads, dropout=config.dropout, batch_first=True
            )
            self.cross_dropout = nn.Dropout(config.dropout)
        else:
            self.cross_attn_norm = None
            self.cross_attn = None
            self.cross_dropout = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass returning classification logits and regression deltas."""

        if x.dim() != 3:
            raise ValueError("TemporalTransformer expects [batch, seq, feat] input tensors")

        _, seq_len, _ = x.shape
        device = x.device

        x = self.input_proj(x)
        pos = self.positional_encoding(x)
        x = x + pos
        x = self.dropout(x)

        attn_mask = self._causal_mask(seq_len, device)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        if self.ssm_encoder is not None and seq_len > self.long_context_threshold:
            memory = self.ssm_encoder(x)
            x_norm = self.cross_attn_norm(x)
            cross_output, _ = self.cross_attn(x_norm, memory, memory)
            x = x + self.cross_dropout(cross_output)

        x = self.final_norm(x)

        pooled = x[:, -1]
        logits_cls = self.cls_head(pooled)
        y_reg = self.reg_head(pooled)

        return {"logits_cls": logits_cls, "y_reg": y_reg}
