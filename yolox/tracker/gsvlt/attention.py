from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dimension: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, dimension * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * expansion, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float, ffn_expansion: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(dimension)
        self.feed_forward = FeedForward(dimension, dropout, ffn_expansion)
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            inputs,
            inputs,
            inputs,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(inputs + attended)
        output = self.output_norm(hidden + self.feed_forward(hidden))
        return output.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class SelfAttentionStack(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        layers: int,
        dropout: float,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SelfAttentionBlock(dimension, heads, dropout, ffn_expansion) for _ in range(layers)]
        )

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output, valid_mask)
        return output


class CrossAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float, ffn_expansion: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(dimension)
        self.feed_forward = FeedForward(dimension, dropout, ffn_expansion)
        self.output_norm = nn.LayerNorm(dimension)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            query,
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(query + attended)
        output = self.output_norm(hidden + self.feed_forward(hidden))
        return output.masked_fill(~query_mask.unsqueeze(-1), 0.0)


class CrossAttentionStack(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        layers: int,
        dropout: float,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(dimension, heads, dropout, ffn_expansion) for _ in range(layers)]
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = query
        for layer in self.layers:
            output = layer(output, memory, query_mask, memory_mask)
        return output
