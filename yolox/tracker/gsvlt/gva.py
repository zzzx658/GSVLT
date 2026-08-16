from __future__ import annotations

import math

import torch
from torch import nn

from .attention import CrossAttentionStack, SelfAttentionStack
from .config import GSVLTConfig
from .types import GVAOutput


def sinusoidal_encoding(offsets: torch.Tensor, dimension: int) -> torch.Tensor:
    if dimension % 2:
        raise ValueError("Sinusoidal encoding dimension must be even")
    offsets = offsets.to(dtype=torch.float32).reshape(-1, 1)
    exponent = torch.arange(0, dimension, 2, device=offsets.device, dtype=torch.float32)
    scale = torch.exp(-math.log(10000.0) * exponent / dimension)
    angles = offsets * scale.reshape(1, -1)
    encoding = torch.zeros(offsets.shape[0], dimension, device=offsets.device)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding


class MotionEncoder(nn.Module):
    def __init__(self, config: GSVLTConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.state_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        shape = sequences.shape
        if len(shape) != 4:
            raise ValueError("Motion sequences must have shape [B, G, N, 8]")
        flattened = sequences.reshape(shape[0] * shape[1], shape[2], shape[3])
        _, (hidden, _) = self.lstm(flattened)
        return hidden[-1].reshape(shape[0], shape[1], -1)


class DetectionEncoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
        )

    def forward(self, detections: torch.Tensor) -> torch.Tensor:
        return self.layers(detections)


class GVA(nn.Module):
    """Group Structure Variation Feature Extraction and Association."""

    def __init__(self, config: GSVLTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dimension = config.hidden_dim
        self.motion_encoder = MotionEncoder(config)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(dimension * 3, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
        )
        self.initial_structure = SelfAttentionStack(
            dimension, config.attention_heads, config.attention_layers, config.dropout
        )
        self.correlation_query = nn.Linear(dimension, dimension)
        self.correlation_key = nn.Linear(dimension, dimension)
        self.refined_structure = SelfAttentionStack(
            dimension, config.attention_heads, config.attention_layers, config.dropout
        )
        self.cross_time_current = CrossAttentionStack(
            dimension, config.attention_heads, config.attention_layers, config.dropout
        )
        self.cross_time_past = CrossAttentionStack(
            dimension, config.attention_heads, config.attention_layers, config.dropout
        )
        self.pgms_norm = nn.LayerNorm(dimension)

        self.detection_encoder = DetectionEncoder(dimension)
        self.detection_attention = SelfAttentionStack(
            dimension, config.attention_heads, config.attention_layers, config.dropout
        )
        self.pgms_to_detection = CrossAttentionStack(
            dimension,
            config.attention_heads,
            1,
            config.dropout,
            ffn_expansion=2,
        )
        self.detection_to_pgms = CrossAttentionStack(
            dimension, config.attention_heads, 1, config.dropout
        )
        self.pgms_attention = SelfAttentionStack(
            dimension, config.attention_heads, config.attention_layers, config.dropout
        )
        self.affinity_head = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dimension, 1),
        )

    def encode_group(
        self,
        past_sequences: torch.Tensor,
        current_sequences: torch.Tensor,
        initial_mask: torch.Tensor,
        survivor_mask: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_features = self.motion_encoder(past_sequences)
        current_features = self.motion_encoder(current_sequences)
        positional = sinusoidal_encoding(time_offsets, self.config.positional_dim)
        positional = positional.unsqueeze(1).expand(-1, past_features.shape[1], -1)
        current_features = self.temporal_fusion(
            torch.cat([current_features, past_features, positional], dim=-1)
        )

        initial_features = self.initial_structure(past_features, initial_mask)
        queries = self.correlation_query(initial_features)
        keys = self.correlation_key(initial_features)
        correlation = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(
            self.config.hidden_dim
        )
        correlation = correlation.masked_fill(~initial_mask.unsqueeze(1), -1e4)
        attention = torch.softmax(correlation, dim=-1)

        core_mask = initial_mask & survivor_mask
        core_mask = core_mask & (attention[:, 0, :] >= self.config.attention_threshold)
        # The target is always retained as index zero and acts as the current anchor.
        core_mask[:, 0] = True

        refined_past = self.refined_structure(past_features, core_mask)
        refined_current = self.refined_structure(current_features, core_mask)
        current_cross = self.cross_time_current(refined_current, refined_past, core_mask, core_mask)
        past_cross = self.cross_time_past(refined_past, refined_current, core_mask, core_mask)
        pgms = self.pgms_norm((current_cross[:, 0] + past_cross[:, 0]) / 2.0)
        return pgms, attention, core_mask

    def score_candidates(
        self,
        pgms: torch.Tensor,
        detections: torch.Tensor,
        detection_mask: torch.Tensor,
    ) -> torch.Tensor:
        detection_features = self.detection_encoder(detections)
        detection_features = self.detection_attention(detection_features, detection_mask)
        pgms_mask = torch.ones(pgms.shape[0], 1, dtype=torch.bool, device=pgms.device)
        pgms_token = self.pgms_attention(pgms.unsqueeze(1), pgms_mask)
        enhanced_pgms = self.pgms_to_detection(
            pgms_token, detection_features, pgms_mask, detection_mask
        )
        enhanced_detections = self.detection_to_pgms(
            detection_features, pgms_token, detection_mask, pgms_mask
        )
        expanded_pgms = enhanced_pgms.expand(-1, detections.shape[1], -1)
        logits = self.affinity_head(
            torch.cat([expanded_pgms, enhanced_detections], dim=-1)
        ).squeeze(-1)
        return logits.masked_fill(~detection_mask, -1e4)

    def forward(
        self,
        past_sequences: torch.Tensor,
        current_sequences: torch.Tensor,
        initial_mask: torch.Tensor,
        survivor_mask: torch.Tensor,
        time_offsets: torch.Tensor,
        detections: torch.Tensor,
        detection_mask: torch.Tensor,
    ) -> GVAOutput:
        pgms, attention, core_mask = self.encode_group(
            past_sequences,
            current_sequences,
            initial_mask,
            survivor_mask,
            time_offsets,
        )
        logits = self.score_candidates(pgms, detections, detection_mask)
        return GVAOutput(logits=logits, pgms=pgms, attention=attention, core_mask=core_mask)

    def affinity(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> GVAOutput:
        output = self.forward(*args, **kwargs)
        return GVAOutput(
            logits=torch.sigmoid(output.logits),
            pgms=output.pgms,
            attention=output.attention,
            core_mask=output.core_mask,
        )
