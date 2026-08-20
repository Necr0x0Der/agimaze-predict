"""Event-triggered 2-D visual-memory Transformer.

The MAP renderer supplies only frame zero.  Each completed ``</ACT>`` triggers
one latent-frame update.  No intermediate map is discretised or supervised in
v0: gradients from the final POS answer flow directly through all feature maps.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tokenizer import PAD_TOKEN_ID, VOCAB_SIZE

PosReadout = Literal["full_text", "visual_only"]


@dataclass(frozen=True)
class VisualTransformerConfig:
    context_length: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_multiplier: int = 4
    dropout: float = 0.0
    canvas_height: int = 9
    canvas_width: int = 17
    visual_d_model: int = 128
    visual_spatial_layers: int = 2
    visual_temporal_layers: int = 2
    temporal_history: int = 8
    pos_readout: PosReadout = "full_text"
    visual_gate_init: float = 0.05
    vocab_size: int = VOCAB_SIZE
    pad_token_id: int = PAD_TOKEN_ID

    def __post_init__(self) -> None:
        if self.context_length < 2 or self.canvas_height <= 0 or self.canvas_width <= 0:
            raise ValueError("context_length and visual canvas dimensions must be positive")
        if min(self.d_model, self.visual_d_model, self.n_heads, self.n_layers) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_model % self.n_heads or self.visual_d_model % self.n_heads:
            raise ValueError("d_model and visual_d_model must both divide n_heads")
        if min(self.mlp_multiplier, self.visual_spatial_layers, self.visual_temporal_layers, self.temporal_history) <= 0:
            raise ValueError("MLP multiplier, visual layer counts, and temporal_history must be positive")
        if self.pos_readout not in {"full_text", "visual_only"}:
            raise ValueError("pos_readout must be 'full_text' or 'visual_only'")
        if not 0.0 < self.visual_gate_init < 1.0:
            raise ValueError("visual_gate_init must be strictly between zero and one")
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"vocab_size must be {VOCAB_SIZE}")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (part.transpose(1, 2) for part in (query, key, value))
        scores = (query @ key.transpose(-2, -1)) * (self.head_dim**-0.5)
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        weights = self.dropout(F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1))
        return self.proj((weights @ value).transpose(1, 2).contiguous().view(batch, length, width))


class CrossAttention(nn.Module):
    """Ordinary all-keys cross-attention; causality is owned by caller states."""

    def __init__(self, query_dim: int, key_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if query_dim % n_heads:
            raise ValueError("query_dim must divide n_heads")
        self.n_heads, self.head_dim = n_heads, query_dim // n_heads
        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(key_dim, query_dim)
        self.value = nn.Linear(key_dim, query_dim)
        self.proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_states: Tensor, key_value_states: Tensor) -> Tensor:
        batch, query_length, width = query_states.shape
        key_length = key_value_states.shape[1]
        query = self.query(query_states).view(batch, query_length, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.key(key_value_states).view(batch, key_length, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.value(key_value_states).view(batch, key_length, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) * (self.head_dim**-0.5)
        weights = self.dropout(F.softmax(scores, dim=-1))
        return self.proj((weights @ value).transpose(1, 2).contiguous().view(batch, query_length, width))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * multiplier), nn.GELU(), nn.Linear(d_model * multiplier, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class CausalTextBlock(nn.Module):
    """Text-only causal encoder used to form complete ACT event contexts."""

    def __init__(self, config: VisualTransformerConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config.d_model, config.mlp_multiplier, config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_1(x))
        return x + self.mlp(self.norm_2(x))


class TextBlock(nn.Module):
    def __init__(self, config: VisualTransformerConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.self_attention = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.norm_cross = nn.LayerNorm(config.d_model)
        self.visual_attention = CrossAttention(config.d_model, config.visual_d_model, config.n_heads, config.dropout)
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config.d_model, config.mlp_multiplier, config.dropout)
        init = math.log(config.visual_gate_init / (1.0 - config.visual_gate_init))
        self.visual_gate_logit = nn.Parameter(torch.tensor(init))

    def forward(self, x: Tensor, visual_states: Tensor) -> Tensor:
        """Read the frame current at each text position.

        ``visual_states`` is [B, T, cells, Dv], so every text token may attend
        to its own event-indexed visual frame without converting the frame into
        ordinary text tokens.
        """
        x = x + self.self_attention(self.norm_1(x))
        batch, length, width = x.shape
        if visual_states.shape[:2] != (batch, length):
            raise ValueError("visual_states must be [batch, text_length, cells, channels]")
        query = self.norm_cross(x).reshape(batch * length, 1, width)
        keys = visual_states.reshape(batch * length, visual_states.shape[2], visual_states.shape[3])
        visual = self.visual_attention(query, keys).reshape(batch, length, width)
        x = x + torch.sigmoid(self.visual_gate_logit) * visual
        return x + self.mlp(self.norm_2(x))


class SpatialBlock(nn.Module):
    def __init__(self, config: VisualTransformerConfig) -> None:
        super().__init__()
        width = config.visual_d_model
        self.norm_1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, config.n_heads, config.dropout)
        # The causal mask is wrong for spatial cells, so use symmetric MHA directly.
        self.symmetric_attention = nn.MultiheadAttention(width, config.n_heads, config.dropout, batch_first=True)
        self.norm_2 = nn.LayerNorm(width)
        self.mlp = FeedForward(width, config.mlp_multiplier, config.dropout)

    def forward(self, frame: Tensor) -> Tensor:
        batch, height, width, channels = frame.shape
        cells = frame.view(batch, height * width, channels)
        attention, _ = self.symmetric_attention(self.norm_1(cells), self.norm_1(cells), self.norm_1(cells), need_weights=False)
        cells = cells + attention
        return (cells + self.mlp(self.norm_2(cells))).view(batch, height, width, channels)


class TemporalBlock(nn.Module):
    def __init__(self, config: VisualTransformerConfig) -> None:
        super().__init__()
        width = config.visual_d_model
        self.norm_1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, config.n_heads, config.dropout)
        self.norm_2 = nn.LayerNorm(width)
        self.mlp = FeedForward(width, config.mlp_multiplier, config.dropout)

    def forward(self, history: Tensor) -> Tensor:
        # [B, frames, H, W, D] -> independent causal temporal stream per cell.
        batch, frames, height, width, channels = history.shape
        cells = history.permute(0, 2, 3, 1, 4).reshape(batch * height * width, frames, channels)
        cells = cells + self.attention(self.norm_1(cells))
        cells = cells + self.mlp(self.norm_2(cells))
        return cells.view(batch, height, width, frames, channels).permute(0, 3, 1, 2, 4)


class VisualMemory(nn.Module):
    def __init__(self, config: VisualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.character_embedding = nn.Embedding(256, config.visual_d_model)
        self.row_embedding = nn.Embedding(config.canvas_height, config.visual_d_model)
        self.column_embedding = nn.Embedding(config.canvas_width, config.visual_d_model)
        self.text_to_visual = nn.Linear(config.d_model, config.visual_d_model)
        self.spatial_blocks = nn.ModuleList([SpatialBlock(config) for _ in range(config.visual_spatial_layers)])
        self.temporal_blocks = nn.ModuleList([TemporalBlock(config) for _ in range(config.visual_temporal_layers)])

    def initial_frame(self, map_chars: Tensor) -> Tensor:
        if map_chars.shape[1:] != (self.config.canvas_height, self.config.canvas_width):
            raise ValueError("visual_maps does not match configured canvas dimensions")
        rows = torch.arange(self.config.canvas_height, device=map_chars.device)[:, None]
        columns = torch.arange(self.config.canvas_width, device=map_chars.device)[None, :]
        frame = self.character_embedding(map_chars)
        frame = frame + self.row_embedding(rows)[None, :, :, :] + self.column_embedding(columns)[None, :, :, :]
        for block in self.spatial_blocks:
            frame = block(frame)
        return frame

    def update(self, history: list[Tensor], action_context: Tensor) -> Tensor:
        frame = history[-1] + self.text_to_visual(action_context)[:, None, None, :]
        for block in self.spatial_blocks:
            frame = block(frame)
        temporal_history = torch.stack([*history[-(self.config.temporal_history - 1) :], frame], dim=1)
        for block in self.temporal_blocks:
            temporal_history = block(temporal_history)
        return temporal_history[:, -1]


class VisualTargetDecoderBlock(nn.Module):
    def __init__(self, config: VisualTransformerConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.self_attention = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.norm_cross = nn.LayerNorm(config.d_model)
        self.cross_attention = CrossAttention(config.d_model, config.visual_d_model, config.n_heads, config.dropout)
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config.d_model, config.mlp_multiplier, config.dropout)

    def forward(self, x: Tensor, visual_states: Tensor) -> Tensor:
        x = x + self.self_attention(self.norm_1(x))
        x = x + self.cross_attention(self.norm_cross(x), visual_states)
        return x + self.mlp(self.norm_2(x))


class VisualTransformer(nn.Module):
    """Text Transformer augmented by a persistent, event-triggered 2-D stream."""

    def __init__(self, config: VisualTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.visual_memory = VisualMemory(config)
        # This causal pass turns the entire completed ACT substring and its
        # preceding action history into the event write context. It does not
        # see a MAP; MAP remains exclusively in the visual channel.
        self.action_blocks = nn.ModuleList([CausalTextBlock(config) for _ in range(config.n_layers)])
        self.text_blocks = nn.ModuleList([TextBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        if config.pos_readout == "visual_only":
            self.target_position_embedding = nn.Embedding(config.context_length, config.d_model)
            self.target_blocks = nn.ModuleList([VisualTargetDecoderBlock(config) for _ in range(config.n_layers)])
            self.target_norm = nn.LayerNorm(config.d_model)
            self.target_output = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.target_output.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _visual_rollout(self, text_seed: Tensor, visual_maps: Tensor, event_positions: Tensor) -> Tensor:
        """Generate one continuous visual feature map per finished ACT event."""
        if event_positions.ndim != 2 or event_positions.shape[0] != text_seed.shape[0]:
            raise ValueError("event_positions must have shape [batch, events]")
        history = [self.visual_memory.initial_frame(visual_maps)]
        sequence_length = text_seed.shape[1]
        for event_index in range(event_positions.shape[1]):
            positions = event_positions[:, event_index]
            valid = positions.ge(0)
            # Invalid padded entries use a harmless position and retain their prior frame.
            safe_positions = positions.clamp(0, sequence_length - 1)
            contexts = text_seed[torch.arange(text_seed.shape[0], device=text_seed.device), safe_positions]
            updated = self.visual_memory.update(history, contexts)
            history.append(torch.where(valid[:, None, None, None], updated, history[-1]))
        return torch.stack(history, dim=1)

    def forward(
        self,
        input_ids: Tensor,
        *,
        visual_maps: Tensor,
        event_positions: Tensor,
        event_counts: Tensor,
        target_input_ids: Tensor | None = None,
    ) -> Tensor:
        if input_ids.ndim != 2 or visual_maps.ndim != 3:
            raise ValueError("input_ids must be [batch,text] and visual_maps [batch,height,width]")
        batch, length = input_ids.shape
        if length > self.config.context_length or event_counts.shape != input_ids.shape:
            raise ValueError("input length/event_counts violate model configuration")
        positions = torch.arange(length, device=input_ids.device)
        text_seed = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :])
        action_states = text_seed
        for block in self.action_blocks:
            action_states = block(action_states)
        frames = self._visual_rollout(action_states, visual_maps, event_positions)
        counts = event_counts.clamp(0, frames.shape[1] - 1)
        visual_for_text = frames[torch.arange(batch, device=input_ids.device)[:, None], counts]
        visual_for_text = visual_for_text.reshape(batch, length, -1, self.config.visual_d_model)
        text_states = text_seed
        # Per-token frame selection preserves continuous reads while writes
        # remain event-triggered only at completed </ACT> delimiters.
        for block in self.text_blocks:
            text_states = block(text_states, visual_for_text)
        if self.config.pos_readout == "full_text":
            return self.output(self.norm(text_states))
        if target_input_ids is None:
            raise ValueError("target_input_ids is required for visual_only readout")
        if target_input_ids.ndim != 2 or target_input_ids.shape[0] != batch:
            raise ValueError("target_input_ids must be [batch, target_length]")
        target_length = target_input_ids.shape[1]
        if target_length == 0 or target_length > self.config.context_length:
            raise ValueError("target_input_ids length violates model configuration")
        target_positions = torch.arange(target_length, device=input_ids.device)
        target_states = self.dropout(
            self.token_embedding(target_input_ids) + self.target_position_embedding(target_positions)[None, :, :]
        )
        final_visual = frames[:, -1].reshape(batch, -1, self.config.visual_d_model)
        for block in self.target_blocks:
            target_states = block(target_states, final_visual)
        return self.target_output(self.target_norm(target_states))


def target_cross_entropy(logits: Tensor, labels: Tensor, *, ignore_index: int = -100) -> Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch and sequence dimensions")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_index)
