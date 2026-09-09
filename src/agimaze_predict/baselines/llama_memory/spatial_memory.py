"""A compact, differentiable 2-D workspace with event-triggered updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SpatialMemoryConfig:
    canvas_height: int = 9
    canvas_width: int = 17
    d_model: int = 256
    n_heads: int = 4
    spatial_layers: int = 2
    memory_tokens: int = 16

    def __post_init__(self) -> None:
        if min(self.canvas_height, self.canvas_width, self.d_model, self.n_heads, self.spatial_layers, self.memory_tokens) <= 0:
            raise ValueError("spatial-memory dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide n_heads")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class SpatialBlock(nn.Module):
    def __init__(self, config: SpatialMemoryConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(config.d_model, config.n_heads, batch_first=True)
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model), nn.GELU(), nn.Linear(4 * config.d_model, config.d_model)
        )

    def forward(self, frame: Tensor) -> Tensor:
        batch, height, width, channels = frame.shape
        cells = frame.reshape(batch, height * width, channels)
        normalized = self.norm_1(cells)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        cells = cells + attended
        return (cells + self.mlp(self.norm_2(cells))).reshape(batch, height, width, channels)


class SpatialWorkspace(nn.Module):
    """Map encoder, recurrent action updater, and learned readout tokens.

    The update is deliberately learned, rather than a hand-coded shift.  A
    shared action vector is written to the prior frame, global spatial blocks
    can condition each cell on walls/rivers elsewhere in the map, and a GRU
    preserves a distinct state at every map cell over the action sequence.
    """

    def __init__(self, config: SpatialMemoryConfig, *, action_embedding_dim: int) -> None:
        super().__init__()
        if action_embedding_dim <= 0:
            raise ValueError("action_embedding_dim must be positive")
        self.config = config
        self.characters = nn.Embedding(256, config.d_model)
        self.rows = nn.Embedding(config.canvas_height, config.d_model)
        self.columns = nn.Embedding(config.canvas_width, config.d_model)
        # The language backbone exposes its embedding width at construction
        # time.  Use it directly rather than LazyLinear: the optimizer and
        # parameter-count logging are created before the first training batch.
        self.action_projection = nn.Linear(action_embedding_dim, config.d_model)
        self.write_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.state_cell = nn.GRUCell(config.d_model, config.d_model)
        self.spatial = nn.ModuleList(SpatialBlock(config) for _ in range(config.spatial_layers))
        self.read_queries = nn.Parameter(torch.empty(config.memory_tokens, config.d_model))
        self.read_attention = nn.MultiheadAttention(config.d_model, config.n_heads, batch_first=True)
        nn.init.normal_(self.read_queries, std=0.02)

    def initial_frame(self, visual_maps: Tensor) -> Tensor:
        batch, height, width = visual_maps.shape
        if (height, width) != (self.config.canvas_height, self.config.canvas_width):
            raise ValueError("visual_maps does not match spatial-memory canvas")
        row_ids = torch.arange(height, device=visual_maps.device)[:, None]
        column_ids = torch.arange(width, device=visual_maps.device)[None, :]
        frame = self.characters(visual_maps)
        frame = frame + self.rows(row_ids)[None] + self.columns(column_ids)[None]
        for block in self.spatial:
            frame = block(frame)
        return frame

    def forward(self, visual_maps: Tensor, action_embeddings: Tensor, action_mask: Tensor) -> Tensor:
        """Return fixed-count soft tokens from the final map frame.

        ``action_embeddings`` is ``[batch, actions, language_hidden]`` and is
        usually a masked mean of frozen Llama token embeddings.  ``action_mask``
        distinguishes actual action events from batch padding.
        """

        if action_embeddings.ndim != 3 or action_mask.shape != action_embeddings.shape[:2]:
            raise ValueError("action embeddings/mask must be [batch, actions, channels]")
        frame = self.initial_frame(visual_maps)
        batch, height, width, channels = frame.shape
        for action_index in range(action_embeddings.shape[1]):
            active = action_mask[:, action_index].bool()
            action = self.action_projection(action_embeddings[:, action_index])
            prior = frame.reshape(batch * height * width, channels)
            write = action[:, None, None, :].expand(-1, height, width, -1).reshape_as(prior)
            gate = torch.sigmoid(self.write_gate(torch.cat((prior, write), dim=-1)))
            updated = self.state_cell(gate * write, prior).reshape_as(frame)
            for block in self.spatial:
                updated = block(updated)
            frame = torch.where(active[:, None, None, None], updated, frame)
        cells = frame.reshape(batch, height * width, channels)
        queries = self.read_queries[None].expand(batch, -1, -1)
        tokens, _ = self.read_attention(queries, cells, cells, need_weights=False)
        return tokens
