"""Hugging Face Llama wrapper which injects 2-D-workspace soft tokens."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .spatial_memory import SpatialMemoryConfig, SpatialWorkspace


class LlamaWithSpatialMemory(nn.Module):
    """Keep Llama unmodified and insert memory tokens before the POS target."""

    def __init__(self, llama: nn.Module, spatial_config: SpatialMemoryConfig) -> None:
        super().__init__()
        self.llama = llama
        self.workspace = SpatialWorkspace(spatial_config)
        hidden_size = int(llama.config.hidden_size)
        self.memory_projection = nn.Linear(spatial_config.d_model, hidden_size, bias=False)

    def _action_embeddings(self, action_input_ids: Tensor, action_attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.llama.get_input_embeddings()(action_input_ids)
        weights = action_attention_mask[..., None].to(embedded.dtype)
        pooled = (embedded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
        return pooled, action_attention_mask.any(dim=2)

    def _insert_memory(
        self, token_embeddings: Tensor, attention_mask: Tensor, labels: Tensor | None, memory: Tensor, prompt_lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        sequences, masks, all_labels = [], [], []
        for index, prompt_length in enumerate(prompt_lengths.tolist()):
            real_length = int(attention_mask[index].sum().item())
            base = token_embeddings[index, :real_length]
            sequences.append(torch.cat((base[:prompt_length], memory[index], base[prompt_length:]), dim=0))
            masks.append(torch.ones(sequences[-1].shape[0], dtype=attention_mask.dtype, device=attention_mask.device))
            if labels is not None:
                current = labels[index, :real_length]
                blanks = torch.full((memory.shape[1],), -100, dtype=labels.dtype, device=labels.device)
                all_labels.append(torch.cat((current[:prompt_length], blanks, current[prompt_length:]), dim=0))
        width = max(sequence.shape[0] for sequence in sequences)
        hidden = token_embeddings.shape[-1]
        embeds = token_embeddings.new_zeros((len(sequences), width, hidden))
        mask = attention_mask.new_zeros((len(sequences), width))
        padded_labels = None if labels is None else labels.new_full((len(sequences), width), -100)
        for index, sequence in enumerate(sequences):
            embeds[index, : sequence.shape[0]] = sequence
            mask[index, : sequence.shape[0]] = masks[index]
            if padded_labels is not None:
                padded_labels[index, : sequence.shape[0]] = all_labels[index]
        return embeds, mask, padded_labels

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        visual_maps: Tensor,
        action_input_ids: Tensor,
        action_attention_mask: Tensor,
        prompt_lengths: Tensor,
        labels: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        actions, action_mask = self._action_embeddings(action_input_ids, action_attention_mask)
        memory = self.memory_projection(self.workspace(visual_maps, actions, action_mask))
        tokens = self.llama.get_input_embeddings()(input_ids)
        inputs_embeds, expanded_mask, expanded_labels = self._insert_memory(
            tokens, attention_mask, labels, memory, prompt_lengths
        )
        return self.llama(inputs_embeds=inputs_embeds, attention_mask=expanded_mask, labels=expanded_labels, **kwargs)
