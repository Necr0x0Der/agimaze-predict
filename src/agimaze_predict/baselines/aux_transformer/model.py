"""Parallel causal main/aux Transformer for the latent-stream experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tokenizer import PAD_TOKEN_ID, VOCAB_SIZE

AuxGateMode = Literal["off", "fixed", "open", "learned"]


@dataclass(frozen=True)
class AuxTransformerConfig:
    context_length: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_multiplier: int = 4
    dropout: float = 0.0
    # A separate causal latent stream. Each source byte before the target owns
    # this many latent slots. AuxTransformer intentionally has no STATE-token
    # compatibility path; that remains entirely in byte_transformer.
    aux_latents_per_token: int = 0
    aux_gate_mode: AuxGateMode = "learned"
    aux_scale: float = 1.0
    aux_gate_init: float = 0.05
    # Optional target-aligned supervision.  A small causal decoder reads only
    # final source-derived aux states plus teacher-forced prior target bytes.
    # It is training-only and is discarded at inference.
    aux_target_weight: float = 0.0
    aux_target_decoder_layers: int = 1
    # The aux model always uses the ordinary byte + PAD input vocabulary.
    vocab_size: int | None = None
    pad_token_id: int = PAD_TOKEN_ID

    def __post_init__(self) -> None:
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("d_model, n_layers, and n_heads must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.mlp_multiplier <= 0:
            raise ValueError("mlp_multiplier must be positive")
        if self.aux_latents_per_token <= 0:
            raise ValueError("aux_latents_per_token must be positive for AuxTransformer")
        if self.aux_gate_mode not in {"off", "fixed", "open", "learned"}:
            raise ValueError("aux_gate_mode must be one of: off, fixed, open, learned")
        if self.aux_scale < 0:
            raise ValueError("aux_scale must be non-negative")
        if not 0.0 < self.aux_gate_init < 1.0:
            raise ValueError("aux_gate_init must be strictly between zero and one")
        if self.aux_target_weight < 0:
            raise ValueError("aux_target_weight must be non-negative")
        if self.aux_target_decoder_layers <= 0:
            raise ValueError("aux_target_decoder_layers must be positive")
        if self.vocab_size is None:
            object.__setattr__(self, "vocab_size", VOCAB_SIZE)
        elif self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"vocab_size must be {VOCAB_SIZE} for AuxTransformer")

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AuxTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, d_model = x.shape
        qkv = self.qkv(x).view(batch_size, sequence_length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) * (self.head_dim**-0.5)
        causal_mask = torch.ones(
            sequence_length, sequence_length, dtype=torch.bool, device=x.device
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model)
        return self.residual_dropout(self.proj(output))


class CausalCrossAttention(nn.Module):
    """Cross-attention whose available key prefix is specified per query.

    ``key_max_indices[b, q]`` is the inclusive final key position visible to a
    query.  The caller supplies only non-negative maxima, ensuring every row
    retains at least one key and cannot produce an all-masked softmax.
    """

    def __init__(self, config: AuxTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key_value = nn.Linear(config.d_model, 2 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, query_states: Tensor, key_value_states: Tensor, key_max_indices: Tensor) -> Tensor:
        batch_size, query_length, d_model = query_states.shape
        key_length = key_value_states.shape[1]
        if key_value_states.shape[0] != batch_size or key_value_states.shape[2] != d_model:
            raise ValueError("query and key/value states must agree on batch and model dimensions")
        if key_max_indices.shape != (batch_size, query_length):
            raise ValueError("key_max_indices must have shape [batch, query_length]")
        if key_length == 0 or key_max_indices.min().item() < 0 or key_max_indices.max().item() >= key_length:
            raise ValueError("key_max_indices must identify a non-empty prefix of key/value states")

        query = self.query(query_states).view(
            batch_size, query_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        key_value = self.key_value(key_value_states).view(
            batch_size, key_length, 2, self.n_heads, self.head_dim
        )
        key, value = key_value.unbind(dim=2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) * (self.head_dim**-0.5)
        key_positions = torch.arange(key_length, device=query_states.device)[None, None, :]
        unavailable = key_positions > key_max_indices[:, :, None]
        scores = scores.masked_fill(unavailable[:, None, :, :], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, query_length, d_model)
        return self.residual_dropout(self.proj(output))


class TransformerBlock(nn.Module):
    """Historical main block, optionally augmented with aux-to-main attention."""

    def __init__(self, config: AuxTransformerConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_multiplier * config.d_model),
            nn.GELU(),
            nn.Linear(config.mlp_multiplier * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.has_aux = config.aux_latents_per_token > 0
        self.aux_gate_mode = config.aux_gate_mode
        self.aux_scale = config.aux_scale
        if self.has_aux:
            self.cross_norm = nn.LayerNorm(config.d_model)
            self.cross_attention = CausalCrossAttention(config)
            if config.aux_gate_mode == "learned":
                initial_logit = math.log(config.aux_gate_init / (1.0 - config.aux_gate_init))
                self.aux_gate_logit = nn.Parameter(torch.tensor(initial_logit))

    def gate_value(self) -> Tensor:
        if not self.has_aux:
            return torch.tensor(0.0, device=self.norm_1.weight.device)
        if self.aux_gate_mode == "off":
            return torch.tensor(0.0, device=self.norm_1.weight.device)
        if self.aux_gate_mode in {"fixed", "open"}:
            return torch.tensor(1.0, device=self.norm_1.weight.device)
        return torch.sigmoid(self.aux_gate_logit)

    def forward(
        self,
        x: Tensor,
        *,
        auxiliary_states: Tensor | None = None,
        aux_key_max_indices: Tensor | None = None,
        disable_aux: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        x = x + self.attention(self.norm_1(x))
        diagnostics: dict[str, Tensor] = {}
        if self.has_aux:
            if auxiliary_states is None or aux_key_max_indices is None:
                raise ValueError("auxiliary states and their causal mask are required when aux is enabled")
            signal = self.cross_attention(self.cross_norm(x), auxiliary_states, aux_key_max_indices)
            gate = self.gate_value()
            # ``open`` is deliberately fully open: it ignores aux_scale so it
            # is the clean no-gating / unit-residual ablation.  ``fixed`` and
            # ``learned`` retain aux_scale as the external experiment control.
            scale = 1.0 if self.aux_gate_mode == "open" else self.aux_scale
            contribution = signal * (0.0 if disable_aux else scale) * gate
            diagnostics = {
                "gate": gate.detach(),
                "cross_residual_ratio": (
                    contribution.norm(dim=-1).mean() / (x.norm(dim=-1).mean() + 1e-8)
                ).detach(),
            }
            x = x + contribution
        return x + self.mlp(self.norm_2(x)), diagnostics


class AuxiliaryTransformerBlock(nn.Module):
    """A causal latent block which reads only the permitted main prefix."""

    def __init__(self, config: AuxTransformerConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = CausalCrossAttention(config)
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_multiplier * config.d_model),
            nn.GELU(),
            nn.Linear(config.mlp_multiplier * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, z: Tensor, main_states: Tensor, main_key_max_indices: Tensor) -> Tensor:
        z = z + self.attention(self.norm_1(z))
        z = z + self.cross_attention(self.cross_norm(z), main_states, main_key_max_indices)
        return z + self.mlp(self.norm_2(z))


class AuxiliaryTargetDecoderBlock(nn.Module):
    """One causal target-decoder block cross-attending to source aux states."""

    def __init__(self, config: AuxTransformerConfig) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = CausalCrossAttention(config)
        self.norm_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_multiplier * config.d_model),
            nn.GELU(),
            nn.Linear(config.mlp_multiplier * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, auxiliary_states: Tensor, aux_key_max_indices: Tensor) -> Tensor:
        x = x + self.attention(self.norm_1(x))
        x = x + self.cross_attention(self.cross_norm(x), auxiliary_states, aux_key_max_indices)
        return x + self.mlp(self.norm_2(x))


class AuxTransformer(nn.Module):
    """GPT-style byte model with an optional parallel causal latent stream.

    The auxiliary stream starts with zero content embeddings plus learned latent
    position embeddings.  Its length and the final permitted source byte are
    supplied by the collator as ``aux_lengths``.  This explicit boundary is the
    no-target-leakage invariant: neither ground-truth POS bytes nor generated
    POS bytes are ever provided to the auxiliary Transformer.
    """

    def __init__(self, config: AuxTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        if config.aux_latents_per_token:
            self.aux_position_embedding = nn.Embedding(
                config.context_length * config.aux_latents_per_token, config.d_model
            )
            self.aux_blocks = nn.ModuleList(
                [AuxiliaryTransformerBlock(config) for _ in range(config.n_layers)]
            )
            if config.aux_target_weight > 0.0:
                self.aux_target_position_embedding = nn.Embedding(config.context_length, config.d_model)
                self.aux_target_blocks = nn.ModuleList(
                    [AuxiliaryTargetDecoderBlock(config) for _ in range(config.aux_target_decoder_layers)]
                )
                self.aux_target_norm = nn.LayerNorm(config.d_model)
                self.aux_target_output = nn.Linear(config.d_model, config.vocab_size, bias=False)
                self.aux_target_output.weight = self.token_embedding.weight
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _auxiliary_masks(self, *, sequence_length: int, aux_lengths: Tensor) -> tuple[Tensor, Tensor, int]:
        """Build the two causal cross-stream masks from source-prefix lengths."""

        if aux_lengths.ndim != 1 or aux_lengths.numel() == 0:
            raise ValueError("aux_lengths must have shape [batch]")
        if aux_lengths.min().item() <= 0 or aux_lengths.max().item() > sequence_length:
            raise ValueError("each aux length must be within the non-empty main input prefix")

        slots = self.config.aux_latents_per_token
        auxiliary_length = int(aux_lengths.max().item()) * slots
        aux_positions = torch.arange(auxiliary_length, device=aux_lengths.device)
        source_positions = aux_positions // slots
        # A latent at j can read main states through its associated source byte.
        # Padded latent rows are harmless because no real main position may read
        # them; keeping their causal source bound valid avoids all-masked rows.
        aux_to_main = source_positions[None, :].expand(aux_lengths.numel(), -1).clone()
        aux_to_main = torch.minimum(aux_to_main, aux_lengths[:, None] - 1)

        # Main position i may read auxiliary slots only when their source byte
        # exists in this example and lies at or before i. The query-wise maximum
        # alone cannot express a per-example key-length constraint, so padded
        # latent keys are assigned a zero value in forward instead.
        main_positions = torch.arange(sequence_length, device=aux_lengths.device)
        main_to_aux = (main_positions[None, :] + 1) * slots - 1
        main_to_aux = main_to_aux.expand(aux_lengths.numel(), -1).clone()
        main_to_aux = torch.minimum(main_to_aux, aux_lengths[:, None] * slots - 1)
        return aux_to_main, main_to_aux, auxiliary_length

    def forward(
        self,
        input_ids: Tensor,
        *,
        aux_lengths: Tensor | None = None,
        disable_aux: bool = False,
        return_aux_diagnostics: bool = False,
        return_auxiliary_states: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]] | tuple[Tensor, Tensor] | tuple[Tensor, dict[str, float], Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.config.context_length}"
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        main_states = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        )

        diagnostics: dict[str, float] = {}
        auxiliary_states: Tensor | None = None
        if self.config.aux_latents_per_token:
            if aux_lengths is None:
                raise ValueError("aux_lengths is required when aux_latents_per_token is positive")
            aux_lengths = aux_lengths.to(device=input_ids.device, dtype=torch.long)
            if aux_lengths.shape != (batch_size,):
                raise ValueError("aux_lengths must have shape [batch]")
            aux_to_main, main_to_aux, auxiliary_length = self._auxiliary_masks(
                sequence_length=sequence_length, aux_lengths=aux_lengths
            )
            aux_positions = torch.arange(auxiliary_length, device=input_ids.device)
            # Content is zero by construction; position embeddings identify
            # distinct latent slots without introducing a latent vocabulary.
            auxiliary_states = self.aux_position_embedding(aux_positions)[None, :, :].expand(
                batch_size, -1, -1
            )
            # Batch padding must not provide an additional latent value for a
            # shorter example. The causal key maxima below ensure these rows
            # are not selected by real main queries; zeroing makes that fact
            # explicit and prevents padded aux self-attention from mattering.
            aux_valid = aux_positions[None, :] < (aux_lengths[:, None] * self.config.aux_latents_per_token)
            auxiliary_states = auxiliary_states * aux_valid[:, :, None]

            for layer, (main_block, aux_block) in enumerate(zip(self.blocks, self.aux_blocks, strict=True)):
                previous_main, previous_aux = main_states, auxiliary_states
                main_states, layer_diagnostics = main_block(
                    previous_main,
                    auxiliary_states=previous_aux,
                    aux_key_max_indices=main_to_aux,
                    disable_aux=disable_aux,
                )
                auxiliary_states = aux_block(previous_aux, previous_main, aux_to_main)
                if return_aux_diagnostics:
                    diagnostics[f"gate/layer_{layer}"] = float(layer_diagnostics["gate"].item())
                    diagnostics[f"cross_residual_ratio/layer_{layer}"] = float(
                        layer_diagnostics["cross_residual_ratio"].item()
                    )
            if return_aux_diagnostics:
                diagnostics["aux/mean_latent_norm"] = float(auxiliary_states.norm(dim=-1).mean().item())
        else:
            for main_block in self.blocks:
                main_states, _ = main_block(main_states)

        logits = self.output(self.norm(main_states))
        if return_aux_diagnostics and return_auxiliary_states:
            assert auxiliary_states is not None
            return logits, diagnostics, auxiliary_states
        if return_aux_diagnostics:
            return logits, diagnostics
        if return_auxiliary_states:
            assert auxiliary_states is not None
            return logits, auxiliary_states
        return logits

    def auxiliary_target_logits(
        self,
        target_input_ids: Tensor,
        *,
        auxiliary_states: Tensor,
        aux_lengths: Tensor,
    ) -> Tensor:
        """Teacher-forced target logits from source-only auxiliary states.

        ``auxiliary_states`` must come from :meth:`forward` on the ordinary
        clean sequence.  The auxiliary stream itself is source bounded by
        ``aux_lengths`` at every layer.  This decoder may see prior target
        bytes in ``target_input_ids``, but target bytes never enter its keys or
        values, so it cannot teach the aux representation by target leakage.
        """

        if not hasattr(self, "aux_target_blocks"):
            raise ValueError("auxiliary target supervision is not enabled in this model configuration")
        if target_input_ids.ndim != 2:
            raise ValueError("target_input_ids must have shape [batch, target_sequence]")
        batch_size, target_length = target_input_ids.shape
        if target_length == 0 or target_length > self.config.context_length:
            raise ValueError("target sequence length must be in [1, context_length]")
        if auxiliary_states.shape[0] != batch_size or auxiliary_states.shape[2] != self.config.d_model:
            raise ValueError("auxiliary_states must agree with target inputs on batch and model dimensions")
        aux_lengths = aux_lengths.to(device=target_input_ids.device, dtype=torch.long)
        if aux_lengths.shape != (batch_size,):
            raise ValueError("aux_lengths must have shape [batch]")
        expected_aux_length = int(aux_lengths.max().item()) * self.config.aux_latents_per_token
        if auxiliary_states.shape[1] != expected_aux_length:
            raise ValueError("auxiliary_states length does not match aux_lengths")

        positions = torch.arange(target_length, device=target_input_ids.device)
        target_states = self.dropout(
            self.token_embedding(target_input_ids)
            + self.aux_target_position_embedding(positions)[None, :, :]
        )
        # Every target query may read the complete source-derived aux prefix
        # for its example, but never padded latent slots from a longer example.
        aux_key_max_indices = (aux_lengths * self.config.aux_latents_per_token - 1)[:, None].expand(
            -1, target_length
        )
        for block in self.aux_target_blocks:
            target_states = block(target_states, auxiliary_states, aux_key_max_indices)
        return self.aux_target_output(self.aux_target_norm(target_states))


def target_cross_entropy(logits: Tensor, labels: Tensor, *, ignore_index: int = -100) -> Tensor:
    """Mean next-byte cross entropy over unmasked target bytes only."""

    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch and sequence dimensions")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_index)
