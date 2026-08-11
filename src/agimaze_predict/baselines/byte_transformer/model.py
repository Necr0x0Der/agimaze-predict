"""A small vanilla decoder-only byte Transformer, trained from scratch."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tokenizer import PAD_TOKEN_ID, VOCAB_SIZE, VOCAB_SIZE_WITH_STATE


@dataclass(frozen=True)
class ByteTransformerConfig:
    context_length: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    mlp_multiplier: int = 4
    dropout: float = 0.0
    state_tokens: int = 0
    # None selects the vocabulary required by state_tokens. An explicit value
    # remains accepted so historical checkpoint model_config dictionaries load.
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
        if self.state_tokens < 0:
            raise ValueError("state_tokens must be non-negative")
        expected_vocab_size = VOCAB_SIZE_WITH_STATE if self.state_tokens else VOCAB_SIZE
        if self.vocab_size is None:
            object.__setattr__(self, "vocab_size", expected_vocab_size)
        elif self.vocab_size != expected_vocab_size:
            raise ValueError(
                "vocab_size must match state_tokens: "
                f"expected {expected_vocab_size} for state_tokens={self.state_tokens}"
            )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ByteTransformerConfig) -> None:
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
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model)
        return self.residual_dropout(self.proj(output))


class TransformerBlock(nn.Module):
    def __init__(self, config: ByteTransformerConfig) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_1(x))
        return x + self.mlp(self.norm_2(x))


class ByteTransformer(nn.Module):
    """Vanilla GPT-style model with learned positions and tied byte embeddings."""

    def __init__(self, config: ByteTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
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

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.config.context_length}"
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        # Padded positions occur only after a sequence's real bytes. Causality
        # prevents real positions from attending to them, while their labels are
        # masked. Avoiding a key-padding mask also prevents all-masked padded
        # query rows from producing NaNs.
        for block in self.blocks:
            x = block(x)
        # STATE_TOKEN_ID is an input-only latent-slot embedding.  It is never a
        # valid prediction target: making it an output class would add a
        # permanent "wrong" logit to every target-byte softmax in the state-slot
        # arm, which is an avoidable and K-dependent ablation confound.  Keep
        # ``self.output`` at the full vocabulary size so checkpoints from the
        # initial implementation remain loadable, but expose only the legacy
        # byte + PAD prediction vocabulary to the loss and decoder.
        return self.output(self.norm(x))[..., :VOCAB_SIZE]


def target_cross_entropy(logits: Tensor, labels: Tensor, *, ignore_index: int = -100) -> Tensor:
    """Mean next-byte cross entropy over unmasked target bytes only."""

    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch and sequence dimensions")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_index)
