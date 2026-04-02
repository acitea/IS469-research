"""Context-conditioned encoder: cross-attention on a frozen BGE backbone.

Architecture:
  1. Frozen backbone (BAAI/bge-base-en-v1.5) encodes chunk and context separately
  2. Trainable cross-attention: chunk hidden states attend to context hidden states
  3. Residual + LayerNorm → mean pool → projection head → 768-dim embedding

For contexts longer than the backbone's 512-token limit, the preceding text
is split into 512-token windows, each encoded separately, and the hidden
states are concatenated before cross-attention.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from contextrag.config import (
    BACKBONE_DIM,
    BACKBONE_MAX_TOKENS,
    BACKBONE_MODEL,
    CROSS_ATTENTION_HEADS,
)

logger = logging.getLogger(__name__)


class ContextConditionedEncoder(nn.Module):
    """Produces chunk embeddings conditioned on preceding document context.

    The backbone is frozen — only the cross-attention layer and projection
    head are trained (~2.4M parameters for 768-dim, 8-head attention).
    """

    def __init__(
        self,
        backbone_name: str = BACKBONE_MODEL,
        d_model: int = BACKBONE_DIM,
        n_heads: int = CROSS_ATTENTION_HEADS,
        max_tokens: int = BACKBONE_MAX_TOKENS,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens

        # Frozen backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable cross-attention: chunk (Q) attends to context (K, V)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    @property
    def device(self) -> torch.device:
        return next(self.projection.parameters()).device

    # ------------------------------------------------------------------
    # Low-level encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_window(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a single window through the frozen backbone.

        Returns per-token hidden states ``(batch, seq_len, d_model)``.
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def _tokenize(self, texts: list[str], max_length: int | None = None) -> dict[str, torch.Tensor]:
        max_length = max_length or self.max_tokens
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

    # ------------------------------------------------------------------
    # Public encoding helpers
    # ------------------------------------------------------------------

    def encode_chunks(self, chunk_texts: list[str]) -> torch.Tensor:
        """Encode chunk texts → per-token hidden states ``(B, M, d_model)``."""
        tok = self._tokenize(chunk_texts)
        return self._encode_window(tok["input_ids"], tok["attention_mask"])

    def encode_contexts(self, context_texts: list[str]) -> torch.Tensor:
        """Encode context texts → per-token hidden states ``(B, N, d_model)``.

        For texts longer than ``max_tokens``, splits into overlapping windows,
        encodes each, and concatenates the hidden states along the sequence
        dimension.  This allows the cross-attention layer to attend over
        arbitrarily long context.
        """
        # Fast path: if all contexts fit in a single window
        tok = self.tokenizer(context_texts, padding=False, truncation=False)
        max_len = max(len(ids) for ids in tok["input_ids"])
        if max_len <= self.max_tokens:
            t = self._tokenize(context_texts)
            return self._encode_window(t["input_ids"], t["attention_mask"])

        # Slow path: windowed encoding for long contexts
        all_hidden: list[torch.Tensor] = []
        for text in context_texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= self.max_tokens - 2:  # room for [CLS] and [SEP]
                t = self._tokenize([text])
                all_hidden.append(self._encode_window(t["input_ids"], t["attention_mask"]).squeeze(0))
            else:
                # Split into windows with 50% overlap
                window_size = self.max_tokens - 2
                step = window_size // 2
                windows: list[torch.Tensor] = []
                for start in range(0, len(ids), step):
                    window_ids = ids[start : start + window_size]
                    # Add special tokens
                    full_ids = [self.tokenizer.cls_token_id] + window_ids + [self.tokenizer.sep_token_id]
                    input_tensor = torch.tensor([full_ids], device=self.device)
                    mask = torch.ones_like(input_tensor)
                    hidden = self._encode_window(input_tensor, mask).squeeze(0)
                    # Drop [CLS] and [SEP] hidden states to avoid duplication
                    windows.append(hidden[1:-1])
                    if start + window_size >= len(ids):
                        break
                all_hidden.append(torch.cat(windows, dim=0))

        # Pad to same sequence length and stack
        max_seq = max(h.size(0) for h in all_hidden)
        padded = torch.zeros(len(all_hidden), max_seq, self.d_model, device=self.device)
        for i, h in enumerate(all_hidden):
            padded[i, : h.size(0)] = h
        return padded

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        chunk_texts: list[str],
        context_texts: list[str],
    ) -> torch.Tensor:
        """Produce context-conditioned chunk embeddings.

        Args:
            chunk_texts: The chunk texts to embed.
            context_texts: The preceding document context for each chunk.

        Returns:
            Tensor of shape ``(batch, d_model)`` — the conditioned embeddings.
        """
        chunk_hidden = self.encode_chunks(chunk_texts)        # (B, M, 768)
        context_hidden = self.encode_contexts(context_texts)  # (B, N, 768)

        # Cross-attention: chunk queries attend to context keys/values
        attended, _ = self.cross_attention(
            query=chunk_hidden,
            key=context_hidden,
            value=context_hidden,
        )

        # Residual connection + layer norm
        conditioned = self.layer_norm(chunk_hidden + attended)  # (B, M, 768)

        # Mean pool over sequence → project
        pooled = conditioned.mean(dim=1)  # (B, 768)
        return self.projection(pooled)    # (B, 768)

    def encode_no_context(self, chunk_texts: list[str]) -> torch.Tensor:
        """Encode chunks without any context (baseline mode).

        Skips cross-attention entirely — produces a pure backbone embedding
        with only the projection head applied.
        """
        chunk_hidden = self.encode_chunks(chunk_texts)  # (B, M, 768)
        pooled = chunk_hidden.mean(dim=1)               # (B, 768)
        return self.projection(pooled)                   # (B, 768)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_trainable(self, path: str) -> None:
        """Save only the trainable parameters (cross-attention + projection)."""
        state = {
            "cross_attention": self.cross_attention.state_dict(),
            "layer_norm": self.layer_norm.state_dict(),
            "projection": self.projection.state_dict(),
        }
        torch.save(state, path)
        logger.info("Saved trainable weights to %s", path)

    def load_trainable(self, path: str, map_location: Optional[str] = None) -> None:
        """Load only the trainable parameters."""
        state = torch.load(path, map_location=map_location or self.device, weights_only=True)
        self.cross_attention.load_state_dict(state["cross_attention"])
        self.layer_norm.load_state_dict(state["layer_norm"])
        self.projection.load_state_dict(state["projection"])
        logger.info("Loaded trainable weights from %s", path)
