"""Embedder implementations: default, contextual with pooling, and Voyage contextualized."""

from __future__ import annotations

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from contextrag.config import (
    OUTPUT_DIMENSION,
    POOLING_STRATEGY,
    CONTEXT_POOLING_OVERLAP_RATIO,
    VOYAGE_API_KEY,
)
from contextrag.models import Chunk, ContextualChunk
from contextrag.openai_utils import embed_texts, count_tokens, split_text_into_token_windows

logger = logging.getLogger(__name__)


class EmbedderBase(ABC):
    """Abstract base class for embedders."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts independently (context-agnostic).
        
        Returns: (N, OUTPUT_DIMENSION) array.
        """
        pass

    @abstractmethod
    def embed_chunks_with_context(
        self,
        chunks: list[Chunk],
        doc_texts: dict[str, str],
    ) -> np.ndarray:
        """Embed chunks with their preceding context.
        
        Returns: (N, OUTPUT_DIMENSION) array.
        """
        pass


class OpenAIDefaultEmbedder(EmbedderBase):
    """Embed chunks standalone using OpenAI text-embedding-3-small (512-dim output)."""

    def __init__(self):
        self.model = "text-embedding-3-small"
        self.output_dimension = OUTPUT_DIMENSION

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenAI API with 512-dim output."""
        logger.debug(f"Embedding {len(texts)} texts with OpenAI (default mode)")
        embeddings = embed_texts(
            texts,
            model=self.model,
            output_dimension=self.output_dimension,
        )
        return np.array(embeddings)

    def embed_chunks_with_context(
        self,
        chunks: list[Chunk],
        doc_texts: dict[str, str],
    ) -> np.ndarray:
        """For default mode, ignore context and embed chunks alone."""
        chunk_texts = [c.text for c in chunks]
        return self.embed(chunk_texts)


class OpenAIContextualEmbedder(EmbedderBase):
    """Embed chunks with pooled preceding context (semantic pooling with 25% overlap)."""

    def __init__(self):
        self.model = "text-embedding-3-small"
        self.output_dimension = OUTPUT_DIMENSION
        self.overlap_ratio = CONTEXT_POOLING_OVERLAP_RATIO
        self.pooling_strategy = POOLING_STRATEGY

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts standalone (for queries, etc.)."""
        logger.debug(f"Embedding {len(texts)} texts with OpenAI (contextual mode, no context)")
        embeddings = embed_texts(
            texts,
            model=self.model,
            output_dimension=self.output_dimension,
        )
        return np.array(embeddings)

    def embed_chunks_with_context(
        self,
        chunks: list[Chunk],
        doc_texts: dict[str, str],
    ) -> np.ndarray:
        """Embed chunks with their preceding context using semantic pooling.
        
        Strategy:
        1. For each chunk, extract preceding document context (as large as possible)
        2. Split preceding context into overlapping windows (25% overlap)
        3. Encode each window independently via OpenAI
        4. Apply max/mean pooling across window embeddings
        5. Concatenate pooled context embedding with chunk embedding
        """
        embeddings_list = []

        for chunk in chunks:
            full_text = doc_texts.get(chunk.doc_file_name, "")
            if not full_text:
                # No preceding context available, embed chunk alone
                chunk_emb = embed_texts(
                    [chunk.text],
                    model=self.model,
                    output_dimension=self.output_dimension,
                )[0]
                embeddings_list.append(chunk_emb)
                continue

            # Extract all preceding context
            preceding_text = full_text[: chunk.char_offset_start]
            if not preceding_text.strip():
                # No preceding context, embed chunk alone
                chunk_emb = embed_texts(
                    [chunk.text],
                    model=self.model,
                    output_dimension=self.output_dimension,
                )[0]
                embeddings_list.append(chunk_emb)
                continue

            # Split preceding context into overlapping windows
            # Calculate max window size to stay within token budget
            # OpenAI's text-embedding-3-small has 8192 token limit
            # Leave buffer for chunk + metadata
            chunk_tokens = count_tokens(chunk.text, model=self.model)
            buffer = 100
            max_context_tokens = 8192 - chunk_tokens - buffer

            if max_context_tokens < 100:
                # Not enough space for context, embed chunk alone
                chunk_emb = embed_texts(
                    [chunk.text],
                    model=self.model,
                    output_dimension=self.output_dimension,
                )[0]
                embeddings_list.append(chunk_emb)
                continue

            # Split context into windows with 25% overlap
            windows = split_text_into_token_windows(
                preceding_text,
                max_window_tokens=max_context_tokens,
                overlap_ratio=self.overlap_ratio,
                model=self.model,
            )

            if not windows:
                # Fallback: embed chunk alone
                chunk_emb = embed_texts(
                    [chunk.text],
                    model=self.model,
                    output_dimension=self.output_dimension,
                )[0]
                embeddings_list.append(chunk_emb)
                continue

            # Embed all windows
            window_embeddings = embed_texts(
                windows,
                model=self.model,
                output_dimension=self.output_dimension,
            )
            window_embeddings = np.array(window_embeddings)  # (num_windows, 512)

            # Apply pooling across windows
            if self.pooling_strategy == "max":
                pooled_context = np.max(window_embeddings, axis=0)
            elif self.pooling_strategy == "mean":
                pooled_context = np.mean(window_embeddings, axis=0)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

            # Embed chunk
            chunk_emb = embed_texts(
                [chunk.text],
                model=self.model,
                output_dimension=self.output_dimension,
            )[0]

            # Concatenate pooled context with chunk embedding
            # This doubles the dimension: 512 + 512 = 1024
            # But the user said all should be 512, so we average instead
            combined_emb = (pooled_context + chunk_emb) / 2.0
            embeddings_list.append(combined_emb)

        return np.array(embeddings_list)


class VoyageContextualEmbedder(EmbedderBase):
    """Embed chunks using Voyage's contextualized chunk embeddings (native context handling)."""

    def __init__(self):
        self.model = "voyage-context-3"
        self.output_dimension = OUTPUT_DIMENSION

        if not VOYAGE_API_KEY:
            raise ValueError(
                "VOYAGE_API_KEY environment variable not set. "
                "Please set it to use the Voyage embedder."
            )

        try:
            import voyageai  # type: ignore
            self.client = voyageai.Client(api_key=VOYAGE_API_KEY)  # type: ignore
        except ImportError:
            raise ImportError(
                "voyageai package not installed. "
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using Voyage (context-agnostic, single text per input)."""
        logger.debug(f"Embedding {len(texts)} texts with Voyage (context-agnostic)")
        # For queries and standalone texts, pass each as a single-item list
        inputs = [[text] for text in texts]
        result = self.client.contextualized_embed(
            inputs=inputs,
            model=self.model,
            input_type=None,
            output_dimension=self.output_dimension,
        )
        embeddings = [r.embeddings[0] for r in result.results]
        return np.array(embeddings)

    def embed_chunks_with_context(
        self,
        chunks: list[Chunk],
        doc_texts: dict[str, str],
    ) -> np.ndarray:
        """Embed chunks using Voyage's native contextualized embeddings.
        
        Groups chunks by document and passes each document's chunks together
        to Voyage (in batches respecting API constraints). Voyage internally 
        encodes each chunk with context from other chunks in the same document.
        
        API Constraints per request:
        - Max 1,000 inputs (outer lists = documents)
        - Max 120K tokens across all inputs
        - Max 16K chunks across all inputs
        """
        logger.debug(f"Embedding {len(chunks)} chunks with Voyage (contextualized)")

        # Group chunks by document and sort by position
        doc_chunks: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if chunk.doc_file_name not in doc_chunks:
                doc_chunks[chunk.doc_file_name] = []
            doc_chunks[chunk.doc_file_name].append(chunk)

        # Sort chunks within each document
        for doc_file in doc_chunks:
            doc_chunks[doc_file].sort(key=lambda c: c.char_offset_start)

        # Build batches respecting Voyage API constraints
        batches: list[tuple[list[list[str]], dict[str, list[Chunk]]]] = []
        current_batch_inputs: list[list[str]] = []
        current_batch_doc_map: dict[str, list[Chunk]] = {}
        current_tokens = 0
        current_chunks = 0

        for doc_file, doc_chunk_list in doc_chunks.items():
            chunk_texts = [c.text for c in doc_chunk_list]
            doc_tokens = sum(count_tokens(text, model=self.model) for text in chunk_texts)
            num_chunks = len(chunk_texts)

            # Check if this document alone exceeds limits
            if doc_tokens > 120_000:
                logger.warning(
                    "Document %s has %d tokens (> 120K limit). "
                    "Consider splitting it before embedding.",
                    doc_file, doc_tokens
                )

            # Check if adding this document exceeds batch limits
            # Use strict limits to ensure we don't exceed constraints
            if (current_batch_inputs and 
                (len(current_batch_inputs) >= 1000 or
                 current_tokens + doc_tokens > 100_000 or  # Use 100K to be safe
                 current_chunks + num_chunks > 10_000)):   # Use 10K to be safe
                # Start a new batch
                batches.append((current_batch_inputs, current_batch_doc_map))
                current_batch_inputs = []
                current_batch_doc_map = {}
                current_tokens = 0
                current_chunks = 0

            # Add document to current batch
            current_batch_inputs.append(chunk_texts)
            current_batch_doc_map[doc_file] = doc_chunk_list
            current_tokens += doc_tokens
            current_chunks += num_chunks

        # Don't forget the last batch
        if current_batch_inputs:
            batches.append((current_batch_inputs, current_batch_doc_map))

        logger.debug(f"Split into {len(batches)} batch(es) for API calls")

        # Call Voyage API for each batch and collect embeddings
        embeddings_by_chunk_id: dict[str, list] = {}

        for batch_idx, (batch_inputs, batch_doc_map) in enumerate(batches):
            logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)}")
            
            result = self.client.contextualized_embed(
                inputs=batch_inputs,
                model=self.model,
                input_type="document",
                output_dimension=self.output_dimension,
            )

            # Map embeddings back to chunk IDs
            # result.results is ordered the same as batch_inputs (which is ordered by doc_file)
            result_idx = 0
            for doc_file in batch_doc_map.keys():
                doc_chunk_list = batch_doc_map[doc_file]
                result_item = result.results[result_idx]
                
                # Each result has embeddings for chunks in this document
                for chunk_idx, chunk in enumerate(doc_chunk_list):
                    embeddings_by_chunk_id[chunk.chunk_id] = result_item.embeddings[chunk_idx]
                
                result_idx += 1

        # Return embeddings in original chunk order
        final_embeddings = [embeddings_by_chunk_id[c.chunk_id] for c in chunks]
        return np.array(final_embeddings)


def get_embedder(mode: str) -> EmbedderBase:
    """Factory function to get embedder by mode."""
    if mode == "default":
        return OpenAIDefaultEmbedder()
    elif mode == "contextual":
        return OpenAIContextualEmbedder()
    elif mode == "voyage":
        return VoyageContextualEmbedder()
    else:
        raise ValueError(
            f"Unknown embedder mode: {mode}. "
            f"Choose from: default, contextual, voyage"
        )
