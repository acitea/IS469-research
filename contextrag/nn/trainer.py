"""Training loop for the context-conditioned encoder.

Uses InfoNCE (NT-Xent) contrastive loss: for each anchor embedding, the
corresponding positive should be closest among all items in the batch.
In-batch negatives provide the contrastive signal.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from contextrag.config import (
    DATABASE_DIR,
    DEFAULT_CONTEXT_WINDOW,
    MODEL_DIR,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_NUM_ARTICLES,
    TRAIN_VAL_SPLIT,
)
from contextrag.nn.dataset import (
    ContrastiveChunkDataset,
    collate_fn,
    load_wikipedia_articles,
)
from contextrag.nn.encoder import ContextConditionedEncoder

logger = logging.getLogger(__name__)


def info_nce_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE (NT-Xent) loss using in-batch negatives.

    For each anchor, the corresponding positive is the target; all other
    positives in the batch serve as negatives.
    """
    # Normalize
    anchor_emb = F.normalize(anchor_emb, dim=-1)
    positive_emb = F.normalize(positive_emb, dim=-1)

    # Similarity matrix: (B, B)
    logits = torch.mm(anchor_emb, positive_emb.t()) / temperature

    # Labels: diagonal (each anchor matches its own positive)
    labels = torch.arange(logits.size(0), device=logits.device)

    return F.cross_entropy(logits, labels)


def train(
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    lr: float = TRAIN_LR,
    num_articles: int = TRAIN_NUM_ARTICLES,
    val_split: float = TRAIN_VAL_SPLIT,
    device_name: str | None = None,
) -> Path:
    """Train the context-conditioned encoder and save the best checkpoint.

    Returns the path to the saved model weights.
    """
    # Resolve device
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Training device: %s", device)

    # Load data
    articles = load_wikipedia_articles(num_articles=num_articles)
    dataset = ContrastiveChunkDataset(articles, context_window_tokens=context_window)

    # Split train/val
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False
    )

    # Model
    model = ContextConditionedEncoder()
    model.to(device)

    # Only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training state
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_path = MODEL_DIR / "best_model.pt"
    log_entries: list[dict] = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        t0 = time.time()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for anchor_chunks, anchor_ctxs, pos_chunks, pos_ctxs in train_bar:
            if context_window == 0:
                anchor_emb = model.encode_no_context(anchor_chunks)
                pos_emb = model.encode_no_context(pos_chunks)
            else:
                anchor_emb = model(anchor_chunks, anchor_ctxs)
                pos_emb = model(pos_chunks, pos_ctxs)

            loss = info_nce_loss(anchor_emb, pos_emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False)
            for anchor_chunks, anchor_ctxs, pos_chunks, pos_ctxs in val_bar:
                if context_window == 0:
                    anchor_emb = model.encode_no_context(anchor_chunks)
                    pos_emb = model.encode_no_context(pos_chunks)
                else:
                    anchor_emb = model(anchor_chunks, anchor_ctxs)
                    pos_emb = model(pos_chunks, pos_ctxs)

                loss = info_nce_loss(anchor_emb, pos_emb)
                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  (%.1fs)",
            epoch, epochs, avg_train_loss, avg_val_loss, elapsed,
        )

        log_entries.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "elapsed_s": round(elapsed, 1),
        })

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_trainable(str(best_path))
            logger.info("  → New best model (val_loss=%.4f)", avg_val_loss)

    # Save training log and config
    log_path = MODEL_DIR / "training_log.json"
    log_path.write_text(json.dumps(log_entries, indent=2))

    config_path = MODEL_DIR / "train_config.json"
    config_path.write_text(json.dumps({
        "context_window": context_window,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_articles": num_articles,
        "best_val_loss": round(best_val_loss, 6),
        "train_samples": train_size,
        "val_samples": val_size,
    }, indent=2))

    logger.info("Training complete. Best val_loss=%.4f saved to %s", best_val_loss, best_path)
    return best_path
