"""
cross_encoder.py
----------------
Cross-encoder fine-tuning and inference for re-ranking.

Architecture:
  Base model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache 2.0)
  Input:  [CLS] query + user_history_summary [SEP] item_title + item_category [SEP]
  Output: relevance score ∈ [0, 1]
  Loss:   Pointwise cross-entropy against silver labels

Why cross-entropy not pairwise hinge:
  Silver labels are continuous per-item scores [0,1], not pairwise annotations.
  Cross-entropy treats each (query, item, label) triple independently and
  matches the base model's pre-training distribution on MS MARCO.
  Pairwise hinge would require constructing (positive, negative) pairs from
  silver labels — an extra step without clear payoff.

Usage (fine-tuning — run on Kaggle GPU):
    from retrieval.cross_encoder import CrossEncoderTrainer
    trainer = CrossEncoderTrainer(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    trainer.train(pairs_path="processed/ce_training_pairs.parquet")
    trainer.save("artifacts/cross_encoder")

Usage (inference — re-ranking):
    from retrieval.cross_encoder import CrossEncoderRanker
    ranker = CrossEncoderRanker.load("artifacts/cross_encoder")
    scores = ranker.predict(query_history_pairs, item_pairs)
    top10  = ranker.rerank(candidates, query, user_history, top_k=10)
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

BASE_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SEQ_LEN = 128


# ── Dataset ───────────────────────────────────────────────────────────────────
class CrossEncoderDataset(Dataset):
    """
    Dataset for cross-encoder fine-tuning.

    Each sample is:
      Left:  "{history_summary}"         (what the user has bought)
      Right: "{item_title} {category}"   (candidate item)
      Label: relevance_label ∈ [0, 1]    (silver label or 0.0 for negatives)

    The tokenizer handles the [CLS] ... [SEP] ... [SEP] formatting
    automatically when given a text_pair input.
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_len:   int = MAX_SEQ_LEN,
    ):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Left side: user history summary
        text_a = str(row["history_summary"])[:256]

        # Right side: item title + category
        text_b = f"{row['item_title']} {row['item_category']}"[:128]

        encoding = self.tokenizer(
            text_a,
            text_b,
            max_length      = self.max_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(float(row["relevance_label"]), dtype=torch.float),
        }


# ── Trainer ───────────────────────────────────────────────────────────────────
class CrossEncoderTrainer:
    """
    Fine-tunes the cross-encoder on (user_history, item, relevance_label) triples.

    Uses pointwise cross-entropy loss — each triple is scored independently.
    The model learns to output high scores for high silver-label items and
    low scores for FAISS-retrieved hard negatives (silver_label = 0.0).
    """

    def __init__(
        self,
        model_name:  str   = BASE_MODEL,
        device:      str   = None,
        max_seq_len: int   = MAX_SEQ_LEN,
    ):
        self.model_name  = model_name
        self.max_seq_len = max_seq_len
        self.device      = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(f"Loading base model: {model_name}")
        logger.info(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Model params: {total_params:,}")

    def train(
        self,
        pairs_path:   str,
        output_dir:   str   = "artifacts/cross_encoder",
        num_epochs:   int   = 3,
        batch_size:   int   = 32,
        learning_rate:float = 2e-5,
        warmup_ratio: float = 0.1,
        val_split:    float = 0.05,
        max_pairs:    int   = None,
    ) -> None:
        """
        Fine-tune the cross-encoder on training pairs.

        Args:
            pairs_path:    Path to ce_training_pairs.parquet
            output_dir:    Where to save the fine-tuned model
            num_epochs:    Training epochs (3 is standard for fine-tuning)
            batch_size:    Batch size (32 fits comfortably on T4 16GB)
            learning_rate: 2e-5 is standard for BERT-style fine-tuning
            warmup_ratio:  Fraction of steps for LR warmup
            val_split:     Fraction of pairs held out for validation
            max_pairs:     Cap total pairs (None = use all)
        """
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-Encoder Fine-Tuning")
        logger.info(f"  Base model:    {self.model_name}")
        logger.info(f"  Pairs:         {pairs_path}")
        logger.info(f"  Epochs:        {num_epochs}")
        logger.info(f"  Batch size:    {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"{'='*60}")

        # ── Load pairs ─────────────────────────────────────────────────────
        logger.info("Loading training pairs...")
        df = pd.read_parquet(pairs_path)
        if max_pairs:
            df = df.sample(min(max_pairs, len(df)), random_state=42)
        logger.info(f"  Total pairs: {len(df):,}")
        logger.info(f"  Positives:   {df['is_positive'].sum():,} ({df['is_positive'].mean():.1%})")

        # ── Train/val split ────────────────────────────────────────────────
        val_size  = int(len(df) * val_split)
        val_df    = df.sample(val_size, random_state=42)
        train_df  = df.drop(val_df.index)
        logger.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

        train_ds = CrossEncoderDataset(train_df, self.tokenizer, self.max_seq_len)
        val_ds   = CrossEncoderDataset(val_df,   self.tokenizer, self.max_seq_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size*2,
                                  shuffle=False, num_workers=2, pin_memory=True)

        # ── Optimizer and scheduler ────────────────────────────────────────
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        # ── Loss function ──────────────────────────────────────────────────
        # BCEWithLogitsLoss = sigmoid + binary cross-entropy
        # Treats each (query, item, label) independently — pointwise
        loss_fn = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── Training loop ──────────────────────────────────────────────────
        for epoch in range(num_epochs):
            ep_start = time.time()
            self.model.train()
            tl, tn = 0.0, 0

            for bidx, batch in enumerate(train_loader):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                logits  = outputs.logits.squeeze(-1)
                loss    = loss_fn(logits, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                tl += loss.item()
                tn += 1

                if bidx % 100 == 0:
                    logger.info(
                        f"  Ep{epoch+1} batch {bidx}/{len(train_loader)} | "
                        f"loss={loss.item():.4f}"
                    )

            scheduler.step()

            # ── Validation ─────────────────────────────────────────────────
            self.model.eval()
            vl, vn = 0.0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids      = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels         = batch["label"].to(self.device)
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask)
                    logits  = outputs.logits.squeeze(-1)
                    loss    = loss_fn(logits, labels)
                    vl += loss.item(); vn += 1
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val  = vl / vn
            is_best  = avg_val < best_val_loss
            if is_best:
                best_val_loss = avg_val
                self.model.save_pretrained(f"{output_dir}/best")
                self.tokenizer.save_pretrained(f"{output_dir}/best")
                logger.info(f"  ✓ Best model saved (val={avg_val:.4f})")

            # Correlation between predicted scores and silver labels
            corr = np.corrcoef(all_preds, all_labels)[0, 1]

            ep_time = time.time() - ep_start
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} — "
                f"train={tl/tn:.4f} val={avg_val:.4f} "
                f"corr={corr:.3f} time={ep_time:.0f}s"
                + (" ← BEST" if is_best else "")
            )

        # Save final model
        self.model.save_pretrained(f"{output_dir}/final")
        self.tokenizer.save_pretrained(f"{output_dir}/final")

        elapsed = time.time() - t0
        logger.info(f"\nFine-tuning complete in {elapsed/60:.1f} min")
        logger.info(f"Best val_loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to: {output_dir}")

    def save(self, output_dir: str) -> None:
        """Save model and tokenizer."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved to {output_dir}")


# ── Inference / Re-ranker ─────────────────────────────────────────────────────
class CrossEncoderRanker:
    """
    Cross-encoder inference wrapper for re-ranking FAISS candidates.

    At demo time:
      1. FAISS retrieves top-100 candidates (Stage 1)
      2. CrossEncoderRanker scores each candidate (Stage 2)
      3. Return top-10 by score

    Each candidate is scored independently — O(100) forward passes.
    With MiniLM-L6 (~22M params) on GPU, 100 candidates takes ~0.5s.
    """

    def __init__(self, model_dir: str, device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=1
        ).to(self.device)
        self.model.eval()
        logger.info(f"CrossEncoderRanker loaded from {model_dir} on {self.device}")

    @classmethod
    def load(cls, model_dir: str, device: str = None) -> "CrossEncoderRanker":
        """Load from local directory or HF hub."""
        return cls(model_dir, device)

    def predict(
        self,
        history_summaries: list,
        item_texts:        list,
        batch_size:        int = 32,
    ) -> np.ndarray:
        """
        Score a list of (history, item) pairs.

        Args:
            history_summaries: list of user history strings
            item_texts:        list of "item_title item_category" strings
            batch_size:        inference batch size

        Returns:
            np.ndarray of scores ∈ [0, 1], shape [len(history_summaries)]
        """
        assert len(history_summaries) == len(item_texts)
        all_scores = []

        for i in range(0, len(history_summaries), batch_size):
            batch_h = history_summaries[i:i+batch_size]
            batch_i = item_texts[i:i+batch_size]

            encoding = self.tokenizer(
                batch_h,
                batch_i,
                max_length     = MAX_SEQ_LEN,
                padding        = "max_length",
                truncation     = True,
                return_tensors = "pt",
            )

            with torch.no_grad():
                input_ids      = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                scores  = torch.sigmoid(outputs.logits.squeeze(-1))
                all_scores.extend(scores.cpu().numpy())

        return np.array(all_scores)

    def rerank(
        self,
        candidates:      list,
        history_summary: str,
        top_k:           int = 10,
    ) -> list:
        """
        Re-rank a list of candidate items for a given user.

        Args:
            candidates:      list of dicts with keys 'asin', 'title', 'category'
            history_summary: user's purchase history as text
            top_k:           number of results to return

        Returns:
            top_k candidates sorted by cross-encoder score (highest first),
            each dict now includes 'ce_score' key
        """
        if not candidates:
            return []

        histories = [history_summary] * len(candidates)
        item_texts = [
            f"{c.get('title', c['asin'])} {c.get('category', '')}"
            for c in candidates
        ]

        scores = self.predict(histories, item_texts)

        # Attach scores and sort
        for i, c in enumerate(candidates):
            c["ce_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
        return ranked[:top_k]
