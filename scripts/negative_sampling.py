"""
negative_sampling.py
--------------------
Three-phase negative sampling for two-tower training.

Phase 1 — Random negatives (epochs 1-2):
  Items the user never interacted with, sampled uniformly from the catalog.
  Fast and scalable. Establishes baseline embedding separation.

Phase 2 — In-batch negatives (epochs 3+):
  Positives from other users in the same training batch automatically become
  negatives. No extra sampling cost. Significantly improves Recall@K because
  the model learns to distinguish between items that are genuinely popular
  (appear frequently as positives) rather than just random items.

Phase 3 — ANN hard negatives (fine-tune, epochs 6+):
  Items semantically close to the positive but still incorrect — mined via
  FAISS nearest-neighbor search against the current item index from the full
  catalog. Example: if positive is "DEWALT 20V drill", a hard negative might
  be "Makita 18V drill" — same category, different brand.

  Requires Config 2 FAISS index to exist. Cannot run in Config 1.

Usage:
    from model.negative_sampling import NegativeSampler
    sampler = NegativeSampler(strategy="in_batch", item_catalog=item_ids)
    neg_items = sampler.sample(batch_user_ids, batch_item_ids, epoch=3)
"""

import logging
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class NegativeSampler:
    """
    Handles all three phases of negative sampling.

    Args:
        strategy:      Starting strategy. Overridden by epoch-based phase
                       switching during training.
        item_catalog:  Array of all valid item indices in the dataset.
                       Used for random negative sampling.
        faiss_index:   Optional FAISS index for ANN hard negative mining.
                       Required for Phase 3 (ann_hard strategy).
        num_neg:       Number of negatives per positive (random strategy).
        phase2_epoch:  Epoch at which to switch from random to in-batch.
        phase3_epoch:  Epoch at which to switch from in-batch to ANN hard.
    """

    def __init__(
        self,
        strategy:     str,
        item_catalog: np.ndarray,
        faiss_index   = None,
        num_neg:      int = 4,
        phase2_epoch: int = 3,
        phase3_epoch: int = 6,
    ):
        self.strategy      = strategy
        self.item_catalog  = item_catalog
        self.faiss_index   = faiss_index
        self.num_neg       = num_neg
        self.phase2_epoch  = phase2_epoch
        self.phase3_epoch  = phase3_epoch
        self._current_phase = "random"

        logger.info(
            f"NegativeSampler initialized: strategy={strategy}, "
            f"catalog_size={len(item_catalog):,}, "
            f"phase2_epoch={phase2_epoch}, phase3_epoch={phase3_epoch}"
        )

    def get_phase(self, epoch: int) -> str:
        """Determine which negative sampling phase applies at this epoch."""
        if epoch >= self.phase3_epoch and self.faiss_index is not None:
            return "ann_hard"
        elif epoch >= self.phase2_epoch:
            return "in_batch"
        else:
            return "random"

    def log_phase_transition(self, epoch: int) -> None:
        """Log when the phase changes between epochs."""
        new_phase = self.get_phase(epoch)
        if new_phase != self._current_phase:
            logger.info(
                f"  Epoch {epoch}: Negative sampling phase → {new_phase.upper()} "
                f"(was {self._current_phase.upper()})"
            )
            self._current_phase = new_phase

    def sample_random(
        self,
        pos_item_ids: torch.Tensor,
        batch_size:   int,
    ) -> torch.Tensor:
        """
        Sample random negatives from the item catalog.

        For each positive item, sample num_neg items that are not the
        positive item. Fast but easy — model will quickly learn to
        distinguish random items.

        Returns:
            Tensor of shape [batch_size, num_neg] with item indices.
        """
        pos_ids = pos_item_ids.cpu().numpy()
        neg_ids = np.zeros((batch_size, self.num_neg), dtype=np.int64)

        for i, pos_id in enumerate(pos_ids):
            sampled = 0
            attempts = 0
            while sampled < self.num_neg and attempts < self.num_neg * 10:
                candidate = random.choice(self.item_catalog)
                if candidate != pos_id:
                    neg_ids[i, sampled] = candidate
                    sampled += 1
                attempts += 1

            # Fill remaining with random if attempts exhausted
            while sampled < self.num_neg:
                neg_ids[i, sampled] = random.choice(self.item_catalog)
                sampled += 1

        return torch.tensor(neg_ids, dtype=torch.long)

    def get_in_batch_negatives(
        self,
        item_embeddings: torch.Tensor,
        labels:          torch.Tensor,
    ) -> torch.Tensor:
        """
        Return a similarity matrix for in-batch negative training.

        In-batch negatives: for user i, all items j≠i in the batch are
        negatives. This is implemented as a full similarity matrix where
        the diagonal is the positive (user_i, item_i) pair.

        The training loss uses this matrix directly — no explicit negative
        sampling needed. The loss function (InfoNCE / NT-Xent) handles it.

        Args:
            item_embeddings: [batch_size, embed_dim] — item vectors
            labels:          [batch_size] — positive item indices (for logging)

        Returns:
            item_embeddings unchanged — the loss function uses the full matrix
        """
        # In-batch negatives are handled at the loss computation level.
        # Return item embeddings as-is; the InfoNCE loss treats off-diagonal
        # elements as negatives automatically.
        return item_embeddings

    def sample_ann_hard(
        self,
        query_embeddings: np.ndarray,
        pos_item_ids:     np.ndarray,
        k_candidates:     int = 50,
    ) -> np.ndarray:
        """
        Mine hard negatives using ANN search against the FAISS index.

        For each user query embedding, retrieve the top-k nearest items.
        Remove the actual positive item. The remaining items are semantically
        close to what the user wants but are not the correct answer —
        these are hard negatives.

        Args:
            query_embeddings: [batch_size, embed_dim] — user+query vectors
            pos_item_ids:     [batch_size] — the actual positive item index
            k_candidates:     How many ANN candidates to retrieve

        Returns:
            [batch_size, num_neg] array of hard negative item indices
        """
        if self.faiss_index is None:
            raise RuntimeError(
                "FAISS index required for ANN hard negative mining. "
                "Build Config 2 index first: python scripts/build_faiss_index.py"
            )

        batch_size = len(query_embeddings)
        # Search k+1 candidates to account for the positive being in results
        distances, indices = self.faiss_index.search(
            query_embeddings.astype(np.float32),
            k_candidates + 1
        )

        hard_negs = np.zeros((batch_size, self.num_neg), dtype=np.int64)

        for i in range(batch_size):
            candidates = indices[i]
            pos_id = pos_item_ids[i]

            # Remove the positive item from candidates
            hard_candidates = candidates[candidates != pos_id]

            # Take the top num_neg hard negatives
            n = min(self.num_neg, len(hard_candidates))
            hard_negs[i, :n] = hard_candidates[:n]

            # Fallback to random for remaining slots
            if n < self.num_neg:
                for j in range(n, self.num_neg):
                    hard_negs[i, j] = random.choice(self.item_catalog)

        return hard_negs

    def update_faiss_index(self, faiss_index) -> None:
        """
        Update the FAISS index used for hard negative mining.
        Called after each epoch when item embeddings are refreshed.
        """
        self.faiss_index = faiss_index
        logger.info("  NegativeSampler: FAISS index updated for hard neg mining")
