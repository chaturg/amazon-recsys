"""
two_tower.py
------------
Two-tower neural network architecture for personalized retrieval.

Architecture:
  User Tower:  user interaction features → MLP → user_embedding [embed_dim]
  Item Tower:  item metadata features    → MLP → item_embedding [embed_dim]
  Projection:  concat(user_emb, query_emb) → linear → combined [embed_dim]

The projection layer learns the optimal weighting of user vs. query signal
during training. For sparse users, the adaptive alpha blends user and query
embeddings before the projection layer — no hardcoded weight needed.

At inference:
  - User embedding is computed from validation history (user tower forward pass)
  - Query embedding is identical to user embedding in this metadata-only setup
    (without query text, the user embedding IS the query vector)
  - Combined vector is searched against the pre-built FAISS item index

Usage:
    from model.two_tower import TwoTowerModel
    from experiments.configs import get_config

    cfg = get_config("config2")
    model = TwoTowerModel(cfg)
    user_emb = model.encode_user(user_features)    # [batch, embed_dim]
    item_emb = model.encode_item(item_features)    # [batch, embed_dim]
    scores   = model.score(user_emb, item_emb)     # [batch] cosine similarity
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable depth and width.
    Used as the backbone for both user and item towers.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout:    float,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final projection to output_dim — no activation
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTowerModel(nn.Module):
    """
    Two-tower model with user tower, item tower, and projection layer.

    Args:
        cfg: TwoTowerConfig dataclass from experiments/configs.py
        user_input_dim: Dimensionality of user feature vector
        item_input_dim: Dimensionality of item feature vector
    """

    def __init__(
        self,
        cfg,
        user_input_dim: int = 7,   # matches len(cfg.user_features)
        item_input_dim: int = 5,   # matches len(cfg.item_features)
    ):
        super().__init__()

        self.cfg       = cfg
        self.embed_dim = cfg.embed_dim

        # ── User tower ────────────────────────────────────────────────────────
        self.user_tower = MLP(
            input_dim  = user_input_dim,
            hidden_dim = cfg.tower_width,
            output_dim = cfg.embed_dim,
            num_layers = cfg.tower_depth,
            dropout    = cfg.dropout,
        )

        # ── Item tower ────────────────────────────────────────────────────────
        self.item_tower = MLP(
            input_dim  = item_input_dim,
            hidden_dim = cfg.tower_width,
            output_dim = cfg.embed_dim,
            num_layers = cfg.tower_depth,
            dropout    = cfg.dropout,
        )

        # ── Projection layer ──────────────────────────────────────────────────
        # Takes concat(user_emb, query_emb) → embed_dim
        # Learns optimal weighting of user vs. query signal during training.
        # For metadata-only setup, user_emb == query_emb, so this projects
        # 2*embed_dim → embed_dim.
        self.projection = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
        )

        self._init_weights()
        self._log_architecture(user_input_dim, item_input_dim)

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _log_architecture(self, user_input_dim: int, item_input_dim: int) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        trainable   = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"TwoTowerModel initialized:\n"
            f"  embed_dim={self.embed_dim} | "
            f"tower_depth={self.cfg.tower_depth} | "
            f"tower_width={self.cfg.tower_width} | "
            f"dropout={self.cfg.dropout}\n"
            f"  user_input={user_input_dim} → {self.embed_dim} | "
            f"item_input={item_input_dim} → {self.embed_dim}\n"
            f"  Total params: {total_params:,} | Trainable: {trainable:,}"
        )

    def encode_user(
        self,
        user_features:    torch.Tensor,
        num_interactions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode user features into a user embedding.

        Applies adaptive alpha blending for sparse users:
          alpha = min(1.0, num_interactions / 5.0)
          combined = alpha * user_emb + (1 - alpha) * mean_emb

        Args:
            user_features:    [batch_size, user_input_dim]
            num_interactions: [batch_size] optional — interaction counts
                              for adaptive sparse user blending

        Returns:
            [batch_size, embed_dim] — L2-normalized user embeddings
        """
        user_emb = self.user_tower(user_features)

        # Adaptive alpha for sparse users
        if num_interactions is not None:
            alpha = torch.clamp(num_interactions.float() / 5.0, 0.0, 1.0)
            alpha = alpha.unsqueeze(-1)  # [batch, 1] for broadcasting

            # Mean embedding as fallback for users with no history
            mean_emb = user_emb.mean(dim=0, keepdim=True).expand_as(user_emb)
            user_emb = alpha * user_emb + (1 - alpha) * mean_emb

        return F.normalize(user_emb, p=2, dim=-1)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """
        Encode item metadata features into an item embedding.

        Args:
            item_features: [batch_size, item_input_dim]

        Returns:
            [batch_size, embed_dim] — L2-normalized item embeddings
        """
        item_emb = self.item_tower(item_features)
        return F.normalize(item_emb, p=2, dim=-1)

    def get_combined_vector(
        self,
        user_emb: torch.Tensor,
        query_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combine user and query embeddings via the projection layer.

        In this metadata-only setup, query_emb defaults to user_emb
        (no separate query text encoder). In a text-augmented version,
        query_emb would come from a sentence transformer.

        Args:
            user_emb:  [batch_size, embed_dim]
            query_emb: [batch_size, embed_dim] optional — defaults to user_emb

        Returns:
            [batch_size, embed_dim] — L2-normalized combined vector
            ready for FAISS ANN search
        """
        if query_emb is None:
            query_emb = user_emb

        combined = torch.cat([user_emb, query_emb], dim=-1)  # [batch, 2*embed_dim]
        combined = self.projection(combined)                   # [batch, embed_dim]
        return F.normalize(combined, p=2, dim=-1)

    def score(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dot product similarity between user and item embeddings.
        Both inputs should already be L2-normalized (cosine similarity).

        Args:
            user_emb: [batch_size, embed_dim]
            item_emb: [batch_size, embed_dim]

        Returns:
            [batch_size] — similarity scores in [-1, 1]
        """
        return (user_emb * item_emb).sum(dim=-1)

    def forward(
        self,
        user_features:    torch.Tensor,
        item_features:    torch.Tensor,
        neg_item_features: Optional[torch.Tensor] = None,
        num_interactions: Optional[torch.Tensor]  = None,
    ) -> dict:
        """
        Full forward pass for training.

        Args:
            user_features:     [batch_size, user_input_dim]
            item_features:     [batch_size, item_input_dim] — positive items
            neg_item_features: [batch_size, num_neg, item_input_dim] — optional negatives
            num_interactions:  [batch_size] — for adaptive alpha

        Returns:
            dict with:
              "user_emb":    [batch_size, embed_dim]
              "item_emb":    [batch_size, embed_dim] — positive items
              "combined":    [batch_size, embed_dim] — projection output
              "pos_scores":  [batch_size] — positive pair similarities
              "neg_embs":    [batch_size, num_neg, embed_dim] — if provided
              "neg_scores":  [batch_size, num_neg] — if provided
        """
        user_emb = self.encode_user(user_features, num_interactions)
        item_emb = self.encode_item(item_features)
        combined = self.get_combined_vector(user_emb)
        pos_scores = self.score(combined, item_emb)

        out = {
            "user_emb":   user_emb,
            "item_emb":   item_emb,
            "combined":   combined,
            "pos_scores": pos_scores,
        }

        if neg_item_features is not None:
            batch_size, num_neg, feat_dim = neg_item_features.shape
            neg_flat = neg_item_features.view(batch_size * num_neg, feat_dim)
            neg_emb_flat = self.encode_item(neg_flat)
            neg_embs = neg_emb_flat.view(batch_size, num_neg, self.embed_dim)

            # Score each negative: [batch_size, num_neg]
            combined_expanded = combined.unsqueeze(1).expand_as(neg_embs)
            neg_scores = (combined_expanded * neg_embs).sum(dim=-1)

            out["neg_embs"]   = neg_embs
            out["neg_scores"] = neg_scores

        return out
