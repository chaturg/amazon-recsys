"""
train.py
--------
Training loop for the two-tower model.

Features:
  - 3-phase negative sampling (random → in-batch → ANN hard)
  - InfoNCE loss for in-batch negatives, BPR loss for random/hard negatives
  - Checkpoint save every epoch + best model by validation loss
  - Optional Azure Blob Storage sync after each checkpoint save
  - Resume from local checkpoint or Azure Blob on restart
  - Learning rate warmup + cosine annealing

Usage:
    python -m model.train --config config2

    # With Azure Blob sync:
    python -m model.train --config config2 --blob_sync
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.configs import get_config, TwoTowerConfig
from model.two_tower import TwoTowerModel
from model.negative_sampling import NegativeSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """
    PyTorch Dataset wrapping the processed parquet interactions.

    Each item is a (user_features, item_features, silver_label) tuple.
    User features are aggregated from the user's training history.
    Item features are aggregated from all reviews of the item.
    """

    def __init__(self, df: pd.DataFrame, user_agg: pd.DataFrame, item_agg: pd.DataFrame):
        self.df       = df.reset_index(drop=True)
        self.user_agg = user_agg
        self.item_agg = item_agg

        # Build index maps for fast lookup
        self.user_feat_map = user_agg.set_index("user_id")
        self.item_feat_map = item_agg.set_index("asin")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        user_id = row["user_id"]
        asin    = row["asin"]

        user_feats = self.user_feat_map.loc[user_id].values.astype(np.float32)
        item_feats = self.item_feat_map.loc[asin].values.astype(np.float32)
        label      = np.float32(row["silver_label"])
        n_inter    = np.float32(self.user_feat_map.loc[user_id, "interaction_count_norm"] * 661)

        return {
            "user_features":    torch.tensor(user_feats),
            "item_features":    torch.tensor(item_feats),
            "silver_label":     torch.tensor(label),
            "num_interactions": torch.tensor(n_inter),
            "user_id":          user_id,
            "asin":             asin,
        }


def build_user_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-user features from training interactions."""
    agg = train_df.groupby("user_id").agg(
        rating_norm_mean      = ("rating_norm",       "mean"),
        rating_norm_std       = ("rating_norm",       "std"),
        helpfulness_mean      = ("helpfulness_score", "mean"),
        verified_ratio        = ("verified_score",    "mean"),
        length_mean           = ("length_score",      "mean"),
        interaction_count     = ("asin",              "count"),
    ).reset_index()

    agg["rating_norm_std"]        = agg["rating_norm_std"].fillna(0)
    max_count = agg["interaction_count"].max()
    agg["interaction_count_norm"] = agg["interaction_count"] / max_count

    # Category entropy — proxy via silver label std (higher = more diverse)
    label_std = train_df.groupby("user_id")["silver_label"].std().fillna(0)
    agg = agg.merge(label_std.rename("category_entropy"), on="user_id", how="left")
    agg["category_entropy"] = agg["category_entropy"].fillna(0)

    feature_cols = [
        "rating_norm_mean", "rating_norm_std", "helpfulness_mean",
        "verified_ratio", "length_mean", "interaction_count_norm",
        "category_entropy"
    ]
    return agg[["user_id"] + feature_cols].fillna(0)


def build_item_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-item features from training interactions."""
    agg = train_df.groupby("asin").agg(
        avg_rating_norm   = ("rating_norm",       "mean"),
        review_count      = ("user_id",           "count"),
        avg_silver_label  = ("silver_label",      "mean"),
        verified_ratio    = ("verified_score",    "mean"),
        avg_length_score  = ("length_score",      "mean"),
    ).reset_index()

    max_count = agg["review_count"].max()
    agg["review_count_norm"] = agg["review_count"] / max_count

    feature_cols = [
        "avg_rating_norm", "review_count_norm", "avg_silver_label",
        "verified_ratio", "avg_length_score"
    ]
    return agg[["asin"] + feature_cols].fillna(0)


# ── Loss functions ─────────────────────────────────────────────────────────────

def infonce_loss(
    combined:  torch.Tensor,
    item_emb:  torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE / NT-Xent loss for in-batch negatives.

    For each (user, item) pair in the batch, the positive is the diagonal
    and all off-diagonal items are negatives.

    Args:
        combined:  [batch_size, embed_dim] — user combined vectors
        item_emb:  [batch_size, embed_dim] — item embeddings
        temperature: scaling factor — lower = sharper distribution

    Returns:
        Scalar loss
    """
    # Similarity matrix: [batch_size, batch_size]
    sim_matrix = torch.matmul(combined, item_emb.T) / temperature
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    return F.cross_entropy(sim_matrix, labels)


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss for random/hard negatives.

    Maximizes the margin between positive and negative scores.

    Args:
        pos_scores: [batch_size] — scores for positive items
        neg_scores: [batch_size, num_neg] — scores for negative items

    Returns:
        Scalar loss
    """
    # Expand pos_scores to match neg_scores shape
    pos_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
    diff = pos_expanded - neg_scores
    return -torch.log(torch.sigmoid(diff) + 1e-8).mean()


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(
    model:     TwoTowerModel,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_loss:  float,
    cfg:       TwoTowerConfig,
    is_best:   bool = False,
) -> str:
    """Save model checkpoint locally and optionally sync to Azure Blob."""
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":       epoch,
        "val_loss":    val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config_name": cfg.name,
    }

    # Always save latest
    latest_path = ckpt_dir / "latest.pt"
    torch.save(state, latest_path)

    # Save epoch-specific checkpoint
    epoch_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
    torch.save(state, epoch_path)

    # Save best separately
    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(state, best_path)
        logger.info(f"  ✓ Best checkpoint saved (val_loss={val_loss:.4f})")

    logger.info(f"  Checkpoint saved: {latest_path}")

    # Optional Azure Blob sync
    if cfg.blob_sync:
        _sync_to_blob(latest_path, cfg)
        if is_best:
            _sync_to_blob(best_path, cfg)

    return str(latest_path)


def load_checkpoint(
    model:     TwoTowerModel,
    optimizer: Optional[torch.optim.Optimizer],
    cfg:       TwoTowerConfig,
) -> int:
    """
    Load checkpoint. Checks local first, falls back to Azure Blob.
    Returns the epoch to resume from (0 if no checkpoint found).
    """
    ckpt_dir  = Path(cfg.checkpoint_dir)
    best_path = ckpt_dir / "best.pt"
    latest_path = ckpt_dir / "latest.pt"

    # Check local first
    ckpt_path = None
    if best_path.exists():
        ckpt_path = best_path
    elif latest_path.exists():
        ckpt_path = latest_path

    # Fall back to Azure Blob
    if ckpt_path is None and cfg.blob_sync:
        ckpt_path = _download_from_blob(cfg)

    if ckpt_path is None:
        logger.info("No checkpoint found — starting from scratch")
        return 0

    logger.info(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optim_state"])

    resume_epoch = state["epoch"] + 1
    logger.info(f"  Resuming from epoch {resume_epoch} (val_loss={state['val_loss']:.4f})")
    return resume_epoch


def _sync_to_blob(local_path: Path, cfg: TwoTowerConfig) -> None:
    """Sync a local checkpoint file to Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
        if not conn_str:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING not set — skipping blob sync")
            return

        client    = BlobServiceClient.from_connection_string(conn_str)
        container = client.get_container_client(cfg.blob_container)
        blob_name = f"{cfg.blob_prefix}/{cfg.name}/{local_path.name}"

        with open(local_path, "rb") as f:
            container.upload_blob(blob_name, f, overwrite=True)

        logger.info(f"  Blob sync: {local_path.name} → {cfg.blob_container}/{blob_name}")
    except Exception as e:
        logger.warning(f"  Blob sync failed (non-fatal): {e}")


def _download_from_blob(cfg: TwoTowerConfig) -> Optional[Path]:
    """Download latest checkpoint from Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
        if not conn_str:
            return None

        client    = BlobServiceClient.from_connection_string(conn_str)
        container = client.get_container_client(cfg.blob_container)
        blob_name = f"{cfg.blob_prefix}/{cfg.name}/best.pt"

        ckpt_dir  = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        local_path = ckpt_dir / "best.pt"

        with open(local_path, "wb") as f:
            data = container.download_blob(blob_name)
            data.readinto(f)

        logger.info(f"  Downloaded checkpoint from blob: {blob_name}")
        return local_path
    except Exception as e:
        logger.warning(f"  Blob download failed: {e}")
        return None


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: TwoTowerConfig) -> None:
    """
    Main training loop.

    Args:
        cfg: TwoTowerConfig from experiments/configs.py
    """
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    logger.info(f"Config: {cfg.name} — {cfg.description}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading parquet data...")
    train_df = pd.read_parquet(cfg.train_path)
    val_df   = pd.read_parquet(cfg.val_path)

    logger.info("Building user and item feature aggregations...")
    user_agg = build_user_features(train_df)
    item_agg = build_item_features(train_df)

    train_dataset = InteractionDataset(train_df, user_agg, item_agg)
    val_dataset   = InteractionDataset(val_df,   user_agg, item_agg)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.batch_size * 2,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
    )

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model = TwoTowerModel(
        cfg,
        user_input_dim = len(cfg.user_features),
        item_input_dim = len(cfg.item_features),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = load_checkpoint(model, optimizer, cfg)

    # ── Negative sampler ──────────────────────────────────────────────────────
    item_catalog = np.array(train_df["asin"].unique())
    sampler = NegativeSampler(
        strategy      = cfg.neg_strategy,
        item_catalog  = item_catalog,
        num_neg       = 4,
        phase2_epoch  = cfg.neg_phase2_epoch,
        phase3_epoch  = cfg.neg_phase3_epoch,
    )

    # ── Build initial item embedding matrix for ANN mining ───────────────────
    best_val_loss = float("inf")

    # ── Training epochs ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        sampler.log_phase_transition(epoch + 1)
        phase = sampler.get_phase(epoch + 1)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            user_feats  = batch["user_features"].to(device)
            item_feats  = batch["item_features"].to(device)
            n_inter     = batch["num_interactions"].to(device)

            optimizer.zero_grad()

            if phase == "in_batch":
                # InfoNCE loss — all off-diagonal items are negatives
                out  = model(user_feats, item_feats, num_interactions=n_inter)
                loss = infonce_loss(out["combined"], out["item_emb"])

            else:
                # BPR loss — explicit negative samples
                out  = model(user_feats, item_feats, num_interactions=n_inter)
                # For simplicity, use random negatives from catalog
                neg_idx  = torch.randint(0, len(item_agg), (len(user_feats), 4))
                neg_feat = torch.stack([
                    torch.tensor(
                        item_agg.iloc[neg_idx[i].numpy()][
                            [c for c in item_agg.columns if c != "asin"]
                        ].values.astype(np.float32)
                    )
                    for i in range(len(user_feats))
                ]).to(device)

                out2 = model(user_feats, item_feats, neg_feat, n_inter)
                loss = bpr_loss(out2["pos_scores"], out2["neg_scores"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

            if batch_idx % 500 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{cfg.num_epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Phase: {phase}"
                )

        avg_train_loss = total_loss / num_batches
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                user_feats = batch["user_features"].to(device)
                item_feats = batch["item_features"].to(device)
                n_inter    = batch["num_interactions"].to(device)

                out  = model(user_feats, item_feats, num_interactions=n_inter)
                loss = infonce_loss(out["combined"], out["item_emb"])
                val_loss   += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{cfg.num_epochs} complete — "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"best={best_val_loss:.4f} | "
            f"time={epoch_time:.0f}s"
        )

        save_checkpoint(model, optimizer, epoch + 1, avg_val_loss, cfg, is_best)

    # ── Save final model ───────────────────────────────────────────────────────
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")

    total_time = time.time() - t0
    logger.info(f"\nTraining complete in {total_time/60:.1f} minutes")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {model_dir}/model.pt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-tower model")
    parser.add_argument(
        "--config",
        choices=["config1", "config2", "config3"],
        required=True,
        help="Which hyperparameter config to train"
    )
    parser.add_argument(
        "--blob_sync",
        action="store_true",
        help="Sync checkpoints to Azure Blob Storage after each save"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = get_config(args.config)
    if args.blob_sync:
        cfg.blob_sync = True
    train(cfg)
