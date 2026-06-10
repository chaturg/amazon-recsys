"""
build_faiss_index.py
--------------------
Standalone script to build the FAISS item index from a trained two-tower model.

Loads the trained item tower, runs all catalog items through it to generate
embeddings, then builds and serializes the FAISS IVF index.

This script must be run after training completes for each config.
Config 3 requires the Config 2 index to exist (for ANN hard neg mining).

Usage:
    python scripts/build_faiss_index.py --config config2

    # With custom paths:
    python scripts/build_faiss_index.py \
        --config config2 \
        --train_path processed/train.parquet \
        --output_path artifacts/faiss_index/config2.bin

Acceptance criterion (PRD Handoff 4):
    FAISS returns top-100 for a sample query with Recall@100 > 0.60
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.configs import get_config, TwoTowerConfig
from model.two_tower import TwoTowerModel
from model.train import build_item_features
from retrieval.faiss_index import FaissIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_index(
    cfg:         TwoTowerConfig,
    train_path:  str,
    output_path: str,
) -> FaissIndex:
    """
    Build FAISS item index from trained model.

    Args:
        cfg:         Config specifying model architecture and index type
        train_path:  Path to train.parquet (for item feature aggregation)
        output_path: Where to save the serialized index

    Returns:
        Built FaissIndex instance
    """
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Building FAISS index for: {cfg.name}")
    logger.info(f"Device: {device}")

    # ── Load trained model ────────────────────────────────────────────────────
    model_path = Path(cfg.model_dir) / "model.pt"
    if not model_path.exists():
        # Try best checkpoint
        ckpt_path = Path(cfg.checkpoint_dir) / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path} or {ckpt_path}. "
                f"Run training first: python -m model.train --config {cfg.name.split('_')[0]}"
            )
        logger.info(f"Loading from checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model = TwoTowerModel(cfg, user_input_dim=7, item_input_dim=5).to(device)
        model.load_state_dict(state["model_state"])
    else:
        logger.info(f"Loading model: {model_path}")
        model = TwoTowerModel(cfg, user_input_dim=7, item_input_dim=5).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # ── Build item feature matrix ──────────────────────────────────────────────
    logger.info("Loading training data for item feature aggregation...")
    train_df = pd.read_parquet(train_path)
    item_agg = build_item_features(train_df)

    feature_cols = [c for c in item_agg.columns if c != "asin"]
    item_ids     = item_agg["asin"].values
    item_feats   = item_agg[feature_cols].values.astype(np.float32)

    logger.info(f"  {len(item_ids):,} unique items to embed")

    # ── Generate item embeddings in batches ────────────────────────────────────
    logger.info("Generating item embeddings...")
    batch_size   = 4096
    n_items      = len(item_feats)
    all_embeddings = np.zeros((n_items, cfg.embed_dim), dtype=np.float32)

    feat_tensor = torch.tensor(item_feats)
    dataset = TensorDataset(feat_tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        start_idx = 0
        for batch_idx, (batch_feats,) in enumerate(loader):
            batch_feats = batch_feats.to(device)
            embeddings  = model.encode_item(batch_feats)
            end_idx     = start_idx + len(embeddings)
            all_embeddings[start_idx:end_idx] = embeddings.cpu().numpy()
            start_idx = end_idx

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"  Embedded {start_idx:,}/{n_items:,} items "
                    f"({start_idx/n_items:.0%})"
                )

    logger.info(f"  All {n_items:,} item embeddings generated")

    # ── Build FAISS index ──────────────────────────────────────────────────────
    faiss_index = FaissIndex(
        embed_dim  = cfg.embed_dim,
        index_type = cfg.faiss_index_type,
        nlist      = cfg.faiss_nlist,
        nprobe     = cfg.faiss_nprobe,
    )
    faiss_index.build(all_embeddings, item_ids)
    faiss_index.log_stats()

    # ── Save index ─────────────────────────────────────────────────────────────
    faiss_index.save(output_path)

    # ── Quick sanity check ────────────────────────────────────────────────────
    logger.info("Running sanity check — querying index with 5 sample vectors...")
    sample_queries = all_embeddings[:5]
    distances, indices = faiss_index.search(sample_queries, k=10)
    logger.info(f"  Sample search returned distances range: "
                f"[{distances.min():.3f}, {distances.max():.3f}]")
    logger.info(f"  Top-1 item for query 0: {faiss_index.item_ids[indices[0, 0]]}")

    elapsed = time.time() - t0
    logger.info(f"\nFAISS index build complete in {elapsed:.1f}s")
    logger.info(f"Index saved to: {output_path}")

    return faiss_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS item index from trained two-tower model"
    )
    parser.add_argument(
        "--config",
        choices=["config1", "config2", "config3"],
        required=True,
        help="Which config's trained model to use"
    )
    parser.add_argument(
        "--train_path",
        default="processed/train.parquet",
        help="Path to train.parquet for item feature aggregation"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output path for the serialized index. "
             "Defaults to cfg.faiss_index_path from configs.py"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = get_config(args.config)

    output_path = args.output_path or cfg.faiss_index_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    build_index(
        cfg         = cfg,
        train_path  = args.train_path,
        output_path = output_path,
    )
