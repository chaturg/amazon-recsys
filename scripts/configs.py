"""
experiments/configs.py
----------------------
Single source of truth for all hyperparameter configurations.

Every training, evaluation, and results script imports from here.
Changing a config in one place propagates to all downstream scripts.

3 configs tell a progressive story:
  Config 1 (baseline):        Random negatives, flat FAISS, no cross-encoder
  Config 2 (better retrieval): In-batch negatives, IVF FAISS, no cross-encoder
  Config 3 (full system):     ANN hard negatives, IVF FAISS, cross-encoder

IMPORTANT SEQUENCING CONSTRAINT:
  Config 3 mines hard negatives from the FAISS index built during Config 2.
  Config 2 must be fully trained and its FAISS index built before Config 3
  training begins. The experiment runner enforces this ordering.

Usage:
    from experiments.configs import CONFIGS, get_config
    cfg = get_config("config2")
    print(cfg.embed_dim)  # 128
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TwoTowerConfig:
    """
    Complete configuration for one two-tower training run.
    All scripts read from this dataclass — never from hardcoded values.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    name: str = "config1_baseline"
    description: str = "Baseline — random negatives, flat FAISS, no cross-encoder"

    # ── Data paths ────────────────────────────────────────────────────────────
    train_path: str = "processed/train.parquet"
    val_path:   str = "processed/val.parquet"
    test_path:  str = "processed/test.parquet"

    # ── Model architecture ────────────────────────────────────────────────────
    embed_dim:   int   = 64       # embedding dimension for both towers
    tower_depth: int   = 2        # number of hidden layers per tower
    tower_width: int   = 256      # hidden layer width
    dropout:     float = 0.1      # dropout rate

    # User tower input features
    user_features: list = field(default_factory=lambda: [
        "rating_norm_mean",       # mean of user's rating_norm scores
        "rating_norm_std",        # std of user's rating_norm scores
        "helpfulness_mean",       # mean helpfulness score
        "verified_ratio",         # fraction of verified purchases
        "length_mean",            # mean review length score
        "interaction_count_norm", # normalized interaction count
        "category_entropy",       # diversity across categories
    ])

    # Item tower input features
    item_features: list = field(default_factory=lambda: [
        "avg_rating_norm",        # normalized average rating
        "review_count_norm",      # normalized review count
        "avg_silver_label",       # mean silver label across all reviewers
        "verified_ratio",         # fraction of verified reviews
        "avg_length_score",       # mean review length score
    ])

    # ── Training ──────────────────────────────────────────────────────────────
    learning_rate:    float = 1e-3
    batch_size:       int   = 1024
    num_epochs:       int   = 10
    weight_decay:     float = 1e-5
    warmup_steps:     int   = 1000
    grad_clip:        float = 1.0

    # ── Negative sampling ─────────────────────────────────────────────────────
    # Phase 1: random negatives (epochs 1 to neg_phase2_epoch-1)
    # Phase 2: in-batch negatives (epochs neg_phase2_epoch to neg_phase3_epoch-1)
    # Phase 3: ANN hard negatives (epochs neg_phase3_epoch onwards)
    neg_strategy:    Literal["random", "in_batch", "ann_hard"] = "random"
    neg_phase2_epoch: int = 3     # switch to in-batch at this epoch
    neg_phase3_epoch: int = 6     # switch to ANN hard negatives at this epoch
    num_neg_samples:  int = 4     # negatives per positive (random strategy)

    # ── FAISS index ───────────────────────────────────────────────────────────
    faiss_index_type: Literal["flat", "ivf"] = "flat"
    faiss_nlist:      int = 100   # IVF: number of Voronoi cells
    faiss_nprobe:     int = 10    # IVF: cells to search at query time
    faiss_index_path: str = "artifacts/faiss_index.bin"

    # ── Cross-encoder ─────────────────────────────────────────────────────────
    use_cross_encoder:      bool  = False
    cross_encoder_model:    str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_lr:       float = 2e-5
    cross_encoder_epochs:   int   = 3
    cross_encoder_max_len:  int   = 128
    cross_encoder_path:     str   = "artifacts/cross_encoder"
    reranker_top_k:         int   = 100   # candidates passed to cross-encoder

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir:   str  = "outputs/checkpoints"
    blob_sync:        bool = False   # set True to sync to Azure Blob
    blob_container:   str  = "recsys-checkpoints"
    blob_prefix:      str  = "checkpoints"

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_top_k:       int  = 10    # top-K for NDCG, Recall, HitRate, MRR
    eval_recall_k:    int  = 100   # top-K for Recall@100 (bi-encoder only)

    # ── Output ────────────────────────────────────────────────────────────────
    model_dir:        str  = "artifacts/two_tower"
    results_csv:      str  = "results/eval_table.csv"


# ── Config 1: Baseline ────────────────────────────────────────────────────────
CONFIG1 = TwoTowerConfig(
    name        = "config1_baseline",
    description = "Baseline — random negatives, flat FAISS, no cross-encoder",
    embed_dim   = 64,
    dropout     = 0.1,
    learning_rate      = 1e-3,
    neg_strategy       = "random",
    neg_phase2_epoch   = 999,   # never switch — random only
    neg_phase3_epoch   = 999,
    faiss_index_type   = "flat",
    use_cross_encoder  = False,
    checkpoint_dir     = "outputs/checkpoints/config1",
    model_dir          = "artifacts/two_tower/config1",
    faiss_index_path   = "artifacts/faiss_index/config1.bin",
)

# ── Config 2: Better retrieval ────────────────────────────────────────────────
CONFIG2 = TwoTowerConfig(
    name        = "config2_better_retrieval",
    description = "Better retrieval — in-batch negatives, IVF FAISS, no cross-encoder",
    embed_dim   = 128,
    dropout     = 0.2,
    learning_rate      = 5e-4,
    neg_strategy       = "in_batch",
    neg_phase2_epoch   = 1,    # start in-batch immediately
    neg_phase3_epoch   = 999,  # never switch to hard negatives
    faiss_index_type   = "ivf",
    faiss_nlist        = 100,
    faiss_nprobe       = 10,
    use_cross_encoder  = False,
    checkpoint_dir     = "outputs/checkpoints/config2",
    model_dir          = "artifacts/two_tower/config2",
    faiss_index_path   = "artifacts/faiss_index/config2.bin",
)

# ── Config 3: Full system ─────────────────────────────────────────────────────
# NOTE: Requires Config 2 FAISS index to exist for ANN hard negative mining.
#       Run Config 2 fully (including FAISS index build) before Config 3.
CONFIG3 = TwoTowerConfig(
    name        = "config3_full_system",
    description = "Full system — ANN hard negatives, IVF FAISS, cross-encoder",
    embed_dim   = 128,
    dropout     = 0.2,
    learning_rate      = 5e-4,
    neg_strategy       = "ann_hard",
    neg_phase2_epoch   = 1,    # start in-batch immediately
    neg_phase3_epoch   = 4,    # switch to ANN hard negatives at epoch 4
    faiss_index_type   = "ivf",
    faiss_nlist        = 100,
    faiss_nprobe       = 20,   # higher nprobe for hard neg mining
    use_cross_encoder  = True,
    cross_encoder_lr   = 2e-5,
    cross_encoder_epochs = 3,
    reranker_top_k     = 100,
    checkpoint_dir     = "outputs/checkpoints/config3",
    model_dir          = "artifacts/two_tower/config3",
    faiss_index_path   = "artifacts/faiss_index/config3.bin",
    # Cross-encoder mines hard negatives from Config 2 FAISS index
    # Set this path after Config 2 training completes
)

# ── Registry ──────────────────────────────────────────────────────────────────
CONFIGS = {
    "config1": CONFIG1,
    "config2": CONFIG2,
    "config3": CONFIG3,
}


def get_config(name: str) -> TwoTowerConfig:
    """
    Retrieve a config by name.

    Args:
        name: One of "config1", "config2", "config3"

    Returns:
        TwoTowerConfig dataclass instance

    Raises:
        ValueError if name not found
    """
    if name not in CONFIGS:
        raise ValueError(
            f"Unknown config '{name}'. "
            f"Available: {list(CONFIGS.keys())}"
        )
    return CONFIGS[name]
