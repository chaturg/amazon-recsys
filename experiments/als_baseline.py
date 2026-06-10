"""
als_baseline.py
---------------
ALS (Alternating Least Squares) collaborative filtering baseline.

Why ALS not SVD:
  ALS is designed for implicit feedback with confidence weighting.
  SVD treats all interactions equally. Since silver labels are calibrated
  confidence scores, ALS is the architecturally correct choice — it uses
  the silver labels as confidence weights directly without binarization.

Confidence formula:
  confidence = 1 + alpha * silver_label
  where alpha=40 (standard production default)
  
  silver_label=0.94 → confidence=38.6  (strong positive)
  silver_label=0.20 → confidence=9.0   (weak positive)
  Missing interaction → confidence=0   (not in sparse matrix)

This is compared against Config 1/2/3 on the same 6 metrics.
ALS cannot use query text — Recall@100 (synthetic) is N/A for ALS.

Usage:
    python experiments/als_baseline.py

    # Or import and run:
    from experiments.als_baseline import run_als_baseline
    results = run_als_baseline(
        train_path="processed/train.parquet",
        val_path="processed/val.parquet",
        top_k=10,
    )
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ALS hyperparameters
ALS_FACTORS        = 128    # match Config 2/3 embed_dim for fair comparison
ALS_ITERATIONS     = 20
ALS_REGULARIZATION = 0.01
ALS_ALPHA          = 40     # confidence scaling: C = 1 + alpha * silver_label


def build_confidence_matrix(
    df:    pd.DataFrame,
    alpha: float = ALS_ALPHA,
) -> tuple:
    """
    Build sparse user-item confidence matrix from silver labels.

    C_ui = 1 + alpha * silver_label_ui  for observed interactions
    C_ui = 0                            for unobserved interactions

    Args:
        df:    DataFrame with user_id, asin, silver_label columns
        alpha: Confidence scaling factor

    Returns:
        (sparse_matrix, user_ids, item_ids)
        sparse_matrix shape: [n_users, n_items]
    """
    user_ids = df["user_id"].unique()
    item_ids = df["asin"].unique()

    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {a: i for i, a in enumerate(item_ids)}

    rows = df["user_id"].map(user_idx).values
    cols = df["asin"].map(item_idx).values
    data = (1 + alpha * df["silver_label"]).values.astype(np.float32)

    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(item_ids))
    )

    logger.info(
        f"  Confidence matrix: {matrix.shape[0]:,} users × "
        f"{matrix.shape[1]:,} items | "
        f"nnz={matrix.nnz:,} | "
        f"density={matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}"
    )
    return matrix, user_ids, item_ids


def run_als_baseline(
    train_path:   str = "processed/train.parquet",
    val_path:     str = "processed/val.parquet",
    results_path: str = "results/eval_table.csv",
    top_k:        int = 10,
    catalog_size: int = 157_462,
    factors:      int = ALS_FACTORS,
    iterations:   int = ALS_ITERATIONS,
    alpha:        float = ALS_ALPHA,
) -> dict:
    """
    Train ALS baseline and evaluate on val set.

    Returns:
        dict with all 6 metrics + config metadata
    """
    try:
        from implicit import als
    except ImportError:
        raise ImportError(
            "implicit library not installed. Run: pip install implicit"
        )

    from eval.metrics import evaluate_rankings, compute_composite_kpi

    t0 = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"ALS Baseline")
    logger.info(f"  factors={factors} | iterations={iterations} | alpha={alpha}")
    logger.info(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    logger.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    # ── Build confidence matrix ────────────────────────────────────────────
    logger.info("Building confidence matrix...")
    conf_matrix, user_ids, item_ids = build_confidence_matrix(train_df, alpha)

    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {a: i for i, a in enumerate(item_ids)}

    # ── Train ALS ──────────────────────────────────────────────────────────
    logger.info("Training ALS model...")
    model = als.AlternatingLeastSquares(
        factors        = factors,
        iterations     = iterations,
        regularization = ALS_REGULARIZATION,
        use_gpu        = False,
    )

    # implicit expects item-user matrix (transposed)
    model.fit(conf_matrix.T)
    logger.info(f"  ALS training complete")

    # ── Generate recommendations for val users ─────────────────────────────
    logger.info("Generating recommendations for val users...")
    val_users = val_df["user_id"].unique()

    # Val ground truth: one item per user (their last interaction)
    val_ground_truth = val_df.set_index("user_id")["asin"].to_dict()

    user_ids_eval   = []
    recommendations = []
    ground_truths   = []

    for uid in val_users:
        if uid not in user_idx:
            continue

        u_idx = user_idx[uid]

        # Get ALS recommendations — returns (item_indices, scores)
        try:
            item_indices, _ = model.recommend(
                u_idx,
                conf_matrix[u_idx],
                N           = top_k,
                filter_already_liked_items = True,
            )
            rec_asins = [item_ids[i] for i in item_indices]
        except Exception:
            rec_asins = []

        gt_asin = val_ground_truth.get(uid)
        if gt_asin is None:
            continue

        user_ids_eval.append(uid)
        recommendations.append(rec_asins)
        ground_truths.append([gt_asin])

    logger.info(f"  Generated recommendations for {len(user_ids_eval):,} users")

    # ── Evaluate ───────────────────────────────────────────────────────────
    logger.info("Computing metrics...")
    results = evaluate_rankings(
        user_ids     = user_ids_eval,
        recommended  = recommendations,
        ground_truth = ground_truths,
        catalog_size = catalog_size,
        k            = top_k,
    )

    elapsed = time.time() - t0

    # Add metadata
    results.update({
        "config":      "als_baseline",
        "description": f"ALS CF baseline (factors={factors}, alpha={alpha})",
        "source":      "real",
        "runtime_min": round(elapsed / 60, 1),
        # ALS cannot use query text — N/A for synthetic recall
        "recall_synthetic":   None,
        "recall_title_proxy": None,
    })

    logger.info(f"\nALS complete in {elapsed/60:.1f} min")

    # ── Save to CSV ────────────────────────────────────────────────────────
    _append_to_csv(results, results_path)

    return results


def _append_to_csv(results: dict, path: str) -> None:
    """Append results row to eval CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([results])

    if Path(path).exists():
        df_existing = pd.read_csv(path)
        # Replace existing row for same config
        df_existing = df_existing[df_existing["config"] != results["config"]]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(path, index=False)
    logger.info(f"  Results saved to {path}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run ALS baseline evaluation")
    parser.add_argument("--train_path",   default="processed/train.parquet")
    parser.add_argument("--val_path",     default="processed/val.parquet")
    parser.add_argument("--results_path", default="results/eval_table.csv")
    parser.add_argument("--top_k",        type=int, default=10)
    args = parser.parse_args()

    results = run_als_baseline(
        train_path   = args.train_path,
        val_path     = args.val_path,
        results_path = args.results_path,
        top_k        = args.top_k,
    )
    print(f"\nALS Results: {results}")
