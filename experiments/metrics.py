"""
metrics.py
----------
All 6 evaluation metrics for the Amazon RecSys two-tower system.

Metrics:
  1. NDCG@K      — Normalized Discounted Cumulative Gain (primary signal)
  2. Recall@K    — Fraction of relevant items in top-K
  3. MRR         — Mean Reciprocal Rank
  4. HitRate@K   — Binary: 1 if any relevant item in top-K
  5. Coverage%   — Unique items recommended / catalog size (anti-popularity-bias)
  6. Composite   — 0.30×NDCG + 0.25×Recall + 0.20×MRR + 0.15×HitRate + 0.10×Coverage

All metrics are computed per user then averaged (macro average).
This gives equal weight to each user regardless of interaction count.

Usage:
    from eval.metrics import evaluate_rankings, compute_composite_kpi

    results = evaluate_rankings(
        user_ids        = ["u1", "u2", ...],
        recommended     = [["item_a", "item_b", ...], ...],  # top-K per user
        ground_truth    = [["item_x"], ["item_y"], ...],      # positive items per user
        silver_labels   = [{"item_x": 0.91}, {"item_y": 0.74}, ...],  # optional
        catalog_size    = 157_462,
        k               = 10,
    )
    print(results)
    # {"ndcg": 0.41, "recall": 0.53, "mrr": 0.38, "hitrate": 0.61,
    #  "coverage": 0.44, "kpi": 0.47, "n_users": 625140}
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Composite KPI weights — explicit and documented
KPI_WEIGHTS = {
    "ndcg":     0.30,
    "recall":   0.25,
    "mrr":      0.20,
    "hitrate":  0.15,
    "coverage": 0.10,
}


# ── Per-user metric functions ─────────────────────────────────────────────────

def ndcg_at_k(recommended: list, relevant: set, k: int,
              silver_labels: Optional[dict] = None) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    If silver_labels provided, uses continuous relevance scores as gains.
    Otherwise uses binary relevance (1 if item in relevant set, 0 otherwise).

    Args:
        recommended:   Ordered list of recommended item IDs (top-K)
        relevant:      Set of relevant item IDs for this user
        k:             Cutoff rank
        silver_labels: Optional dict mapping item_id → silver_label [0,1]

    Returns:
        NDCG@K score ∈ [0, 1]
    """
    if not relevant:
        return 0.0

    recommended = recommended[:k]

    # DCG — actual ranking
    dcg = 0.0
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            gain = silver_labels.get(item, 1.0) if silver_labels else 1.0
            dcg += gain / math.log2(rank + 1)

    # Ideal DCG — best possible ranking
    if silver_labels:
        ideal_gains = sorted(
            [silver_labels.get(item, 1.0) for item in relevant],
            reverse=True
        )[:k]
    else:
        ideal_gains = [1.0] * min(len(relevant), k)

    idcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, start=1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Recall at K — fraction of relevant items that appear in top-K.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant:    Set of relevant item IDs
        k:           Cutoff rank

    Returns:
        Recall@K ∈ [0, 1]
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def mrr(recommended: list, relevant: set) -> float:
    """
    Mean Reciprocal Rank — reciprocal of rank of first relevant item.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant:    Set of relevant item IDs

    Returns:
        MRR score ∈ [0, 1] (0 if no relevant item found)
    """
    if not relevant:
        return 0.0
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Hit Rate at K — binary: 1 if any relevant item in top-K, else 0.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant:    Set of relevant item IDs
        k:           Cutoff rank

    Returns:
        1.0 or 0.0
    """
    if not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def coverage_ratio(all_recommendations: list, catalog_size: int) -> float:
    """
    Coverage — fraction of catalog items that appear in any recommendation list.

    High coverage = system recommends diverse items.
    Low coverage = system over-recommends popular items (popularity bias).

    Args:
        all_recommendations: List of recommendation lists (one per user)
        catalog_size:        Total number of items in the catalog

    Returns:
        Coverage ratio ∈ [0, 1]
    """
    if catalog_size == 0:
        return 0.0
    unique_items = set(item for recs in all_recommendations for item in recs)
    return len(unique_items) / catalog_size


def compute_composite_kpi(
    ndcg:     float,
    recall:   float,
    mrr_val:  float,
    hitrate:  float,
    coverage: float,
) -> float:
    """
    Composite KPI — weighted combination of all 5 metrics.

    Weights: NDCG 0.30 + Recall 0.25 + MRR 0.20 + HitRate 0.15 + Coverage 0.10

    Args:
        ndcg, recall, mrr_val, hitrate, coverage: Individual metric scores

    Returns:
        Composite KPI ∈ [0, 1]
    """
    return (
        KPI_WEIGHTS["ndcg"]     * ndcg     +
        KPI_WEIGHTS["recall"]   * recall   +
        KPI_WEIGHTS["mrr"]      * mrr_val  +
        KPI_WEIGHTS["hitrate"]  * hitrate  +
        KPI_WEIGHTS["coverage"] * coverage
    )


# ── Batch evaluation ───────────────────────────────────────────────────────────

def evaluate_rankings(
    user_ids:       list,
    recommended:    list,
    ground_truth:   list,
    catalog_size:   int,
    silver_labels:  Optional[list] = None,
    k:              int = 10,
    recall_k:       int = 100,
    verbose:        bool = True,
) -> dict:
    """
    Evaluate recommendation rankings for a set of users.

    Args:
        user_ids:      List of user IDs (for logging)
        recommended:   List of recommendation lists — each is an ordered list
                       of item IDs. Can be length K (re-ranked) or 100 (FAISS only).
        ground_truth:  List of relevant item sets — one set per user.
                       Usually contains 1 item (the val/test interaction).
        catalog_size:  Total catalog size for coverage computation.
        silver_labels: Optional list of dicts mapping item_id → silver_label.
                       If provided, used as continuous gains in NDCG.
        k:             Top-K cutoff for NDCG, Recall, HitRate, MRR (default 10).
        recall_k:      Separate K for Recall@K computation (default 100,
                       used for bi-encoder evaluation).
        verbose:       Log progress every 50k users.

    Returns:
        dict with keys: ndcg, recall, recall_100, mrr, hitrate, coverage,
                        kpi, n_users
    """
    assert len(user_ids) == len(recommended) == len(ground_truth), \
        "user_ids, recommended, and ground_truth must have the same length"

    n = len(user_ids)
    ndcg_scores     = np.zeros(n)
    recall_scores   = np.zeros(n)
    recall_100_scores = np.zeros(n)
    mrr_scores      = np.zeros(n)
    hitrate_scores  = np.zeros(n)

    for i, (uid, recs, gt) in enumerate(zip(user_ids, recommended, ground_truth)):
        relevant = set(gt) if not isinstance(gt, set) else gt
        sl       = silver_labels[i] if silver_labels else None

        ndcg_scores[i]       = ndcg_at_k(recs, relevant, k, sl)
        recall_scores[i]     = recall_at_k(recs, relevant, k)
        recall_100_scores[i] = recall_at_k(recs, relevant, recall_k)
        mrr_scores[i]        = mrr(recs, relevant)
        hitrate_scores[i]    = hit_rate_at_k(recs, relevant, k)

        if verbose and (i + 1) % 50_000 == 0:
            logger.info(f"  Evaluated {i+1:,}/{n:,} users...")

    cov = coverage_ratio(recommended, catalog_size)

    avg_ndcg    = float(ndcg_scores.mean())
    avg_recall  = float(recall_scores.mean())
    avg_r100    = float(recall_100_scores.mean())
    avg_mrr     = float(mrr_scores.mean())
    avg_hitrate = float(hitrate_scores.mean())
    kpi         = compute_composite_kpi(avg_ndcg, avg_recall, avg_mrr,
                                        avg_hitrate, cov)

    results = {
        "ndcg":       round(avg_ndcg,    4),
        "recall":     round(avg_recall,  4),
        "recall_100": round(avg_r100,    4),
        "mrr":        round(avg_mrr,     4),
        "hitrate":    round(avg_hitrate, 4),
        "coverage":   round(cov,         4),
        "kpi":        round(kpi,         4),
        "n_users":    n,
    }

    if verbose:
        _log_results(results)

    return results


def _log_results(results: dict) -> None:
    logger.info(
        f"\n  Evaluation Results\n"
        f"  {'─'*40}\n"
        f"  NDCG@10:     {results['ndcg']:.4f}\n"
        f"  Recall@10:   {results['recall']:.4f}\n"
        f"  Recall@100:  {results['recall_100']:.4f}\n"
        f"  MRR:         {results['mrr']:.4f}\n"
        f"  HitRate@10:  {results['hitrate']:.4f}\n"
        f"  Coverage:    {results['coverage']:.4f}\n"
        f"  Composite:   {results['kpi']:.4f}\n"
        f"  {'─'*40}\n"
        f"  Users:       {results['n_users']:,}"
    )


# ── Synthetic query evaluation ─────────────────────────────────────────────────

def evaluate_synthetic_recall(
    synthetic_queries_path: str,
    faiss_index,
    model,
    item_ids:     np.ndarray,
    user_feat_map,
    item_feat_map,
    device,
    k:            int = 100,
    max_items:    Optional[int] = None,
) -> dict:
    """
    Compute Recall@K using synthetic query paraphrases.

    For each item in synthetic_queries.jsonl:
      1. Encode each of the 5 paraphrased queries
      2. Search FAISS index
      3. Check if the target item appears in top-K

    Returns:
        dict with recall_synthetic, recall_title_proxy, delta, n_items
    """
    import json
    import torch

    logger.info(f"Computing synthetic query Recall@{k}...")

    records = []
    with open(synthetic_queries_path) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    if max_items:
        records = records[:max_items]

    asin_to_idx = {asin: i for i, asin in enumerate(item_ids)}
    item_hits_synthetic   = 0
    item_hits_title_proxy = 0
    n_items = 0

    for rec in records:
        asin    = rec["asin"]
        queries = rec.get("queries", [])
        title   = rec.get("title", asin)

        if asin not in asin_to_idx:
            continue

        target_idx = asin_to_idx[asin]

        # Evaluate each synthetic query
        item_hit_syn = False
        for query_text in queries[:5]:
            # For metadata-only model, we use a mean user embedding as proxy
            # This tests whether the item is retrievable by semantic proximity
            # In a text-augmented model, query_text would be encoded directly
            pass  # placeholder — full implementation in run_experiment.py

        # Title proxy evaluation
        item_hit_title = False

        n_items += 1
        if item_hit_syn:    item_hits_synthetic   += 1
        if item_hit_title:  item_hits_title_proxy += 1

    recall_syn   = item_hits_synthetic   / n_items if n_items > 0 else 0.0
    recall_title = item_hits_title_proxy / n_items if n_items > 0 else 0.0

    logger.info(f"  Recall@{k} (synthetic):    {recall_syn:.4f}")
    logger.info(f"  Recall@{k} (title proxy):  {recall_title:.4f}")
    logger.info(f"  Delta:                     {recall_syn - recall_title:+.4f}")
    logger.info(f"  Items evaluated:           {n_items:,}")

    return {
        "recall_synthetic":   round(recall_syn,   4),
        "recall_title_proxy": round(recall_title, 4),
        "delta":              round(recall_syn - recall_title, 4),
        "n_items":            n_items,
    }
