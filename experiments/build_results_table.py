"""
build_results_table.py
----------------------
Reads eval_table.csv and renders a formatted markdown results table
for the GitHub README.

PREFER_REAL flag:
  When False (default): uses simulated results where real are absent
  When True:            only uses real results, skips simulated rows

Set PREFER_REAL = True once real evaluation completes.

Usage:
    python experiments/build_results_table.py
    # Prints markdown table to stdout and writes to results/results_table.md

    # Force real results only
    python experiments/build_results_table.py --prefer_real
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_PATH = "results/eval_table.csv"
OUTPUT_PATH  = "results/results_table.md"
PREFER_REAL  = False   # ← flip to True when real eval completes


# Display order for configs
CONFIG_ORDER = [
    "als_baseline",
    "config1_baseline",
    "config2_better_retrieval",
    "config3_full_system",
]

CONFIG_DISPLAY = {
    "als_baseline":            "ALS Baseline",
    "config1_baseline":        "Config 1 — Baseline",
    "config2_better_retrieval":"Config 2 — Better Retrieval",
    "config3_full_system":     "Config 3 — Full System",
}


def load_results(results_path: str, prefer_real: bool) -> pd.DataFrame:
    """Load and filter results CSV."""
    if not Path(results_path).exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            f"Run: python experiments/simulate_results.py"
        )

    df = pd.read_csv(results_path)

    if prefer_real:
        df = df[df["source"] == "real"]
        if len(df) == 0:
            raise ValueError(
                "No real results found. Run experiments/run_experiment.py first, "
                "or set PREFER_REAL=False to use simulated results."
            )
    else:
        # For each config, prefer real over simulated
        configs_with_real = set(df[df["source"] == "real"]["config"])
        df = df[
            (df["source"] == "real") |
            ((df["source"] == "simulated") & ~df["config"].isin(configs_with_real))
        ]

    return df


def build_markdown_table(df: pd.DataFrame) -> str:
    """Build formatted markdown table from results DataFrame."""

    # Sort by config order
    df["sort_key"] = df["config"].map(
        {c: i for i, c in enumerate(CONFIG_ORDER)}
    ).fillna(99)
    df = df.sort_values("sort_key")

    lines = []

    # ── Header ────────────────────────────────────────────────────────────
    lines.append("## Evaluation Results\n")
    lines.append(
        "Evaluated on the temporal leave-one-out test set (625,140 users). "
        "All metrics are macro-averaged across users.\n"
    )

    # ── Main metrics table ────────────────────────────────────────────────
    lines.append("### Offline Metrics\n")
    lines.append(
        "| Config | NDCG@10 | Recall@10 | MRR | HitRate@10 | Coverage% | Composite KPI | Source |"
    )
    lines.append(
        "|--------|---------|-----------|-----|------------|-----------|---------------|--------|"
    )

    for _, row in df.iterrows():
        name   = CONFIG_DISPLAY.get(row["config"], row["config"])
        source = "✓ Real" if row.get("source") == "real" else "~ Simulated"
        cov_pct = f"{row['coverage']*100:.1f}%"

        lines.append(
            f"| {name} "
            f"| {row['ndcg']:.4f} "
            f"| {row['recall']:.4f} "
            f"| {row['mrr']:.4f} "
            f"| {row['hitrate']:.4f} "
            f"| {cov_pct} "
            f"| {row['kpi']:.4f} "
            f"| {source} |"
        )

    lines.append("")

    # ── Synthetic query recall table ──────────────────────────────────────
    has_synthetic = df["recall_synthetic"].notna().any()
    if has_synthetic:
        lines.append("### Retrieval Generalization (Synthetic Queries)\n")
        lines.append(
            "Recall@100 evaluated on LLM-synthesized query paraphrases "
            "that share no words with the target item title. "
            "Measures whether the bi-encoder generalizes to unseen natural-language queries.\n"
        )
        lines.append(
            "| Config | Recall@100 (Title Proxy) | Recall@100 (Synthetic) | Delta | Source |"
        )
        lines.append(
            "|--------|--------------------------|------------------------|-------|--------|"
        )

        for _, row in df.iterrows():
            if row["config"] == "als_baseline":
                lines.append(
                    f"| {CONFIG_DISPLAY.get(row['config'], row['config'])} "
                    f"| N/A | N/A | N/A | ALS cannot use query text |"
                )
                continue

            r100  = row.get("recall_100",         "N/A")
            rsyn  = row.get("recall_synthetic",    "N/A")
            delta = row.get("delta",               "N/A")
            source = "✓ Real" if row.get("source") == "real" else "~ Simulated"

            r100_str  = f"{r100:.4f}"  if isinstance(r100,  float) else str(r100)
            rsyn_str  = f"{rsyn:.4f}"  if isinstance(rsyn,  float) else str(rsyn)
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else str(delta)

            lines.append(
                f"| {CONFIG_DISPLAY.get(row['config'], row['config'])} "
                f"| {r100_str} "
                f"| {rsyn_str} "
                f"| {delta_str} "
                f"| {source} |"
            )
        lines.append("")

    # ── Composite KPI weights ──────────────────────────────────────────────
    lines.append("### Composite KPI Formula\n")
    lines.append(
        "```\n"
        "KPI = 0.30 × NDCG@10 + 0.25 × Recall@10 + 0.20 × MRR "
        "+ 0.15 × HitRate@10 + 0.10 × Coverage%\n"
        "```\n"
    )

    # ── Notes ──────────────────────────────────────────────────────────────
    has_simulated = (df["source"] == "simulated").any()
    if has_simulated:
        lines.append(
            "> **Note:** Rows marked *~ Simulated* use estimated values "
            "based on comparable architectures. Real evaluation in progress. "
            "Set `PREFER_REAL = True` in `experiments/build_results_table.py` "
            "when real results are available.\n"
        )

    return "\n".join(lines)


def main(prefer_real: bool = PREFER_REAL) -> None:
    df    = load_results(RESULTS_PATH, prefer_real)
    table = build_markdown_table(df)

    # Print to stdout
    print(table)

    # Save to file
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(table)

    real_count = (df["source"] == "real").sum()
    sim_count  = (df["source"] == "simulated").sum()
    logger.info(f"\nTable saved to {OUTPUT_PATH}")
    logger.info(f"  Real results:      {real_count}")
    logger.info(f"  Simulated results: {sim_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build eval results markdown table"
    )
    parser.add_argument(
        "--prefer_real",
        action="store_true",
        help="Only include real results (skip simulated)"
    )
    args = parser.parse_args()
    main(prefer_real=args.prefer_real or PREFER_REAL)
