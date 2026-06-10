"""
pipeline.py
-----------
Orchestrator — runs all data pipeline steps in order and writes
train/val/test parquet files to the output directory.

Steps:
  1. Load raw JSONL            (loader.py)
  2. Clean text                (cleaner.py)
  3. Remove cold-start         (filters.py)
  4. Generate silver labels    (silver_labels.py)
  5. Temporal leave-one-out    (splitter.py)
  6. Write parquet files       (this file)

Usage (command line):
    python -m data.pipeline \
        --input  raw/Tools_and_Home_Improvement.jsonl \
        --output processed/

Usage (Python):
    from data.pipeline import run
    run(input_path="raw/Tools_and_Home_Improvement.jsonl",
        output_dir="processed/")

Acceptance criterion (PRD Handoff 2):
    pipeline.py completes without error.
    3 parquet files written: train.parquet, val.parquet, test.parquet.
    Silver label histogram looks reasonable (not degenerate).
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from data.cleaner       import clean_reviews
from data.filters       import remove_cold_start, log_interaction_stats
from data.loader        import load_reviews
from data.silver_labels import generate_silver_labels
from data.splitter      import time_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Columns written to parquet — keeps file size small
# The model training code reads exactly these columns
OUTPUT_COLS = [
    "user_id",
    "asin",
    "rating",
    "silver_label",
    "timestamp",
    "split",
    # Signal columns kept for debugging and ablation studies
    "rating_norm",
    "helpfulness_score",
    "verified_score",
    "length_score",
    # Text kept for cross-encoder training pairs and LLM judge context
    "text",
]


def run(input_path: str, output_dir: str) -> dict:
    """
    Execute the full data pipeline.

    Args:
        input_path: Path to the raw Amazon Reviews JSONL file.
        output_dir: Directory to write train.parquet, val.parquet,
                    test.parquet.

    Returns:
        dict with keys "train", "val", "test" mapping to the
        output file paths (useful when calling from other scripts).
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Amazon RecSys — Data Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    logger.info("\n[1/5] Loading raw reviews...")
    df = load_reviews(input_path)

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    logger.info("\n[2/5] Cleaning review text...")
    df = clean_reviews(df)

    # ── Step 3: Filter cold-start ─────────────────────────────────────────────
    logger.info("\n[3/5] Removing cold-start users and items...")
    df = remove_cold_start(df)
    log_interaction_stats(df, label="post-filter")

    # ── Step 4: Silver labels ─────────────────────────────────────────────────
    logger.info("\n[4/5] Generating silver labels...")
    df = generate_silver_labels(df)

    # ── Step 5: Split ─────────────────────────────────────────────────────────
    logger.info("\n[5/5] Computing temporal leave-one-out split...")
    train, val, test = time_split(df)

    # ── Step 6: Write parquet ─────────────────────────────────────────────────
    logger.info("\nWriting parquet files...")
    output_paths = {}

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        # Keep only output columns that exist in this split
        cols = [c for c in OUTPUT_COLS if c in split_df.columns]
        out_path = output_dir / f"{split_name}.parquet"
        split_df[cols].to_parquet(out_path, index=False)
        output_paths[split_name] = str(out_path)
        logger.info(f"  Wrote {len(split_df):,} rows → {out_path}")

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"{'=' * 60}")

    _print_summary(train, val, test, output_dir)
    _check_silver_label_sanity(train)

    return output_paths


def _print_summary(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    output_dir: Path,
) -> None:
    """Print a human-readable summary table after the run."""
    total = len(train) + len(val) + len(test)
    print("\n" + "─" * 50)
    print("  Pipeline Summary")
    print("─" * 50)
    print(f"  Output dir : {output_dir}")
    print(f"  Train      : {len(train):>8,} interactions  "
          f"({train['user_id'].nunique():,} users)")
    print(f"  Val        : {len(val):>8,} interactions  "
          f"({val['user_id'].nunique():,} users)")
    print(f"  Test       : {len(test):>8,} interactions  "
          f"({test['user_id'].nunique():,} users)")
    print(f"  Total      : {total:>8,} interactions")
    print(f"  Items      : {train['asin'].nunique():,} unique (train)")
    print(f"  Silver lbl : mean={train['silver_label'].mean():.3f}  "
          f"std={train['silver_label'].std():.3f}")
    print("─" * 50 + "\n")


def _check_silver_label_sanity(train: pd.DataFrame) -> None:
    """
    Warn if the silver label distribution looks degenerate.
    A healthy distribution should not have >80% of labels in a single bin.
    """
    bins = pd.cut(train["silver_label"], bins=5)
    max_bin_pct = bins.value_counts(normalize=True).max()
    if max_bin_pct > 0.80:
        logger.warning(
            f"Silver label distribution looks degenerate: "
            f"{max_bin_pct:.1%} of labels fall in one bin. "
            f"Check silver_labels.py signal weights."
        )
    else:
        logger.info(
            f"Silver label sanity check passed "
            f"(max bin concentration: {max_bin_pct:.1%})"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Amazon RecSys data pipeline — raw JSONL → train/val/test parquet"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw Amazon Reviews JSONL file "
             "(e.g. raw/Tools_and_Home_Improvement.jsonl)"
    )
    parser.add_argument(
        "--output",
        default="processed/",
        help="Output directory for parquet files (default: processed/)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(input_path=args.input, output_dir=args.output)
