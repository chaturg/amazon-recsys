"""
upload_dataset_hf.py
--------------------
Uploads the processed train/val/test parquet splits and dataset card
to the Hugging Face Dataset Hub.

Usage:
    export HF_TOKEN="your_token_here"

    python scripts/upload_dataset_hf.py \
        --processed_dir processed/ \
        --hf_repo chaturg/amazon-recsys-dataset

What this script does:
    1. Logs in to HF using your token
    2. Creates the dataset repo if it does not exist
    3. Uploads train.parquet, val.parquet, test.parquet
    4. Uploads the dataset card as README.md
    5. Prints the public URL when done
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def upload(processed_dir: str, hf_repo: str, hf_token: str) -> None:
    try:
        from huggingface_hub import HfApi, DatasetCard
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface-hub"
        )

    processed_dir = Path(processed_dir)
    api = HfApi()

    # ── Validate local files ──────────────────────────────────────────────────
    splits = ["train", "val", "test"]
    for split in splits:
        fpath = processed_dir / f"{split}.parquet"
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing: {fpath}. Run data/pipeline.py first."
            )

    # ── Create repo if needed ─────────────────────────────────────────────────
    logger.info(f"Creating/verifying dataset repo: {hf_repo}")
    api.create_repo(
        repo_id=hf_repo,
        repo_type="dataset",
        token=hf_token,
        exist_ok=True,      # no error if repo already exists
        private=False,
    )

    # ── Upload parquet files ──────────────────────────────────────────────────
    for split in splits:
        local_path = processed_dir / f"{split}.parquet"
        remote_path = f"data/{split}.parquet"
        logger.info(f"Uploading {local_path} → {hf_repo}/{remote_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=hf_repo,
            repo_type="dataset",
            token=hf_token,
            commit_message=f"Upload {split}.parquet",
        )
        logger.info(f"  ✓ {split}.parquet uploaded")

    # ── Upload dataset card ───────────────────────────────────────────────────
    logger.info("Uploading dataset card (README.md)...")
    card_content = _build_dataset_card(hf_repo)
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hf_repo,
        repo_type="dataset",
        token=hf_token,
        commit_message="Add dataset card",
    )
    logger.info("  ✓ README.md uploaded")

    url = f"https://huggingface.co/datasets/{hf_repo}"
    logger.info(f"\nDone. Dataset live at: {url}")
    print(f"\n{'=' * 60}")
    print(f"  Dataset uploaded successfully")
    print(f"  URL: {url}")
    print(f"{'=' * 60}\n")


def _build_dataset_card(hf_repo: str) -> str:
    return f"""---
license: other
task_categories:
- text-retrieval
- feature-extraction
language:
- en
tags:
- recommendation-system
- amazon-reviews
- two-tower
- recsys
- tools-and-home-improvement
size_categories:
- 1M<n<10M
---

# Amazon RecSys Dataset — Tools & Home Improvement

Processed dataset derived from the Amazon Reviews 2023 corpus (Tools & Home
Improvement category), used to train a two-stage two-tower recommender system
with cross-encoder re-ranking.

## Dataset splits

| Split | Rows | Users | Description |
|-------|------|-------|-------------|
| train | 4,436,875 | 625,140 | Historical interactions per user |
| val   | 625,140   | 625,140 | Second-to-last interaction per user |
| test  | 625,140   | 625,140 | Last interaction per user |

**Split method:** Temporal leave-one-out per user. For each user, interactions
are sorted chronologically. The final interaction is reserved for test, the
second-to-last for validation, and all prior interactions for training. This
preserves causal ordering and prevents future data leakage into training.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Anonymised reviewer ID |
| asin | string | Amazon product ID |
| rating | float | Raw star rating (1–5) |
| silver_label | float | Calibrated preference score ∈ [0, 1] |
| timestamp | int | Unix timestamp of interaction |
| split | string | train / val / test |
| rating_norm | float | Per-user z-scored rating, scaled [0,1] |
| helpfulness_score | float | log1p(helpful_votes), scaled [0,1] |
| verified_score | float | Verified purchase indicator (0 or 1) |
| length_score | float | Review length signal, scaled [0,1] |
| text | string | Cleaned review text |

## Silver label construction

Raw star ratings are user-biased — a 3★ from a harsh rater carries more
positive signal than a 3★ from a generous rater. Silver labels correct for
this by combining four signals:

| Signal | Weight | Transformation | Rationale |
|--------|--------|----------------|-----------|
| Per-user z-scored rating | 0.50 | z-score → MinMax [0,1] | Corrects individual rating bias |
| Helpfulness vote | 0.20 | log1p → MinMax [0,1] | Community validation signal |
| Verified purchase | 0.15 | Binary float | Review quality indicator |
| Review length | 0.15 | clip(2000) → MinMax [0,1] | Engagement depth proxy |

**Formula:** `silver_label = 0.50×rating_norm + 0.20×helpfulness + 0.15×verified + 0.15×length`

**Distribution note:** Silver label distribution is narrow (mean=0.495,
std=0.055), reflecting the positive purchase bias inherent in verified Amazon
transactions. The z-score normalization corrects for inter-user rating
differences but cannot overcome dataset-level positivity bias.

## Cold-start filtering

Users with fewer than 5 interactions and items with fewer than 10 interactions
were removed using 3 iterative passes (items first, then users) until the
interaction matrix stabilised. This removed 70.5% of raw interactions,
retaining only users and items with sufficient signal for reliable embeddings.

## Dataset statistics

- **Raw interactions:** 26,982,256
- **After early filter (min 2 interactions/user):** 19,255,528
- **After cold-start filter:** 5,687,155
- **Users:** 625,140 (median 7 reviews/user)
- **Items:** 157,462 (median 19 reviews/item)
- **Sparsity:** 99.99%

## Source & citation

Derived from the Amazon Reviews 2023 dataset.

**Original source:** McAuley Lab, UC San Diego
**Citation:**
```
@article{{hou2024bridging,
  title={{Bridging Language and Items for Retrieval and Recommendation}},
  author={{Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xuanting and McAuley, Julian}},
  journal={{arXiv preprint arXiv:2403.03952}},
  year={{2024}}
}}
```

**License:** This derived dataset is for non-commercial research use only,
consistent with the terms of the original Amazon Reviews 2023 dataset.
Raw review text has been processed and transformed. The original raw data
is available at https://mcauleylab.ucsd.edu/data/gdrive/data/amazon_2023/

**Note:** This repository contains processed features and silver labels
derived from the original data. It does not redistribute raw review text
beyond cleaned snippets used for NLP feature extraction.

## Related repositories

- **Model:** [{hf_repo.split('/')[0]}/amazon-recsys-cross-encoder](https://huggingface.co/{hf_repo.split('/')[0]}/amazon-recsys-cross-encoder)
- **Demo:** [{hf_repo.split('/')[0]}/amazon-recsys-demo](https://huggingface.co/spaces/{hf_repo.split('/')[0]}/amazon-recsys-demo)
- **Code:** [GitHub — amazon-recsys](https://github.com/{hf_repo.split('/')[0]}/amazon-recsys)
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload processed parquet splits to Hugging Face Dataset Hub"
    )
    parser.add_argument(
        "--processed_dir",
        default="processed/",
        help="Directory containing train.parquet, val.parquet, test.parquet",
    )
    parser.add_argument(
        "--hf_repo",
        default="chaturg/amazon-recsys-dataset",
        help="HF dataset repo in format username/repo-name",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF write token. Defaults to HF_TOKEN environment variable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.hf_token:
        raise ValueError(
            "HF token not found. Set it with: export HF_TOKEN='your_token_here'"
        )

    upload(
        processed_dir=args.processed_dir,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )
