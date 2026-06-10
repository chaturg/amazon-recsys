"""
upload_model_hf.py
------------------
Pushes the fine-tuned cross-encoder model to HF Model Hub.

Uploads:
  - Model weights (pytorch_model.bin or model.safetensors)
  - Tokenizer files (tokenizer.json, vocab.txt, etc.)
  - Model card (README.md)

Usage:
    export HF_TOKEN="your_token_here"
    python scripts/upload_model_hf.py \
        --model_dir artifacts/cross_encoder/best \
        --hf_repo   chaturg/amazon-recsys-cross-encoder
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


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: cross-encoder/ms-marco-MiniLM-L-6-v2
tags:
- cross-encoder
- reranking
- recommendation-system
- amazon-reviews
- tools-and-home-improvement
language:
- en
---

# Amazon RecSys Cross-Encoder Re-ranker

Fine-tuned cross-encoder for re-ranking retrieval candidates in a two-stage
recommendation system for the Amazon Reviews Tools & Home Improvement domain.

## Model details

- **Base model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache 2.0)
- **Task:** Pointwise relevance scoring for product re-ranking
- **Domain:** Amazon Tools & Home Improvement (McAuley Lab 2023)
- **Fine-tuning data:** FAISS top-100 candidates with silver label relevance scores

## Input format

```
[CLS] {user_history_summary} [SEP] {item_title} {item_category} [SEP]
```

**Left side:** User purchase history summary — titles of items the user has
bought, in reverse chronological order, comma-separated.

**Right side:** Candidate item title and category from the Amazon product catalog.

**Example:**
```
[CLS] bought DEWALT 20V MAX drill, Stanley screwdrivers, Irwin drill bits
[SEP] Makita 18V LXT Drill Driver Combo Kit Power Tools [SEP]
```

## Loss function

**Pointwise cross-entropy (BCEWithLogitsLoss)** against silver labels.

Silver labels are continuous per-item scores ∈ [0,1] derived from:
- Per-user z-scored ratings (weight 0.15)
- Helpfulness votes (weight 0.15)
- Verified purchase flag (weight 0.40)
- Review length (weight 0.30)

Pointwise cross-entropy was chosen over pairwise hinge loss because:
1. Silver labels are continuous per-item scores, not pairwise annotations
2. Cross-entropy matches the base model's MS MARCO pre-training distribution
3. No margin hyperparameter required

## Training details

- **Training pairs:** FAISS top-100 candidates per user (Config 3 index)
- **Positive label:** Silver label of the item the user actually purchased
- **Negative label:** 0.0 for FAISS-retrieved items not in user history
- **Epochs:** 3
- **Learning rate:** 2e-5 with cosine annealing
- **Max sequence length:** 128 tokens
- **Batch size:** 32

## Usage

```python
from retrieval.cross_encoder import CrossEncoderRanker

ranker = CrossEncoderRanker.load("chaturg/amazon-recsys-cross-encoder")

# Re-rank FAISS candidates
candidates = [
    {"asin": "B001", "title": "DEWALT 20V Drill", "category": "Power Tools"},
    {"asin": "B002", "title": "Makita 18V Drill", "category": "Power Tools"},
    # ... up to 100 candidates from FAISS
]

history = "bought DEWALT drill, Stanley screwdrivers, 3M safety glasses"
top10 = ranker.rerank(candidates, history_summary=history, top_k=10)
```

## Architecture context

This cross-encoder is Stage 2 in a two-stage retrieval pipeline:

1. **Stage 1 (Two-Tower + FAISS IVF):** Retrieves top-100 candidates in ~0.3s
2. **Stage 2 (Cross-Encoder):** Re-ranks top-100 to top-10 in ~0.5s

The cross-encoder applies full bidirectional attention over the concatenated
[user_history, item] sequence — more accurate than the bi-encoder but only
feasible on the ~100 candidate shortlist, not the full 157k item catalog.

## Related repositories

- **Dataset:** [chaturg/amazon-recsys-dataset](https://huggingface.co/datasets/chaturg/amazon-recsys-dataset)
- **Demo:** [chaturg/amazon-recsys-demo](https://huggingface.co/spaces/chaturg/amazon-recsys-demo)
- **Code:** [GitHub — amazon-recsys](https://github.com/chaturg/amazon-recsys)

## Citation

```
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and
          Chen, Xuanting and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
"""


def upload(model_dir: str, hf_repo: str, hf_token: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("pip install huggingface-hub")

    api      = HfApi()
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Run cross-encoder fine-tuning first."
        )

    # ── Create repo ────────────────────────────────────────────────────────
    logger.info(f"Creating/verifying model repo: {hf_repo}")
    api.create_repo(
        repo_id    = hf_repo,
        repo_type  = "model",
        token      = hf_token,
        exist_ok   = True,
        private    = False,
    )

    # ── Upload model files ─────────────────────────────────────────────────
    logger.info(f"Uploading model files from {model_dir}...")
    api.upload_folder(
        folder_path    = str(model_dir),
        repo_id        = hf_repo,
        repo_type      = "model",
        token          = hf_token,
        commit_message = "Upload fine-tuned cross-encoder weights and tokenizer",
        ignore_patterns = ["*.log", "__pycache__", "*.pyc"],
    )
    logger.info("  ✓ Model files uploaded")

    # ── Upload model card ──────────────────────────────────────────────────
    logger.info("Uploading model card...")
    card_bytes = MODEL_CARD_TEMPLATE.encode("utf-8")
    api.upload_file(
        path_or_fileobj = card_bytes,
        path_in_repo    = "README.md",
        repo_id         = hf_repo,
        repo_type       = "model",
        token           = hf_token,
        commit_message  = "Add model card",
    )
    logger.info("  ✓ Model card uploaded")

    url = f"https://huggingface.co/{hf_repo}"
    logger.info(f"\nDone. Model live at: {url}")
    print(f"\n{'='*60}")
    print(f"  Cross-encoder uploaded successfully")
    print(f"  URL: {url}")
    print(f"{'='*60}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push fine-tuned cross-encoder to HF Model Hub"
    )
    parser.add_argument(
        "--model_dir",
        default="artifacts/cross_encoder/best",
        help="Local path to fine-tuned model directory"
    )
    parser.add_argument(
        "--hf_repo",
        default="chaturg/amazon-recsys-cross-encoder",
        help="HF model repo in format username/repo-name"
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF write token (defaults to HF_TOKEN env var)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.hf_token:
        raise ValueError("HF token not found. Set HF_TOKEN environment variable.")
    upload(
        model_dir = args.model_dir,
        hf_repo   = args.hf_repo,
        hf_token  = args.hf_token,
    )
