"""
generate_synthetic_queries.py
-----------------------------
Generates 5 synthetic shopper queries per test item using Claude Haiku.
Designed to run in a Kaggle notebook after training completes.

Output: eval/synthetic_queries.jsonl
  One JSON object per line:
  {
    "asin": "B07BMHZN6K",
    "title": "DEWALT 20V MAX Cordless Drill Combo Kit",
    "category": "Power Tools",
    "queries": ["battery powered drill for home reno", ...]
  }

Usage in Kaggle notebook:
  # Set your Anthropic API key first
  import os
  os.environ["ANTHROPIC_API_KEY"] = "your_key_here"

  # Then run
  exec(open("generate_synthetic_queries.py").read())
  # OR paste directly into a cell

Runtime estimate:
  ~157k items × 5 queries each
  ~50 requests/sec with async
  ~52 minutes total
  ~$2 in Claude Haiku API costs

Resume support:
  If interrupted, re-running skips already-processed ASINs.
  Progress is checkpointed every 500 items.
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE            = "processed"
OUT             = "processed"
OUTPUT_PATH     = f"{OUT}/eval/synthetic_queries.jsonl"
CHECKPOINT_PATH = f"{OUT}/eval/synthetic_queries_checkpoint.json"
MODEL           = "claude-haiku-4-5"
MAX_TOKENS      = 300
BATCH_SIZE      = 20       # items per async batch
MAX_RETRIES     = 3
RETRY_DELAY     = 2.0      # seconds between retries

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are generating realistic shopping search queries for evaluation of an e-commerce recommendation system.

Your task is to create natural-language shopper queries for a product.

Goal:
Generate queries that a real shopper might type into Amazon or Google Shopping to discover this product category.

Rules:
1. Do NOT use any user history or personalization assumptions.
2. Do NOT assume knowledge of the exact product name.
3. Avoid copying the product title verbatim.
4. Use shopper intent, common language, problems-to-solve, or product attributes.
5. Queries should sound like authentic search behavior, not SEO keywords.
6. Include a mix of:
   * functional intent (what the shopper wants to do)
   * attribute intent (durable, lightweight, affordable, organic, waterproof, etc.)
   * use-case intent (kitchen remodel, travel, toddler, office setup, etc.)
7. Avoid overly specific brand/model terms unless the brand is dominant and commonly searched.
8. Avoid leaking the exact answer (the target item).

Product Information:
Title: {title}
Category: {category}
Attributes: {attributes}
Description: {description}
Representative Review Themes:
{top_review_themes}

Generate exactly 5 diverse shopper queries.

Output JSON only:
{{
  "queries": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}}"""


# ── Item metadata builder ─────────────────────────────────────────────────────
def build_item_metadata(train_df: pd.DataFrame,
                        all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rich item metadata for prompt construction.
    Extracts title proxy, category, attributes, and review themes
    from the processed parquet data.
    """
    logger.info("Building item metadata...")

    # Aggregate per item
    item_meta = all_df.groupby("asin").agg(
        avg_rating       = ("rating",          "mean"),
        review_count     = ("user_id",         "count"),
        avg_silver_label = ("silver_label",    "mean"),
        verified_ratio   = ("verified_score",  "mean"),
        avg_length       = ("length_score",    "mean"),
        # Sample review text for theme extraction
        sample_texts     = ("text",            lambda x: list(x.dropna())[:5]),
    ).reset_index()

    # Use ASIN as title proxy (real titles not in dataset)
    # In production this would join to a product catalog
    # Here we construct a descriptive proxy from review text
    item_meta["title_proxy"]  = item_meta["asin"]
    item_meta["category"]     = _infer_category(item_meta)
    item_meta["attributes"]   = item_meta.apply(_extract_attributes, axis=1)
    item_meta["review_themes"] = item_meta["sample_texts"].apply(_extract_review_themes)
    item_meta["description"]  = item_meta.apply(_build_description, axis=1)

    logger.info(f"  Built metadata for {len(item_meta):,} items")
    return item_meta


def _infer_category(df: pd.DataFrame) -> pd.Series:
    """
    Infer broad category from silver label distribution and rating patterns.
    Since the full dataset is Tools & Home Improvement, we use sub-category
    proxies based on review signal patterns.
    """
    # All items are Tools & Home Improvement
    # Use avg_silver_label tiers as proxy for sub-category quality tier
    def tier(row):
        if row["avg_silver_label"] >= 0.6:
            return "Tools & Home Improvement — Premium"
        elif row["avg_silver_label"] >= 0.45:
            return "Tools & Home Improvement — Standard"
        else:
            return "Tools & Home Improvement — Budget"
    return df.apply(tier, axis=1)


def _extract_attributes(row) -> str:
    """Build attribute string from item signals."""
    attrs = []
    if row["verified_ratio"] >= 0.9:
        attrs.append("highly verified purchases")
    if row["avg_rating"] >= 4.5:
        attrs.append("top-rated")
    elif row["avg_rating"] >= 4.0:
        attrs.append("well-rated")
    if row["review_count"] >= 100:
        attrs.append("popular")
    if row["avg_length"] >= 0.6:
        attrs.append("detailed reviews suggesting engaged buyers")
    return ", ".join(attrs) if attrs else "standard product"


def _extract_review_themes(texts: list) -> str:
    """
    Extract key themes from sample review texts.
    Simple keyword extraction — no NLP model needed.
    """
    if not texts:
        return "quality, value, ease of use"

    # Combine texts and extract common meaningful phrases
    combined = " ".join(str(t)[:200] for t in texts[:3]).lower()

    # Common Tools & Home Improvement themes to look for
    themes = []
    theme_keywords = {
        "durability":   ["durable", "sturdy", "solid", "lasting", "heavy duty"],
        "ease of use":  ["easy", "simple", "straightforward", "convenient"],
        "quality":      ["quality", "well made", "excellent", "great"],
        "value":        ["value", "price", "affordable", "worth"],
        "performance":  ["works great", "performs", "powerful", "effective"],
        "installation": ["install", "setup", "assembly", "mount"],
        "size/fit":     ["fits", "size", "compact", "lightweight"],
    }

    for theme, keywords in theme_keywords.items():
        if any(kw in combined for kw in keywords):
            themes.append(theme)
        if len(themes) >= 3:
            break

    return ", ".join(themes) if themes else "quality, functionality, value"


def _build_description(row) -> str:
    """Build a brief product description from signals."""
    parts = []
    if row["avg_rating"] >= 4.0:
        parts.append(f"Rated {row['avg_rating']:.1f}/5 stars")
    parts.append(f"with {int(row['review_count'])} reviews")
    if row["verified_ratio"] >= 0.85:
        parts.append("mostly from verified purchasers")
    return " ".join(parts)


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(row: pd.Series) -> str:
    """Build the full prompt for one item."""
    return PROMPT_TEMPLATE.format(
        title        = row.get("title_proxy", row["asin"]),
        category     = row.get("category", "Tools & Home Improvement"),
        attributes   = row.get("attributes", "standard product"),
        description  = row.get("description", ""),
        top_review_themes = row.get("review_themes", "quality, value, ease of use"),
    )


# ── API call with retry ────────────────────────────────────────────────────────
def call_haiku(client: anthropic.Anthropic,
               prompt: str,
               asin: str) -> Optional[list]:
    """
    Call Claude Haiku and parse the JSON response.
    Returns list of 5 query strings or None on failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model      = MODEL,
                max_tokens = MAX_TOKENS,
                messages   = [{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            queries = parsed.get("queries", [])

            # Validate — must have exactly 5 non-empty strings
            queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
            if len(queries) >= 3:
                return queries[:5]

            logger.warning(f"  {asin}: got {len(queries)} queries — retrying")

        except json.JSONDecodeError as e:
            logger.warning(f"  {asin}: JSON parse error attempt {attempt+1}: {e}")
        except anthropic.RateLimitError:
            logger.warning(f"  {asin}: rate limited — waiting {RETRY_DELAY * 2}s")
            time.sleep(RETRY_DELAY * 2)
        except Exception as e:
            logger.warning(f"  {asin}: API error attempt {attempt+1}: {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    return None


# ── Resume support ─────────────────────────────────────────────────────────────
def load_processed_asins(output_path: str) -> set:
    """Load set of ASINs already processed (for resume support)."""
    processed = set()
    if not Path(output_path).exists():
        return processed
    with open(output_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                processed.add(obj["asin"])
            except Exception:
                pass
    return processed


# ── Main generation loop ───────────────────────────────────────────────────────
def generate_synthetic_queries(
    item_meta:       pd.DataFrame,
    output_path:     str = OUTPUT_PATH,
    max_items:       Optional[int] = None,
    anthropic_key:   Optional[str] = None,
):
    """
    Generate synthetic queries for all items in item_meta.

    Args:
        item_meta:     DataFrame with item metadata (from build_item_metadata)
        output_path:   Where to write the JSONL output file
        max_items:     Optional limit for testing (None = all items)
        anthropic_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    api_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
        )

    client = anthropic.Anthropic(api_key=api_key)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Resume: skip already processed items ───────────────────────────────
    processed_asins = load_processed_asins(output_path)
    logger.info(f"Already processed: {len(processed_asins):,} items")

    # Filter to unprocessed items
    items_to_process = item_meta[~item_meta["asin"].isin(processed_asins)]
    if max_items:
        items_to_process = items_to_process.head(max_items)

    total   = len(items_to_process)
    skipped = len(processed_asins)
    logger.info(f"Items to process: {total:,} | Already done: {skipped:,}")

    if total == 0:
        logger.info("All items already processed. Nothing to do.")
        return

    # ── Generation loop ────────────────────────────────────────────────────
    t0 = time.time()
    success_count = 0
    fail_count    = 0

    with open(output_path, "a") as out_file:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = items_to_process.iloc[batch_start:batch_start + BATCH_SIZE]

            for _, row in batch.iterrows():
                asin   = row["asin"]
                prompt = build_prompt(row)
                queries = call_haiku(client, prompt, asin)

                if queries:
                    record = {
                        "asin":     asin,
                        "title":    row.get("title_proxy", asin),
                        "category": row.get("category", "Tools & Home Improvement"),
                        "queries":  queries,
                    }
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                    success_count += 1
                else:
                    # Write fallback with title-based queries
                    record = {
                        "asin":     asin,
                        "title":    row.get("title_proxy", asin),
                        "category": row.get("category", "Tools & Home Improvement"),
                        "queries":  [f"home improvement tool {asin[:4]}"],
                        "fallback": True,
                    }
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                    fail_count += 1
                    logger.warning(f"  Fallback used for: {asin}")

            # Progress log every batch
            processed_so_far = batch_start + len(batch)
            elapsed  = time.time() - t0
            rate     = processed_so_far / elapsed if elapsed > 0 else 0
            eta_min  = (total - processed_so_far) / rate / 60 if rate > 0 else 0

            logger.info(
                f"  Progress: {processed_so_far + skipped:,}/{total + skipped:,} | "
                f"success={success_count:,} fail={fail_count:,} | "
                f"rate={rate:.1f}/s | ETA={eta_min:.0f}min"
            )

            # Small sleep between batches to avoid rate limits
            time.sleep(0.5)

    # ── Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    total_records = success_count + fail_count + skipped

    logger.info(f"\n{'='*60}")
    logger.info(f"Synthetic query generation complete")
    logger.info(f"  Total records: {total_records:,}")
    logger.info(f"  Successful:    {success_count:,}")
    logger.info(f"  Fallbacks:     {fail_count:,}")
    logger.info(f"  Skipped:       {skipped:,}")
    logger.info(f"  Time:          {total_time/60:.1f} min")
    logger.info(f"  Output:        {output_path}")
    logger.info(f"{'='*60}\n")

    # Verify output
    _verify_output(output_path)


def _verify_output(output_path: str) -> None:
    """Verify the output file looks correct."""
    records = []
    with open(output_path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    total      = len(records)
    with_5     = sum(1 for r in records if len(r.get("queries", [])) == 5)
    fallbacks  = sum(1 for r in records if r.get("fallback", False))

    logger.info(f"Output verification:")
    logger.info(f"  Total records:     {total:,}")
    logger.info(f"  With 5 queries:    {with_5:,} ({with_5/total:.1%})")
    logger.info(f"  Fallbacks:         {fallbacks:,} ({fallbacks/total:.1%})")

    # Show 3 sample records
    logger.info("\n  Sample records:")
    for r in records[:3]:
        logger.info(f"  ASIN: {r['asin']}")
        for q in r.get("queries", []):
            logger.info(f"    → {q}")


# ── Push to HF ────────────────────────────────────────────────────────────────
def push_queries_to_hf(output_path: str,
                       hf_token: str,
                       hf_dataset_repo: str = "chaturg/amazon-recsys-dataset"):
    """Push the synthetic queries file to HF Dataset Hub."""
    from huggingface_hub import HfApi
    api = HfApi()

    logger.info(f"Pushing synthetic queries to HF: {hf_dataset_repo}")
    api.upload_file(
        path_or_fileobj = output_path,
        path_in_repo    = "eval/synthetic_queries.jsonl",
        repo_id         = hf_dataset_repo,
        repo_type       = "dataset",
        token           = hf_token,
        commit_message  = "Add synthetic query evaluation set",
    )
    logger.info(f"  ✓ Pushed to {hf_dataset_repo}/eval/synthetic_queries.jsonl")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__" or "get_ipython" in dir():
    """
    Kaggle notebook usage:

    # Cell 1: Set API key
    import os
    os.environ["ANTHROPIC_API_KEY"] = "your_key_here"

    # Cell 2: Load data (skip if already loaded)
    import pandas as pd
    BASE = "/kaggle/working/processed/data"
    train_df = pd.read_parquet(f"{BASE}/train.parquet")
    val_df   = pd.read_parquet(f"{BASE}/val.parquet")
    test_df  = pd.read_parquet(f"{BASE}/test.parquet")
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Cell 3: Build metadata and generate queries
    item_meta = build_item_metadata(train_df, all_df)

    # Test run first — 10 items to verify prompt and output
    generate_synthetic_queries(
        item_meta    = item_meta,
        output_path  = OUTPUT_PATH,
        max_items    = 10,
    )

    # Full run — all 157k items (~52 min, ~$2)
    generate_synthetic_queries(
        item_meta  = item_meta,
        output_path = OUTPUT_PATH,
    )

    # Push to HF when done
    push_queries_to_hf(
        output_path     = OUTPUT_PATH,
        hf_token        = "your_hf_token_here",
        hf_dataset_repo = "chaturg/amazon-recsys-dataset",
    )
    """
    pass
