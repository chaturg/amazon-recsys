"""
run_synthetic_queries.py
------------------------
Generates 5 synthetic shopper queries per item using Claude Haiku.
Uses real product titles where available (53.4% coverage),
falls back to review text proxy for remaining items.

Usage:
    # Set API key first
    export ANTHROPIC_API_KEY="your_key_here"

    # Test run — 10 items
    python scripts/run_synthetic_queries.py --max_items 10

    # Full run — all 157k items (~52 min, ~$2)
    nohup python scripts/run_synthetic_queries.py \
        > eval/synthetic_queries_run.log 2>&1 &

    # Resume after interruption (automatically skips done items)
    python scripts/run_synthetic_queries.py

Output:
    eval/synthetic_queries.jsonl
    One JSON object per line:
    {
      "asin": "B000LNS094",
      "title": "Baldwin Estate 4754.030 Square Beveled...",
      "category": "Tools & Home Improvement > Electrical",
      "queries": ["decorative wall plate for GFCI outlet", ...],
      "has_real_title": true
    }
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import pandas as pd
import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = "processed"
OUT_PATH = "eval/synthetic_queries.jsonl"
MODEL    = "claude-haiku-4-5"

# ── Prompt ────────────────────────────────────────────────────────────────────
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
{review_themes}

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


# ── Feature helpers ───────────────────────────────────────────────────────────
def extract_review_themes(texts: list) -> str:
    """Extract key themes from sample review texts."""
    if not texts:
        return "quality, value, ease of use"
    combined = " ".join(str(t)[:200] for t in texts[:3]).lower()
    themes = []
    theme_map = {
        "durability":   ["durable", "sturdy", "solid", "lasting", "heavy duty"],
        "ease of use":  ["easy", "simple", "straightforward", "convenient"],
        "quality":      ["quality", "well made", "excellent", "great"],
        "value":        ["value", "price", "affordable", "worth"],
        "performance":  ["works great", "performs", "powerful", "effective"],
        "installation": ["install", "setup", "assembly", "mount"],
        "size/fit":     ["fits", "size", "compact", "lightweight"],
    }
    for theme, keywords in theme_map.items():
        if any(kw in combined for kw in keywords):
            themes.append(theme)
        if len(themes) >= 3:
            break
    return ", ".join(themes) if themes else "quality, functionality, value"


def review_title_proxy(texts: list) -> str:
    """Extract title proxy from review text for items without real titles."""
    for t in texts:
        t = t.strip()
        if len(t) > 20:
            first = t.split(".")[0].strip()
            if len(first) > 15:
                return first[:80]
    return "home improvement product"


def build_attributes(row: pd.Series) -> str:
    """Build attribute string from item signals."""
    attrs = []
    price = row.get("price", None)
    if isinstance(price, float) and price > 0:
        if price < 25:    attrs.append("budget-friendly")
        elif price < 75:  attrs.append("mid-range price")
        else:             attrs.append("premium price point")
    if row.get("verified_ratio", 0) >= 0.9:  attrs.append("highly verified purchases")
    if row.get("avg_rating", 0) >= 4.5:      attrs.append("top-rated")
    elif row.get("avg_rating", 0) >= 4.0:    attrs.append("well-rated")
    if row.get("review_count", 0) >= 100:    attrs.append("popular item")
    return ", ".join(attrs) if attrs else "standard product"


# ── Metadata builder ──────────────────────────────────────────────────────────
def build_item_metadata(
    train_df:  pd.DataFrame,
    all_df:    pd.DataFrame,
    titles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build rich item metadata for prompt construction.
    Joins real product titles where available (53.4% of items).
    Falls back to review text proxy for remaining items.
    """
    logger.info("Building item metadata...")

    # Aggregate review signals per item
    agg = all_df.groupby("asin").agg(
        avg_rating    = ("rating",         "mean"),
        review_count  = ("user_id",        "count"),
        verified_ratio= ("verified_score", "mean"),
        avg_length    = ("length_score",   "mean"),
        sample_texts  = ("text", lambda x: list(x.dropna().astype(str))[:5]),
    ).reset_index()

    # Join real product titles
    agg = agg.merge(
        titles_df[["asin", "title", "categories", "description", "price"]],
        on="asin", how="left",
    )

    # Resolve title — real title or review proxy
    agg["title_final"] = agg.apply(
        lambda r: r["title"]
        if isinstance(r.get("title"), str) and len(r["title"]) > 5
        else review_title_proxy(r["sample_texts"]),
        axis=1,
    )

    # Resolve category
    agg["category_final"] = agg["categories"].fillna("Tools & Home Improvement")

    # Build attributes and themes
    agg["attributes"]    = agg.apply(build_attributes, axis=1)
    agg["review_themes"] = agg["sample_texts"].apply(extract_review_themes)

    # Resolve description — real or signal-based
    agg["desc_final"] = agg.apply(
        lambda r: r["description"][:200]
        if isinstance(r.get("description"), str) and len(r["description"]) > 10
        else f"Rated {r['avg_rating']:.1f}/5 with {int(r['review_count'])} reviews",
        axis=1,
    )

    # Flag items with real titles
    agg["has_real_title"] = agg["title"].apply(
        lambda t: isinstance(t, str) and len(t) > 5
    )

    real = agg["has_real_title"].sum()
    logger.info(
        f"  Items: {len(agg):,} | "
        f"Real titles: {real:,} ({real/len(agg):.1%}) | "
        f"Review proxy: {len(agg)-real:,} ({(len(agg)-real)/len(agg):.1%})"
    )
    return agg


# ── API call ──────────────────────────────────────────────────────────────────
def call_haiku(
    client: anthropic.Anthropic,
    row:    pd.Series,
    retries: int = 3,
) -> list:
    """Call Claude Haiku and parse the JSON response. Returns list of queries or None."""
    prompt = PROMPT_TEMPLATE.format(
        title         = row["title_final"],
        category      = row["category_final"],
        attributes    = row["attributes"],
        description   = row["desc_final"],
        review_themes = row["review_themes"],
    )

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model      = MODEL,
                max_tokens = 300,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed  = json.loads(raw)
            queries = [
                q.strip() for q in parsed.get("queries", [])
                if isinstance(q, str) and q.strip()
            ]
            if len(queries) >= 3:
                return queries[:5]

            logger.warning(f"  Only {len(queries)} queries returned — retrying")

        except anthropic.RateLimitError:
            wait = 5 * (attempt + 1)
            logger.warning(f"  Rate limited — waiting {wait}s")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            logger.warning(f"  JSON parse error attempt {attempt+1}: {e}")
            time.sleep(2)
        except Exception as e:
            logger.warning(f"  API error attempt {attempt+1}: {e}")
            time.sleep(2)

    return None


# ── Resume support ─────────────────────────────────────────────────────────────
def load_done_asins(path: str) -> set:
    """Load set of already-processed ASINs for resume support."""
    done = set()
    if not Path(path).exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line.strip())["asin"])
            except Exception:
                pass
    logger.info(f"  Resuming: {len(done):,} items already processed")
    return done


# ── Push to HF ────────────────────────────────────────────────────────────────
def push_to_hf(output_path: str, hf_token: str,
               repo_id: str = "chaturg/amazon-recsys-dataset") -> None:
    """Push synthetic_queries.jsonl to HF Dataset Hub."""
    from huggingface_hub import HfApi
    api = HfApi()
    logger.info(f"Pushing to HF: {repo_id}...")
    api.upload_file(
        path_or_fileobj = output_path,
        path_in_repo    = "eval/synthetic_queries.jsonl",
        repo_id         = repo_id,
        repo_type       = "dataset",
        token           = hf_token,
        commit_message  = "Add synthetic query evaluation set",
    )
    logger.info(f"  ✓ Pushed to {repo_id}/eval/synthetic_queries.jsonl")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(max_items: int = None) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY='your_key'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Load data
    logger.info("Loading data...")
    train_df  = pd.read_parquet(f"{BASE}/train.parquet")
    val_df    = pd.read_parquet(f"{BASE}/val.parquet")
    test_df   = pd.read_parquet(f"{BASE}/test.parquet")
    titles_df = pd.read_parquet(f"{BASE}/item_titles.parquet")
    all_df    = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"  Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}")

    # Build item metadata
    item_meta = build_item_metadata(train_df, all_df, titles_df)
    if max_items:
        item_meta = item_meta.head(max_items)
        logger.info(f"  Limited to {max_items} items for testing")

    # Resume support
    done_asins = load_done_asins(OUT_PATH)
    todo       = item_meta[~item_meta["asin"].isin(done_asins)]
    total_todo = len(todo)
    logger.info(f"  To process: {total_todo:,} | Already done: {len(done_asins):,}")

    if total_todo == 0:
        logger.info("All items already processed.")
        _show_samples(OUT_PATH)
        return

    # Generation loop
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    t0        = time.time()
    success   = 0
    fallbacks = 0

    with open(OUT_PATH, "a") as out_file:
        for i, (_, row) in enumerate(todo.iterrows()):
            queries = call_haiku(client, row)

            record = {
                "asin":          row["asin"],
                "title":         row["title_final"],
                "category":      row["category_final"],
                "queries":       queries if queries else ["home improvement product"],
                "has_real_title":bool(row["has_real_title"]),
            }
            if not queries:
                record["fallback"] = True
                fallbacks += 1
            else:
                success += 1

            out_file.write(json.dumps(record) + "\n")
            out_file.flush()

            # Progress every 20 items
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed if elapsed > 0 else 1
                eta_min = (total_todo - i - 1) / rate / 60
                total_done = len(done_asins) + i + 1
                logger.info(
                    f"  {total_done:,}/{len(item_meta)+len(done_asins):,} | "
                    f"success={success:,} fallback={fallbacks:,} | "
                    f"rate={rate:.1f}/s ETA={eta_min:.0f}min"
                )

            # Gentle rate limiting — 10 req/sec
            time.sleep(0.1)

    # Summary
    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation complete")
    logger.info(f"  Success:   {success:,}")
    logger.info(f"  Fallbacks: {fallbacks:,}")
    logger.info(f"  Time:      {elapsed/60:.1f} min")
    logger.info(f"  Output:    {OUT_PATH}")
    logger.info(f"{'='*60}")
    _show_samples(OUT_PATH)


def _show_samples(path: str) -> None:
    """Print 3 sample records with real titles."""
    logger.info("\nSample records (real titles):")
    count = 0
    try:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("has_real_title") and not r.get("fallback"):
                    logger.info(f"  Title: {r['title'][:70]}")
                    for q in r["queries"]:
                        logger.info(f"    → {q}")
                    logger.info("")
                    count += 1
                    if count >= 3:
                        break
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic shopper queries for RecSys eval"
    )
    parser.add_argument(
        "--max_items", type=int, default=None,
        help="Limit to N items for testing (omit for full 157k run)"
    )
    parser.add_argument(
        "--push_to_hf", action="store_true",
        help="Push output to HF Dataset Hub after generation"
    )
    parser.add_argument(
        "--hf_token", type=str, default="",
        help="HF write token (required if --push_to_hf)"
    )
    args = parser.parse_args()

    run(max_items=args.max_items)

    if args.push_to_hf:
        if not args.hf_token:
            logger.error("--hf_token required when using --push_to_hf")
            sys.exit(1)
        push_to_hf(OUT_PATH, args.hf_token)
