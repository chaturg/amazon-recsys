# ============================================================
# Amazon RecSys — Cross-Encoder Fine-Tuning (Kaggle)
# Base: cross-encoder/ms-marco-MiniLM-L-6-v2
# Loss: pointwise cross-entropy against silver labels
# ============================================================

# ── CELL 1: Install dependencies ─────────────────────────────────────────────
# !pip install transformers huggingface-hub -q

# ── CELL 2: Imports and setup ─────────────────────────────────────────────────
import os
import gc
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")   # set via os.environ["HF_TOKEN"] = "..."
HF_MODEL_REPO = "chaturg/amazon-recsys-cross-encoder"
HF_DATA_REPO  = "chaturg/amazon-recsys-dataset"
OUT           = "/kaggle/working"
MAX_SEQ_LEN   = 128

print(f"Device: {DEVICE}")
print(f"GPU:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU MB: {torch.cuda.get_device_properties(0).total_memory/1e6:.0f} MB")


# ── CELL 3: Pull training pairs from HF ──────────────────────────────────────
print("Downloading cross-encoder training pairs from HF...")
pairs_path = hf_hub_download(
    repo_id   = HF_DATA_REPO,
    filename  = "processed/ce_training_pairs.parquet",
    repo_type = "dataset",
    token     = HF_TOKEN,
    local_dir = OUT,
)
print(f"Downloaded: {pairs_path}")

df = pd.read_parquet(pairs_path)
print(f"\nTraining pairs: {len(df):,}")
print(f"Positives:      {df['is_positive'].sum():,} ({df['is_positive'].mean():.1%})")
print(f"Negatives:      {(~df['is_positive']).sum():,}")
print(f"\nSample pair:")
row = df[df["is_positive"]].iloc[0]
print(f"  History: {row['history_summary'][:80]}")
print(f"  Item:    {row['item_title'][:80]}")
print(f"  Label:   {row['relevance_label']:.3f}")


# ── CELL 4: Dataset ───────────────────────────────────────────────────────────
class CrossEncoderDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_SEQ_LEN):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        text_a = str(row["history_summary"])[:256]
        text_b = f"{row['item_title']} {row['item_category']}"[:128]

        enc = self.tokenizer(
            text_a, text_b,
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(float(row["relevance_label"]), dtype=torch.float),
        }


# ── CELL 5: Push helpers ──────────────────────────────────────────────────────
def push_model_to_hf(model_dir: str, commit_msg: str = "Upload cross-encoder") -> None:
    """Push fine-tuned model to HF Model Hub."""
    api = HfApi()
    print(f"\nPushing {model_dir} to HF...")
    api.upload_folder(
        folder_path    = model_dir,
        repo_id        = HF_MODEL_REPO,
        repo_type      = "model",
        token          = HF_TOKEN,
        commit_message = commit_msg,
        ignore_patterns = ["*.log", "__pycache__"],
    )
    print(f"  ✓ Pushed to {HF_MODEL_REPO}")


# ── CELL 6: Fine-tuning function ──────────────────────────────────────────────
def finetune_cross_encoder(
    df:            pd.DataFrame,
    output_dir:    str   = f"{OUT}/cross_encoder",
    num_epochs:    int   = 3,
    batch_size:    int   = 32,
    learning_rate: float = 2e-5,
    val_split:     float = 0.05,
    max_pairs:     int   = None,
) -> str:
    """
    Fine-tune cross-encoder and push to HF after each epoch.

    Returns path to best model directory.
    """
    print(f"\n{'='*60}")
    print(f"Cross-Encoder Fine-Tuning")
    print(f"  Base:    {BASE_MODEL}")
    print(f"  Epochs:  {num_epochs}")
    print(f"  Batch:   {batch_size}")
    print(f"  LR:      {learning_rate}")
    print(f"{'='*60}")
    t0 = time.time()

    # Optionally cap pairs for testing
    if max_pairs:
        df = df.sample(min(max_pairs, len(df)), random_state=42)
        print(f"  Using {len(df):,} pairs (capped)")

    # Train/val split
    val_size = int(len(df) * val_split)
    val_df   = df.sample(val_size, random_state=42)
    train_df = df.drop(val_df.index)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    # Load tokenizer and model
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=1
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    # Datasets and loaders
    train_ds = CrossEncoderDataset(train_df, tokenizer)
    val_ds   = CrossEncoderDataset(val_df,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn   = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_dir      = f"{output_dir}/best"
    Path(best_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        ep_start = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tl, tn = 0.0, 0
        for bidx, batch in enumerate(train_loader):
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            lbls  = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            out   = model(input_ids=ids, attention_mask=mask)
            logits = out.logits.squeeze(-1)
            loss  = loss_fn(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tl += loss.item(); tn += 1

            if bidx % 100 == 0:
                gpu_mb = torch.cuda.memory_allocated()/1e6
                print(f"  Ep{epoch+1} batch {bidx}/{len(train_loader)} | "
                      f"loss={loss.item():.4f} | GPU={gpu_mb:.0f}MB")

        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        vl, vn = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids   = batch["input_ids"].to(DEVICE)
                mask  = batch["attention_mask"].to(DEVICE)
                lbls  = batch["label"].to(DEVICE)
                out   = model(input_ids=ids, attention_mask=mask)
                logits = out.logits.squeeze(-1)
                loss  = loss_fn(logits, lbls)
                vl += loss.item(); vn += 1
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())

        avg_val = vl / vn
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  ✓ Best model saved (val={avg_val:.4f})")

        # Correlation between predicted scores and silver labels
        corr = np.corrcoef(all_preds, all_labels)[0, 1]
        ep_time = time.time() - ep_start

        print(f"Epoch {epoch+1}/{num_epochs} — "
              f"train={tl/tn:.4f} val={avg_val:.4f} "
              f"label_corr={corr:.3f} time={ep_time:.0f}s"
              + (" ← BEST" if is_best else ""))

        # Push after each epoch — crash-proof
        push_model_to_hf(best_dir, commit_msg=f"Epoch {epoch+1} — val={avg_val:.4f}")

    # Save final model
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Fine-tuning complete in {elapsed/60:.1f} min")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Best model:    {best_dir}")
    print(f"{'='*60}\n")

    return best_dir


# ── CELL 7: Quick inference test ──────────────────────────────────────────────
def test_inference(model_dir: str) -> None:
    """Verify the fine-tuned model can score candidate pairs."""
    print("\nRunning inference test...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=1
    ).to(DEVICE)
    model.eval()

    test_pairs = [
        # (history, item) — expect high score
        ("bought dewalt drill, stanley screwdrivers, irwin drill bits",
         "DEWALT 20V MAX Cordless Drill Combo Kit Power Tools"),
        # (history, item) — expect lower score
        ("bought dewalt drill, stanley screwdrivers, irwin drill bits",
         "Garden Hose 50ft Expandable Water Hose Garden & Outdoor"),
        # (history, item) — similar but different brand
        ("bought dewalt drill, stanley screwdrivers, irwin drill bits",
         "Makita 18V LXT Brushless Drill Driver Kit Power Tools"),
    ]

    histories = [p[0] for p in test_pairs]
    items     = [p[1] for p in test_pairs]

    enc = tokenizer(histories, items, max_length=MAX_SEQ_LEN,
                    padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        out    = model(input_ids=enc["input_ids"].to(DEVICE),
                       attention_mask=enc["attention_mask"].to(DEVICE))
        scores = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()

    print("\nInference test results:")
    for (h, item), score in zip(test_pairs, scores):
        print(f"  Score={score:.3f} | Item: {item[:60]}")
    print()

    # Sanity check: DeWalt drill should score higher than garden hose
    assert scores[0] > scores[1], \
        "WARNING: DeWalt drill scored lower than garden hose — model may not be learning"
    print("  ✓ Sanity check passed: relevant item scores higher than irrelevant item")


# ── CELL 8: Run everything ────────────────────────────────────────────────────
# Set HF token first if not already set:
# import os; os.environ["HF_TOKEN"] = "your_token_here"

# Fine-tune
best_model_dir = finetune_cross_encoder(
    df         = df,
    output_dir = f"{OUT}/cross_encoder",
    num_epochs = 3,
    batch_size = 32,
)

# Test inference
test_inference(best_model_dir)

# Final push with model card
api = HfApi()
model_card = """---
license: apache-2.0
base_model: cross-encoder/ms-marco-MiniLM-L-6-v2
tags:
- cross-encoder
- reranking
- recommendation-system
- amazon-reviews
language:
- en
---

# Amazon RecSys Cross-Encoder Re-ranker

Fine-tuned on Amazon Tools & Home Improvement reviews for two-stage recommendation.

## Input
`[CLS] user_history_summary [SEP] item_title item_category [SEP]`

## Loss
Pointwise cross-entropy against silver labels (not pairwise hinge).
Silver labels are continuous [0,1] scores combining verified purchase,
review length, helpfulness, and per-user z-scored ratings.

## Dataset
[chaturg/amazon-recsys-dataset](https://huggingface.co/datasets/chaturg/amazon-recsys-dataset)
"""
api.upload_file(
    path_or_fileobj = model_card.encode("utf-8"),
    path_in_repo    = "README.md",
    repo_id         = HF_MODEL_REPO,
    repo_type       = "model",
    token           = HF_TOKEN,
    commit_message  = "Add model card",
)

print(f"\nAll done. Model at: https://huggingface.co/{HF_MODEL_REPO}")
