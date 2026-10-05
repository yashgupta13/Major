"""
01_encode_impressions.py — Phase 1: Text Encoder
=================================================
Encodes section_impression (fallback: section_findings) for all TRAIN records
using a pretrained medical language model (ClinicalBERT / BioLinkBERT).

Reads from:
    ../df_chexpert_plus_240401.csv   (main table)
    ../impression.csv                (14 pathology labels per image)
    ../report.csv                    (14 pathology labels per image — full report)

Writes to:
    outputs/text_embeddings_train.npy    (N × 768 float32 array)
    outputs/embedding_index_map.csv      (N rows — maps row index → study metadata)

Run:
    cd PersonB/phase1_text_encoder
    python 01_encode_impressions.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

from utils.config import (
    MAIN_CSV, IMPRESSION_CSV, REPORT_CSV,
    TEXT_EMBEDDINGS_PATH, EMBEDDING_INDEX_MAP,
    TEXT_ENCODER_MODEL, TEXT_ENCODER_MAX_LENGTH, TEXT_ENCODER_BATCH_SIZE,
    TRAIN_SPLIT, PATHOLOGY_LABELS,
)
from utils.text_utils import add_text_target_column, print_coverage_report, estimate_token_lengths


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("\n[1/5] Loading CSVs...")

    print(f"  Reading main table: {MAIN_CSV}")
    df = pd.read_csv(MAIN_CSV, low_memory=False)
    print(f"  Total rows: {len(df):,} | Columns: {len(df.columns)}")

    print(f"  Reading impression labels: {IMPRESSION_CSV}")
    imp_labels = pd.read_csv(IMPRESSION_CSV)

    print(f"  Reading report labels: {REPORT_CSV}")
    rep_labels = pd.read_csv(REPORT_CSV)

    # Filter to train split only — never encode val/test into the retrieval index
    df_train = df[df["split"] == TRAIN_SPLIT].copy()
    print(f"  Train records: {len(df_train):,} (split='{TRAIN_SPLIT}')")

    # Merge impression pathology labels (primary) onto train set
    df_train = df_train.merge(
        imp_labels.add_prefix("imp_"),
        left_on="path_to_image",
        right_on="imp_path_to_image",
        how="left"
    ).drop(columns=["imp_path_to_image"], errors="ignore")

    # Merge report pathology labels (recall) — rename to avoid collision
    df_train = df_train.merge(
        rep_labels.add_prefix("rep_"),
        left_on="path_to_image",
        right_on="rep_path_to_image",
        how="left"
    ).drop(columns=["rep_path_to_image"], errors="ignore")

    print(f"  After label merge: {len(df_train):,} rows, {len(df_train.columns)} columns")
    return df_train


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Text preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_texts(df_train: pd.DataFrame):
    print("\n[2/5] Preparing text targets...")

    df_train = add_text_target_column(df_train)
    print_coverage_report(df_train)

    # Remove records with no usable text
    df_valid = df_train[df_train["text_target"].notna()].copy().reset_index(drop=True)
    dropped = len(df_train) - len(df_valid)
    print(f"  Records with usable text: {len(df_valid):,} (dropped {dropped:,} with no text)")

    # Quick token length check
    stats = estimate_token_lengths(df_valid["text_target"].tolist())
    print(f"  Token length estimate — mean: {stats['mean']} | median: {stats['median']} "
          f"| p99: {stats['p99']} | max: {stats['max']} | >400 words: {stats['pct_over_400w']}%")

    # Extract study_id from path: train/patientXXXXX/studyYYYYY/...
    def extract_ids(path):
        parts = str(path).replace("\\", "/").split("/")
        patient = next((p for p in parts if p.startswith("patient")), None)
        study   = next((p for p in parts if p.startswith("study")),   None)
        return patient, study

    df_valid[["parsed_patient_id", "parsed_study_id"]] = df_valid["path_to_image"].apply(
        lambda p: pd.Series(extract_ids(p))
    )

    return df_valid


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load encoder
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder():
    print(f"\n[3/5] Loading encoder: {TEXT_ENCODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_MODEL)
    model = AutoModel.from_pretrained(TEXT_ENCODER_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  Running on: {device.upper()}")
    return tokenizer, model, device


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Batch encode
# ─────────────────────────────────────────────────────────────────────────────

def batch_encode(texts: list, tokenizer, model, device: str) -> np.ndarray:
    """
    Encode a list of strings using mean-pooling over token embeddings.
    Returns float32 array of shape (N, hidden_size).
    """
    all_embeddings = []
    batch_size = TEXT_ENCODER_BATCH_SIZE
    
    # Use bfloat16 autocast if supported, otherwise float16 (ignored if CPU doesn't support it well, but helps if it does)
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding batches"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=TEXT_ENCODER_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type == 'cpu' else torch.float16):
            output = model(**encoded)

        # Mean pool over token dimension (excluding padding)
        attention_mask = encoded["attention_mask"]
        token_embeddings = output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        all_embeddings.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df_valid: pd.DataFrame, embeddings: np.ndarray):
    print(f"\n[5/5] Saving outputs...")

    # Save embeddings
    np.save(str(TEXT_EMBEDDINGS_PATH), embeddings)
    print(f"  Saved embeddings → {TEXT_EMBEDDINGS_PATH}  shape={embeddings.shape}")

    # Build and save the index map (row index ↔ study metadata)
    imp_label_cols = [f"imp_{l}" for l in PATHOLOGY_LABELS]
    rep_label_cols = [f"rep_{l}" for l in PATHOLOGY_LABELS]

    keep_cols = [
        "path_to_image", "deid_patient_id", "patient_report_date_order",
        "frontal_lateral", "ap_pa", "split",
        "text_target", "text_source",
        "parsed_patient_id", "parsed_study_id",
    ] + [c for c in imp_label_cols if c in df_valid.columns] \
      + [c for c in rep_label_cols if c in df_valid.columns]

    index_map = df_valid[[c for c in keep_cols if c in df_valid.columns]].copy()
    index_map.to_csv(str(EMBEDDING_INDEX_MAP), index=False)
    print(f"  Saved index map  → {EMBEDDING_INDEX_MAP}  rows={len(index_map):,}")

    # Sanity check
    assert len(embeddings) == len(index_map), (
        f"Embedding count {len(embeddings)} != index map rows {len(index_map)}"
    )
    print("\n  ✅ Phase 1 complete. Ready for Phase 2 (FAISS indexing).")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Encode impressions.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to encode for testing.")
    args = parser.parse_args()

    print("=" * 60)
    print("  PHASE 1 — Text Encoder")
    print(f"  Model: {TEXT_ENCODER_MODEL}")
    if args.limit:
        print(f"  LIMIT: {args.limit} records")
    print("=" * 60)

    df_train  = load_data()
    df_valid  = prepare_texts(df_train)
    
    if args.limit:
        df_valid = df_valid.head(args.limit)

    tokenizer, model, device = load_encoder()

    print(f"\n[4/5] Encoding {len(df_valid):,} texts...")
    embeddings = batch_encode(df_valid["text_target"].tolist(), tokenizer, model, device)

    save_outputs(df_valid, embeddings)
