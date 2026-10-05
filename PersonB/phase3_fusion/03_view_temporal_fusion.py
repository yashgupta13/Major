"""
03_view_temporal_fusion.py — Phase 3: Spatial & Temporal Fusion
================================================================
Handles two types of context fusion (coordinated with Person A):

  A) SPATIAL FUSION — merge per-view image embeddings from the same study
     (frontal + lateral → single study embedding)

  B) TEMPORAL FUSION — retrieve prior study impressions for the same patient
     to provide longitudinal context in the RAG prompt

This script does NOT modify the FAISS index.  The fused embeddings are used
as QUERY embeddings at inference time (Pipeline 2). If Person A provides
pre-fused embeddings, you can skip the spatial fusion functions here.

Reads from:
    outputs/faiss_metadata.csv               (from Phase 2)
    outputs/embedding_index_map.csv          (from Phase 1)
    PersonA/outputs/image_embeddings.npy     (from Person A — coordinate with them)

Writes to:
    outputs/temporal_context_cache.csv       (pre-computed prior impressions)

Run:
    cd PersonB/phase3_fusion
    python 03_view_temporal_fusion.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from typing import Optional

from utils.config import (
    FAISS_METADATA_PATH, EMBEDDING_INDEX_MAP, OUTPUTS_DIR,
    MAIN_CSV,
)


# ─────────────────────────────────────────────────────────────────────────────
# PART A — SPATIAL FUSION (Multi-View)
# ─────────────────────────────────────────────────────────────────────────────

def fuse_views(view_embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Merge multiple per-view image embeddings from the same study into a
    single study-level embedding.

    Strategy:
        - 1 view  → return as-is (81.9% of studies — most common case)
        - 2 views → average pooling
        - 3 views → average pooling (rare: 0.95% of studies)

    This is the SIMPLE approach. For better performance, use attention-weighted
    fusion after contrastive training (coordinate with Person A).

    Args:
        view_embeddings: list of 1-D np.arrays, each shape (dim,)

    Returns:
        Single embedding of shape (dim,)
    """
    if len(view_embeddings) == 0:
        raise ValueError("view_embeddings list is empty — cannot fuse.")

    if len(view_embeddings) == 1:
        return view_embeddings[0].copy()

    # Average pooling over all views
    fused = np.mean(np.stack(view_embeddings, axis=0), axis=0)
    return fused


def fuse_study_embeddings(
    per_image_embeddings: np.ndarray,
    image_to_study_map: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Apply fuse_views() to group all per-image embeddings by study_id.

    Args:
        per_image_embeddings : (N_images, dim) array — one embedding per image
        image_to_study_map   : DataFrame with columns [path_to_image, parsed_study_id]
                               in the same row order as per_image_embeddings

    Returns:
        (study_embeddings, study_meta):
            study_embeddings : (N_studies, dim) array — one embedding per study
            study_meta       : DataFrame with study-level metadata
    """
    assert len(per_image_embeddings) == len(image_to_study_map), (
        "Embedding count must match image_to_study_map rows"
    )

    image_to_study_map = image_to_study_map.copy()
    image_to_study_map["_emb_idx"] = range(len(image_to_study_map))

    study_embeddings = []
    study_records    = []

    for study_id, group in image_to_study_map.groupby("parsed_study_id", sort=False):
        idxs = group["_emb_idx"].tolist()
        views = [per_image_embeddings[i] for i in idxs]
        fused = fuse_views(views)
        study_embeddings.append(fused)

        # Keep first row's metadata for the study record
        first_row = group.iloc[0].to_dict()
        first_row["n_views_fused"] = len(idxs)
        study_records.append(first_row)

    fused_array = np.stack(study_embeddings, axis=0)
    study_meta  = pd.DataFrame(study_records).drop(columns=["_emb_idx"], errors="ignore")

    print(f"  Fused {len(image_to_study_map):,} images → {len(fused_array):,} studies")
    single_view = (study_meta["n_views_fused"] == 1).sum()
    multi_view  = (study_meta["n_views_fused"] > 1).sum()
    print(f"  Single-view: {single_view:,} ({single_view/len(study_meta)*100:.1f}%)")
    print(f"  Multi-view:  {multi_view:,} ({multi_view/len(study_meta)*100:.1f}%)")

    return fused_array, study_meta


# ─────────────────────────────────────────────────────────────────────────────
# PART B — TEMPORAL FUSION (Prior Studies)
# ─────────────────────────────────────────────────────────────────────────────

def get_temporal_context(
    patient_id: str,
    current_study_order: int,
    metadata: pd.DataFrame,
    max_prior: int = 1,
) -> Optional[list[dict]]:
    """
    Retrieve prior study impression(s) for the same patient, most recent first.

    IMPORTANT RULES:
      - Only returns impressions from the TRAIN split (never val/test)
      - If no valid prior studies exist, returns None (graceful fallback)
      - The temporal branch is disabled entirely if this returns None
        (report as a dataset limitation, do NOT force it)

    Args:
        patient_id          : de-identified patient ID string
        current_study_order : patient_report_date_order for the current study
        metadata            : the FAISS metadata DataFrame (train split only)
        max_prior           : max number of prior studies to return (default 1)

    Returns:
        List of dicts [{'impression': str, 'study_order': int}, ...] sorted
        most-recent-first, or None if no valid prior studies exist.
    """
    # Filter: same patient, earlier study, train split only
    prior = metadata[
        (metadata["deid_patient_id"] == patient_id) &
        (metadata["patient_report_date_order"] < current_study_order) &
        (metadata.get("split", pd.Series(["train"] * len(metadata))) == "train")
    ].copy()

    if prior.empty:
        return None  # No longitudinal data — skip temporal branch

    # Sort by most recent first
    prior.sort_values("patient_report_date_order", ascending=False, inplace=True)
    prior = prior.head(max_prior)

    results = []
    for _, row in prior.iterrows():
        imp = row.get("text_target") or row.get("impression")
        if imp:
            results.append({
                "impression":   str(imp),
                "study_order":  int(row["patient_report_date_order"]),
            })

    return results if results else None


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute temporal context cache (optional — speeds up Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_cache(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute the most recent prior study impression for every record in
    metadata that has one. Saves to temporal_context_cache.csv.

    The RAG pipeline can do a simple CSV lookup instead of re-computing each time.
    """
    print("\n  Building temporal context cache...")

    # Fast vectorized approach using pandas groupby and shift
    cache = metadata.copy()
    
    # Sort by patient and study order
    cache.sort_values(["deid_patient_id", "patient_report_date_order"], ascending=[True, True], inplace=True)
    
    # Create the text target column to shift
    cache["_imp_text"] = cache.apply(lambda row: row.get("text_target") or row.get("impression"), axis=1)
    
    # Shift to get the previous study's text and order
    cache["prior_impression"] = cache.groupby("deid_patient_id")["_imp_text"].shift(1)
    cache["prior_study_order"] = cache.groupby("deid_patient_id")["patient_report_date_order"].shift(1)
    
    cache["has_temporal_context"] = cache["prior_impression"].notna()
    
    # Keep only the needed columns
    keep_cols = ["path_to_image", "deid_patient_id", "patient_report_date_order", 
                 "has_temporal_context", "prior_impression", "prior_study_order"]
    
    cache = cache[keep_cols]

    cache_path = OUTPUTS_DIR / "temporal_context_cache.csv"
    cache.to_csv(str(cache_path), index=False)

    n_with = cache["has_temporal_context"].sum()
    print(f"  Records with temporal context: {n_with:,} ({n_with/len(cache)*100:.1f}%)")
    print(f"  Cache saved → {cache_path}")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 3 — Spatial & Temporal Fusion")
    print("=" * 60)

    # Load metadata from Phase 2
    print(f"\nLoading metadata from Phase 2: {FAISS_METADATA_PATH}")
    metadata = pd.read_csv(str(FAISS_METADATA_PATH))
    print(f"  Rows: {len(metadata):,}")

    # ── Spatial fusion (demo — real use requires Person A's image embeddings) ──
    print("\n[A] Spatial Fusion")
    print("  Waiting for Person A's per-image embeddings...")
    print("  When ready, call: fuse_study_embeddings(image_embeddings, index_map)")
    print("  fuse_views() is available for ad-hoc use.")

    # ── Temporal context ───────────────────────────────────────────────────────
    print("\n[B] Temporal Context")

    # Quick demo on first 5 patients
    demo_rows = metadata.head(20)
    demo_with_ctx = 0
    for _, row in demo_rows.iterrows():
        ctx = get_temporal_context(
            row["deid_patient_id"],
            row["patient_report_date_order"],
            metadata,
        )
        if ctx:
            demo_with_ctx += 1
            print(f"  Patient {row['deid_patient_id']} — prior impression: "
                  f"{ctx[0]['impression'][:60]}...")

    print(f"\n  Demo: {demo_with_ctx}/{len(demo_rows)} rows have temporal context.")

    # Pre-compute full temporal cache
    print("\n  Building full temporal context cache (may take a few minutes)...")
    cache = build_temporal_cache(metadata)

    print("\n  ✅ Phase 3 complete.")
    print("     fuse_views()              — spatial fusion ready")
    print("     get_temporal_context()    — temporal lookup ready")
    print("     temporal_context_cache.csv — pre-computed cache saved")
    print("\n  Next: run Phase 4 (RAG pipeline).")
