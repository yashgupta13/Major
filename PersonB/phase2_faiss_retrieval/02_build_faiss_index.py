"""
02_build_faiss_index.py — Phase 2: FAISS Vector Database
=========================================================
Builds a FAISS similarity search index over the text embeddings produced in
Phase 1. Also implements and tests the retrieve_top_k() function.

Reads from:
    outputs/text_embeddings_train.npy    (from Phase 1)
    outputs/embedding_index_map.csv      (from Phase 1)
    ../impression.csv                    (for label metadata merge — already
                                          merged in Phase 1 but kept for ref)

Writes to:
    outputs/faiss_train.index            (searchable FAISS index)
    outputs/faiss_metadata.csv           (per-row metadata for retrieval results)

Run:
    cd PersonB/phase2_faiss_retrieval
    python 02_build_faiss_index.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import faiss

from utils.config import (
    TEXT_EMBEDDINGS_PATH, EMBEDDING_INDEX_MAP,
    FAISS_INDEX_PATH, FAISS_METADATA_PATH,
    FAISS_TOP_K, PATHOLOGY_LABELS, TRAIN_SPLIT,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load Phase 1 outputs
# ─────────────────────────────────────────────────────────────────────────────

def load_phase1_outputs():
    print("\n[1/5] Loading Phase 1 outputs...")

    embeddings = np.load(str(TEXT_EMBEDDINGS_PATH)).astype("float32")
    print(f"  Embeddings shape: {embeddings.shape}")

    meta = pd.read_csv(str(EMBEDDING_INDEX_MAP))
    print(f"  Index map rows: {len(meta):,}")

    assert len(embeddings) == len(meta), (
        f"Embedding count {len(embeddings)} != metadata rows {len(meta)}"
    )

    # Hard safety check — test/val records must NEVER be indexed
    if "split" in meta.columns:
        non_train = meta[meta["split"] != TRAIN_SPLIT]
        assert len(non_train) == 0, (
            f"❌ LEAKAGE DETECTED: {len(non_train)} non-train records found in index!"
        )
        print(f"  ✅ Split check passed — all {len(meta):,} records are train split.")

    return embeddings, meta


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build FAISS index
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat inner-product index (= cosine similarity after L2 normalization).
    For very large indexes (>500k vectors), switch to IndexIVFFlat for speed.
    """
    print("\n[2/5] Building FAISS index...")

    # Normalize embeddings → cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]  # embedding dimension (768 for ClinicalBERT)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    print(f"  Index type: IndexFlatIP (exact cosine similarity)")
    print(f"  Dimension:  {d}")
    print(f"  Vectors:    {index.ntotal:,}")

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"  Saved → {FAISS_INDEX_PATH}")

    return index


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build metadata store
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean metadata CSV for retrieval results.
    Keeps: identifiers, text, impression pathology labels.
    """
    print("\n[3/5] Building metadata store...")

    # Columns to keep for each retrieved result
    id_cols = [
        "path_to_image", "deid_patient_id", "patient_report_date_order",
        "frontal_lateral", "ap_pa", "split",
        "text_target", "text_source",
        "parsed_patient_id", "parsed_study_id",
    ]
    imp_label_cols = [f"imp_{l}" for l in PATHOLOGY_LABELS]

    keep = [c for c in id_cols + imp_label_cols if c in meta.columns]
    metadata = meta[keep].copy()

    # Rename imp_* back to plain label names for easier downstream use
    rename_map = {f"imp_{l}": l for l in PATHOLOGY_LABELS if f"imp_{l}" in metadata.columns}
    metadata.rename(columns=rename_map, inplace=True)

    metadata.to_csv(str(FAISS_METADATA_PATH), index=False)
    print(f"  Saved → {FAISS_METADATA_PATH}  (rows={len(metadata):,})")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Retrieval function
# ─────────────────────────────────────────────────────────────────────────────

def make_retriever(index: faiss.Index, metadata: pd.DataFrame):
    """
    Returns a retrieve_top_k() function bound to the given index + metadata.
    This is the function Person A's query embeddings will call at test time.
    """

    def retrieve_top_k(query_embedding: np.ndarray, k: int = FAISS_TOP_K) -> list[dict]:
        """
        Retrieve the top-K most similar indexed studies for a query embedding.

        Args:
            query_embedding : 1-D np.ndarray of shape (dim,)  — can be an image
                              embedding (from Person A) or a text embedding.
            k               : number of results to return.

        Returns:
            List of k dicts, each containing:
                study_id            parsed study folder name
                patient_id          de-identified patient ID
                impression          the indexed impression text
                pathology_labels    dict of 14 CheXpert label values
                similarity_score    cosine similarity (0–1)
                view_type           frontal / lateral
        """
        query = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query)

        scores, indices = index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            row = metadata.iloc[idx]
            label_dict = {l: row.get(l) for l in PATHOLOGY_LABELS}
            results.append({
                "study_id":         row.get("parsed_study_id"),
                "patient_id":       row.get("deid_patient_id"),
                "impression":       row.get("text_target"),
                "pathology_labels": label_dict,
                "similarity_score": float(score),
                "view_type":        row.get("frontal_lateral"),
            })

        return results

    return retrieve_top_k


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(retrieve_top_k, embeddings: np.ndarray, metadata: pd.DataFrame):
    """
    Quick sanity check: use the 1st embedding as a query, confirm top-1 is itself.
    """
    print("\n[4/5] Smoke test — querying with first embedding...")

    result = retrieve_top_k(embeddings[0], k=5)

    print(f"  Top-{len(result)} results:")
    for i, r in enumerate(result):
        print(f"    [{i+1}] study={r['study_id']}  score={r['similarity_score']:.4f}  "
              f"view={r['view_type']}")
        if i == 0:
            print(f"         impression preview: {str(r['impression'])[:80]}...")

    # Top-1 should be the query itself (score ≈ 1.0)
    assert result[0]["similarity_score"] > 0.99, (
        f"Top-1 similarity {result[0]['similarity_score']:.4f} is unexpectedly low!"
    )
    print("  ✅ Self-retrieval check passed (top-1 score ≈ 1.0)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Retrieval evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieve_fn, query_embeddings: np.ndarray,
                true_study_ids: list, k: int = 5) -> float:
    """
    Compute Recall@K: fraction of queries where the relevant study appears in top-K.

    Args:
        retrieve_fn      : the retrieve_top_k() function
        query_embeddings : (N, dim) array of query embeddings
        true_study_ids   : list of N expected study IDs (ground truth)
        k                : retrieval cutoff

    Returns:
        Recall@K as a float in [0, 1]
    """
    hits = 0
    for query_emb, true_id in zip(query_embeddings, true_study_ids):
        results = retrieve_fn(query_emb, k=k)
        retrieved_ids = [r["study_id"] for r in results]
        if true_id in retrieved_ids:
            hits += 1
    return hits / len(query_embeddings)


def mean_reciprocal_rank(retrieve_fn, query_embeddings: np.ndarray,
                         true_study_ids: list, k: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) at k.
    MRR = average of 1/rank of the first correct result (0 if not in top-K).
    """
    rr_sum = 0.0
    for query_emb, true_id in zip(query_embeddings, true_study_ids):
        results = retrieve_fn(query_emb, k=k)
        for rank, r in enumerate(results, start=1):
            if r["study_id"] == true_id:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(query_embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 2 — FAISS Vector Database")
    print("=" * 60)

    embeddings, meta = load_phase1_outputs()

    # Keep a copy of original (un-normalized) embeddings for smoke test
    embeddings_raw = embeddings.copy()

    index    = build_faiss_index(embeddings)
    metadata = build_metadata(meta)

    retrieve_top_k = make_retriever(index, metadata)

    smoke_test(retrieve_top_k, embeddings_raw, metadata)

    print("\n[5/5] Evaluation helpers available:")
    print("  recall_at_k(retrieve_top_k, query_embeddings, true_ids, k=5)")
    print("  mean_reciprocal_rank(retrieve_top_k, query_embeddings, true_ids, k=10)")
    print("\n  ✅ Phase 2 complete. FAISS index ready.")
    print(f"     Index: {FAISS_INDEX_PATH}")
    print(f"     Meta:  {FAISS_METADATA_PATH}")
    print("\n  Next: run Phase 3 (fusion) then Phase 4 (RAG pipeline).")
