"""
04_rag_pipeline.py — Phase 4: RAG Prompt + LLM Report Generation
=================================================================
Full end-to-end pipeline:
  1. Take a query embedding (from Person A's vision encoder)
  2. Retrieve top-K similar past cases via FAISS (from Phase 2)
  3. Fetch optional temporal context (from Phase 3)
  4. Build a structured RAG prompt
  5. Call LLM (Mistral-7B via Ollama OR GPT-4o-mini) with temperature=0
  6. Parse and save the generated Findings + Impression report

Reads from:
    outputs/faiss_train.index            (from Phase 2)
    outputs/faiss_metadata.csv           (from Phase 2)
    outputs/temporal_context_cache.csv   (from Phase 3 — optional)
    ../df_chexpert_plus_240401.csv        (for validation set records)
    ../impression.csv                    (for current-study pathology labels)

Writes to:
    outputs/rag_val_outputs.csv          (generated reports for Person C's evaluation)

Run:
    cd PersonB/phase4_rag_pipeline
    python 04_rag_pipeline.py

    # For OpenAI fallback:
    set OPENAI_API_KEY=<your-key>
    python 04_rag_pipeline.py --llm openai
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
import faiss
import requests
from tqdm import tqdm
from typing import Optional

from utils.config import (
    MAIN_CSV, IMPRESSION_CSV,
    FAISS_INDEX_PATH, FAISS_METADATA_PATH,
    RAG_VAL_OUTPUTS_PATH, OUTPUTS_DIR,
    FAISS_TOP_K, PATHOLOGY_LABELS,
    OLLAMA_URL, OLLAMA_MODEL,
    OPENAI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    TRAIN_SPLIT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────

def load_resources():
    print("\n[1/6] Loading FAISS index and metadata...")
    index    = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = pd.read_csv(str(FAISS_METADATA_PATH))
    print(f"  FAISS index: {index.ntotal:,} vectors")
    print(f"  Metadata:    {len(metadata):,} rows")

    # Temporal context cache (pre-computed in Phase 3)
    temporal_cache_path = OUTPUTS_DIR / "temporal_context_cache.csv"
    temporal_cache = None
    if temporal_cache_path.exists():
        temporal_cache = pd.read_csv(str(temporal_cache_path))
        n_with = temporal_cache["has_temporal_context"].sum()
        print(f"  Temporal cache: {len(temporal_cache):,} rows ({n_with:,} with prior context)")
    else:
        print("  Temporal cache not found — Phase 3 not run yet. Temporal context disabled.")

    return index, metadata, temporal_cache


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_top_k(index: faiss.Index, metadata: pd.DataFrame,
                   query_embedding: np.ndarray, k: int = FAISS_TOP_K) -> list[dict]:
    """
    Retrieve the k most similar indexed studies for a query embedding.

    Args:
        index           : loaded FAISS index
        metadata        : the FAISS metadata DataFrame
        query_embedding : 1-D np.array of shape (dim,)
        k               : number of results

    Returns:
        List of result dicts
    """
    query = query_embedding.reshape(1, -1).astype("float32")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = metadata.iloc[idx]
        results.append({
            "study_id":         row.get("parsed_study_id"),
            "patient_id":       row.get("deid_patient_id"),
            "impression":       str(row.get("text_target", "")),
            "pathology_labels": {l: row.get(l) for l in PATHOLOGY_LABELS},
            "similarity_score": float(score),
            "view_type":        row.get("frontal_lateral"),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Temporal context lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_temporal_context_from_cache(path_to_image: str,
                                     temporal_cache: Optional[pd.DataFrame]) -> Optional[str]:
    """Look up pre-computed prior impression from Phase 3 cache."""
    if temporal_cache is None:
        return None
    row = temporal_cache[temporal_cache["path_to_image"] == path_to_image]
    if row.empty or not row.iloc[0]["has_temporal_context"]:
        return None
    return str(row.iloc[0]["prior_impression"])


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_prompt(
    current_label_dict: dict,
    retrieved_cases: list[dict],
    temporal_impression: Optional[str] = None,
) -> str:
    """
    Build the structured RAG prompt sent to the LLM.

    Priority:  Current study findings  >  Retrieved similar cases  >  Temporal context
    The LLM is instructed to ONLY generate findings supported by the current study.

    Args:
        current_label_dict   : 14-label dict for the current study (from impression.csv)
        retrieved_cases      : output of retrieve_top_k()
        temporal_impression  : prior study impression string (or None)

    Returns:
        Formatted prompt string
    """
    # ── Current study summary from pathology labels ──────────────────────────
    positive  = [l for l, v in current_label_dict.items() if v == 1.0]
    uncertain = [l for l, v in current_label_dict.items() if v == -1.0]
    negative  = [l for l, v in current_label_dict.items() if v == 0.0]

    if positive:
        current_summary = f"Confirmed: {', '.join(positive)}."
    else:
        current_summary = "No confirmed findings."

    if uncertain:
        current_summary += f" Possible/uncertain: {', '.join(uncertain)}."
    if negative:
        current_summary += f" Absent: {', '.join(negative[:5])}."  # top-5 negatives

    # ── Retrieved similar cases ───────────────────────────────────────────────
    retrieved_text = ""
    for i, case in enumerate(retrieved_cases[:5], 1):
        retrieved_text += (
            f"Case {i} (similarity {case['similarity_score']:.3f}):\n"
            f"  {case['impression']}\n\n"
        )

    # ── Temporal context ──────────────────────────────────────────────────────
    temporal_section = ""
    if temporal_impression:
        temporal_section = (
            f"\nHISTORICAL CONTEXT (most recent prior study for this patient):\n"
            f"  {temporal_impression}\n"
        )

    # ── Assemble prompt ───────────────────────────────────────────────────────
    prompt = f"""You are a radiology report assistant. Your task is to generate a structured chest X-ray report.

CURRENT STUDY — DETECTED FINDINGS:
{current_summary}

RETRIEVED SIMILAR CASES (supporting reference only):
{retrieved_text}{temporal_section}
INSTRUCTIONS:
- Generate a report based ONLY on the current study findings listed above.
- Use retrieved cases as stylistic and clinical references, not as facts about this patient.
- Do NOT mention that you retrieved similar cases.
- Do NOT invent findings that are not listed in the current study.
- Output EXACTLY in this format (two lines, no extra text):

Findings: [detailed observations about the chest radiograph]
Impression: [concise clinical summary and assessment]"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# LLM calls
# ─────────────────────────────────────────────────────────────────────────────

def generate_ollama(prompt: str, model_name: str) -> str:
    """Call local Ollama REST API with the specified model."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama not running. Start it with: ollama serve\n"
            f"Then pull the model: ollama pull {model_name}"
        )


def parse_generated_report(text: str) -> tuple[str, str]:
    """
    Extract Findings and Impression from the LLM output.
    Returns ("", "") if parsing fails.
    """
    findings, impression = "", ""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("findings:"):
            findings = line[len("findings:"):].strip()
        elif line.lower().startswith("impression:"):
            impression = line[len("impression:"):].strip()
    return findings, impression


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def generate_report_for_study(
    query_embedding: np.ndarray,
    current_label_dict: dict,
    path_to_image: str,
    index: faiss.Index,
    metadata: pd.DataFrame,
    temporal_cache: Optional[pd.DataFrame],
    model_name: str = "llama3",
    k: int = FAISS_TOP_K,
) -> dict:
    """
    Full RAG pipeline for a single study.

    Args:
        query_embedding    : 1-D np.array — image embedding from Person A
        current_label_dict : 14 pathology labels for the current study
        path_to_image      : image path string (used for temporal cache lookup)
        index              : loaded FAISS index
        metadata           : FAISS metadata DataFrame
        temporal_cache     : Phase 3 temporal context cache (or None)
        llm_backend        : "ollama" or "openai"
        k                  : number of retrieved cases

    Returns:
        dict with generated report + metadata for evaluation
    """
    # Step 1: Retrieve
    retrieved = retrieve_top_k(index, metadata, query_embedding, k=k)

    # Step 2: Temporal context
    temporal_imp = get_temporal_context_from_cache(path_to_image, temporal_cache)

    # Step 3: Build prompt
    prompt = build_rag_prompt(current_label_dict, retrieved, temporal_imp)

    # Step 4: Generate
    generated_text = generate_ollama(prompt, model_name=model_name)

    # Step 5: Parse
    findings, impression = parse_generated_report(generated_text)

    return {
        "path_to_image":         path_to_image,
        "generated_findings":    findings,
        "generated_impression":  impression,
        "full_generated_text":   generated_text,
        "temporal_context_used": temporal_imp is not None,
        "n_retrieved":           len(retrieved),
        "top1_similarity":       retrieved[0]["similarity_score"] if retrieved else None,
        "retrieved_study_ids":   [r["study_id"] for r in retrieved],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch run on validation set
# ─────────────────────────────────────────────────────────────────────────────

def run_on_validation_set(
    index: faiss.Index,
    metadata: pd.DataFrame,
    temporal_cache: Optional[pd.DataFrame],
    model_name: str = "llama3",
    n_samples: int = 30,
):
    """
    Run the RAG pipeline on a sample of validation cases.
    Uses text embeddings as proxy query until Person A's image embeddings arrive.
    Saves results to rag_val_outputs.csv for Person C's evaluation.
    """
    print(f"\n[5/6] Running RAG on {n_samples} validation records...")

    # Load validation records from the main CSV
    df = pd.read_csv(str(MAIN_CSV), low_memory=False)
    imp_labels = pd.read_csv(str(IMPRESSION_CSV))

    # Use records NOT in train — proxy for val set until Person A delivers splits
    df_val = df[df["split"] != TRAIN_SPLIT].copy()
    if len(df_val) == 0:
        # If no official val split records in CSV, sample from train for demo
        print("  No non-train records found — sampling from train for demo.")
        df_val = df[df["split"] == TRAIN_SPLIT].sample(n=n_samples, random_state=42)

    df_val = df_val.merge(imp_labels, on="path_to_image", how="left")
    df_val = df_val.head(n_samples)

    print(f"  Validation sample: {len(df_val)} records")

    # Load text embeddings as PROXY queries (until image embeddings from Person A)
    text_emb_path = OUTPUTS_DIR / "text_embeddings_train.npy"
    emb_map_path  = OUTPUTS_DIR / "embedding_index_map.csv"

    use_proxy = text_emb_path.exists() and emb_map_path.exists()
    if use_proxy:
        print("  Using text embeddings as proxy query embeddings (Person A's not yet available).")
        proxy_embs = np.load(str(text_emb_path)).astype("float32")
        emb_map    = pd.read_csv(str(emb_map_path))
    else:
        print("  ⚠  No embeddings found. Run Phase 1 first, then Phase 2.")
        return

    results = []
    for i, (_, row) in enumerate(tqdm(df_val.iterrows(), total=len(df_val))):
        path = row.get("path_to_image", "")

        # Find matching embedding row (or use row i as proxy)
        match = emb_map[emb_map["path_to_image"] == path]
        if not match.empty:
            emb_idx = match.index[0]
        else:
            emb_idx = min(i, len(proxy_embs) - 1)  # fallback

        query_emb = proxy_embs[emb_idx]

        # Build label dict for current study
        label_dict = {l: row.get(l) for l in PATHOLOGY_LABELS}

        try:
            result = generate_report_for_study(
                query_embedding=query_emb,
                current_label_dict=label_dict,
                path_to_image=path,
                index=index,
                metadata=metadata,
                temporal_cache=temporal_cache,
                model_name=model_name,
            )
        except Exception as e:
            print(f"  ⚠ Error on record {i}: {e}")
            result = {
                "path_to_image": path,
                "generated_findings": "",
                "generated_impression": "",
                "full_generated_text": f"ERROR: {e}",
                "temporal_context_used": False,
                "n_retrieved": 0,
                "top1_similarity": None,
                "retrieved_study_ids": [],
            }

        # Add reference text for Person C's evaluation
        result["reference_impression"] = row.get("section_impression", "")
        result["reference_findings"]   = row.get("section_findings", "")
        result["true_labels"]          = str(label_dict)
        results.append(result)

    output_df = pd.DataFrame(results)
    output_df.to_csv(str(RAG_VAL_OUTPUTS_PATH), index=False)

    print(f"\n[6/6] Saved {len(output_df)} generated reports → {RAG_VAL_OUTPUTS_PATH}")

    # Quick summary
    temporal_used = output_df["temporal_context_used"].sum()
    empty_reports = (output_df["generated_impression"] == "").sum()
    print(f"  Temporal context used: {temporal_used}/{len(output_df)}")
    print(f"  Reports with empty impression: {empty_reports}/{len(output_df)}")
    print("\n  ✅ Phase 4 complete. Hand rag_val_outputs.csv to Person C for evaluation.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline — Phase 4")
    parser.add_argument("--model", type=str, default="llama3",
                        help="Ollama model to use (default: llama3)")
    parser.add_argument("--samples", type=int, default=30,
                        help="Number of validation cases to run (default: 30)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  PHASE 4 — RAG Pipeline  (Model: {args.model})")
    print("=" * 60)

    index, metadata, temporal_cache = load_resources()

    print(f"\n[2/6] LLM backend: Ollama")
    print(f"  Model: {args.model}")
    print(f"  URL:   {OLLAMA_URL}")
    print("  Make sure Ollama is running: ollama serve")
    print(f"  And model is pulled: ollama pull {args.model}")

    print("\n[3/6] Prompt template ready (build_rag_prompt)")
    print("[4/6] Parser ready (parse_generated_report)")

    run_on_validation_set(
        index, metadata, temporal_cache,
        model_name=args.model,
        n_samples=args.samples,
    )
