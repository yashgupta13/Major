"""
text_utils.py — Text normalization helpers shared across all Person B phases.
"""

import re
import pandas as pd
from typing import Optional


# ── De-identification normalization ───────────────────────────────────────

def normalize_deid(text: Optional[str]) -> Optional[str]:
    """
    Replace de-identification placeholders (___) with the token [DEID].
    CheXpert Plus: 4.19% of reports contain ___ placeholder runs.

    Args:
        text: raw report text or None

    Returns:
        Normalized string, or None if the input was empty/null.
    """
    if pd.isna(text) or str(text).strip() == "":
        return None
    # Collapse any run of 2+ underscores into a single [DEID] token
    cleaned = re.sub(r"_{2,}", "[DEID]", str(text))
    return cleaned.strip()


# ── Primary text target selection ─────────────────────────────────────────

def select_text_target(row: pd.Series) -> Optional[str]:
    """
    Pick the best available text for a record:
      1. section_impression  (present in 99.93% of records)
      2. section_findings    (fallback — present in only 26.6%)
      3. None                (record cannot be embedded — skip)

    Args:
        row: a single DataFrame row with at least the two section columns

    Returns:
        Normalized text or None.
    """
    impression = normalize_deid(row.get("section_impression"))
    if impression:
        return impression

    findings = normalize_deid(row.get("section_findings"))
    if findings:
        return findings

    return None


# ── Batch application ─────────────────────────────────────────────────────

def add_text_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply select_text_target to every row and store result in 'text_target'.
    Also adds 'text_source' column to track which section was used.

    Args:
        df: the main CheXpert Plus DataFrame

    Returns:
        df with two new columns: 'text_target', 'text_source'
    """
    targets, sources = [], []
    for _, row in df.iterrows():
        imp = normalize_deid(row.get("section_impression"))
        if imp:
            targets.append(imp)
            sources.append("impression")
        else:
            fnd = normalize_deid(row.get("section_findings"))
            if fnd:
                targets.append(fnd)
                sources.append("findings")
            else:
                targets.append(None)
                sources.append("none")

    df = df.copy()
    df["text_target"] = targets
    df["text_source"] = sources
    return df


# ── Quick coverage report ─────────────────────────────────────────────────

def print_coverage_report(df: pd.DataFrame) -> None:
    """Print a summary of how many records have usable text."""
    total = len(df)
    imp_ok = (df.get("text_source") == "impression").sum()
    fnd_ok = (df.get("text_source") == "findings").sum()
    none_ok = (df.get("text_source") == "none").sum()

    print(f"\n{'='*50}")
    print(f"Text Coverage Report (n={total:,})")
    print(f"{'='*50}")
    print(f"  section_impression used : {imp_ok:>8,}  ({imp_ok/total*100:.2f}%)")
    print(f"  section_findings  used  : {fnd_ok:>8,}  ({fnd_ok/total*100:.2f}%)")
    print(f"  No text available       : {none_ok:>8,}  ({none_ok/total*100:.2f}%)")
    print(f"{'='*50}\n")


# ── Token length estimation ───────────────────────────────────────────────

def estimate_token_lengths(texts: list, sample_n: int = 5000) -> dict:
    """
    Quick whitespace-token estimate of lengths (no tokenizer needed).
    Real sub-word counts will be slightly higher.
    Use this for a fast sanity check before running the full encoder.

    Returns dict with mean, median, p90, p99, max, pct_over_512.
    """
    import numpy as np
    lengths = [len(str(t).split()) for t in texts if t is not None]
    lengths = lengths[:sample_n]
    arr = np.array(lengths)
    pct_over = (arr > 400).mean() * 100  # 400 words ≈ 512 sub-word tokens
    return {
        "mean":        round(arr.mean(), 1),
        "median":      int(np.median(arr)),
        "p90":         int(np.percentile(arr, 90)),
        "p99":         int(np.percentile(arr, 99)),
        "max":         int(arr.max()),
        "pct_over_400w": round(pct_over, 2),
    }
