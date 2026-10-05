"""
verify_setup.py — Quick setup verification for Person B workspace
=================================================================
Run this FIRST to confirm all data files are accessible and
libraries are installed correctly.

    cd PersonB
    python verify_setup.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 55)
print("  Person B — Setup Verification")
print("=" * 55)

errors = []

# ── 1. Check Python version ───────────────────────────────────
import platform
py_ver = sys.version_info
print(f"\n✅ Python {py_ver.major}.{py_ver.minor}.{py_ver.micro} ({platform.system()})")
if py_ver < (3, 10):
    errors.append("Python 3.10+ recommended (type hints used).")

# ── 2. Check required libraries ───────────────────────────────
REQUIRED = {
    "pandas":       "pandas",
    "numpy":        "numpy",
    "torch":        "torch",
    "transformers": "transformers",
    "faiss":        "faiss",
    "tqdm":         "tqdm",
    "requests":     "requests",
}

print("\nLibrary check:")
for display, import_name in REQUIRED.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  ✅ {display:<15} {ver}")
    except ImportError:
        print(f"  ❌ {display:<15} NOT INSTALLED")
        errors.append(f"Missing: pip install {display}")

# ── 3. Check data files ───────────────────────────────────────
from pathlib import Path
from utils.config import MAIN_CSV, IMPRESSION_CSV, REPORT_CSV, OUTPUTS_DIR

print("\nData file check:")
for label, path in [
    ("df_chexpert_plus_240401.csv", MAIN_CSV),
    ("impression.csv",               IMPRESSION_CSV),
    ("report.csv",                   REPORT_CSV),
]:
    if Path(path).exists():
        size_mb = Path(path).stat().st_size / 1_048_576
        print(f"  ✅ {label:<35} ({size_mb:.0f} MB)")
    else:
        print(f"  ❌ {label:<35} NOT FOUND at {path}")
        errors.append(f"Missing data file: {path}")

# ── 4. Check outputs directory ────────────────────────────────
print(f"\nOutputs directory: {OUTPUTS_DIR}")
if OUTPUTS_DIR.exists():
    files = list(OUTPUTS_DIR.iterdir())
    print(f"  ✅ Exists — {len(files)} file(s) inside")
    for f in files:
        size_mb = f.stat().st_size / 1_048_576
        print(f"     {f.name} ({size_mb:.1f} MB)")
else:
    print("  ⚠  Outputs directory empty — run Phase 1 first.")

# ── 5. Quick data peek ────────────────────────────────────────
print("\nData quick peek (first 2 rows of impression.csv):")
try:
    import pandas as pd
    imp = pd.read_csv(str(IMPRESSION_CSV), nrows=2)
    print(f"  Columns: {list(imp.columns)}")
    print(f"  Sample path: {imp['path_to_image'].iloc[0]}")
except Exception as e:
    errors.append(f"Could not read impression.csv: {e}")

# ── 6. Ollama check (optional) ────────────────────────────────
print("\nOllama check (optional — needed for Phase 4):")
try:
    import requests as req
    r = req.get("http://localhost:11434/api/tags", timeout=2)
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"  ✅ Ollama running — models: {models if models else '(none pulled yet)'}")
    if not any("mistral" in m for m in models):
        print("  ℹ  Mistral not pulled yet. Run: ollama pull mistral:7b-instruct-v0.3")
except Exception:
    print("  ℹ  Ollama not running (OK for now — needed in Phase 4)")
    print("     Start with: ollama serve")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
if errors:
    print(f"  ❌ {len(errors)} issue(s) found:")
    for e in errors:
        print(f"     • {e}")
    print("\n  Fix the above, then re-run this script.")
else:
    print("  ✅ All checks passed! Ready to start Phase 1.")
    print("\n  Next step:")
    print("     cd phase1_text_encoder")
    print("     python 01_encode_impressions.py")
print("=" * 55)
