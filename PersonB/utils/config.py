"""
config.py — Shared paths and constants for all Person B scripts.
Edit DATA_ROOT if your data is in a different location.
"""

from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
# PersonB/ lives inside the project root
PERSONB_ROOT = Path(__file__).resolve().parent.parent  # d:/SEM 7/Major/Major/PersonB
PROJECT_ROOT = PERSONB_ROOT.parent                     # d:/SEM 7/Major/Major

# ── Input data files (read-only — shared with all team members) ───────────
MAIN_CSV      = PROJECT_ROOT / "df_chexpert_plus_240401.csv"
IMPRESSION_CSV = PROJECT_ROOT / "impression.csv"
REPORT_CSV     = PROJECT_ROOT / "report.csv"

# ── Output directory (Person B writes here only) ──────────────────────────
OUTPUTS_DIR = PERSONB_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Output file paths ─────────────────────────────────────────────────────
TEXT_EMBEDDINGS_PATH   = OUTPUTS_DIR / "text_embeddings_train.npy"
EMBEDDING_INDEX_MAP    = OUTPUTS_DIR / "embedding_index_map.csv"
FAISS_INDEX_PATH       = OUTPUTS_DIR / "faiss_train.index"
FAISS_METADATA_PATH    = OUTPUTS_DIR / "faiss_metadata.csv"
RAG_VAL_OUTPUTS_PATH   = OUTPUTS_DIR / "rag_val_outputs.csv"

# ── Person A outputs (once they are ready, update these paths) ────────────
# Person A saves their image embeddings here — coordinate with them.
PERSON_A_EMBEDDINGS_DIR = PROJECT_ROOT / "PersonA" / "outputs"  # update if needed
IMAGE_EMBEDDINGS_PATH   = PERSON_A_EMBEDDINGS_DIR / "image_embeddings.npy"
DATASET_TRAIN_CSV       = PERSON_A_EMBEDDINGS_DIR / "dataset_train.csv"
DATASET_VAL_CSV         = PERSON_A_EMBEDDINGS_DIR / "dataset_val.csv"
DATASET_TEST_CSV        = PERSON_A_EMBEDDINGS_DIR / "dataset_test.csv"

# ── Model configuration ───────────────────────────────────────────────────
# Text encoder — change to BioLinkBERT-large for higher quality
# (needs more RAM/VRAM)
TEXT_ENCODER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
# TEXT_ENCODER_MODEL = "michiyasunaga/BioLinkBERT-large"
# TEXT_ENCODER_MODEL = "StanfordAIMI/RadBERT"

TEXT_ENCODER_MAX_LENGTH = 512  # Safe — only 0.85% of impressions exceed this
TEXT_ENCODER_BATCH_SIZE = 64   # Reduce to 32 if OOM

# ── FAISS configuration ───────────────────────────────────────────────────
FAISS_TOP_K = 5          # Default retrieval count
EMBEDDING_DIM = 768      # ClinicalBERT hidden size (1024 for BioLinkBERT-large)

# ── 14 CheXpert pathology labels ─────────────────────────────────────────
PATHOLOGY_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# ── LLM configuration ─────────────────────────────────────────────────────
# Option A: local Mistral/Llama3 via Ollama (free, runs on ~8GB RAM)
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3"

# Option B: OpenAI API (set env var OPENAI_API_KEY)
OPENAI_MODEL  = "gpt-4o-mini"

LLM_TEMPERATURE = 0.0   # Deterministic — required for reproducibility
LLM_MAX_TOKENS  = 512

# ── Split column values ───────────────────────────────────────────────────
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
# NOTE: The official CheXpert Plus valid set has only 234 images.
# Person A will create a proper 80/10/10 patient-level split.
# Once that is ready, use DATASET_TRAIN_CSV / DATASET_VAL_CSV above.
