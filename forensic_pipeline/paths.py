from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "outputs"

PROMPT_BANK_JSON = CONFIG_DIR / "prompt_bank.json"
IMAGE_EVIDENCE_CSV = OUTPUT_DIR / "image_evidence.csv"
OPENCLIP_OUTPUTS_CSV = OUTPUT_DIR / "openclip_outputs.csv"
TEXT_EVIDENCE_CSV = OUTPUT_DIR / "text_evidence.csv"
FUSED_DECISIONS_CSV = OUTPUT_DIR / "multimodal_fused_decisions.csv"
FORENSIC_TIMELINE_CSV = RAW_DATA_DIR / "forensic_multimodal_evidence.csv"


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value
