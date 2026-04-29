from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from forensic_pipeline.paths import FORENSIC_TIMELINE_CSV, IMAGE_EVIDENCE_CSV, TEXT_EVIDENCE_CSV, project_path
from forensic_pipeline.pipeline_utils import drop_blank_rows, safe_str


# -----------------------------
# Config
# -----------------------------
FORENSIC_REPORT_CSV = FORENSIC_TIMELINE_CSV
DEFAULT_OUTPUT_CSV = TEXT_EVIDENCE_CSV
ASSOC_SOURCE = "assoc"
OCR_SOURCE = "ocr"

# Default (you can override from CLI)
DEFAULT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# Known fast fallback model
FALLBACK_MODEL = "facebook/bart-large-mnli"

HYPOTHESIS_TEMPLATE = "This text is about {}."

FROZEN_LABELS: List[str] = [
    "neutral_or_contextual",
    "abusive_or_obscene_language",
    "harassment_or_intimidation",
    "threat_of_violence",
    "incitement_or_endorsement_of_violence",
    "hate_or_bias_based_content",
    "weapon_related_text",
    "criminal_admission_or_description",
    "sexual_violence_or_exploitation",
    "self_harm_or_suicide_risk",
]


# -----------------------------
# Helpers
# -----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def media_id_from_image_path(image_path: str) -> str:
    return Path(image_path).stem


def split_context_text(text: str) -> List[str]:
    cleaned = safe_str(text).strip()
    return [cleaned] if cleaned else []


def texts_from_image_evidence(media_id: str, text_source: str, csv_path: str = IMAGE_EVIDENCE_CSV) -> List[str]:
    path = Path(csv_path)
    if not path.exists():
        return []

    df = drop_blank_rows(pd.read_csv(path))
    if "media_id" not in df.columns:
        return []

    sub = df[df["media_id"].astype(str) == str(media_id)]
    if sub.empty:
        return []

    row = sub.iloc[-1]
    if text_source == ASSOC_SOURCE:
        return split_context_text(row.get("assoc_text_concat", ""))
    if text_source == OCR_SOURCE:
        return split_context_text(row.get("ocr_text", ""))
    raise ValueError(f"text_source must be '{ASSOC_SOURCE}' or '{OCR_SOURCE}', got: {text_source}")


def texts_from_forensic_report(media_id: str, csv_path: str = FORENSIC_REPORT_CSV) -> List[str]:
    report_path = Path(csv_path)
    if not report_path.exists():
        return []

    print("[A] Reading forensic report CSV...")
    df = pd.read_csv(report_path)

    required_cols = {"event_type", "media_id", "message_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    print("[B] Filtering text rows for media_id...")
    df_sub = df[(df["event_type"] == "text") & (df["media_id"].astype(str) == str(media_id))].copy()
    return [t.strip() for t in df_sub["message_text"].fillna("").astype(str).tolist() if t and t.strip()]


def get_entailment_index(model) -> int:
    """
    Find entailment class index for NLI-style models.
    Falls back to last logit if not found.
    """
    label2id = getattr(model.config, "label2id", None) or {}
    for k, v in label2id.items():
        if "entail" in str(k).lower():
            return int(v)
    return int(model.config.num_labels - 1)


def ensure_csv(csv_path: Path, columns: List[str]) -> None:
    if not csv_path.exists():
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)


def append_row(csv_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    ensure_csv(csv_path, columns)

    df = pd.read_csv(csv_path)

    # Ensure schema is stable
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    df.loc[len(df)] = {c: row.get(c, "") for c in df.columns}
    df.to_csv(csv_path, index=False)


def print_scores(scores: Dict[str, float]) -> None:
    for lbl, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{lbl}: {sc:.4f}")


# -----------------------------
# Zero-shot text classification (no pipeline)
# -----------------------------
def classify_texts_zero_shot(
    texts: List[str],
    device: str,
    model_name: str,
) -> Dict[str, float]:
    print(f"[1] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print("[2] Loading model weights...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval().to(device)

    print("[3] Model ready.")
    entail_idx = get_entailment_index(model)

    # Aggregate across messages using MAX per label (simple + effective)
    agg_scores = {lbl: 0.0 for lbl in FROZEN_LABELS}

    print(f"[4] Classifying {len(texts)} text row(s)...")
    for i, text in enumerate(texts, start=1):
        t = str(text).strip()
        if not t:
            continue

        premises = [t] * len(FROZEN_LABELS)
        hyps = [HYPOTHESIS_TEMPLATE.format(lbl.replace("_", " ")) for lbl in FROZEN_LABELS]

        batch = tokenizer(
            premises,
            hyps,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**batch).logits
            entail_logits = logits[:, entail_idx]

        probs = torch.softmax(entail_logits, dim=0).detach().cpu().tolist()
        for lbl, p in zip(FROZEN_LABELS, probs):
            agg_scores[lbl] = max(agg_scores[lbl], float(p))

        print(f"    processed {i}/{len(texts)}")

    print("[5] Done scoring.")
    return agg_scores


# -----------------------------
# Main (CLIP-style)
# -----------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m forensic_pipeline.case_text path/to/image.jpg [output_csv] [model_name] [assoc|ocr]")
        print(f"Default model: {DEFAULT_MODEL}")
        print(f"Fallback model: {FALLBACK_MODEL}")
        raise SystemExit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_csv = project_path(sys.argv[2]) if len(sys.argv) >= 3 else Path(DEFAULT_OUTPUT_CSV)
    model_name = sys.argv[3] if len(sys.argv) >= 4 else DEFAULT_MODEL
    text_source = sys.argv[4].strip().lower() if len(sys.argv) >= 5 else ASSOC_SOURCE
    if text_source not in {ASSOC_SOURCE, OCR_SOURCE}:
        raise ValueError(f"text_source must be '{ASSOC_SOURCE}' or '{OCR_SOURCE}', got: {text_source}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    media_id = media_id_from_image_path(image_path)
    print("Media ID:", media_id)
    print("Text source:", text_source)

    print("[A] Reading image evidence text context...")
    texts = texts_from_image_evidence(media_id, text_source)
    if not texts and text_source == ASSOC_SOURCE:
        texts = texts_from_forensic_report(media_id)
    print(f"[C] Found {len(texts)} matching text row(s).")

    mapped_text = " | ".join(texts)[:15000] if texts else ""

    # Master CSV schema (one row per run, one file for all media_id)
    columns = [
        "created_utc",
        "image_path",
        "media_id",
        "text_source",
        "has_text_match",
        "n_text_rows",
        "mapped_text",
        "best_label",
        "best_score",
        *[f"score_{lbl}" for lbl in FROZEN_LABELS],
    ]

    # Always print the mapped text(s) before scores
    print("\nMapped forensic text(s):")
    if texts:
        for i, t in enumerate(texts, start=1):
            print(f"[{i}] {t}")
    else:
        print("(none)")

    # No match: still append a row
    if not texts:
        scores = {lbl: 0.0 for lbl in FROZEN_LABELS}
        scores["neutral_or_contextual"] = 1.0

        print("\nText classification scores:")
        print_scores(scores)
        print("\nBest label: neutral_or_contextual (1.0000)")

        row = {
            "created_utc": utc_now(),
            "image_path": str(Path(image_path).resolve()),
            "media_id": media_id,
            "text_source": text_source,
            "has_text_match": False,
            "n_text_rows": 0,
            "mapped_text": "",
            "best_label": "neutral_or_contextual",
            "best_score": 1.0,
        }
        for lbl in FROZEN_LABELS:
            row[f"score_{lbl}"] = float(scores.get(lbl, 0.0))

        append_row(output_csv, row, columns)
        print(f"\nSaved to CSV: {output_csv.resolve()}")
        return

    # Run model
    print(f"\n[D] Running zero-shot model: {model_name}")
    scores = classify_texts_zero_shot(texts, device=device, model_name=model_name)

    best_label = max(scores.items(), key=lambda x: x[1])[0]
    best_score = float(scores[best_label])

    print("\nText classification scores:")
    print_scores(scores)
    print(f"\nBest label: {best_label} ({best_score:.4f})")

    row = {
        "created_utc": utc_now(),
        "image_path": str(Path(image_path).resolve()),
        "media_id": media_id,
        "text_source": text_source,
        "has_text_match": True,
        "n_text_rows": len(texts),
        "mapped_text": mapped_text,
        "best_label": best_label,
        "best_score": best_score,
    }
    for lbl in FROZEN_LABELS:
        row[f"score_{lbl}"] = float(scores.get(lbl, 0.0))

    append_row(output_csv, row, columns)
    print(f"\nSaved to CSV: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
