from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


# -----------------------------
# Inputs / Outputs
# -----------------------------
DEFAULT_IMAGE_EVIDENCE = "image_evidence.csv"
DEFAULT_CLIP_OUTPUTS = "openclip_outputs.csv"
DEFAULT_TEXT_EVIDENCE = "text_evidence.csv"
DEFAULT_OUT = "multimodal_decisions.csv"

# -----------------------------
# Decision thresholds (v1)
# -----------------------------
TEXT_RISK_THRESHOLD = 0.25
CLIP_RISK_THRESHOLD = 0.20

HIGH_RISK_TEXT_LABELS = {
    "threat_of_violence",
    "incitement_or_endorsement_of_violence",
    "sexual_violence_or_exploitation",
    "self_harm_or_suicide_risk",
    "criminal_admission_or_description",
    "hate_or_bias_based_content",
}

# CLIP prompt-bank classes you treat as concerning (adjust to your prompt_bank.json)
HIGH_RISK_CLIP_CLASSES = {
    "hate_or_extremist_ideology",
    "threat_or_intimidation",
    "violence_or_gore",
    "weapon_or_armed_person",
    "criminal_activity",
    "sexual_content",
    "self_harm",
    "obscene_or_insulting_content",
}


# -----------------------------
# Helpers
# -----------------------------
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def latest_per_media(df: pd.DataFrame, media_col: str, time_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    if time_col not in df.columns:
        # if no time column, keep last row per media_id as-is
        return df.drop_duplicates(subset=[media_col], keep="last").copy()
    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce", utc=True)
    tmp = tmp.sort_values([media_col, time_col], ascending=[True, True])
    return tmp.drop_duplicates(subset=[media_col], keep="last").copy()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def parse_probs_json(s: Any) -> Dict[str, float]:
    try:
        if s is None:
            return {}
        txt = str(s)
        if not txt.strip():
            return {}
        d = json.loads(txt)
        return {str(k): float(v) for k, v in d.items()}
    except Exception:
        return {}


def any_text_risk(scores: Dict[str, float]) -> bool:
    for lbl in HIGH_RISK_TEXT_LABELS:
        if float(scores.get(f"score_{lbl}", 0.0)) >= TEXT_RISK_THRESHOLD:
            return True
    return False


def any_clip_risk(top_class: str, top_prob: float, triage: str) -> bool:
    if safe_str(triage).lower() == "review":
        return True
    if top_class in HIGH_RISK_CLIP_CLASSES and float(top_prob) >= CLIP_RISK_THRESHOLD:
        return True
    return False


def build_text_score_dict(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in row.index:
        if str(k).startswith("score_"):
            try:
                out[k] = float(row[k])
            except Exception:
                out[k] = 0.0
    return out


# -----------------------------
# Main fusion
# -----------------------------
def main() -> None:
    # CLI:
    # python fuse_multimodal_decisions.py [image_evidence.csv] [openclip_outputs.csv] [text_evidence.csv] [out.csv]
    image_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(DEFAULT_IMAGE_EVIDENCE)
    clip_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path(DEFAULT_CLIP_OUTPUTS)
    text_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path(DEFAULT_TEXT_EVIDENCE)
    out_path = Path(sys.argv[4]) if len(sys.argv) >= 5 else Path(DEFAULT_OUT)

    img = read_csv(image_path)
    clip = read_csv(clip_path)
    txt = read_csv(text_path)

    # Only image evidences drive the output
    if "media_id" not in img.columns:
        raise ValueError("image_evidence.csv must include 'media_id'")

    # Keep latest row per media_id from CLIP + text outputs (append-only files)
    clip_latest = latest_per_media(clip, "media_id", "created_utc")
    txt_latest = latest_per_media(txt, "media_id", "created_utc")

    # Merge
    df = img.merge(
        clip_latest[["media_id", "top_class", "top_prob", "triage", "probs_json", "created_utc"]].rename(
            columns={"created_utc": "clip_created_utc", "probs_json": "clip_probs_json"}
        ),
        on="media_id",
        how="left",
    ).merge(
        txt_latest,
        on="media_id",
        how="left",
        suffixes=("", "_text"),
    )

    # Build decision rows
    out_rows: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        media_id = safe_str(r.get("media_id"))
        file_path = safe_str(r.get("file_path"))
        file_name = safe_str(r.get("file_name"))

        # From image_evidence.csv
        embedded_text_flag = bool(r.get("embedded_text_flag")) if pd.notna(r.get("embedded_text_flag")) else False
        ocr_text = safe_str(r.get("ocr_text"))
        has_ocr_text = embedded_text_flag and bool(ocr_text.strip())

        associated_text_flag = bool(r.get("associated_text_flag")) if pd.notna(r.get("associated_text_flag")) else False
        assoc_text_concat = safe_str(r.get("assoc_text_concat"))
        has_assoc_text_from_cases = associated_text_flag and bool(assoc_text_concat.strip())

        # From text_evidence.csv (already mapped to media_id)
        has_text_match = bool(r.get("has_text_match")) if pd.notna(r.get("has_text_match")) else False
        mapped_text = safe_str(r.get("mapped_text"))

        has_assoc_text = has_text_match or has_assoc_text_from_cases or bool(mapped_text.strip())

        # From CLIP
        clip_top_class = safe_str(r.get("top_class"))
        clip_top_prob = float(r.get("top_prob")) if pd.notna(r.get("top_prob")) else 0.0
        clip_triage = safe_str(r.get("triage"))
        clip_probs = parse_probs_json(r.get("clip_probs_json"))

        # Text scores
        text_best_label = safe_str(r.get("best_label"))
        text_best_score = float(r.get("best_score")) if pd.notna(r.get("best_score")) else 0.0
        text_scores = build_text_score_dict(r)

        # Decide source of final decision
        if has_assoc_text:
            decision_source = "image_plus_associated_text"
            final_label = text_best_label if text_best_label else "neutral_or_contextual"
            final_score = text_best_score
            needs_review = (final_label != "neutral_or_contextual") or any_text_risk(text_scores)
        else:
            decision_source = "image_only"
            final_label = clip_top_class if clip_top_class else "unknown"
            final_score = clip_top_prob
            needs_review = any_clip_risk(clip_top_class, clip_top_prob, clip_triage)

        # OCR note (we are not classifying OCR-only text in this fusion script)
        ocr_text_unscored = bool(has_ocr_text and not has_assoc_text and (not text_best_label))

        out_rows.append(
            {
                "media_id": media_id,
                "file_path": file_path,
                "file_name": file_name,
                "has_assoc_text": bool(has_assoc_text),
                "has_ocr_text": bool(has_ocr_text),
                "ocr_text_unscored": bool(ocr_text_unscored),
                "decision_source": decision_source,
                "final_label": final_label,
                "final_score": float(final_score),
                "needs_review": bool(needs_review),
                # keep provenance
                "text_best_label": text_best_label,
                "text_best_score": float(text_best_score),
                "mapped_text": mapped_text,
                "clip_top_class": clip_top_class,
                "clip_top_prob": float(clip_top_prob),
                "clip_triage": clip_triage,
                # optional JSON fields for audit
                "clip_probs_json": json.dumps(clip_probs, ensure_ascii=False) if clip_probs else "",
            }
        )

    out_df = pd.DataFrame(out_rows)

    out_df.to_csv(out_path, index=False)
    print(f"Saved fused decisions CSV: {out_path.resolve()}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
