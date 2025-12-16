from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd


# =============================
# CONFIG
# =============================
IMAGE_EVIDENCE_CSV = "image_evidence.csv"
CLIP_CSV = "openclip_outputs.csv"
TEXT_CSV = "text_evidence.csv"
OUT_CSV = "multimodal_fused_decisions.csv"

# --- Text source column in text_evidence.csv ---
TEXT_SOURCE_COL = "text_source"  # must be 'assoc' or 'ocr' per row
ASSOC_VALUE = "assoc"
OCR_VALUE = "ocr"

# --- Frozen label set (must match your text classifier outputs: score_<label>) ---
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

# --- Weights (v2 policy you agreed with) ---
W_IMG = 1.0
W_ASSOC = 1.2
W_OCR = 1.0

# --- Review logic (conservative triage) ---
RISK_THRESHOLD = 0.25
HIGH_RISK_LABELS = {
    "threat_of_violence",
    "incitement_or_endorsement_of_violence",
    "sexual_violence_or_exploitation",
    "self_harm_or_suicide_risk",
    "criminal_admission_or_description",
    "hate_or_bias_based_content",
}

# --- CLIP → Frozen mapping ---
# If your CLIP prompt classes ALREADY equal frozen labels, leave this empty {} and we will use identity.
# Otherwise, fill this mapping with your prompt-bank class names as keys.
CLIP_TO_FROZEN: Dict[str, str] = {
    # Examples (edit to match your prompt_bank.json classes):
    # "weapon_or_armed_person": "weapon_related_text",
    # "violence_or_gore": "incitement_or_endorsement_of_violence",
    # "threat_or_intimidation": "threat_of_violence",
    # "hate_symbol_or_extremist": "hate_or_bias_based_content",
    # "self_harm": "self_harm_or_suicide_risk",
    # "sexual_violence": "sexual_violence_or_exploitation",
}


# =============================
# Helpers
# =============================
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def normalize_case_label(x: Any) -> str:
    return safe_str(x).strip().lower()


def infer_case_type(case_label: str) -> str:
    """
    Return one of: assoc_text, ocr_text, no_text, unknown
    Supports common variants.
    """
    cl = normalize_case_label(case_label)

    # numeric / compact
    if cl in {"case_1", "case1", "1"}:
        return "ocr_text"
    if cl in {"case_2", "case2", "2"}:
        return "assoc_text"
    if cl in {"case_3", "case3", "3"}:
        return "no_text"

    # keyword style
    if any(k in cl for k in ["assoc", "associated", "caption", "report_text", "message_text", "post_text"]):
        return "assoc_text"
    if any(k in cl for k in ["ocr", "embedded", "in_image_text", "meme_text"]):
        return "ocr_text"
    if any(k in cl for k in ["no_text", "notext", "image_only", "none"]):
        return "no_text"

    return "unknown"


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default


def latest_per_media(df: pd.DataFrame, media_col: str = "media_id", time_col: str = "created_utc") -> pd.DataFrame:
    if df.empty:
        return df
    if media_col not in df.columns:
        raise ValueError(f"CSV missing '{media_col}'")
    if time_col in df.columns:
        tmp = df.copy()
        tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce", utc=True)
        tmp = tmp.sort_values([media_col, time_col], ascending=[True, True])
        return tmp.drop_duplicates(subset=[media_col], keep="last").copy()
    return df.drop_duplicates(subset=[media_col], keep="last").copy()


def latest_text_by_source(
    text_df: pd.DataFrame,
    media_id: str,
    source_value: str,
    time_col: str = "created_utc",
) -> Optional[pd.Series]:
    df = text_df
    if df.empty:
        return None
    if "media_id" not in df.columns:
        raise ValueError("text_evidence.csv missing 'media_id' column")
    if TEXT_SOURCE_COL not in df.columns:
        raise ValueError(
            f"text_evidence.csv must contain '{TEXT_SOURCE_COL}' with values '{ASSOC_VALUE}' and '{OCR_VALUE}'. "
            f"Found columns: {list(df.columns)}"
        )

    sub = df[(df["media_id"].astype(str) == str(media_id)) & (df[TEXT_SOURCE_COL].astype(str) == source_value)].copy()
    if sub.empty:
        return None
    if time_col in sub.columns:
        sub[time_col] = pd.to_datetime(sub[time_col], errors="coerce", utc=True)
        sub = sub.sort_values(time_col, ascending=True)
    return sub.iloc[-1]


def extract_text_scores(row: Optional[pd.Series]) -> Dict[str, float]:
    scores = {lbl: 0.0 for lbl in FROZEN_LABELS}
    if row is None:
        return scores
    for lbl in FROZEN_LABELS:
        col = f"score_{lbl}"
        if col in row.index:
            scores[lbl] = to_float(row.get(col), 0.0)
    return scores


def best_from_scores(scores: Dict[str, float]) -> Tuple[str, float]:
    if not scores:
        return "unknown", 0.0
    lbl = max(scores.items(), key=lambda x: x[1])[0]
    return lbl, float(scores[lbl])


def parse_clip_probs_json(x: Any) -> Dict[str, float]:
    try:
        s = safe_str(x)
        if not s:
            return {}
        d = json.loads(s)
        return {str(k): float(v) for k, v in d.items()}
    except Exception:
        return {}


def clip_to_frozen_scores(
    clip_row: Optional[pd.Series],
) -> Dict[str, float]:
    """
    Convert CLIP outputs to frozen-label scores.
    Preferred: use probs_json dict if present.

    Strategy:
      - If probs_json exists:
          map each clip_class -> frozen_label
          aggregate by MAX into frozen space
      - Else:
          fall back to top_class/top_prob (only one label gets a score)
    """
    frozen = {lbl: 0.0 for lbl in FROZEN_LABELS}
    if clip_row is None:
        return frozen

    clip_probs = {}
    if "probs_json" in clip_row.index:
        clip_probs = parse_clip_probs_json(clip_row.get("probs_json"))

    # If CLIP probs are already in frozen labels (identity)
    # we treat keys that match frozen labels directly.
    if clip_probs:
        for k, v in clip_probs.items():
            k_norm = str(k)

            if k_norm in FROZEN_LABELS:
                frozen[k_norm] = max(frozen[k_norm], float(v))
                continue

            if CLIP_TO_FROZEN:
                mapped = CLIP_TO_FROZEN.get(k_norm)
                if mapped and mapped in frozen:
                    frozen[mapped] = max(frozen[mapped], float(v))

        return frozen

    # Fallback: only top-1 available
    top_class = safe_str(clip_row.get("top_class"))
    top_prob = to_float(clip_row.get("top_prob"), 0.0)

    if top_class in FROZEN_LABELS:
        frozen[top_class] = max(frozen[top_class], top_prob)
    elif CLIP_TO_FROZEN and top_class in CLIP_TO_FROZEN:
        mapped = CLIP_TO_FROZEN[top_class]
        if mapped in frozen:
            frozen[mapped] = max(frozen[mapped], top_prob)

    return frozen


def any_high_risk(scores: Dict[str, float]) -> bool:
    return any(scores.get(lbl, 0.0) >= RISK_THRESHOLD for lbl in HIGH_RISK_LABELS)


def weighted_fuse(
    img_scores: Dict[str, float],
    assoc_scores: Dict[str, float],
    ocr_scores: Dict[str, float],
    w_img: float,
    w_assoc: float,
    w_ocr: float,
) -> Dict[str, float]:
    fused = {}
    denom = w_img + w_assoc + w_ocr
    if denom <= 0:
        return {lbl: 0.0 for lbl in FROZEN_LABELS}

    for lbl in FROZEN_LABELS:
        fused[lbl] = (
            w_img * img_scores.get(lbl, 0.0) +
            w_assoc * assoc_scores.get(lbl, 0.0) +
            w_ocr * ocr_scores.get(lbl, 0.0)
        ) / denom
    return fused


# =============================
# Main
# =============================
def main() -> None:
    # python fusion_runner.py [image_evidence.csv] [openclip_outputs.csv] [text_evidence.csv] [out.csv]
    image_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(IMAGE_EVIDENCE_CSV)
    clip_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path(CLIP_CSV)
    text_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path(TEXT_CSV)
    out_path = Path(sys.argv[4]) if len(sys.argv) >= 5 else Path(OUT_CSV)

    img_df = read_csv(image_path)
    clip_df = read_csv(clip_path)
    text_df = read_csv(text_path)

    if "media_id" not in img_df.columns:
        raise ValueError("image_evidence.csv must contain 'media_id'")
    if "case_label" not in img_df.columns:
        raise ValueError("image_evidence.csv must contain 'case_label' (required for routing)")

    # Reduce CLIP to latest per media_id (append-only file)
    clip_latest = latest_per_media(clip_df, "media_id", "created_utc")
    clip_latest_indexed = clip_latest.set_index("media_id", drop=False)

    out_rows: List[Dict[str, Any]] = []

    for _, r in img_df.iterrows():
        media_id = safe_str(r.get("media_id"))
        case_label = safe_str(r.get("case_label"))
        case_type = infer_case_type(case_label)

        # Always image evidence universe only
        clip_row = None
        if media_id in clip_latest_indexed.index:
            clip_row = clip_latest_indexed.loc[media_id]

        # Pull latest assoc and OCR rows from the combined text_evidence.csv
        assoc_row = latest_text_by_source(text_df, media_id, ASSOC_VALUE)
        ocr_row = latest_text_by_source(text_df, media_id, OCR_VALUE)

        # Scores per modality (in frozen label space)
        img_scores = clip_to_frozen_scores(clip_row)
        assoc_scores = extract_text_scores(assoc_row)
        ocr_scores = extract_text_scores(ocr_row)

        # Case-label controlled modality participation
        w_img = W_IMG if clip_row is not None else 0.0

        if case_type == "no_text":
            w_assoc = 0.0
            w_ocr = 0.0
        else:
            # Allow both if available; case_type influences which is expected but does not block if both exist
            w_assoc = W_ASSOC if assoc_row is not None else 0.0
            w_ocr = W_OCR if ocr_row is not None else 0.0

        fused_scores = weighted_fuse(img_scores, assoc_scores, ocr_scores, w_img, w_assoc, w_ocr)
        final_label, final_score = best_from_scores(fused_scores)

        needs_review = any_high_risk(fused_scores)

        # Per-modality best (for audit)
        img_best_label, img_best_score = best_from_scores(img_scores)
        assoc_best_label, assoc_best_score = best_from_scores(assoc_scores)
        ocr_best_label, ocr_best_score = best_from_scores(ocr_scores)

        assoc_text = safe_str(assoc_row.get("mapped_text")) if assoc_row is not None and "mapped_text" in assoc_row.index else ""
        ocr_text = safe_str(ocr_row.get("mapped_text")) if ocr_row is not None and "mapped_text" in ocr_row.index else ""

        row_out: Dict[str, Any] = {
            "media_id": media_id,
            "case_label": case_label,
            "case_type": case_type,
            "has_clip": bool(clip_row is not None),
            "has_assoc_text_scores": bool(assoc_row is not None),
            "has_ocr_text_scores": bool(ocr_row is not None),
            "w_img": float(w_img),
            "w_assoc": float(w_assoc),
            "w_ocr": float(w_ocr),
            "final_label": final_label,
            "final_score": float(final_score),
            "needs_review": bool(needs_review),
            "clip_best_label": img_best_label,
            "clip_best_score": float(img_best_score),
            "assoc_best_label": assoc_best_label,
            "assoc_best_score": float(assoc_best_score),
            "ocr_best_label": ocr_best_label,
            "ocr_best_score": float(ocr_best_score),
            "assoc_text": assoc_text,
            "ocr_text": ocr_text,
        }

        # Save fused per-label scores as columns for transparency
        for lbl in FROZEN_LABELS:
            row_out[f"fused_{lbl}"] = float(fused_scores.get(lbl, 0.0))

        out_rows.append(row_out)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)
    print(f"Saved fused decisions: {out_path.resolve()} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
