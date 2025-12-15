"""
Case 3 finalize: route decision for image-only evidence

Case 3 is not a detector. It finalizes the decision using outputs of:
- Case 1 (OCR embedded text flag)
- Case 2 (associated SMS text flag)

Rules:
- If Case 1 or Case 2 has not run (missing flags): case_label = "incomplete"
- If embedded_text_flag == False AND associated_text_flag == False:
      case_label = "case_3_image_only" (eligible for image-only OpenCLIP stage)
- Otherwise:
      case_label = "not_case_3" (evidence has text context, so not image-only)

Usage:
  python case_3.py IMG_0346
  python case_3.py IMG_0346 image_evidence.csv

Dependencies:
  pip install pandas
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd


# -----------------------------
# Evidence CSV schema
# -----------------------------
# This extends your Case 1/2 schema with case_label + case_reason.
EVIDENCE_COLUMNS = [
    "case_id",
    "media_id",
    "file_path",
    "file_name",
    "file_hash_sha256",
    "image_width",
    "image_height",

    "thread_id",
    "anchor_event_id",
    "anchor_timestamp_utc",

    "ocr_ran",
    "embedded_text_flag",
    "ocr_text",
    "ocr_mean_confidence",
    "ocr_character_count",
    "ocr_text_coverage",

    "assoc_check_ran",
    "associated_text_flag",
    "assoc_message_count",
    "assoc_window_before_s",
    "assoc_window_after_s",
    "assoc_text_concat",
    "assoc_messages_json",


    "has_any_text_context",
    "analysis_mode",

    # Case 3 outputs
    "case_label",
    "case_reason",

    "last_updated_utc",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_evidence_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Evidence CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in EVIDENCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def truthy(val: Any) -> bool:
    """
    Handles pandas weirdness: booleans, NaN, empty strings.
    """
    if val is None:
        return False
    s = str(val).strip().lower()
    if s in ("", "nan", "none"):
        return False
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    # If it’s already a bool or numeric-like, fall back
    try:
        return bool(val)
    except Exception:
        return False


def is_missing_flag(val: Any) -> bool:
    s = str(val).strip().lower()
    return s in ("", "nan", "none")


def recompute_derived_fields(df: pd.DataFrame, idx: int) -> None:
    embedded_raw = df.at[idx, "embedded_text_flag"]
    assoc_raw = df.at[idx, "associated_text_flag"]

    embedded_known = not is_missing_flag(embedded_raw)
    assoc_known = not is_missing_flag(assoc_raw)

    embedded = truthy(embedded_raw) if embedded_known else False
    associated = truthy(assoc_raw) if assoc_known else False

    has_any = embedded or associated
    df.at[idx, "has_any_text_context"] = has_any
    df.at[idx, "analysis_mode"] = "multimodal" if has_any else "image_only"


def finalize_case3_for_media(df: pd.DataFrame, media_id: str) -> Dict[str, Any]:
    if not (df["media_id"] == media_id).any():
        raise ValueError(f"media_id {media_id} not found in image_evidence.csv")

    idx = df.index[df["media_id"] == media_id][0]

    # Ensure derived fields are consistent with flags
    recompute_derived_fields(df, idx)

    # Check that Case 1 and Case 2 were actually run
    ocr_ran = df.at[idx, "ocr_ran"]
    assoc_ran = df.at[idx, "assoc_check_ran"]

    embedded_flag = df.at[idx, "embedded_text_flag"]
    assoc_flag = df.at[idx, "associated_text_flag"]

    # Incomplete if either check wasn't run or flag missing
    if not truthy(ocr_ran) or is_missing_flag(embedded_flag):
        df.at[idx, "case_label"] = "incomplete"
        df.at[idx, "case_reason"] = "Case 1 not completed or embedded_text_flag missing."
        df.at[idx, "last_updated_utc"] = now_utc()
        return {"media_id": media_id, "case_label": "incomplete", "case_reason": df.at[idx, "case_reason"]}

    if not truthy(assoc_ran) or is_missing_flag(assoc_flag):
        df.at[idx, "case_label"] = "incomplete"
        df.at[idx, "case_reason"] = "Case 2 not completed or associated_text_flag missing."
        df.at[idx, "last_updated_utc"] = now_utc()
        return {"media_id": media_id, "case_label": "incomplete", "case_reason": df.at[idx, "case_reason"]}

    embedded = truthy(embedded_flag)
    associated = truthy(assoc_flag)

    # Case 3 applies only when image-only (no text context)
    if (not embedded) and (not associated):
        df.at[idx, "case_label"] = "case_3_image_only"
        df.at[idx, "case_reason"] = "No usable embedded OCR text and no associated SMS text. Route to image-only analysis."
    else:
        df.at[idx, "case_label"] = "not_case_3"
        df.at[idx, "case_reason"] = "Text context exists (embedded OCR or associated SMS). Not eligible for image-only case."

    df.at[idx, "last_updated_utc"] = now_utc()

    return {
        "media_id": media_id,
        "case_label": df.at[idx, "case_label"],
        "case_reason": df.at[idx, "case_reason"],
        "embedded_text_flag": embedded,
        "associated_text_flag": associated,
        "analysis_mode": df.at[idx, "analysis_mode"],
        "has_any_text_context": df.at[idx, "has_any_text_context"],
    }


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Defaults
    MEDIA_ID = "IMG_0346"
    EVIDENCE_CSV = "image_evidence.csv"

    # CLI:
    #   python case_3.py <media_id> [evidence_csv]
    if len(sys.argv) >= 2 and str(sys.argv[1]).strip():
        MEDIA_ID = sys.argv[1]
    if len(sys.argv) >= 3 and str(sys.argv[2]).strip():
        EVIDENCE_CSV = sys.argv[2]

    df = ensure_evidence_csv(EVIDENCE_CSV)
    result = finalize_case3_for_media(df, MEDIA_ID)

    # Save updates
    df.to_csv(EVIDENCE_CSV, index=False)

    print(result)
    print(f"\nUpdated evidence CSV: {EVIDENCE_CSV} (media_id={MEDIA_ID})")
