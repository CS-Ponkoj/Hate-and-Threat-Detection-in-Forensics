"""
Case 3 (Best Practice Finalizer): Image-only classification

What this script does
- Reads image_evidence.csv (one row per media_id)
- Ensures required columns exist (adds them if missing)
- Ensures the target media_id row exists (creates a minimal row if missing)
- Determines Case 3 only when BOTH Case 1 and Case 2 have completed
- Updates: case_label, case_reason, has_any_text_context, analysis_mode, last_updated_utc

Case 3 definition (strict, forensic-safe)
- OCR ran AND embedded_text_flag == False
- Association check ran AND associated_text_flag == False

If checks are incomplete: case_label = "incomplete"
If any text channel exists: case_label = "not_case_3"
If both channels absent: case_label = "case_3_image_only"

Dependencies:
  pip install pandas
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pandas as pd


@dataclass
class Case3Config:
    # Forensic best practice: do not label Case 3 unless both checks ran.
    require_both_checks: bool = True


# Columns we expect to exist in image_evidence.csv for this pipeline.
# If your earlier scripts already create these, this script will just reuse them.
REQUIRED_COLUMNS = [
    "case_id",
    "media_id",

    # Case 1 fields
    "ocr_ran",
    "embedded_text_flag",

    # Case 2 fields
    "assoc_check_ran",
    "associated_text_flag",

    # Derived / final
    "has_any_text_context",
    "analysis_mode",
    "last_updated_utc",

    # Case 3 labeling
    "case_label",
    "case_reason",
]


def _now_utc_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_bool_safe(x) -> Optional[bool]:
    """
    Converts common CSV representations to bool.
    Returns None if empty/unknown.
    """
    if x is None:
        return None

    # pandas may store actual booleans
    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()
    if s == "":
        return None
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    # Keep columns in a stable order (required first, then any extra user columns)
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    df = df[REQUIRED_COLUMNS + extra_cols]
    return df


def ensure_row(df: pd.DataFrame, media_id: str, case_id: str = "") -> pd.DataFrame:
    """
    Best practice: never crash just because the row doesn't exist.
    Create a minimal row so the pipeline can proceed and mark it incomplete.
    """
    if not (df["media_id"].astype(str) == str(media_id)).any():
        idx = len(df)
        # Create an empty row for all columns
        df.loc[idx] = {col: "" for col in df.columns}
        df.at[idx, "media_id"] = media_id
        if case_id:
            df.at[idx, "case_id"] = case_id
        df.at[idx, "case_label"] = "incomplete"
        df.at[idx, "case_reason"] = "Row created by Case 3 finalizer. Run Case 1 and Case 2 first."
        df.at[idx, "last_updated_utc"] = _now_utc_z()
    return df


def finalize_case3_for_media(df: pd.DataFrame, media_id: str, cfg: Case3Config) -> Dict[str, Any]:
    df = ensure_columns(df)
    df = ensure_row(df, media_id)

    idx = df.index[df["media_id"].astype(str) == str(media_id)][0]

    ocr_ran = to_bool_safe(df.at[idx, "ocr_ran"])
    embedded = to_bool_safe(df.at[idx, "embedded_text_flag"])

    assoc_ran = to_bool_safe(df.at[idx, "assoc_check_ran"])
    associated = to_bool_safe(df.at[idx, "associated_text_flag"])

    # Completeness check
    if cfg.require_both_checks:
        complete = (ocr_ran is True) and (assoc_ran is True) and (embedded is not None) and (associated is not None)
    else:
        # Still require the flags to be known before labeling
        complete = (embedded is not None) and (associated is not None)

    if not complete:
        df.at[idx, "case_label"] = "incomplete"
        df.at[idx, "case_reason"] = (
            "Cannot label Case 3 yet. Require: ocr_ran==True and assoc_check_ran==True with known flags."
        )

        # Keep derived fields consistent even in incomplete state (best effort)
        has_any = (embedded is True) or (associated is True)
        df.at[idx, "has_any_text_context"] = bool(has_any)
        df.at[idx, "analysis_mode"] = "multimodal" if has_any else (df.at[idx, "analysis_mode"] or "")
        df.at[idx, "last_updated_utc"] = _now_utc_z()

        return {
            "media_id": media_id,
            "case_label": "incomplete",
            "analysis_mode": df.at[idx, "analysis_mode"],
            "has_any_text_context": df.at[idx, "has_any_text_context"],
            "reason": df.at[idx, "case_reason"],
            "ocr_ran": ocr_ran,
            "embedded_text_flag": embedded,
            "assoc_check_ran": assoc_ran,
            "associated_text_flag": associated,
        }

    # Final Case 3 decision
    is_case3 = (embedded is False) and (associated is False)

    if is_case3:
        case_label = "case_3_image_only"
        case_reason = "OCR ran and found no usable embedded text; association check ran and found no associated SMS text."
        analysis_mode = "image_only"
        has_any = False
    else:
        case_label = "not_case_3"
        case_reason = "At least one text channel is available (embedded OCR text or associated SMS text)."
        analysis_mode = "multimodal"
        has_any = True

    df.at[idx, "case_label"] = case_label
    df.at[idx, "case_reason"] = case_reason
    df.at[idx, "has_any_text_context"] = has_any
    df.at[idx, "analysis_mode"] = analysis_mode
    df.at[idx, "last_updated_utc"] = _now_utc_z()

    return {
        "media_id": media_id,
        "case_label": case_label,
        "analysis_mode": analysis_mode,
        "has_any_text_context": has_any,
        "reason": case_reason,
        "ocr_ran": ocr_ran,
        "embedded_text_flag": embedded,
        "assoc_check_ran": assoc_ran,
        "associated_text_flag": associated,
    }


def main():
    # ------------- CONFIG -------------
    IMAGE_EVIDENCE_CSV = "image_evidence.csv"
    MEDIA_ID = "text-image-title6"     # <-- change this
    CASE_ID = "CASE-2025-001"          # optional (used if row must be created)
    CFG = Case3Config(require_both_checks=True)
    # ----------------------------------

    # Load or create evidence CSV
    if os.path.exists(IMAGE_EVIDENCE_CSV):
        df = pd.read_csv(IMAGE_EVIDENCE_CSV)
    else:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = ensure_columns(df)
    df = ensure_row(df, media_id=MEDIA_ID, case_id=CASE_ID)

    result = finalize_case3_for_media(df, MEDIA_ID, cfg=CFG)
    print(result)

    df.to_csv(IMAGE_EVIDENCE_CSV, index=False)
    print(f"\nUpdated evidence CSV: {IMAGE_EVIDENCE_CSV} (media_id={MEDIA_ID})")


if __name__ == "__main__":
    main()
