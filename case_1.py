"""
Case 1 check: "Image has usable embedded text" (OCR-trigger)
+ updates an image evidence CSV (one row per image evidence item).

This script:
- runs OCR on an image
- computes mean confidence, character length, and text coverage ratio
- applies deterministic thresholds
- prints a structured decision record
- writes/updates image_evidence.csv with Case 1 fields

Dependencies:
  pip install pytesseract pillow pandas
System requirement:
  Tesseract OCR must be installed and available on PATH.
  - Windows: install Tesseract and set pytesseract.pytesseract.tesseract_cmd if needed
"""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from PIL import Image, ImageOps

import pytesseract
from pytesseract import Output

# Windows-only: uncomment/adjust if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------
# Thresholds and OCR utilities
# -----------------------------

@dataclass
class Case1Thresholds:
    # pytesseract confidence is 0..100, and -1 for invalid
    C_min: float = 70.0
    L_min_chars: int = 8
    A_min: float = 0.005  # 0.5% of image area


def _safe_open_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Light, reproducible preprocessing
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    return gray


_noise_only_re = re.compile(r"^[\W_]+$")  # only symbols/underscores
_time_like_re = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")  # 12:34 or 12:34:56
_date_like_re = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")  # 2025-12-14


def _is_noise_token(t: str) -> bool:
    t = (t or "").strip()
    if not t:
        return True
    if _noise_only_re.match(t):
        return True
    if _time_like_re.match(t):
        return True
    if _date_like_re.match(t):
        return True
    if len(t) <= 1:
        return True
    return False


def detect_case1_embedded_text(
    image_path: str,
    thresholds: Case1Thresholds = Case1Thresholds(),
    ocr_lang: str = "eng",
    tesseract_config: str = "--oem 3 --psm 6",
) -> Dict[str, Any]:
    """
    Returns:
      {
        "embedded_text_flag": bool,
        "extracted_text": str,
        "metrics": {...},
        "thresholds": {...},
        "reason": "..."
      }
    """
    img = _safe_open_image(image_path)
    w, h = img.size
    img_work = _preprocess_for_ocr(img)

    data = pytesseract.image_to_data(
        img_work, lang=ocr_lang, config=tesseract_config, output_type=Output.DICT
    )

    tokens: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    n = len(data.get("text", []))

    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf_raw = data["conf"][i]

        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        if conf < 0:
            continue
        if _is_noise_token(text):
            continue

        x, y, bw, bh = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        tokens.append((text, conf, (x, y, bw, bh)))

    if not tokens:
        return {
            "embedded_text_flag": False,
            "extracted_text": "",
            "metrics": {
                "mean_confidence": None,
                "character_count": 0,
                "text_coverage_ratio": 0.0,
                "token_count": 0,
                "image_width": w,
                "image_height": h,
            },
            "thresholds": thresholds.__dict__,
            "reason": "No usable OCR tokens detected (after filtering).",
        }

    extracted_text = " ".join([t[0] for t in tokens]).strip()

    confidences = [t[1] for t in tokens]
    mean_conf = sum(confidences) / max(1, len(confidences))

    char_count = len(re.sub(r"\s+", "", extracted_text))
    token_count = len(tokens)

    total_text_area = 0
    for _, _, (_, _, bw, bh) in tokens:
        if bw > 0 and bh > 0:
            total_text_area += bw * bh

    img_area = max(1, w * h)
    coverage = total_text_area / img_area

    pass_conf = mean_conf >= thresholds.C_min
    pass_len = char_count >= thresholds.L_min_chars
    pass_cov = coverage >= thresholds.A_min

    embedded_flag = pass_conf and pass_len and pass_cov

    failed = []
    if not pass_conf:
        failed.append(f"mean_confidence {mean_conf:.2f} < {thresholds.C_min}")
    if not pass_len:
        failed.append(f"character_count {char_count} < {thresholds.L_min_chars}")
    if not pass_cov:
        failed.append(f"text_coverage_ratio {coverage:.6f} < {thresholds.A_min}")

    reason = "Passed OCR thresholds." if embedded_flag else ("Failed: " + "; ".join(failed))

    return {
        "embedded_text_flag": embedded_flag,
        "extracted_text": extracted_text,
        "metrics": {
            "mean_confidence": mean_conf,
            "character_count": char_count,
            "text_coverage_ratio": coverage,
            "token_count": token_count,
            "image_width": w,
            "image_height": h,
        },
        "thresholds": thresholds.__dict__,
        "reason": reason,
    }


# -----------------------------
# Evidence CSV update utilities
# -----------------------------

EVIDENCE_COLUMNS = [
    # Identification
    "case_id",
    "media_id",
    "file_path",
    "file_name",
    "file_hash_sha256",
    "image_width",
    "image_height",

    # Optional linking context (SMS thread) - can be empty now; Case 2 can fill later
    "thread_id",
    "anchor_event_id",
    "anchor_timestamp_utc",

    # Case 1 fields (OCR)
    "ocr_ran",
    "embedded_text_flag",
    "ocr_text",
    "ocr_mean_confidence",
    "ocr_character_count",
    "ocr_text_coverage",

    # Case 2 fields (associated text) - filled by the other script
    "assoc_check_ran",
    "associated_text_flag",
    "assoc_message_count",
    "assoc_window_before_s",
    "assoc_window_after_s",

    # Derived decision fields
    "has_any_text_context",
    "analysis_mode",

    # Audit
    "last_updated_utc",
]


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_evidence_csv(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Add any missing columns for forward compatibility
        for col in EVIDENCE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[EVIDENCE_COLUMNS]
        return df

    # Create empty CSV with schema
    df = pd.DataFrame(columns=EVIDENCE_COLUMNS)
    df.to_csv(csv_path, index=False)
    return df


def upsert_image_row(
    df: pd.DataFrame,
    case_id: str,
    media_id: str,
    image_path: str,
    w: int,
    h: int,
    file_hash: str,
) -> pd.DataFrame:
    if "media_id" not in df.columns:
        df["media_id"] = ""

    if (df["media_id"] == media_id).any():
        idx = df.index[df["media_id"] == media_id][0]
    else:
        idx = len(df)
        df.loc[idx] = {col: "" for col in EVIDENCE_COLUMNS}

    df.at[idx, "case_id"] = case_id
    df.at[idx, "media_id"] = media_id
    df.at[idx, "file_path"] = image_path
    df.at[idx, "file_name"] = os.path.basename(image_path)
    df.at[idx, "file_hash_sha256"] = file_hash
    df.at[idx, "image_width"] = w
    df.at[idx, "image_height"] = h

    return df


def update_case1_fields(
    df: pd.DataFrame,
    media_id: str,
    ocr_result: Dict[str, Any],
) -> pd.DataFrame:
    idx = df.index[df["media_id"] == media_id][0]

    df.at[idx, "ocr_ran"] = True
    df.at[idx, "embedded_text_flag"] = bool(ocr_result.get("embedded_text_flag", False))
    df.at[idx, "ocr_text"] = ocr_result.get("extracted_text", "")

    metrics = ocr_result.get("metrics", {}) or {}
    df.at[idx, "ocr_mean_confidence"] = metrics.get("mean_confidence")
    df.at[idx, "ocr_character_count"] = metrics.get("character_count")
    df.at[idx, "ocr_text_coverage"] = metrics.get("text_coverage_ratio")

    # Derived fields (note: associated_text_flag may be empty until Case 2 runs)
    embedded = bool(df.at[idx, "embedded_text_flag"]) if str(df.at[idx, "embedded_text_flag"]).strip() != "" else False
    associated_raw = df.at[idx, "associated_text_flag"]
    associated = bool(associated_raw) if str(associated_raw).strip() != "" else False

    has_any = embedded or associated
    df.at[idx, "has_any_text_context"] = has_any
    df.at[idx, "analysis_mode"] = "multimodal" if has_any else "image_only"

    df.at[idx, "last_updated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return df


# -----------------------------
# Main
# -----------------------------

def derive_media_id_from_path(image_path: str) -> str:
    # Example: "IMG_0346.jpg" -> "IMG_0346"
    base = os.path.basename(image_path)
    return os.path.splitext(base)[0]


if __name__ == "__main__":
    # ----------- CONFIG -----------
    CASE_ID = "CASE-2025-001"
    IMAGE_PATH = r"Case_1_data/text-image-title.png"   # <-- change this
    EVIDENCE_CSV = "image_evidence.csv"                 # output CSV

    # Optional override if you want a specific media_id
    MEDIA_ID: Optional[str] = None  # e.g., "IMG_0346"

    # OCR thresholds (tune if needed)
    THRESHOLDS = Case1Thresholds(C_min=70.0, L_min_chars=8, A_min=0.005)
    OCR_LANG = "eng"
    TESS_CONFIG = "--oem 3 --psm 6"
    # ------------------------------

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    media_id = MEDIA_ID or derive_media_id_from_path(IMAGE_PATH)

    # Run OCR Case 1 check
    result = detect_case1_embedded_text(
        IMAGE_PATH,
        thresholds=THRESHOLDS,
        ocr_lang=OCR_LANG,
        tesseract_config=TESS_CONFIG,
    )
    print(result)

    # Prepare evidence CSV and upsert row
    df = ensure_evidence_csv(EVIDENCE_CSV)

    img = _safe_open_image(IMAGE_PATH)
    w, h = img.size
    file_hash = sha256_file(IMAGE_PATH)

    df = upsert_image_row(
        df=df,
        case_id=CASE_ID,
        media_id=media_id,
        image_path=IMAGE_PATH,
        w=w,
        h=h,
        file_hash=file_hash,
    )

    # Update Case 1 fields
    df = update_case1_fields(df, media_id=media_id, ocr_result=result)

    # Save CSV
    df.to_csv(EVIDENCE_CSV, index=False)
    print(f"\nUpdated evidence CSV: {EVIDENCE_CSV} (media_id={media_id})")
