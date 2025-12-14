"""
Case 1 check: "Image has usable embedded text" (OCR-trigger)

This script:
- runs OCR on an image
- computes mean confidence, character/word length, and text coverage ratio
- applies deterministic thresholds
- returns a structured decision record

Dependencies:
  pip install pytesseract pillow
System requirement:
  Tesseract OCR must be installed on your OS and available on PATH.
  - Windows: install Tesseract and set pytesseract.pytesseract.tesseract_cmd if needed
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from PIL import Image, ImageOps
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    import pytesseract
    from pytesseract import Output
except ImportError as e:
    raise ImportError(
        "Missing dependency. Install with: pip install pytesseract"
    ) from e


@dataclass
class Case1Thresholds:
    # Conservative defaults (tune as needed)
    C_min: float = 70.0      # pytesseract conf is 0..100, and -1 for invalid 
                              # OCR confidence is normalized or treated on a 0–100 scale
                              # Threshold C_min = 70 means “70 out of 100”
    L_min_chars: int = 8     # min character count in extracted text
    A_min: float = 0.005     # 0.5% of image area (text coverage in image)


def _safe_open_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    # For OCR, RGB or L works; we normalize here
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Light, non-destructive preprocessing that helps OCR in many cases
    gray = ImageOps.grayscale(img)
    # Optional: increase contrast a bit; keep it simple and reproducible
    gray = ImageOps.autocontrast(gray)
    return gray


_noise_only_re = re.compile(r"^[\W_]+$")  # only symbols/underscores
_time_like_re = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")  # 12:34 or 12:34:56
_date_like_re = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")  # 2025-12-14


def _is_noise_token(t: str) -> bool:
    t = t.strip()
    if not t:
        return True
    if _noise_only_re.match(t):
        return True
    if _time_like_re.match(t):
        return True
    if _date_like_re.match(t):
        return True
    # Single short UI-ish tokens are often noise; keep minimal
    if len(t) <= 1:
        return True
    return False


def detect_case1_embedded_text(
    image_path: str,
    thresholds: Case1Thresholds = Case1Thresholds(),
    ocr_lang: str = "eng",
) -> Dict[str, Any]:
    """
    Returns a record:
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

    # Run OCR in "data" mode to get confidences and bounding boxes
    data = pytesseract.image_to_data(img_work, lang=ocr_lang, output_type=Output.DICT)

    # Extract tokens with conf and boxes
    tokens: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    n = len(data.get("text", []))

    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf_raw = data["conf"][i]

        # conf can be '-1' strings; handle safely
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        if conf < 0:
            continue
        if _is_noise_token(text):
            continue

        x, y, bw, bh = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
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

    # Metrics
    confidences = [t[1] for t in tokens]
    mean_conf = sum(confidences) / len(confidences)

    char_count = len(re.sub(r"\s+", "", extracted_text))  # exclude whitespace
    token_count = len(tokens)

    # Coverage: sum of bounding box areas / image area (naive sum; acceptable for heuristic)
    total_text_area = 0
    for _, _, (x, y, bw, bh) in tokens:
        if bw > 0 and bh > 0:
            total_text_area += bw * bh

    img_area = max(1, w * h)
    coverage = total_text_area / img_area

    # Apply thresholds
    pass_conf = mean_conf >= thresholds.C_min
    pass_len = char_count >= thresholds.L_min_chars
    pass_cov = coverage >= thresholds.A_min

    embedded_flag = pass_conf and pass_len and pass_cov

    # Reason string (for auditability)
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


if __name__ == "__main__":
    # Example usage (replace with your image path)
    img_path = "Case_1_data/text-image-title.png"
    result = detect_case1_embedded_text(img_path)
    print(result)
