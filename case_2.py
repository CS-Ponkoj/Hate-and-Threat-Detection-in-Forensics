"""
Case 2 check: "Image has associated nearby text" (SMS thread context)

This script:
- loads an SMS export timeline CSV (events)
- finds the image event row for a given media_id (anchor)
- searches for nearby TEXT events within a time window in the same thread
- updates image_evidence.csv (one row per media_id) with Case 2 fields

Usage:
  python case_2.py IMG_0346
  python case_2.py IMG_0346 Case_2_data/forensic_multimodal_evidence.csv

Dependencies:
  pip install pandas
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


# -----------------------------
# Config + helpers
# -----------------------------

@dataclass
class AssocTextConfig:
    window_before_s: int = 120
    window_after_s: int = 120
    exclude_system_like: bool = True
    drop_text_exact: Tuple[str, ...] = (
        "",
        "sent an attachment",
        "photo",
        "image",
    )


def _parse_utc(ts: str) -> datetime:
    # Accepts "2025-01-12T14:23:11Z"
    if (ts or "").endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def is_meaningful_text(text: str, cfg: AssocTextConfig) -> bool:
    t = _norm(text)
    if t in cfg.drop_text_exact:
        return False
    if len(t) <= 1:
        return False
    return True


def _truthy(v: Any) -> bool:
    """Robust bool parsing for values read from CSV."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n", "", "none", "nan"):
        return False
    # fallback: non-empty string treated as True is dangerous; default False
    return False


# -----------------------------
# Load SMS events + detection
# -----------------------------

def load_events(csv_path: str) -> List[Dict[str, str]]:
    events: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)
    return events


def find_image_anchor(events: List[Dict[str, str]], media_id: str) -> Dict[str, Any]:
    """
    Finds the image event row for media_id.
    Expected columns in SMS timeline CSV:
      event_type, media_id, thread_id, event_id, utc_timestamp, sender_id
    """
    m = _norm(media_id)
    for row in events:
        if _norm(row.get("event_type", "")) == "image" and _norm(row.get("media_id", "")) == m:
            ts = _parse_utc(row["utc_timestamp"])
            return {
                "thread_id": row.get("thread_id", ""),
                "anchor_event_id": row.get("event_id", ""),
                "anchor_timestamp": ts,
                "anchor_sender_id": row.get("sender_id", ""),
                "row": row,
            }
    raise ValueError(f"Image event for media_id={media_id} not found in SMS CSV: {media_id}")


def detect_associated_text_sms(
    sms_csv_path: str,
    media_id: str,
    cfg: AssocTextConfig = AssocTextConfig(),
) -> Dict[str, Any]:
    """
    Returns:
      {
        media_id, thread_id, anchor_event_id, anchor_timestamp_utc,
        associated_text_flag, associated_messages (list),
        window_before_s, window_after_s, reason
      }
    """
    events = load_events(sms_csv_path)
    try:
         anchor = find_image_anchor(events, media_id)
    except ValueError:
        # media_id not present in SMS export (common) -> treat as no associated text
      return {
        "media_id": media_id,
        "thread_id": "",
        "anchor_event_id": "",
        "anchor_timestamp_utc": "",
        "assoc_check_ran": True,
        "associated_text_flag": False,
        "assoc_message_count": 0,
        "associated_messages": [],
        "window_before_s": cfg.window_before_s,
        "window_after_s": cfg.window_after_s,
        "reason": "No anchor image event found in SMS export for this media_id; treating as no associated text.",
      }

    thread_id = anchor["thread_id"]
    t0 = anchor["anchor_timestamp"]

    t_min = t0.timestamp() - cfg.window_before_s
    t_max = t0.timestamp() + cfg.window_after_s

    candidates: List[Dict[str, Any]] = []

    for row in events:
        if row.get("thread_id", "") != thread_id:
            continue

        if _norm(row.get("event_type", "")) != "text":
            continue

        if cfg.exclude_system_like and _norm(row.get("sender_id", "")) in ("system",):
            continue

        try:
            t = _parse_utc(row["utc_timestamp"]).timestamp()
        except Exception:
            continue

        if not (t_min <= t <= t_max):
            continue

        msg_text = row.get("message_text", "")
        if not is_meaningful_text(msg_text, cfg):
            continue

        candidates.append(
            {
                "event_id": row.get("event_id", ""),
                "utc_timestamp": row.get("utc_timestamp", ""),
                "sender_id": row.get("sender_id", ""),
                "message_text": msg_text,
            }
        )

    associated_flag = len(candidates) > 0
    reason = (
        f"Found {len(candidates)} text message(s) within ±{cfg.window_before_s}/{cfg.window_after_s} seconds."
        if associated_flag
        else f"No meaningful text messages found within ±{cfg.window_before_s}/{cfg.window_after_s} seconds."
    )

    return {
        "media_id": media_id,
        "thread_id": thread_id,
        "anchor_event_id": anchor["anchor_event_id"],
        "anchor_timestamp_utc": anchor["anchor_timestamp"].isoformat().replace("+00:00", "Z"),
        "associated_text_flag": associated_flag,
        "associated_messages": candidates,
        "window_before_s": cfg.window_before_s,
        "window_after_s": cfg.window_after_s,
        "reason": reason,
    }


# -----------------------------
# Evidence CSV update (matches case_1.py schema + adds text fields)
# -----------------------------

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

    # NEW (your requirements)
    "assoc_text_concat",
    "assoc_messages_json",

    "has_any_text_context",
    "analysis_mode",
    "last_updated_utc",

    "case_label",
    "case_reason"
]


def ensure_evidence_csv(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for col in EVIDENCE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[EVIDENCE_COLUMNS]

    df = pd.DataFrame(columns=EVIDENCE_COLUMNS)
    df.to_csv(csv_path, index=False)
    return df


def upsert_min_row_if_missing(df: pd.DataFrame, media_id: str) -> pd.DataFrame:
    if "media_id" not in df.columns:
        df["media_id"] = ""
    if (df["media_id"] == media_id).any():
        return df
    idx = len(df)
    df.loc[idx] = {col: "" for col in EVIDENCE_COLUMNS}
    df.at[idx, "media_id"] = media_id
    return df


def recompute_derived_fields(df: pd.DataFrame, idx: int):
    embedded_raw = df.at[idx, "embedded_text_flag"]
    assoc_raw = df.at[idx, "associated_text_flag"]

    embedded = _truthy(embedded_raw)
    associated = _truthy(assoc_raw)

    has_any = embedded or associated
    df.at[idx, "has_any_text_context"] = has_any
    df.at[idx, "analysis_mode"] = "multimodal" if has_any else "image_only"
    df.at[idx, "last_updated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def update_case2_fields(
    evidence_csv: str,
    result: Dict[str, Any],
) -> None:
    df = ensure_evidence_csv(evidence_csv)
    media_id = result["media_id"]
    df = upsert_min_row_if_missing(df, media_id)

    idx = df.index[df["media_id"] == media_id][0]

    # Link context
    df.at[idx, "thread_id"] = result.get("thread_id", "")
    df.at[idx, "anchor_event_id"] = result.get("anchor_event_id", "")
    df.at[idx, "anchor_timestamp_utc"] = result.get("anchor_timestamp_utc", "")

    # Case 2 fields
    messages = result.get("associated_messages", []) or []

    df.at[idx, "assoc_check_ran"] = True
    df.at[idx, "associated_text_flag"] = bool(result.get("associated_text_flag", False))
    df.at[idx, "assoc_message_count"] = len(messages)
    df.at[idx, "assoc_window_before_s"] = int(result.get("window_before_s", 0))
    df.at[idx, "assoc_window_after_s"] = int(result.get("window_after_s", 0))

    # NEW: store text for later classification
    # readable text
    df.at[idx, "assoc_text_concat"] = " ".join(
        [m.get("message_text", "").strip() for m in messages if (m.get("message_text", "") or "").strip()]
    ).strip()
    # full structured list
    df.at[idx, "assoc_messages_json"] = json.dumps(messages, ensure_ascii=False)

    # Derived
    recompute_derived_fields(df, idx)

    df.to_csv(evidence_csv, index=False)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # ----------- CONFIG (defaults) -----------
    MEDIA_ID = "IMG_0346"  # fallback
    SMS_TIMELINE_CSV = "forensic_multimodal_evidence.csv"
    EVIDENCE_CSV = "image_evidence.csv"

    CFG = AssocTextConfig(window_before_s=120, window_after_s=120)
    # ----------------------------------------

    # CLI override:
    #   python case_2.py <media_id> [sms_timeline_csv]
    if len(sys.argv) >= 2 and str(sys.argv[1]).strip():
        MEDIA_ID = sys.argv[1]
    if len(sys.argv) >= 3 and str(sys.argv[2]).strip():
        SMS_TIMELINE_CSV = sys.argv[2]

    if not os.path.exists(SMS_TIMELINE_CSV):
        raise FileNotFoundError(f"SMS timeline CSV not found: {SMS_TIMELINE_CSV}")

    result = detect_associated_text_sms(SMS_TIMELINE_CSV, MEDIA_ID, cfg=CFG)
    print(result)

    update_case2_fields(EVIDENCE_CSV, result)

    print(f"\nUpdated evidence CSV: {EVIDENCE_CSV} (media_id={MEDIA_ID})")
