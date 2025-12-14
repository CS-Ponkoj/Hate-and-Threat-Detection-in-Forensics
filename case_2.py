"""
Case 2 check (SMS associated text):
- Reads a single SMS-thread timeline CSV (your forensic_multimodal_evidence.csv)
- Finds the image anchor event for a given media_id
- Collects nearby TEXT events within a configurable window
- Outputs a structured decision record
- Updates/creates image_evidence.csv (one row per image evidence item)

Dependencies:
  pip install pandas
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


# -----------------------------
# Config + parsing utilities
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
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def load_events(csv_path: str) -> List[Dict[str, str]]:
    events: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)
    return events


def find_image_anchor(events: List[Dict[str, str]], media_id: str) -> Dict[str, Any]:
    for row in events:
        if _norm(row.get("event_type", "")) == "image" and _norm(row.get("media_id", "")) == _norm(media_id):
            ts = _parse_utc(row["utc_timestamp"])
            return {
                "thread_id": row.get("thread_id", ""),
                "anchor_event_id": row.get("event_id", ""),
                "anchor_timestamp": ts,
                "anchor_sender_id": row.get("sender_id", ""),
                "row": row,
            }
    raise ValueError(f"Image event for media_id={media_id} not found in CSV: {csv_path}")


def is_meaningful_text(text: str, cfg: AssocTextConfig) -> bool:
    t = _norm(text)
    if t in cfg.drop_text_exact:
        return False
    if len(t) <= 1:
        return False
    return True


# -----------------------------
# Case 2 detection
# -----------------------------

def detect_associated_text_sms(
    sms_timeline_csv: str,
    media_id: str,
    cfg: AssocTextConfig = AssocTextConfig(),
) -> Dict[str, Any]:
    events = load_events(sms_timeline_csv)
    anchor = find_image_anchor(events, media_id)

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
# Evidence CSV update utilities
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

    # Case 1 (OCR) fields (filled by Case 1 script)
    "ocr_ran",
    "embedded_text_flag",
    "ocr_text",
    "ocr_mean_confidence",
    "ocr_character_count",
    "ocr_text_coverage",

    # Case 2 fields (this script fills)
    "assoc_check_ran",
    "associated_text_flag",
    "assoc_message_count",
    "assoc_window_before_s",
    "assoc_window_after_s",

    # Derived
    "has_any_text_context",
    "analysis_mode",
    "last_updated_utc",
]


def ensure_evidence_csv(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for col in EVIDENCE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[EVIDENCE_COLUMNS]
        return df

    df = pd.DataFrame(columns=EVIDENCE_COLUMNS)
    df.to_csv(csv_path, index=False)
    return df


def upsert_media_row_minimal(df: pd.DataFrame, case_id: str, media_id: str) -> pd.DataFrame:
    if (df["media_id"] == media_id).any():
        return df
    idx = len(df)
    df.loc[idx] = {col: "" for col in EVIDENCE_COLUMNS}
    df.at[idx, "case_id"] = case_id
    df.at[idx, "media_id"] = media_id
    return df


def update_case2_fields(df: pd.DataFrame, media_id: str, assoc_result: Dict[str, Any]) -> pd.DataFrame:
    idx = df.index[df["media_id"] == media_id][0]

    df.at[idx, "assoc_check_ran"] = True
    df.at[idx, "associated_text_flag"] = bool(assoc_result.get("associated_text_flag", False))
    df.at[idx, "assoc_message_count"] = len(assoc_result.get("associated_messages", []))
    df.at[idx, "assoc_window_before_s"] = assoc_result.get("window_before_s")
    df.at[idx, "assoc_window_after_s"] = assoc_result.get("window_after_s")

    # Also fill linking context if present
    df.at[idx, "thread_id"] = assoc_result.get("thread_id", df.at[idx, "thread_id"])
    df.at[idx, "anchor_event_id"] = assoc_result.get("anchor_event_id", df.at[idx, "anchor_event_id"])
    df.at[idx, "anchor_timestamp_utc"] = assoc_result.get("anchor_timestamp_utc", df.at[idx, "anchor_timestamp_utc"])

    # Derived fields (OCR may or may not already exist)
    embedded_raw = df.at[idx, "embedded_text_flag"]
    embedded = bool(embedded_raw) if str(embedded_raw).strip() != "" else False

    associated = bool(df.at[idx, "associated_text_flag"]) if str(df.at[idx, "associated_text_flag"]).strip() != "" else False

    has_any = embedded or associated
    df.at[idx, "has_any_text_context"] = has_any
    df.at[idx, "analysis_mode"] = "multimodal" if has_any else "image_only"

    df.at[idx, "last_updated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return df


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    # ----------- CONFIG -----------
    CASE_ID = "CASE-2025-001"
    SMS_TIMELINE_CSV = r"Case_2_data/forensic_multimodal_evidence.csv"  # your timeline CSV
    MEDIA_ID = "IMG_0346"

    IMAGE_EVIDENCE_CSV = "image_evidence.csv"  # updated/created here

    CFG = AssocTextConfig(window_before_s=120, window_after_s=120, exclude_system_like=True)
    # ------------------------------

    if not os.path.exists(SMS_TIMELINE_CSV):
        raise FileNotFoundError(f"SMS timeline CSV not found: {SMS_TIMELINE_CSV}")

    # Run Case 2 detection
    result = detect_associated_text_sms(SMS_TIMELINE_CSV, MEDIA_ID, cfg=CFG)
    print(result)

    # Update the image evidence CSV
    df = ensure_evidence_csv(IMAGE_EVIDENCE_CSV)
    df = upsert_media_row_minimal(df, case_id=CASE_ID, media_id=MEDIA_ID)
    df = update_case2_fields(df, media_id=MEDIA_ID, assoc_result=result)
    df.to_csv(IMAGE_EVIDENCE_CSV, index=False)

    print(f"\nUpdated evidence CSV: {IMAGE_EVIDENCE_CSV} (media_id={MEDIA_ID})")
