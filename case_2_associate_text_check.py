from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class AssocTextConfig:
    # Time window in seconds around the image event
    window_before_s: int = 120
    window_after_s: int = 120

    # Whether to exclude system events (recommended)
    exclude_system_like: bool = True

    # Heuristic patterns that are typically non-content placeholders in exports
    drop_text_exact: Tuple[str, ...] = (
        "",
        "sent an attachment",
        "photo",
        "image",
    )


def _parse_utc(ts: str) -> datetime:
    # Accepts "2025-01-12T14:23:11Z"
    # If you ever have "+00:00" style, this will still work with a small tweak.
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
    """
    Finds the image event row for media_id.
    Returns dict with anchor info or raises ValueError if not found.
    """
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
    raise ValueError(f"Image event for media_id={media_id} not found in CSV.")


def is_meaningful_text(text: str, cfg: AssocTextConfig) -> bool:
    t = _norm(text)
    if t in cfg.drop_text_exact:
        return False
    # Drop very short reaction-like content
    if len(t) <= 1:
        return False
    return True


def detect_associated_text_sms(
    csv_path: str,
    media_id: str,
    cfg: AssocTextConfig = AssocTextConfig(),
) -> Dict[str, Any]:
    """
    Checks if there is associated SMS text around an image event in the same thread.

    Returns:
      {
        "media_id": ...,
        "associated_text_flag": bool,
        "associated_messages": [ {event_id, utc_timestamp, sender_id, message_text}, ... ],
        "window_before_s": ...,
        "window_after_s": ...,
        "reason": ...
      }
    """
    events = load_events(csv_path)
    anchor = find_image_anchor(events, media_id)

    thread_id = anchor["thread_id"]
    t0 = anchor["anchor_timestamp"]

    t_min = t0.timestamp() - cfg.window_before_s
    t_max = t0.timestamp() + cfg.window_after_s

    candidates: List[Dict[str, Any]] = []

    for row in events:
        if row.get("thread_id", "") != thread_id:
            continue

        event_type = _norm(row.get("event_type", ""))
        if event_type != "text":
            continue

        # Optional: exclude system-like rows by simple sender role
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


if __name__ == "__main__":
    # Example run
    CSV_PATH = "Case_2_data/forensic_multimodal_evidence.csv"
    MEDIA_ID = "IMG_0346"
    result = detect_associated_text_sms(CSV_PATH, MEDIA_ID)
    print(result)
