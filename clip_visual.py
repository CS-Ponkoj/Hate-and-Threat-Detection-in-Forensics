"""
CLIP Visual Analysis – Single Script Version

What this script does:
1. Takes an image path
2. Derives media_id
3. Performs evidence intake (hash, dimensions)
4. Runs CLIP zero-shot image classification
5. Creates only_image_output.csv if missing
6. Appends one forensic-style output row per run

Usage:
  python clip_visual.py --image-path "Case_1_data/IMG_0346.jpg"

Dependencies:
  pip install torch torchvision transformers pillow pandas
"""

from __future__ import annotations

import argparse
import os
import json
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Dict, Any, List

import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


# -------------------- CONFIG --------------------

MODEL_NAME = "openai/clip-vit-base-patch32"

PROMPTS = [
    "a neutral everyday image",
    "an image depicting harassment or intimidation",
    "an image depicting a violent threat",
    "an image containing hate symbols or extremist imagery",
]

OUTPUT_CSV = "only_image_output.csv"

OUTPUT_COLUMNS = [
    "run_id",
    "created_utc",
    "media_id",
    "file_name",
    "file_path",
    "file_hash_sha256",
    "image_width",
    "image_height",
    "stage",
    "model_name",
    "device",
    "decision",
    "risk_score",
    "top_label",
    "top_score",
    "output_json",
]

STAGE_NAME = "clip_visual"


# -------------------- UTILS --------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_id() -> str:
    return f"{now_utc()}_{secrets.token_hex(3)}"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_csv(path: str):
    if not os.path.exists(path):
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(path, index=False)


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# -------------------- CLIP --------------------

def run_clip(image: Image.Image, device: str) -> Dict[str, Any]:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    inputs = processor(
        text=PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=0).cpu().tolist()

    return {
        "probs": probs,
        "top_label": PROMPTS[int(max(range(len(probs)), key=lambda i: probs[i]))],
        "top_score": max(probs),
    }


def decision_from_probs(probs: List[float]) -> Dict[str, Any]:
    p_harass = probs[1]
    p_threat = probs[2]
    p_hate = probs[3]

    risk = max(p_harass, p_threat, p_hate)

    if risk >= 0.60:
        decision = "high"
    elif risk >= 0.40:
        decision = "review"
    else:
        decision = "low"

    return {
        "decision": decision,
        "risk_score": risk,
    }


# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True)
    args = parser.parse_args()

    image_path = os.path.abspath(args.image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    media_id = os.path.splitext(os.path.basename(image_path))[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evidence intake
    image = load_image(image_path)
    w, h = image.size
    file_hash = sha256_file(image_path)

    # CLIP
    clip_out = run_clip(image, device=device)
    decision_out = decision_from_probs(clip_out["probs"])

    # Prepare output row
    ensure_csv(OUTPUT_CSV)
    df = pd.read_csv(OUTPUT_CSV)

    payload = {
        "prompts": PROMPTS,
        "probs": clip_out["probs"],
        "device": device,
        "model": MODEL_NAME,
        "decision": decision_out,
    }

    row = {
        "run_id": run_id(),
        "created_utc": now_utc(),
        "media_id": media_id,
        "file_name": os.path.basename(image_path),
        "file_path": image_path,
        "file_hash_sha256": file_hash,
        "image_width": w,
        "image_height": h,
        "stage": STAGE_NAME,
        "model_name": MODEL_NAME,
        "device": device,
        "decision": decision_out["decision"],
        "risk_score": decision_out["risk_score"],
        "top_label": clip_out["top_label"],
        "top_score": clip_out["top_score"],
        "output_json": json.dumps(payload, ensure_ascii=False),
    }

    df.loc[len(df)] = row
    df.to_csv(OUTPUT_CSV, index=False)

    print("CLIP analysis complete")
    print(f"media_id: {media_id}")
    print(f"decision: {decision_out['decision']}")
    print(f"risk_score: {decision_out['risk_score']:.3f}")
    print(f"output CSV updated: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
