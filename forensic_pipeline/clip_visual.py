"""
openclip_test.py

OpenCLIP ViT-L/14 prompt-ensembled classification using a JSON prompt bank,
and appends results to an output CSV.

Folder requirements:
  - openclip_test.py
  - prompt_bank.json   (same folder)

Usage:
  python openclip_test.py path/to/image.jpg
  python openclip_test.py Case_1_data\\IMG_0346.jpg

Install:
  pip install open_clip_torch torch torchvision pillow pandas
"""

from __future__ import annotations

import json
import sys
import secrets
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import pandas as pd
import torch
from PIL import Image
import open_clip

from forensic_pipeline.paths import OPENCLIP_OUTPUTS_CSV, PROMPT_BANK_JSON, project_path


# ---------- Model config ----------
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"  # Valid tag in your open_clip install

# Output CSV (created if missing)
OUTPUT_CSV = OPENCLIP_OUTPUTS_CSV

# CSV schema (stable, append-only)
CSV_COLUMNS = [
    "run_id",
    "created_utc",
    "media_id",
    "image_path",
    "device",
    "model_name",
    "pretrained_tag",
    "prompt_bank_file",
    "top_class",
    "top_prob",
    "triage",
    "probs_json",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_run_id() -> str:
    return f"{now_utc()}_{secrets.token_hex(3)}"


def safe_media_id(image_path: str) -> str:
    # media_id is file stem; change later if you want to map to your evidence CSV
    return Path(image_path).stem


# ---------- Prompt bank loader ----------
def load_prompt_bank(json_filename: str | Path = PROMPT_BANK_JSON) -> Dict[str, List[str]]:
    path = project_path(json_filename)

    if not path.exists():
        raise FileNotFoundError(f"Prompt bank JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        bank = json.load(f)

    if not isinstance(bank, dict) or not bank:
        raise ValueError("Prompt bank JSON must be a non-empty object: {class_name: [prompts...]}")

    for cls, prompts in bank.items():
        if not isinstance(cls, str) or not cls.strip():
            raise ValueError("Each class name must be a non-empty string.")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"Class '{cls}' must map to a non-empty list of prompts.")
        if any((not isinstance(p, str) or not p.strip()) for p in prompts):
            raise ValueError(f"Class '{cls}' contains an empty or non-string prompt.")
    return bank


# ---------- OpenCLIP helpers ----------
def load_model(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


def encode_image(model, preprocess, image_path: str, device: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat  # [1, D]


def encode_class_features(
    model,
    tokenizer,
    prompt_bank: Dict[str, List[str]],
    device: str,
) -> Tuple[List[str], torch.Tensor]:
    """
    Returns class_names and class_features.
    Each class feature = normalized mean of its prompt embeddings.
    """
    class_names: List[str] = []
    class_feats: List[torch.Tensor] = []

    with torch.no_grad():
        for cls, prompts in prompt_bank.items():
            tokens = tokenizer(prompts).to(device)         # [P, ...]
            txt = model.encode_text(tokens)                # [P, D]
            txt = txt / txt.norm(dim=-1, keepdim=True)     # normalize each prompt embedding

            cls_feat = txt.mean(dim=0, keepdim=True)       # [1, D] average prompts
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)

            class_names.append(cls)
            class_feats.append(cls_feat)

    return class_names, torch.cat(class_feats, dim=0)      # [C, D]


def triage_level(top_class: str) -> str:
    """
    Simple forensic triage mapping.
    Adjust sets to match your prompt_bank.json keys.
    """
    high = {
        "hate_or_extremist_ideology",
        "physical_violence_or_criminal_act",
        "threat_of_violence_or_crime",
        "weapon_present",
    }
    review = {
        "harassment_or_intimidation",
        "obscene_or_insulting_content",
    }

    if top_class in high:
        return "high"
    if top_class in review:
        return "review"
    return "low"


# ---------- CSV logging ----------
def ensure_output_csv(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(path, index=False)


def append_output_row(csv_path: str, row: Dict[str, Any]):
    ensure_output_csv(csv_path)
    df = pd.read_csv(csv_path)

    # Enforce schema order; fill missing keys with empty
    ordered = {col: row.get(col, "") for col in CSV_COLUMNS}
    df.loc[len(df)] = ordered
    df.to_csv(csv_path, index=False)


# ---------- Main ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python openclip_test.py path/to/image.jpg")
        raise SystemExit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    prompt_bank_file = PROMPT_BANK_JSON
    prompt_bank = load_prompt_bank(prompt_bank_file)

    model, preprocess, tokenizer = load_model(device)

    img_feat = encode_image(model, preprocess, image_path, device)
    class_names, class_feats = encode_class_features(model, tokenizer, prompt_bank, device)

    with torch.no_grad():
        logits = img_feat @ class_feats.T         # [1, C]
        probs = logits.softmax(dim=-1)[0].cpu()   # [C]

    # Build result dict
    probs_dict = {cls: float(p) for cls, p in zip(class_names, probs.tolist())}
    sorted_pairs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)

    top_class, top_prob = sorted_pairs[0]
    triage = triage_level(top_class)

    # Print
    print("\nOpenCLIP ViT-L/14 (prompt-bank JSON + ensembling) results:\n")
    for cls, p in sorted_pairs:
        print(f"{cls:32s} → {p:.4f}")

    print("\nTop prediction:")
    print(f"{top_class} ({top_prob:.4f})")
    print("Triage:", triage)

    # Append to CSV
    row = {
        "run_id": make_run_id(),
        "created_utc": now_utc(),
        "media_id": safe_media_id(image_path),
        "image_path": str(Path(image_path).resolve()),
        "device": device,
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED,
        "prompt_bank_file": str(prompt_bank_file),
        "top_class": top_class,
        "top_prob": top_prob,
        "triage": triage,
        "probs_json": json.dumps(probs_dict, ensure_ascii=False),
    }

    append_output_row(OUTPUT_CSV, row)
    print(f"\nSaved to CSV: {Path(OUTPUT_CSV).resolve()}")


if __name__ == "__main__":
    main()
