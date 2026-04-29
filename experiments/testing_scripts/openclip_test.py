"""
openclip_test.py

OpenCLIP ViT-L/14 prompt-ensembled classification using a JSON prompt bank.

Folder requirements:
  - openclip_test.py
  - prompt_bank.json   (same folder)

Usage:
  python openclip_test.py path/to/image.jpg
  python openclip_test.py Case_1_data\\IMG_0346.jpg

Install:
  pip install open_clip_torch torch torchvision pillow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
import open_clip


# ---------- Model config ----------
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"  # Valid for your open_clip install


# ---------- Prompt bank loader ----------
def load_prompt_bank(json_filename: str = "prompt_bank.json") -> Dict[str, List[str]]:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / json_filename

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
    Returns:
      class_names: list[str]
      class_feats: tensor [C, D]
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
    Adjust as needed to match your course framing.
    """
    high = {"hate_or_extremist_ideology", "physical_violence_or_criminal_act", "threat_of_violence_or_crime", "weapon_present"}
    review = {"harassment_or_intimidation", "obscene_or_insulting_content"}

    if top_class in high:
        return "high"
    if top_class in review:
        return "review"
    return "low"


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

    prompt_bank = load_prompt_bank("prompt_bank.json")
    model, preprocess, tokenizer = load_model(device)

    img_feat = encode_image(model, preprocess, image_path, device)
    class_names, class_feats = encode_class_features(model, tokenizer, prompt_bank, device)

    with torch.no_grad():
        logits = img_feat @ class_feats.T        # [1, C]
        probs = logits.softmax(dim=-1)[0].cpu()  # [C]

    # Print sorted results
    pairs = sorted(zip(class_names, probs.tolist()), key=lambda x: x[1], reverse=True)

    print("\nOpenCLIP ViT-L/14 (prompt-bank JSON + ensembling) results:\n")
    for cls, p in pairs:
        print(f"{cls:32s} → {p:.4f}")

    top_class, top_prob = pairs[0]
    print("\nTop prediction:")
    print(f"{top_class} ({top_prob:.4f})")
    print("Triage:", triage_level(top_class))


if __name__ == "__main__":
    main()
