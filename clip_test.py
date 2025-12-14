"""
Minimal CLIP example: image + text prompts → printed probabilities

Usage:
  python clip_minimal.py path/to/image.jpg
"""

import sys
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 2. Load image
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")

# 3. Define text prompts (labels)
labels = [
    "a neutral everyday image",
    "an image depicting harassment",
    "an image depicting a violent threat",
    "an image containing hate symbols",
]

# 4. Prepare inputs for CLIP
inputs = processor(
    text=labels,
    images=image,
    return_tensors="pt",
    padding=True,
)

# 5. Run CLIP
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits_per_image  # shape: [1, num_labels]
    probs = torch.softmax(logits, dim=1)[0]

# 6. Print results
print("\nCLIP image classification results:\n")
for label, prob in zip(labels, probs):
    print(f"{label:45s} → {prob.item():.4f}")

# 7. Top prediction
top_idx = probs.argmax().item()
print("\nTop prediction:")
print(f"{labels[top_idx]} ({probs[top_idx].item():.4f})")
