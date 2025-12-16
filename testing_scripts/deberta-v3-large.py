from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -----------------------------
# Frozen forensic text taxonomy (v1.0)
# -----------------------------
FROZEN_LABELS: List[str] = [
    "neutral_or_contextual",
    "abusive_or_obscene_language",
    "harassment_or_intimidation",
    "threat_of_violence",
    "incitement_or_endorsement_of_violence",
    "hate_or_bias_based_content",
    "weapon_related_text",
    "criminal_admission_or_description",
    "sexual_violence_or_exploitation",
    "self_harm_or_suicide_risk",
]

# DeBERTa v3 large model designed for zero-shot classification
MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# Hypothesis template used for NLI-style zero-shot
HYPOTHESIS_TEMPLATE = "This text is about {}."


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    score: float
    scores: Dict[str, float]


def _normalize(text: str) -> str:
    return (text or "").strip()


def _entailment_index(model) -> int:
    """
    Find the entailment logit index from the model's label mapping.
    Works across common label conventions.
    """
    label2id = getattr(model.config, "label2id", None) or {}
    # Common keys
    for key in ("entailment", "ENTAILMENT", "Entailment"):
        if key in label2id:
            return int(label2id[key])

    # Sometimes labels are like {"CONTRADICTION":0,"NEUTRAL":1,"ENTAILMENT":2}
    # If not found, fallback to last logit (common for NLI heads)
    return model.config.num_labels - 1


def classify_text(text: str) -> ClassificationResult:
    text = _normalize(text)
    if not text:
        return ClassificationResult(
            label="neutral_or_contextual",
            score=0.0,
            scores={"neutral_or_contextual": 0.0},
        )

    # Load tokenizer/model (slow tokenizer avoids known fast conversion issues)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    entail_idx = _entailment_index(model)

    # Build premise-hypothesis pairs
    pairs: List[Tuple[str, str]] = [
        (text, HYPOTHESIS_TEMPLATE.format(label.replace("_", " ")))
        for label in FROZEN_LABELS
    ]

    # Tokenize as a batch
    batch = tokenizer(
        [p for p, _ in pairs],
        [h for _, h in pairs],
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**batch).logits  # shape: [num_labels_candidates, num_nli_classes]

    # Use entailment score for each candidate label
    entail_logits = logits[:, entail_idx]  # shape: [num_candidates]

    # Convert to probabilities across candidates (single-label decision)
    probs = torch.softmax(entail_logits, dim=0).detach().cpu().tolist()

    scores = {label: float(prob) for label, prob in zip(FROZEN_LABELS, probs)}
    best_label = max(scores.items(), key=lambda x: x[1])[0]
    best_score = scores[best_label]

    return ClassificationResult(label=best_label, score=best_score, scores=scores)


def main() -> None:
    if len(sys.argv) >= 2:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    result = classify_text(text)

    # stdout: single label only (pipeline-safe)
    print(result.label)

    # stderr: debug info
    print(f"score={result.score:.4f}", file=sys.stderr)
    for lbl in FROZEN_LABELS:
        print(f"{lbl}: {result.scores.get(lbl, 0.0):.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
