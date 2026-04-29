from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

from transformers import pipeline

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

# Recommended baseline: zero-shot NLI
# First run downloads model and caches it locally.
DEFAULT_MODEL = "facebook/bart-large-mnli"


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    score: float
    scores: Dict[str, float]  # label -> score


def _normalize_text(text: str) -> str:
    return (text or "").strip()


def classify_text(
    text: str,
    labels: List[str] = FROZEN_LABELS,
    model_name: str = DEFAULT_MODEL,
) -> ClassificationResult:
    """
    Zero-shot text classifier.
    Returns exactly one label from the frozen taxonomy.

    Notes:
    - multi_label=False forces a single-label style ranking.
    - If text is empty, defaults to neutral_or_contextual.
    """
    text = _normalize_text(text)
    if not text:
        return ClassificationResult(
            label="neutral_or_contextual",
            score=0.0,
            scores={"neutral_or_contextual": 0.0},
        )

    zs = pipeline("zero-shot-classification", model=model_name)

    # Single-label style decision
    out = zs(
        sequences=text,
        candidate_labels=labels,
        multi_label=False,
    )

    label_scores: List[Tuple[str, float]] = list(zip(out["labels"], out["scores"]))
    label_scores.sort(key=lambda x: x[1], reverse=True)

    best_label, best_score = label_scores[0]
    scores_dict = {lbl: float(sc) for lbl, sc in label_scores}

    return ClassificationResult(
        label=best_label,
        score=float(best_score),
        scores=scores_dict,
    )


def main() -> None:
    # Input comes from argv or stdin
    if len(sys.argv) >= 2:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    result = classify_text(text)

    # Primary output: one label only (easy to consume in pipelines)
    print(result.label)

    # Optional debug info to stderr (won't break stdout parsing)
    print(f"score={result.score:.4f}", file=sys.stderr)
    for lbl in FROZEN_LABELS:
        if lbl in result.scores:
            print(f"{lbl}: {result.scores[lbl]:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
