from __future__ import annotations

import json
import sys

import pandas as pd

from forensic_pipeline import case_3, fusion_runner
from forensic_pipeline.pipeline_utils import drop_blank_rows, truthy


def test_truthy_handles_csv_strings() -> None:
    assert truthy("True") is True
    assert truthy("1") is True
    assert truthy("False") is False
    assert truthy("") is False
    assert truthy(float("nan")) is False


def test_case3_routes_false_strings_to_image_only() -> None:
    df = pd.DataFrame(
        [
            {
                "media_id": "img1",
                "ocr_ran": "True",
                "embedded_text_flag": "False",
                "assoc_check_ran": "True",
                "associated_text_flag": "False",
            }
        ]
    )
    for col in case_3.EVIDENCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    result = case_3.finalize_case3_for_media(df, "img1")

    assert result["case_label"] == "case_3_image_only"
    assert result["analysis_mode"] == "image_only"


def test_clip_prompt_scores_map_to_frozen_labels() -> None:
    row = pd.Series(
        {
            "media_id": "img1",
            "probs_json": json.dumps(
                {
                    "neutral": 0.1,
                    "weapon_present": 0.8,
                    "hate_or_extremist_ideology": 0.6,
                }
            ),
        }
    )

    scores = fusion_runner.clip_to_frozen_scores(row)

    assert scores["weapon_related_text"] == 0.8
    assert scores["hate_or_bias_based_content"] == 0.6
    assert scores["neutral_or_contextual"] == 0.1


def test_fusion_runner_drops_blank_image_rows(tmp_path, monkeypatch) -> None:
    image_csv = tmp_path / "image_evidence.csv"
    clip_csv = tmp_path / "openclip_outputs.csv"
    text_csv = tmp_path / "text_evidence.csv"
    out_csv = tmp_path / "out.csv"

    pd.DataFrame(
        [
            {"media_id": "", "case_label": ""},
            {"media_id": "img1", "case_label": "case_3_image_only"},
        ]
    ).to_csv(image_csv, index=False)
    pd.DataFrame(
        [
            {
                "media_id": "img1",
                "created_utc": "2025-01-01T00:00:00Z",
                "probs_json": json.dumps({"weapon_present": 0.9, "neutral": 0.1}),
            }
        ]
    ).to_csv(clip_csv, index=False)
    pd.DataFrame(columns=["media_id", "created_utc", "text_source"]).to_csv(text_csv, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m forensic_pipeline.fusion_runner", str(image_csv), str(clip_csv), str(text_csv), str(out_csv)],
    )

    fusion_runner.main()

    out = pd.read_csv(out_csv)
    assert out["media_id"].tolist() == ["img1"]
    assert out.loc[0, "final_label"] == "weapon_related_text"


def test_drop_blank_rows_uses_media_id() -> None:
    df = pd.DataFrame([{"media_id": "", "value": "x"}, {"media_id": "img1", "value": ""}])

    cleaned = drop_blank_rows(df)

    assert cleaned["media_id"].tolist() == ["img1"]
