# Hate and Threat Detection in Digital Forensics

## Overview
This project implements an image-centric digital forensic pipeline for detecting
hate, threat, violence, weapon, and related risk signals in image evidence from
messaging environments.

The codebase is organized as a small Python package. Run active commands with
`python -m forensic_pipeline.<module>` from the repository root.

## Project Layout
```text
forensic_pipeline/      Core package modules
config/                 Prompt banks and model configuration
data/raw/               Input evidence and source timelines
data/outputs/           Generated CSV outputs
docs/                   Paper, presentation, and visual assets
experiments/            Exploratory one-off scripts
legacy/                 Retired scripts kept for reference
tests/                  Lightweight regression tests
```

## Requirements
- Python 3.9+
- Tesseract OCR installed and available on `PATH`
- Python dependencies from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows, if Tesseract is not on `PATH`, update `_TESSERACT_EXE` in
`forensic_pipeline/case_1.py`.

## Exact Run Order
Run these commands from the project root.

### 1. Run Case 1/2/3 routing for one image
`forensic_pipeline.run_cases` runs OCR routing, associated text routing, and
Case 3 finalization.

```bash
python -m forensic_pipeline.run_cases data/raw/Case_1_data/IMG_0346.jpg
```

This updates `data/outputs/image_evidence.csv`.

### 2. Run visual classification
Run OpenCLIP for the same image. This appends to
`data/outputs/openclip_outputs.csv`.

```bash
python -m forensic_pipeline.clip_visual data/raw/Case_1_data/IMG_0346.jpg
```

### 3. Run associated-text classification
This reads associated text from `data/outputs/image_evidence.csv` and appends an
`assoc` row to `data/outputs/text_evidence.csv`.

```bash
python -m forensic_pipeline.case_text data/raw/Case_1_data/IMG_0346.jpg data/outputs/text_evidence.csv MoritzLaurer/deberta-v3-large-zeroshot-v2.0 assoc
```

### 4. Run OCR-text classification when Case 1 found embedded text
This appends an `ocr` row to `data/outputs/text_evidence.csv`.

```bash
python -m forensic_pipeline.case_text data/raw/Case_1_data/IMG_0346.jpg data/outputs/text_evidence.csv MoritzLaurer/deberta-v3-large-zeroshot-v2.0 ocr
```

### 5. Fuse modalities
Use the official fusion runner:

```bash
python -m forensic_pipeline.fusion_runner data/outputs/image_evidence.csv data/outputs/openclip_outputs.csv data/outputs/text_evidence.csv data/outputs/multimodal_fused_decisions.csv
```

The output is `data/outputs/multimodal_fused_decisions.csv`.

### 6. Optional summary

```bash
python -m forensic_pipeline.case_stats
```

## Active Entry Points
- `forensic_pipeline.run_cases`: runs Case 1, Case 2, and Case 3 for one image
- `forensic_pipeline.case_1`: OCR embedded text detection
- `forensic_pipeline.case_2`: nearby associated text detection
- `forensic_pipeline.case_3`: final routing to text-context or image-only evidence
- `forensic_pipeline.clip_visual`: OpenCLIP visual scoring
- `forensic_pipeline.case_text`: zero-shot text scoring with `assoc` or `ocr` source
- `forensic_pipeline.fusion_runner`: official multimodal fusion
- `forensic_pipeline.case_stats`: summary of routing outputs

## Data Files
- `data/outputs/image_evidence.csv`: one row per image artifact and routing state
- `data/outputs/openclip_outputs.csv`: append-only visual model results
- `data/outputs/text_evidence.csv`: append-only text model results with `text_source`
- `data/outputs/multimodal_fused_decisions.csv`: official fused output

## Label Flow
Text classification uses the frozen labels in `forensic_pipeline/case_text.py`
and `forensic_pipeline/fusion_runner.py`. CLIP prompt-bank labels from
`config/prompt_bank.json` are mapped into those frozen labels during fusion.

## Tests
Run the lightweight regression suite:

```bash
python -m pytest -q
```

The tests cover CSV boolean parsing, blank evidence row cleanup, Case 3 routing,
CLIP-to-frozen label mapping, and fusion behavior.

## Author

**Ponkoj Shill**  
AI/ML researcher and Ph.D. candidate in Computer Science

- [GitHub](https://github.com/CS-Ponkoj)
- [Portfolio](https://ponkoj.com)

## License

No license file is currently included. Please contact the author before reusing the repository beyond review, education, or fair-use evaluation.
