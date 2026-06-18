# Model Card — Linear B Classifier (`lb-classifier`)

## Overview
A ConvNeXt-Tiny image classifier that identifies a cropped Linear B sign by its
**B-code**, mapped to a phonetic reading where established. Stage 2 of the
BOTHROS pipeline (the unified detector supplies the crops).

- **Architecture:** ConvNeXt-Tiny (`timm convnext_tiny.in12k_ft_in1k`), 224px.
- **Weights:** `lb_labeller_v3_calibrated.pth` (~107 MB).
- **Classes:** 142 B-codes; outputs mapped to readings (e.g. `da`, `ro`, `pa`)
  via the bundled class→reading map. Temperature-calibrated (T≈0.856).

## Training data
~8,500 sign crops from the linearb.xyz imagemap + DĀMOS-validated tablets,
selected by a **triple-guard** rule (alignment top-1 ∧ committee agreement ∧ no
box overlap) for clean labels, **excluding all DĀMOS test-split tablets** at
extraction time. Label smoothing 0.1 + mixup 0.2.

> Quality over quantity: a larger but looser 12.9k-crop variant scored *worse*
> (61.4% vs 65.9% oracle). The clean triple-guard set is the better classifier.

## Performance (held-out, leak-free; DĀMOS)
- Oracle top-1 **64.5%** / top-5 73.8% (classifier on gold boxes).
- **On detected signs the classifier is ~86–90% top-1/top-5** — the pipeline
  ceiling is detection, not classification.
- In the full pipeline (unified detector @ conf-filter 0.30): per-line F1
  **76.5%**, end-to-end sign top-1 ~63%, **CER 0.44** (beats the DeepScribe
  cuneiform reference, 0.669).

## Intended use
Identifying detected Linear B signs for research, transliteration assistance,
DĀMOS-style corpus work. Not for commercial use (CC BY-NC-SA 4.0).

## Limitations
- Vocabulary is 142 classes vs the ~180 full Linear B inventory — rare signs and
  some logograms/numerals are under-represented (clean-data-limited).
- Oracle top-1 (64.5%) is below the DeepScribe reference (74%); this is a
  clean-training-data ceiling, not a calibration problem (calibration is good).
- Trained on the available photo/facsimile corpora; unusual imaging may be OOD.
