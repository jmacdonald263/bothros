# Model Card — Linear A Classifier (`la-classifier`)

## Overview
A ConvNeXt-Tiny image classifier that identifies a cropped Linear A sign by its
canonical **AB-code**. Stage 2 of the BOTHROS pipeline (the unified detector
supplies the crops).

- **Architecture:** ConvNeXt-Tiny (`timm convnext_tiny.in12k_ft_in1k`), 224px.
- **Weights:** `la_clean_v1_best.pth` (~107 MB).
- **Classes:** 374, spanning the **full SigLA-scale Linear A inventory** (~370
  signs) — not just the ~90-sign syllabary. Composition: ~88 AB-code
  syllabograms + logograms (`*NNN`-series, written without the asterisk, e.g.
  `120`, `131B`) + fractions + numerals + 59 ligatures. Names canonicalised
  (reading/AB-code "twin" names merged, `*` stripped, `+`→`_`). **Long-tailed:**
  ~120 signs are well-sampled; most are rare (median ~7 training crops/class)
  with lower accuracy. The headline oracle reflects the common signs.

## Training data
~48,984 sign crops from SigLA, lineara.xyz and GORILA — a **mix of facsimile
line-drawings and tablet photographs, the majority (~75%) photographic**
(lineara.xyz "Inscription" photos + photo hard-crops + imagemap, plus SigLA/
facsimile drawings). **Excluding every held-out tablet** (canonical manifest
applied at crop-extraction time, not just at evaluation). Trained from ImageNet
init with **label smoothing 0.1 + mixup 0.2** (calibration), val accuracy 0.593.

## Performance (held-out, leak-free)
- **Oracle top-1: 79.3%** (classifier on gold boxes) — beats the DeepScribe
  cuneiform reference (74%). Top-5: 83.9%.
- 95% CI ≈ [70, 89] on n=87 box-bearing held-out signs (thin sample — reported
  with its interval).
- In the full pipeline (with the unified detector @ conf-filter 0.25):
  per-line F1 **64.9%**, end-to-end sign top-1 **68.7%**.

## Naming convention (important for consumers)
Outputs are **canonical AB-codes** (e.g. `AB01`, `AB81`). The GORILA convention
markers — leading `*`, and `+`/`_` ligature separators — are normalised away;
reading/AB-code "twin" names are collapsed to one code. A mapping to phonetic
values is provided where the value is established.

## Intended use
Identifying detected Linear A signs for research, transliteration assistance,
and corpus work. Not for commercial use (CC BY-NC-SA 4.0).

## Limitations
- Long-tailed sign inventory: rare signs have few training examples and lower
  accuracy.
- Trained on **both facsimiles and photographs**, so both are in-distribution.
  The hard cases are very low-resolution or heavily-damaged tablets, and
  detection recall (not classification) is the main end-to-end limiter.
- The oracle figure isolates the classifier; end-to-end accuracy is lower
  because the detector misses some signs (see the detector card).
