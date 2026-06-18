# BOTHROS — Aegean sign reading from photographs

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20746759-blue)](https://doi.org/10.5281/zenodo.20746759)
[![weights: Hugging Face](https://img.shields.io/badge/weights-Hugging%20Face-yellow)](https://huggingface.co/JMacD263/bothros)
[![code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)

**Photograph an ancient Aegean tablet → get the signs on it, identified by
catalogue code and (where known) phonetic value.** Linear A and Linear B.

BOTHROS is a two-stage computer-vision pipeline:

```
photo ─▶ [1] unified Aegean sign detector (YOLO11) ─▶ sign boxes
              │
              └▶ [2] per-script classifier (ConvNeXt-Tiny) ─▶ AB-codes / B-codes + readings
```

A **single** detector localises signs for both scripts (Linear A and Linear B
signs are visually cognate; sign detection is class-agnostic). A small
script-specific classifier then identifies each detected sign. The output maps
signs to their catalogue codes (AB-codes for Linear A, B-codes for Linear B) and
their phonetic readings where established.

> **Status: research preview / draft.** Numbers below are held-out and
> leak-free (see *Evaluation* for exactly what that means). This is an honest
> engineering result, not a solved problem — read the caveats.

---

## Why this exists

No sign-segmentation-and-classification benchmark against the **DĀMOS** corpus
(Linear B) or **SigLA** (Linear A) had been published for a tool of this type.
BOTHROS provides one. The headline claim to a researcher is concrete and
checkable: *"evaluated on held-out DĀMOS / SigLA tablets, leak-free, achieving
X — and here is the reproduction command."*

The closest comparable published system is **DeepScribe** (Williams et al.,
JOCCH 2025) for Elamite cuneiform. It is a different script and corpus, so the
comparison is a cross-domain reference point, not a head-to-head — but it is the
only quantitative yardstick that exists for this class of tool.

---

## Results (held-out, leak-free)

All numbers are on **held-out test tablets** the models never saw in training —
neither the classifier crops nor the detector boxes (a tablet-ID-level split
enforced at extraction time, not just at evaluation time). Metrics:

| metric | what it measures | Linear A | Linear B | DeepScribe (cuneiform ref) |
|---|---|---|---|---|
| **Classifier oracle top-1** | classifier on gold boxes | **79.3%** | 64.5% | 74% |
| Classifier oracle top-5 | " | 83.9% | 73.8% | 91% |
| **Pipeline E2E sign top-1** | full pipeline, per sign | 68.7% | 63.8% | 56.3% |
| Pipeline E2E sign top-5 | " | 75.3% | 67.0% | 79.3% |
| **Pipeline per-line F1** | detect+classify+align, in-line | 64.9% | 76.5% | — |
| **CER** (lower is better) | character error rate (flat) | ~0.48 | **0.44** | 0.669 |
| Detection mAP@50 | detector only | n/m | ~0.47 | 0.78 |

*All pipeline numbers use the single shipped `aegean-unified` detector (E2E at
high recall; per-line F1 + CER at the precise operating points, conf-filter 0.25
LA / 0.30 LB). Detection mAP separately measured for Linear B only (n/m = not
measured for Linear A; both scripts are detector-recall-bound).*

**Honest reading of this table:**

- **Where we lead:** Linear A classifier oracle top-1 (79.3% vs 74%); both
  scripts on **CER** (0.44 / 0.48 vs 0.669) and on **end-to-end sign top-1**
  (68.7% / 63.8% vs 56.3%).
- **Where we trail:** oracle top-5 (both below 91%), end-to-end top-5, and
  **detection mAP** — the detector is the bottleneck. A sign the detector never
  finds cannot be classified, which caps end-to-end recall around 60–69%. The
  Linear B *classifier* is 86–90% top-1/top-5 *on detected signs*; the ceiling
  is detection, not classification.
- The Linear A oracle (79.3%) rests on a thin held-out sample (n=87 box-bearing
  signs, 95% CI ≈ [70, 89]) — reported with its interval, not as a point claim.

Full methodology, per-tablet statistics, IoU thresholds, and reproduction
commands are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Quickstart

```bash
pip install -e .
python3 scripts/download_weights.py          # fetches weights from Hugging Face
python3 -m bothros read your_tablet.jpg --script la   # Linear A
python3 -m bothros read your_tablet.jpg --script lb   # Linear B
```

Tablet images are not bundled (the source corpora are research-only — see
[`examples/`](examples/) for where to get one).

Output: detected signs with bounding boxes, catalogue codes, phonetic readings,
and per-sign confidence — as JSON and an annotated overlay image.

---

## Models

One detector, two classifiers (weights hosted on Hugging Face + archived on
Zenodo with a DOI):

| component | model | notes |
|---|---|---|
| detector (both scripts) | `aegean-unified` (YOLO11s) | trained on SigLA + lineara.xyz + DĀMOS boxes — facsimiles **and** photographs (~half photo); test-split excluded |
| Linear A classifier | `la-classifier` (ConvNeXt-Tiny) | AB-codes, label-smoothing + mixup calibrated |
| Linear B classifier | `lb-classifier` (ConvNeXt-Tiny) | B-codes → readings, calibrated |

See the model cards in [`docs/`](docs/) for training data, intended use, and
limitations.

---

## Evaluation — what "leak-free" means here

Earlier internal numbers for this project were inflated by **train/test
leakage**: the split was enforced when *evaluating* but not when *extracting
training crops or training the detector*, so models had effectively seen test
tablets. Every number above is from a pipeline rebuilt with a canonical
held-out tablet-ID manifest excluded at **extraction** time. Where a metric is
compared to DeepScribe, the same metric definition is used (e.g. CER is the
standard flat character error rate, not a more lenient per-line variant).

---

## Licence

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Model weights:** CC BY-NC-SA 4.0 (see [`WEIGHTS_LICENSE`](WEIGHTS_LICENSE)) —
  derived from corpora (GORILA, DĀMOS, SigLA) with non-commercial research terms.

## Citation

```bibtex
@software{bothros2026,
  title     = {BOTHROS: Aegean sign reading from photographs},
  author    = {MacDonald, Jamie},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20746759},
  url       = {https://doi.org/10.5281/zenodo.20746759}
}
```

## Acknowledgements

Built on [DĀMOS](https://damos.hf.uio.no) (Linear B), [SigLA](https://sigla.phis.me)
and GORILA (Linear A), the [lineara.xyz](https://lineara.xyz) /
[LinearBExplorer](https://github.com/mwenge/LinearBExplorer) corpora,
[LiBER](https://liber.cnr.it) (Linear B photographs), Ultralytics YOLO, and timm.
Detector
architecture is extensible — the unified Aegean base can be fine-tuned onto
Cretan Hieroglyphic, Cypro-Minoan, or other scripts.
