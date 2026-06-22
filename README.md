# BOTHROS — Aegean sign reading from photographs

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20746759-blue)](https://doi.org/10.5281/zenodo.20746759)
[![weights: Hugging Face](https://img.shields.io/badge/weights-Hugging%20Face-yellow)](https://huggingface.co/JMacD263/linear-a-linear-b-bothros)
[![demo: Hugging Face Spaces](https://img.shields.io/badge/demo-%F0%9F%A4%97%20Spaces-orange)](https://huggingface.co/spaces/JMacD263/bothros-demo)
[![research: findings](https://img.shields.io/badge/research-findings-purple)](https://github.com/jmacdonald263/bothros-research)
[![code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)

**Photograph an ancient Aegean tablet → get the signs on it, identified by
catalogue code and (where known) phonetic value.** Linear A and Linear B.

> **The name** — a [*bóthros*](https://en.wikipedia.org/wiki/Bothros) (βόθρος) is the sacrificial pit Odysseus digs in
> the *Odyssey*, pouring libations into it so the spirits of the dead rise to
> speak with him. A fitting name for a tool that reads scripts no one has
> spoken aloud in three thousand years.

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

The closest comparable published system is **DeepScribe** — Williams et al.,
*"DeepScribe: Localization and Classification of Elamite Cuneiform Signs Via Deep
Learning"*, JOCCH 2025 ([arXiv:2306.01268](https://arxiv.org/abs/2306.01268) ·
[code](https://github.com/edwardclem/deepscribe) ·
[project](https://voices.uchicago.edu/ochre/project/deepscribe/)). It is a
different script and corpus, so the comparison is a cross-domain reference point,
not a head-to-head — but it is the only quantitative yardstick that exists for
this class of tool. (DeepScribe reference figures below are from that paper.)

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
| Detection mAP@50 | detector only | 0.60 | ~0.55* | 0.78 |

*All pipeline numbers use the single shipped `aegean-unified` detector — E2E at
high recall; per-line F1 + CER at the precise operating points (conf-filter 0.25
LA / 0.30 LB). Detection mAP@50 measured vs held-out GT boxes (LA n=51, clean
GT; LB n=24, GT registration imperfect → conservative figure). Both scripts are
detector-recall-bound.*

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

**No install — try the [🤗 live demo](https://huggingface.co/spaces/JMacD263/bothros-demo):** upload a tablet photo, get the signs. Or run locally:

```bash
pip install -e .
python3 scripts/download_weights.py          # ~232 MB of weights from the public HF repo — no token needed
python3 -m bothros read your_tablet.jpg --script la   # Linear A
python3 -m bothros read your_tablet.jpg --script lb   # Linear B
```

**Tested with** Python 3.12.9 · torch 2.11 · timm 1.0.26 · ultralytics 8.4.30 — exact
pins in [`requirements-lock.txt`](requirements-lock.txt) (`pyproject.toml` carries only
floors; install the lock to reproduce the benchmark numbers).

Tablet images are not bundled (the source corpora are research-only — see
[`examples/`](examples/) for a runnable, openly-licensed example).

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
limitations, and [`docs/DECISIONS.md`](docs/DECISIONS.md) for *why* YOLO11 + ConvNeXt-Tiny
(with honest limits).

---

## Cross-script transfer — the unified detector, validated from first principles

Shipping a *single* detector for both scripts isn't just convenience — it's
empirically justified. A detector trained on **Linear B alone**, having never seen
a Linear A tablet, reads Linear A **zero-shot at 60.7% per-line F1 / 66.2%
end-to-end sign top-1** — close to the 64.7% of the dedicated LA-trained detector.
Sign *detection* transfers across the two scripts because Linear A and Linear B
signs are visually cognate; only the *classifier* needs to be script-specific. This
is the basis for the one `aegean-unified` detector, and suggests the same base will
extend to related scripts (Cretan Hieroglyphic, Cypro-Minoan) with little or no
script-specific detection data. Full numbers: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
(2026-06-17, cross-script detector transfer).

---

## Evaluation — what "leak-free" means here

Earlier internal numbers for this project were inflated by **train/test
leakage**: the split was enforced when *evaluating* but not when *extracting
training crops or training the detector*, so models had effectively seen test
tablets. Every number above is from a pipeline rebuilt with a canonical
held-out tablet-ID manifest excluded at **extraction** time. Where a metric is
compared to DeepScribe, the same metric definition is used (e.g. CER is the
standard flat character error rate, not a more lenient per-line variant).

**Reproduce it yourself:** the Linear A classifier-oracle (79.3%) runs from this repo
on public artifacts — see [`eval/`](eval/) (the held-out GT + harness ship here; you
supply the tablet images). LB's oracle used tight human-traced crops from the
[linearb.xyz imagemap](https://github.com/mwenge/LinearBExplorer); `eval/README.md`
explains how to reproduce it from that source.

---

## Research — companion findings

Exploratory findings built on this pipeline live in a separate repo:
**[jmacdonald263/bothros-research](https://github.com/jmacdonald263/bothros-research)** —
*what a vision model recovers from Aegean sign shapes that isn't obvious by eye.* Every
claim is anchored to published ground truth with a baseline/significance test (or it
doesn't ship), and negatives are featured — including a **retracted** result kept as a
worked example. Highlights: Linear A ↔ Linear B sign correspondences recovered by visual
similarity (secure-tier P@5 = 100%), scribal-hand agreement with Skelton's palaeography,
and a model-free script-structure comparison. Each finding reports both published weight
sets (release + held-out benchmark).

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

Built on several open corpora, each credited for what it actually provides:

- **Linear A** — inscription **images** from [**lineara.xyz**](https://lineara.xyz)
  (mwenge; ultimately from GORILA, Godart & Olivier); hand-traced **sign boxes** and the
  **AB-code sign catalogue** from [**SigLA**](https://sigla.phis.me) (**Ester Salgarella
  & Simon Castellan**) and the lineara.xyz imagemap.
- **Linear B** — transcriptions from [**DĀMOS**](https://damos.hf.uio.no) (Federico
  Aurora), the hand-traced sign **imagemap (boxes)** from
  [**LinearBExplorer**](https://github.com/mwenge/LinearBExplorer) (mwenge), and
  photographs from [**LiBER**](https://liber.cnr.it).

Plus Ultralytics YOLO and timm. The unified Aegean detector architecture is extensible —
the base can be fine-tuned onto Cretan Hieroglyphic, Cypro-Minoan, or other scripts.
