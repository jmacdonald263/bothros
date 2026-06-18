# Methodology

How BOTHROS reads Aegean signs from photographs, and — just as important — how
its numbers were made honest. Written for researchers who want to reproduce,
adapt, or audit the approach.

---

## The problem

Given a photograph or facsimile of a Linear A or Linear B tablet, locate each
sign and identify it by catalogue code (and reading, where known). Two sub-tasks
that fail differently: **detection** (where are the signs?) and **classification**
(which sign is this?). Treating them separately lets each be measured and
improved on its own.

## Architecture

```
photo ─▶ unified Aegean detector (YOLO11s) ─▶ boxes ─▶ per-script classifier (ConvNeXt-Tiny) ─▶ code + reading
```

- **One detector for both scripts.** Sign *detection* is class-agnostic, and
  Linear A / Linear B signs are visually cognate (a shared Bronze-Age Aegean
  glyph repertoire). A single detector trained on both corpora localises signs
  for either script — and a Linear-B-only detector already reads Linear A at
  ~61% F1 *zero-shot*, confirming the transfer.
- **A small classifier per script.** ConvNeXt-Tiny, ~107 MB, identifying the
  cropped sign by AB-code (Linear A) or B-code→reading (Linear B).

## Bootstrap from (almost) zero manual annotation

The training data was bootstrapped, not hand-labelled tablet by tablet:

1. **Facsimile detection** from existing bounding-box annotations
   (lineara.xyz / LinearBExplorer imagemaps, SigLA, DĀMOS).
2. **Sign crops** extracted by a triple-guard rule — a crop is kept as training
   data only when transcription alignment, a classifier committee, and box
   geometry all agree. This yields clean labels without manual review.
3. **Classification** trained on the clean crops, easy-to-hard.

A recurring lesson: **clean beats large.** A bigger but looser Linear B crop set
(12.9k crops) scored *worse* than a smaller triple-guarded one (8.5k) — 61.4% vs
65.9% oracle. Label noise costs more than data volume buys.

## Calibration

Classifiers are trained with **label smoothing (0.1) + mixup (0.2)** and
temperature-scaled after training. This matters because an over-confident wrong
prediction is worse than a hedged one for downstream use: it keeps the
recall→precision "cliff" small, so the precise-mode numbers a researcher would
actually trust are close to the optimistic ones.

## The integrity story: finding and fixing train/test leakage

The single most important methodological correction. Early numbers for this
project were **inflated by train/test leakage**: a held-out tablet split was
enforced when *evaluating*, but **not** when extracting classifier crops or
training the detector. Models had therefore seen "held-out" tablets during
training, and the reported accuracies were not what a reviewer would get on
genuinely unseen data.

The fix was structural, not cosmetic:

- A **canonical held-out manifest** (tablet IDs) is now excluded at
  **extraction time** — every crop and every detector box from those tablets is
  removed before training, not just filtered at evaluation.
- The whole pipeline was **rebuilt** from these manifests. Several datasets were
  caught leaking in the process (e.g. one Linear B detector set contained 62
  test tablets).
- **Naming artifacts** that had silently suppressed *measured* accuracy were
  also fixed: GORILA convention markers (`*NNN` vs `NNN`, `+`/`_` ligature
  separators) and AB-code/reading "twin" names were reconciled to one canonical
  label. Reconciling these alone moved the Linear A oracle from 60.9% to 79.3% —
  the model was already right; the scoring was double-counting one sign as two.
- Metric definitions were made consistent with the comparison target (CER is the
  standard flat character error rate, not a more lenient per-line variant).

The numbers in the README are the post-fix, leak-free results.

## Results and the DeepScribe reference

The only published comparable system is DeepScribe (Williams et al., JOCCH 2025)
for Elamite cuneiform — a different script and corpus, so a cross-domain
reference point, not a head-to-head. Against it, leak-free: Linear A leads on
classifier oracle top-1 (79.3% vs 74%); both scripts lead on CER and on
end-to-end sign top-1; both trail on detection mAP and top-5. **The detector is
the bottleneck** — the classifiers score 86–90% on detected signs, but held-out
detection recall caps the end-to-end result around 60–69%. See the README table
and `BENCHMARKS.md`.

## Limitations and honest framing

- Detection recall (~60–70% held-out) is the ceiling; dense/small/damaged signs
  are missed.
- The Linear A oracle (79.3%) rests on a thin box-bearing held-out sample (n=87,
  95% CI ≈ [70, 89]) — reported with its interval.
- Linear B vocabulary is 142 classes vs the ~180 full inventory (clean-data
  limited); rare signs and some logograms are under-represented.
- Models are trained on a **mix of facsimiles and photographs** (roughly half
  the detector data and a majority of the classifier crops are photographic), so
  both are in-distribution. The hard cases are very low-resolution or
  heavily-damaged tablets and the long sign tail — not "photos" per se.

## Reproducing / extending

The detector is a class-agnostic Aegean base: fine-tune it onto Cretan
Hieroglyphic, Cypro-Minoan, or other scripts with modest data, then train a
small classifier for the new sign inventory. Training infrastructure is not in
this public repo (it needs corpus credentials); the held-out manifests and
benchmark commands in `BENCHMARKS.md` make the evaluation reproducible.
