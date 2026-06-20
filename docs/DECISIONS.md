# Architecture decisions

Why the pipeline is shaped the way it is. Numbers trace to
[`BENCHMARKS.md`](BENCHMARKS.md) / the development repo's `FINDINGS.md`; where the
record doesn't settle a question, this says so rather than inventing a reason.

---

## 1. One YOLO11 detector, not a per-script HRCenterNet ensemble

**HRCenterNet was not a failure** — it was the Phase-1 detector and it worked. Fine-tuned
on facsimiles it reached **F1 = 0.873 @ IoU≥0.3** (P=0.99, R=0.78) and **0.837 @ IoU≥0.5**,
the latter *above* the measured inter-annotator agreement ceiling. Its weakness was
narrower: pretrained zero-shot it was useless (F1 ≈ 0.09 — it's a Chinese-character
detector), so it needed separate fine-tuning per script and was facsimile-tuned, while
the project's real target is **photographs**.

The pipeline moved HRCenterNet → YOLO8s → **YOLO11s** for reasons the record supports:

- **One model spans both scripts.** Sign detection is class-agnostic and Linear A/B
  signs are visually cognate, so a single detector localises either script. This is
  empirically validated, not assumed: a **Linear-B-only detector reads Linear A at
  ~61% F1 zero-shot** (see [`BENCHMARKS.md`](BENCHMARKS.md) cross-script transfer). One
  unified model replaces a pair of script-specific HRCenterNet models.
- **Built-in regularisation.** A RetinaNet (ResNet-50+FPN) baseline overfit this
  dataset rapidly (best val at epoch 2–3, then diverged). YOLO's mosaic/mixup/scale
  augmentation regularises without a custom pipeline.
- **Simpler, photo-robust validation.** Labels are validated by detection *count*, not
  by matching classifications to a transcription — faster, and doesn't require the
  classifier to work on photos during detector development.
- **Operational simplicity over a marginal gain.** A 3-detector ensemble was explicitly
  cancelled: a 2-detector test showed ensembling *added noise* in precise mode, and the
  triple ensemble's edge (≈78% vs ≈75% for a single YOLO8s) didn't justify the
  complexity. One maintained model beats a brittle committee.

**Honest limits.** There is no direct HRCenterNet-vs-YOLO11 mAP head-to-head on an
identical test set (they were strong in different domains/eras), and the docs don't
record why YOLO11s specifically over 8s/10s — it's the current ultralytics line the
last detectors were trained on.

---

## 2. ConvNeXt-Tiny as the per-script classifier

A small ConvNeXt-Tiny (~28M params, ~107 MB) identifies each cropped sign.

- **Accuracy.** 92.2% val on Linear B crops vs 87.8% for a ResNet-18 baseline (+4.4 pp);
  ~84.5% (LB) / ~89.5% (LA) top-1 on isolated gold crops.
- **Bigger didn't help.** ConvNeXt-**Base** (88M) converged to the *same* 58–60% test
  top-1 as Tiny — the ceiling is the data/detector, not classifier capacity. EfficientNet-B0
  matched ResNet-18 (no gain); a DINOv2 backbone was a dead end (domain gap too large).
  So Tiny is the size/accuracy sweet spot, not a compromise.
- **Calibration is deliberate.** Trained with **label smoothing 0.1 + mixup 0.2** and
  temperature-scaled, because an over-confident wrong prediction is worse than a hedged
  one downstream. This keeps the recall→precision "cliff" small, so the precise-mode
  numbers a researcher would trust stay close to the optimistic ones.
- **Recipe note.** Fine-tune from ImageNet weights at **LR=1e-4** — LR=1e-3 collapses to
  chance.

**Honest limits.** The comparison set is ResNet-18 / EfficientNet-B0 / ConvNeXt-Base /
DINOv2 — there's no systematic sweep against ViT/Swin variants. The claim is "Tiny is a
well-justified, well-calibrated choice," not "provably optimal across all backbones."

---

See [`BENCHMARKS.md`](BENCHMARKS.md) for the full evaluation record and the
development repo's `FINDINGS.md` for the phase-by-phase history behind these calls.
