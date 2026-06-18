# Model Card — Unified Aegean Sign Detector (`aegean-unified`)

## Overview
A single YOLO11s object detector that localises individual signs on photographs
and facsimiles of Linear A and Linear B tablets. Detection is **class-agnostic**
(one class, `sign`): Linear A and Linear B signs are visually cognate, so one
detector serves both scripts. It is the first stage of the BOTHROS pipeline; a
script-specific classifier identifies each detected sign.

- **Architecture:** YOLO11s (Ultralytics), inference resolution 1280px.
- **Weights:** `yolo_aegean_full_chinit_best.pt` (~19 MB).
- **Input:** RGB tablet image. **Output:** sign bounding boxes + confidences.

## Training data
- **Linear A** (2,976 images): SigLA + lineara.xyz human-traced sign boxes
  (`la_yolo_expanded_v5`).
- **Linear B** (2,150 images): DĀMOS-validated + linearb.xyz imagemap +
  corpus.db boxes (`yolo_lb_gt_v1`), **with all test-split tablets removed**.
- Warm-started from `clean_human_1280` (a balanced LA+LB base), then trained on
  the combined full set for 100 epochs (AdamW, no horizontal flip — Aegean signs
  are not mirror-symmetric).

## Leak safety
Test tablets are excluded at the **tablet-ID level** from the training set
(checked against the canonical held-out manifests; the build also caught and
removed 62 Linear B test tablets that an earlier dataset had leaked). No
held-out tablet contributes any training box.

## Performance (held-out, with the per-script classifiers)
| script | conf-filter | per-line F1 | precision | recall |
|---|---|---|---|---|
| Linear A | 0.25 | 64.9% | 70% | 61% |
| Linear B | 0.30 | 76.5% | 86% | 69% |

Detection mAP@50 on held-out is ~0.47 — **the detector is the pipeline
bottleneck.** Held-out recall caps end-to-end performance; the classifiers score
substantially higher on the signs the detector does find.

## Intended use
Research on Aegean scripts: sign localisation as input to classification,
transliteration assistance, corpus annotation. **Not** for commercial use
(weights are CC BY-NC-SA 4.0).

## Limitations
- Recall ~60–70% on held-out tablets; dense, small, or damaged signs are missed.
- Trained on facsimiles + the available photo corpora; very different imaging
  conditions (lighting, resolution, medium) may degrade detection.
- Class-agnostic: it finds signs, it does not identify them (that is stage 2).

## Extensibility
Because detection is class-agnostic and Aegean glyphs share morphology, this
base transfers across scripts: a Linear-B-only detector already reads Linear A
at ~61% F1 zero-shot. Researchers can fine-tune this detector onto Cretan
Hieroglyphic, Cypro-Minoan, or other scripts with modest data.
