# Benchmarks

Final leak-free results. Every number is on **held-out test tablets** excluded
at crop-extraction and detector-training time (tablet-ID-level split via a
canonical manifest), not merely filtered at evaluation. See `METHODOLOGY.md` for
how earlier leaked numbers were found and corrected.

## Scorecard

| metric | Linear A | Linear B | DeepScribe (cuneiform ref) | leader |
|---|---|---|---|---|
| Classifier oracle top-1 | **79.3%** | 64.5% | 74% | LA |
| Classifier oracle top-5 | 83.9% | 73.8% | 91% | DS |
| Pipeline E2E sign top-1 | **66.9%** | **62.6%** | 56.3% | LA, LB |
| Pipeline E2E sign top-5 | 72.4% | 64.8% | 79.3% | DS |
| Pipeline per-line F1 | 64.9% | 76.5% | — | — |
| CER (flat, lower better) | ~0.49 | **0.44** | 0.669 | LA, LB |
| Detection mAP@50 | ~0.47 | ~0.47 | 0.78 | DS |

- **Held-out sizes:** LA = 133 tablets (oracle on n=87 box-bearing signs, 95% CI
  ≈ [70, 89]); LB = 320 tablets (DĀMOS).
- **Operating points:** the full-pipeline numbers use the unified detector with a
  post-classification confidence filter of **0.25 (LA)** / **0.30 (LB)**, chosen
  by sweeping cf ∈ {0,…,0.40}. LA peaks at 0.25 (recall falls fast past it); LB
  at 0.30 (multi-line tablets tolerate a tighter precision filter).
- **The detector is the bottleneck:** classifiers score 86–90% top-1/top-5 on
  *detected* signs; held-out detection recall (~60–70%) caps end-to-end.

## Comparison framing

DeepScribe (Williams et al., JOCCH 2025) is Elamite cuneiform — a different
script and corpus. The comparison is a cross-domain reference, not a
head-to-head; CER uses the same standard (flat) definition on both sides. The
honest summary: **competitive with DeepScribe**, leading on classifier oracle
top-1 (LA), CER (both), and end-to-end sign top-1 (both); trailing on detection
mAP and top-5.

## Cross-script transfer

A detector trained on **Linear B only** (never any Linear A tablet) reads Linear
A at **60.7% F1 / 66.2% E2E top-1 zero-shot** — essentially matching a
Linear-A-trained detector. This motivates the single unified Aegean detector and
suggests the base is extensible to other Aegean scripts.

## Reproduction

Held-out manifests and the evaluation script define the numbers. With the
weights downloaded and a held-out tablet image:

```bash
python3 -m bothros read <held_out_tablet>.jpg --script la --conf-filter 0.25
python3 -m bothros read <held_out_tablet>.jpg --script lb --conf-filter 0.30
```

Per-sign oracle / E2E top-k and CER are computed against the corpus
transliterations; the full evaluation harness (with ground-truth alignment) is
part of the development repository, not this inference package.
