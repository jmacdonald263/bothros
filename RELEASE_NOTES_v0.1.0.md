# BOTHROS v0.1.0

First public release. Reads signs on Linear A and Linear B tablet photographs:
one unified Aegean detector (YOLO11) → per-script ConvNeXt-Tiny classifier →
catalogue codes + phonetic readings.

## Highlights
- **Single unified Aegean sign detector** for both scripts (sign detection is
  class-agnostic; LA/LB signs are visually cognate). Extensible to other scripts.
- **Leak-free benchmarks** on held-out DĀMOS (LB) and SigLA (LA) tablets — split
  enforced at extraction time, not just evaluation.
- Competitive with the DeepScribe cuneiform reference: Linear A leads on
  classifier oracle top-1 (79.3% vs 74%); both scripts lead on CER (0.44 / ~0.48
  vs 0.669) and end-to-end sign top-1.

## Results (held-out, leak-free)
| metric | Linear A | Linear B |
|---|---|---|
| classifier oracle top-1 | 79.3% | 64.5% |
| pipeline E2E sign top-1 | 68.7% | 63.8% |
| pipeline per-line F1 | 64.9% | 76.5% |
| CER (lower better) | ~0.48 | 0.44 |

See `docs/BENCHMARKS.md` for full methodology, CIs, and caveats (the detector is
the bottleneck; the LA oracle is a thin-n result reported with its interval).

## Install
```bash
pip install -e .
python3 scripts/download_weights.py   # weights from huggingface.co/JMacD263/bothros (public, no token)
python3 -m bothros read your_tablet.jpg --script la
```

## Licence
Code MIT; model weights CC BY-NC-SA 4.0 (hosted on Hugging Face).
