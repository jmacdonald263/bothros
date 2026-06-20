"""Publish BOTHROS weights to the Hugging Face Hub (JMacD263/linear-a-linear-b-bothros).

No CLI needed. Get a WRITE token at https://huggingface.co/settings/tokens, then:

    HF_TOKEN=hf_xxx python3 scripts/push_to_hf.py            # private
    HF_TOKEN=hf_xxx python3 scripts/push_to_hf.py --public   # + make public

Uploads the staged weights in release/weights/ — the unified detector, the two
classifiers, and the B-code->reading map — to a (private by default) model repo.
Weights are CC BY-NC-SA 4.0; the upload includes that licence note.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

HF_REPO = "JMacD263/linear-a-linear-b-bothros"
WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
FILES = [
    "yolo_aegean_unified.pt",
    "la_classifier.pth",
    "lb_classifier.pth",
    "lb_class_to_reading.json",
    # full-data 'release' variants (max-capability, NOT benchmarkable)
    "yolo_aegean_release.pt",
    "la_classifier_release.pth",
    "lb_classifier_release.pth",
]

MODEL_CARD = """---
license: cc-by-nc-sa-4.0
pipeline_tag: object-detection
language:
  - gmy
tags:
  - linear-a
  - linear-b
  - aegean-scripts
  - epigraphy
  - ancient-languages
  - object-detection
  - image-classification
  - digital-humanities
  - ocr
  - yolo
  - convnext
  - minoan
  - mycenaean
  - greek
---

# BOTHROS — Linear A & Linear B sign reading from photographs

Weights for the [BOTHROS](https://github.com/jmacdonald263/bothros) pipeline:
photograph an ancient Aegean tablet, get the signs on it by catalogue code and
reading.

**🤗 Try the [live demo](https://huggingface.co/spaces/JMacD263/bothros-demo)** — no install, upload a photo.

> **The name** — a [*bóthros*](https://en.wikipedia.org/wiki/Bothros) (βόθρος) is the pit Odysseus digs in
> the *Odyssey*, pouring libations so the spirits of the dead rise to speak with him. Apt for a
> tool that reads scripts silent for three thousand years.

- **`yolo_aegean_unified.pt`** — one YOLO11s detector localising signs for *both*
  scripts (sign detection is class-agnostic; Linear A and Linear B signs are
  visually cognate).
- **`la_classifier.pth` / `lb_classifier.pth`** — ConvNeXt-Tiny classifiers
  (AB-codes for Linear A; B-codes + readings for Linear B).
- **`lb_class_to_reading.json`** — Linear B B-code → phonetic reading map.

## Results (held-out, leak-free)

| metric | Linear A | Linear B | DeepScribe (cuneiform ref) |
|---|---|---|---|
| classifier oracle top-1 | 79.3% | 64.5% | 74% |
| pipeline E2E sign top-1 | 68.7% | 63.8% | 56.3% |
| pipeline per-line F1 | 64.9% | 76.5% | — |
| CER (lower better) | ~0.48 | 0.44 | 0.669 |

*Cross-script: a Linear-B-only detector reads Linear A at **60.7% F1 zero-shot** — the
basis for shipping one unified `aegean-unified` detector for both scripts.*

## Benchmark vs release weights

Two sets ship here. **Benchmark** (`yolo_aegean_unified.pt`, `la_classifier.pth`,
`lb_classifier.pth`) — strict held-out split; the numbers above are theirs; use these
to reproduce/compare. **Release** (`*_release`) — retrained on the **full data incl.
the held-out split**: max capability + broader coverage (LB 148 vs 142 classes), but
**NOT benchmarkable** (they have seen the test tablets — cite the benchmark numbers,
not these). Fetch with `download_weights.py --release`; run with `bothros read … --release`.

## Usage
```bash
pip install bothros            # or: pip install -e . from the GitHub repo
python3 scripts/download_weights.py
python3 -m bothros read your_tablet.jpg --script la   # or --script lb
```

## Licence
**CC BY-NC-SA 4.0** — derived from research-only corpora: lineara.xyz + GORILA (Linear
A images), SigLA + lineara.xyz (Linear A sign boxes + AB-code catalogue), DĀMOS +
LinearBExplorer (Linear B).
The pipeline source code is MIT (see the GitHub repo). Non-commercial use only.

## Citation
DOI [10.5281/zenodo.20746759](https://doi.org/10.5281/zenodo.20746759) ·
code + docs: https://github.com/jmacdonald263/bothros
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true",
                    help="set the repo public after upload")
    ap.add_argument("--flip-only", action="store_true",
                    help="skip uploads, only set visibility (use with --public)")
    args = ap.parse_args()

    missing = [f for f in FILES if not (WEIGHTS / f).exists()]
    if missing:
        raise SystemExit(f"Missing staged weights in {WEIGHTS}: {missing}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (a WRITE token from "
                         "https://huggingface.co/settings/tokens), e.g.\n"
                         "  HF_TOKEN=hf_xxx python3 scripts/push_to_hf.py --public")
    api = HfApi(token=token)
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
    if not args.flip_only:
        for f in FILES:
            print(f"Uploading {f} ...")
            api.upload_file(path_or_fileobj=str(WEIGHTS / f), path_in_repo=f,
                            repo_id=HF_REPO, repo_type="model")
    # model card (frontmatter drives HF discoverability) — always (re)pushed,
    # so --flip-only can refresh it without re-uploading weights.
    print("Uploading model card (README.md) ...")
    api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md",
                    repo_id=HF_REPO, repo_type="model")

    if args.public:
        # hub 1.x: update_repo_settings (update_repo_visibility was removed)
        api.update_repo_settings(HF_REPO, private=False, repo_type="model")
        print(f"Repo is now PUBLIC: https://huggingface.co/{HF_REPO}")
    else:
        print(f"Uploaded (private): https://huggingface.co/{HF_REPO}  "
              f"(re-run with --public when ready)")


if __name__ == "__main__":
    main()
