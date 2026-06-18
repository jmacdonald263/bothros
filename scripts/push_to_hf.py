"""Publish BOTHROS weights to the Hugging Face Hub (JMacD263/bothros).

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

HF_REPO = "JMacD263/bothros"
WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
FILES = [
    "yolo_aegean_unified.pt",
    "la_classifier.pth",
    "lb_classifier.pth",
    "lb_class_to_reading.json",
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
---

# BOTHROS — Aegean sign reading from photographs (Linear A + Linear B)

Weights for the [BOTHROS](https://github.com/jmacdonald263/bothros) pipeline:
photograph an ancient Aegean tablet, get the signs on it by catalogue code and
reading.

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
| pipeline E2E sign top-1 | 66.9% | 62.6% | 56.3% |
| pipeline per-line F1 | 64.9% | 76.5% | — |
| CER (lower better) | ~0.49 | 0.44 | 0.669 |

## Usage
```bash
pip install bothros            # or: pip install -e . from the GitHub repo
python3 scripts/download_weights.py
python3 -m bothros read your_tablet.jpg --script la   # or --script lb
```

## Licence
**CC BY-NC-SA 4.0** — derived from research-only corpora (GORILA, DĀMOS, SigLA).
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
