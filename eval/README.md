# Reproducible evaluation (classifier oracle)

A stripped-down **classifier-oracle** harness — enough to reproduce the project
README's **Linear A classifier-oracle** number from public artifacts. *Oracle* means
the classifier is handed the correct sign locations (gold boxes), so it measures
classifier quality with the detector removed from the loop.

The tablet **images are not bundled** (the corpora are research-only) — you fetch them
from the open corpus explorer (instructions below). What ships here is the held-out
**ground truth**: per-sign **bounding boxes** (in the
[lineara.xyz](https://github.com/mwenge/lineara.xyz) inscription-image coordinate
space) + **AB-code sign labels** (identified via the SigLA catalogue), CC BY-NC-SA 4.0 —
see each JSON's `_attribution`.

## Files
- `eval_classifier_oracle.py` — the harness (self-contained; helpers vendored in `eval_utils.py`).
- `eval_utils.py` — sign normalisation + confidence intervals, copied verbatim from the
  development repo so results match byte-for-byte.
- `heldout/la_oracle.json` — Linear A held-out GT: per-tablet sign labels (AB-codes) +
  bounding boxes, **no images**. 52 tablets / 87 box-bearing signs.

## Reproduce the Linear A oracle (README: 79.3% top-1)

The 52 LA tablet images come from **[lineara.xyz](https://github.com/mwenge/lineara.xyz)**
(the open Linear A corpus explorer; images ultimately from GORILA). Each `la_oracle.json`
entry's `inscription` field is the tablet ID, and its image path ends
`<ID>-Inscription.jpg` (lineara.xyz's naming). Fetch those inscription images from
lineara.xyz and place them so the relative paths in `la_oracle.json` resolve under
`--images-root`. **Use lineara.xyz's images** — the boxes are in that coordinate space,
so a different scan of the same tablet won't align. Then:

```bash
python3 scripts/download_weights.py        # if not already done
python3 eval/eval_classifier_oracle.py \
    --classifier weights/la_classifier.pth \
    --gt eval/heldout/la_oracle.json \
    --images-root /path/to/your/tablet/images
```

Expected (verified byte-for-byte against the development harness, 2026-06-20):

```
Top-1 79.3% (69/87) · Top-3 81.6% · Top-5 83.9% · mean recall 71.9%
cluster-bootstrap 95% CI (tablets): [70.0%, 89.2%]
```

These match the README's Linear A oracle row exactly.

## Reproducing the Linear B oracle (via linearb.xyz)

The README's **LB** classifier-oracle (64.5%) was measured on **tight, human-traced
sign crops from the [LinearBExplorer / linearb.xyz](https://github.com/mwenge/LinearBExplorer)
imagemap** — each crop is one hand-traced sign box. We rebuilt that exact 904-crop
test set from its manifest, so the boxes + B-code labels are known.

The catch: the box coordinates live in **that project's own image coordinate space**.
They're tightly coupled to its specific facsimile/photo images — cropping the same
boxes from a *different* facsimile scan misaligns badly (verified: ~43/255 mean pixel
difference, dropping the oracle from 64.5% to ~12%). So the LB oracle is reproducible
**from the LinearBExplorer imagemap itself** — its boxes and matching images together,
which is the canonical source we built from — rather than from a standalone box file
that wouldn't align to any other images. Clone
[LinearBExplorer](https://github.com/mwenge/LinearBExplorer) (MIT-licensed), use its
imagemap boxes + images, crop, and run a classifier as in `eval_classifier_oracle.py`.

The LB **pipeline / end-to-end** numbers are reproducible on any LB tablet photo via
`python3 -m bothros read --script lb` with the shipped detector + classifier.

## "Held-out / leak-free"

These tablets were excluded from training at **extraction** time (the canonical
held-out tablet-ID manifest is applied when building crops and training the detector,
not just at evaluation), so the classifier never saw them. The full pipeline-eval
harness (detector + ground-truth alignment, per-line F1, CER) lives in the development
repository; this package ships the oracle slice that's cleanly reproducible.

## Licence

- **GT sign-labels + boxes:** CC BY-NC-SA 4.0 (attribution in each JSON's `_attribution`).
  Sign labels derive from SigLA (Ester Salgarella & Simon Castellan); boxes are original
  to BOTHROS.
- **Harness code:** MIT.
- **Tablet images:** not included — obtain from the research-only corpora.
