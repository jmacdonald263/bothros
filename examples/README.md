# Example tablets

The Linear A / Linear B facsimiles and photographs used to develop BOTHROS come
from corpora with research-only / non-commercial terms (GORILA, DĀMOS, SigLA,
lineara.xyz, LinearBExplorer) and **cannot be redistributed here**.

To try the pipeline on a real tablet:

1. Obtain a tablet image from one of the open corpora, e.g.:
   - Linear A: https://sigla.phis.me/ or https://lineara.xyz
   - Linear B: https://damos.hf.uio.no/ or [LinearBExplorer](https://github.com/mwenge/LinearBExplorer)
2. Run:
   ```bash
   python3 -m bothros read path/to/your_tablet.jpg --script la   # or --script lb
   ```

Outputs `<name>_signs.json` (codes, readings, boxes, confidences) and
`<name>_overlay.png` (annotated image).

> Tip: the models were trained on **both** facsimile line-drawings and tablet
> photographs, so both work. Clear, high-resolution images do best; very
> low-resolution or heavily-damaged tablets are the hard cases (the detector
> under- or over-fires on them).
