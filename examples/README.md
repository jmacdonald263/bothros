# Example tablets

Three ways to try BOTHROS on a real tablet.

## 1. The live demo (no install)

Upload any tablet photo at the **[🤗 bothros-demo](https://huggingface.co/spaces/JMacD263/bothros-demo)**.

## 2. Run on a clearly-licensed example image

These are openly licensed tablet **photographs** on Wikimedia Commons — download one
and run the pipeline (URLs verified resolving + licensed as listed, 2026-06-20):

```bash
# Linear B — Pylos tablet Tn 316 (National Archaeological Museum, Athens)
curl -L -o pylos_Tn316.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/2/25/Linear_B_tablet%2C_Pylos%2C_13th_century_BC%2C_NAMA_Tn_316%2C_191137.jpg"
python3 -m bothros read pylos_Tn316.jpg --script lb

# Linear A — tablet from Zakros (Archaeological Museum of Sitia)
curl -L -o zakros_LA.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/6/6b/Linear_A_tablet_from_Zakros%2C_AM_Sitia%2C_258622.jpg"
python3 -m bothros read zakros_LA.jpg --script la
```

| image | script | licence | credit |
|---|---|---|---|
| [Pylos Tn 316](https://commons.wikimedia.org/wiki/File:Linear_B_tablet,_Pylos,_13th_century_BC,_NAMA_Tn_316,_191137.jpg) | Linear B | CC BY-SA 4.0 | © Zde, via Wikimedia Commons |
| [Zakros tablet](https://commons.wikimedia.org/wiki/File:Linear_A_tablet_from_Zakros,_AM_Sitia,_258622.jpg) | Linear A | CC BY-SA 4.0 | © Zde, via Wikimedia Commons |
| [Knossos KN 1815](https://commons.wikimedia.org/wiki/File:Knossos_KN_1815.jpg) | Linear B | CC0 (public domain) | ShlomoKatzav — no attribution needed (lower-res) |

CC BY-SA 4.0 requires attribution + share-alike if you *redistribute* the image; CC0
has no conditions. These are museum photographs (not facsimile line-drawings) — the
pipeline was trained on both and handles either.

> These are out-of-domain photos (museum display shots, oblique lighting, glass
> glare), so expect the honest end-to-end behaviour from the README, not the oracle
> numbers. A real test, not a cherry-pick.

## 3. The full development corpora

The Linear A / Linear B facsimiles and photographs used to *develop* BOTHROS come
from corpora with research-only / non-commercial terms and **cannot be redistributed
here**. To work with them directly:

- Linear A: [SigLA](https://sigla.phis.me/) (Salgarella & Castellan) or [lineara.xyz](https://lineara.xyz)
- Linear B: [DĀMOS](https://damos.hf.uio.no/) or [LinearBExplorer](https://github.com/mwenge/LinearBExplorer); photographs via [LiBER](https://liber.cnr.it)

Output for any run: `<name>_signs.json` (codes, readings, boxes, confidences) and
`<name>_overlay.png` (annotated image). Clear, high-resolution images do best;
very low-resolution or heavily-damaged tablets are the hard cases.
