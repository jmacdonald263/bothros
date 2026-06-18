"""BOTHROS CLI:  python3 -m bothros read <image> --script {la|lb}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import BothrosPipeline, render_overlay, signs_to_json

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"
DEFAULTS = {
    "detector": WEIGHTS / "yolo_aegean_unified.pt",
    "la": WEIGHTS / "la_classifier.pth",
    "lb": WEIGHTS / "lb_classifier.pth",
    # B-code -> reading map is bundled in-package (small, public lookup)
    "lb_map": Path(__file__).resolve().parent / "lb_class_to_reading.json",
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bothros")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="read the signs on a tablet photo")
    r.add_argument("image", type=Path)
    r.add_argument("--script", required=True, choices=["la", "lb"])
    r.add_argument("--detector", type=Path, default=None)
    r.add_argument("--classifier", type=Path, default=None)
    r.add_argument("--class-map", type=Path, default=None,
                   help="B-code->reading JSON (LB)")
    r.add_argument("--conf-filter", type=float, default=None,
                   help="override per-script default (la 0.25 / lb 0.30)")
    r.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args(argv)

    detector = args.detector or DEFAULTS["detector"]
    classifier = args.classifier or DEFAULTS[args.script]
    class_map = args.class_map or (DEFAULTS["lb_map"] if args.script == "lb" else None)
    for p in (detector, classifier):
        if not Path(p).exists():
            sys.exit(f"Missing weights: {p}\nRun: python3 scripts/download_weights.py")

    pipe = BothrosPipeline(detector, classifier, args.script, class_map)
    signs = pipe.read(args.image, conf_filter=args.conf_filter)

    stem = Path(args.image).stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{stem}_signs.json"
    overlay_path = args.out_dir / f"{stem}_overlay.png"
    json.dump(signs_to_json(signs), open(json_path, "w"), indent=2, ensure_ascii=False)
    render_overlay(args.image, signs, overlay_path)

    print(f"{len(signs)} signs read from {args.image.name} ({args.script.upper()})")
    for s in signs:
        rd = f" [{s.reading}]" if s.reading else ""
        print(f"  {s.code}{rd}  conf={s.confidence:.2f}  box={s.box}")
    print(f"\nJSON:    {json_path}\nOverlay: {overlay_path}")


if __name__ == "__main__":
    main()
