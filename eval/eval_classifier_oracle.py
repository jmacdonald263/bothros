"""Oracle classifier-only evaluation: feed GT bounding boxes directly to the
classifier and measure top-K accuracy. Isolates classifier quality (no detector).
This reproduces the README's "classifier oracle top-1/3/5" numbers.

You supply the tablet IMAGES (the corpora are research-only and not bundled — see
README.md); the GT sign-labels + boxes ship in eval/heldout/.

Usage (run from the repo root, after `download_weights.py`):
    # Linear A
    python3 eval/eval_classifier_oracle.py \\
        --classifier weights/la_classifier.pth \\
        --gt eval/heldout/la_oracle.json \\
        --images-root .

    # Linear B (needs the B-code -> reading map shipped with the weights)
    python3 eval/eval_classifier_oracle.py \\
        --classifier weights/lb_classifier.pth \\
        --gt eval/heldout/lb_oracle.json \\
        --mapping weights/lb_class_to_reading.json \\
        --images-root .

`--images-root` is prepended to each GT entry's relative image path. Place the
tablet images so those relative paths resolve (the inscription ID is in each
entry's "inscription" field to help you locate them in the corpus).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
import timm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import _normalize_sign, cluster_bootstrap_ci, wilson_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--mapping", default=None, type=Path,
                    help="Optional class->reading mapping JSON (Linear B)")
    ap.add_argument("--images-root", default=Path("."), type=Path,
                    help="Prepended to each GT entry's relative image path (default: .)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Per-item records JSON (default: oracle_<clf>_<gt>.json in cwd)")
    args = ap.parse_args()
    if args.output is None:
        args.output = Path(f"oracle_{args.classifier.stem}_{args.gt.stem}.json")

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.classifier, map_location=device, weights_only=False)
    class_map = ck["class_map"]
    idx_to_cls = {v: k for k, v in class_map.items()}
    Tsc = float(ck.get("temperature_scale", 1.0))
    backbone = ck.get("backbone", "convnext_tiny")
    model_id = backbone if "." in backbone else f"{backbone}.in12k_ft_in1k"
    model = timm.create_model(model_id, pretrained=False, num_classes=len(class_map))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    mapping = json.load(open(args.mapping)) if args.mapping else {}
    def to_reading(c): return mapping.get(c, c)

    def canon(s):  # strip '*' + unify ligature sep: *301==301, *a+*b==a_b
        return s.lstrip("*").replace("+", "_") if s else s

    gt = json.load(open(args.gt))
    total = top1 = top3 = top5 = 0
    per_class_n: Counter = Counter()
    per_class_correct: Counter = Counter()
    n_tablets = 0
    missing = 0
    records: list[dict] = []

    for img_path_key, entry in gt.items():
        if img_path_key.startswith("_"):  # skip _license / _attribution metadata keys
            continue
        img_path = args.images_root / img_path_key
        if not img_path.exists():
            missing += 1
            continue
        img = Image.open(img_path).convert("RGB")
        gt_signs = entry.get("gt_signs", [])
        gt_bboxes = entry.get("gt_bboxes", [])
        if not gt_signs or not gt_bboxes:
            continue
        n_pairs = min(len(gt_signs), len(gt_bboxes))
        n_tablets += 1
        for i in range(n_pairs):
            b = gt_bboxes[i]
            x = b.get("x", 0); y = b.get("y", 0)
            w = b.get("width", 0); h = b.get("height", 0)
            if w <= 0 or h <= 0:
                continue
            try:
                crop = img.crop((x, y, x + w, y + h))
            except Exception:
                continue
            inp = tf(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(inp).squeeze(0) / Tsc
                probs = logits.softmax(0)
            top5_vals, top5_idx = torch.topk(probs, k=min(5, probs.shape[0]))
            top5_classes = [canon(_normalize_sign(to_reading(idx_to_cls[int(top5_idx[k].item())])))
                            for k in range(top5_vals.shape[0])]
            target = canon(_normalize_sign(gt_signs[i]))
            if not target:
                continue
            total += 1
            per_class_n[target] += 1
            if top5_classes[0] == target:
                top1 += 1
                per_class_correct[target] += 1
            if target in top5_classes[:3]:
                top3 += 1
            if target in top5_classes[:5]:
                top5 += 1
            records.append({
                "tablet": img_path_key, "gt_idx": i, "target": target,
                "top5": top5_classes,
                "top5_probs": [round(float(v), 5) for v in top5_vals.tolist()],
                "correct1": top5_classes[0] == target,
                "correct3": target in top5_classes[:3],
                "correct5": target in top5_classes[:5],
            })

    if total == 0:
        print("No oracle eval data scored.")
        if missing:
            print(f"  {missing} tablet image(s) not found under --images-root "
                  f"{args.images_root} — obtain them from the corpora (see eval/README.md).")
        return

    mean_recall = sum(per_class_correct[c] / per_class_n[c]
                      for c in per_class_n if per_class_n[c] > 0) / len(per_class_n)
    per_tablet: dict[str, list] = {}
    for r in records:
        per_tablet.setdefault(r["tablet"], []).append(r["correct1"])
    w_lo, w_hi = wilson_ci(top1, total)
    cb = cluster_bootstrap_ci(per_tablet, lambda xs: sum(xs) / len(xs))

    print("\n=== ORACLE CLASSIFIER EVAL ===")
    print(f"Classifier: {args.classifier}")
    print(f"GT source: {args.gt}")
    if missing:
        print(f"NOTE: {missing} tablet image(s) not found — scored only what resolved.")
    print(f"Tablets: {n_tablets}, GT-sign-bbox pairs scored: {total}, classes touched: {len(per_class_n)}\n")
    print(f"Top-1 accuracy:  {top1/total*100:.1f}%  ({top1}/{total})")
    print(f"  Wilson 95% CI (iid signs):          [{w_lo*100:.1f}%, {w_hi*100:.1f}%]")
    print(f"  Cluster-bootstrap 95% CI (tablets): [{cb['ci_lo']*100:.1f}%, {cb['ci_hi']*100:.1f}%]")
    print(f"Top-3 accuracy:  {top3/total*100:.1f}%  ({top3}/{total})")
    print(f"Top-5 accuracy:  {top5/total*100:.1f}%  ({top5}/{total})")
    print(f"Mean recall (balanced acc): {mean_recall*100:.1f}%")

    per_class = {c: {"n": per_class_n[c], "recall": per_class_correct[c] / per_class_n[c]}
                 for c in sorted(per_class_n)}
    out = {
        "meta": {
            "classifier": str(args.classifier), "gt": str(args.gt),
            "gt_sha256": hashlib.sha256(args.gt.read_bytes()).hexdigest()[:12],
            "mapping": str(args.mapping) if args.mapping else None,
            "temperature_scale": Tsc, "date": date.today().isoformat(),
            "n_tablets": n_tablets, "n_pairs": total,
            "top1": top1 / total, "top3": top3 / total, "top5": top5 / total,
            "mean_recall": mean_recall,
            "top1_wilson_ci": [w_lo, w_hi], "top1_cluster_ci": [cb["ci_lo"], cb["ci_hi"]],
        },
        "per_class": per_class, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nPer-item records → {args.output}")


if __name__ == "__main__":
    main()
