"""BOTHROS inference pipeline: photo -> sign detection -> classification.

Two stages:
  1. Unified Aegean detector (YOLO11s) localises signs (class-agnostic).
  2. Per-script ConvNeXt-Tiny classifier identifies each detected sign.

Inference-only (no ground-truth alignment / evaluation). For the benchmark
methodology see docs/BENCHMARKS.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import timm
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Per-script defaults established by the conf-filter sweep (docs/BENCHMARKS.md).
DETECTOR_CONF = 0.10          # YOLO box threshold (high recall; classifier filters)
CONF_FILTER = {"la": 0.25, "lb": 0.30}   # post-classification confidence floor
DEDUP_IOU = 0.20              # class-aware NMS on final predictions


def _canon_la(s: str) -> str:
    """Normalise GORILA convention markers on a Linear A AB-code."""
    if not s:
        return s
    return s.lstrip("*").replace("+", "_")


@dataclass
class Sign:
    box: list[int]        # [x, y, w, h]
    code: str             # AB-code (LA) or B-code (LB)
    reading: str | None   # phonetic value where known
    confidence: float


class BothrosPipeline:
    def __init__(self, detector_path, classifier_path, script,
                 class_map_path=None, device=None):
        assert script in ("la", "lb"), "script must be 'la' or 'lb'"
        self.script = script
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.detector = YOLO(str(detector_path))

        ck = torch.load(classifier_path, map_location=self.device, weights_only=False)
        self.class_map = ck["class_map"]
        self.idx_to_cls = {v: k for k, v in self.class_map.items()}
        self.temperature = float(ck.get("temperature_scale", 1.0))
        backbone = ck.get("backbone", "convnext_tiny")
        model_id = backbone if "." in backbone else f"{backbone}.in12k_ft_in1k"
        self.model = timm.create_model(model_id, pretrained=False,
                                       num_classes=len(self.class_map))
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.tf = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # B-code -> reading (LB); LA codes are their own label
        self.reading_map = (json.load(open(class_map_path))
                            if class_map_path and Path(class_map_path).exists() else {})

    def _classify(self, crop: Image.Image):
        x = self.tf(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = (self.model(x).squeeze(0) / self.temperature).softmax(0)
        conf, idx = torch.max(probs, 0)
        cls = self.idx_to_cls[int(idx)]
        return cls, float(conf)

    def read(self, image_path, conf_filter=None):
        conf_filter = CONF_FILTER[self.script] if conf_filter is None else conf_filter
        img = Image.open(image_path).convert("RGB")
        det = self.detector(str(image_path), conf=DETECTOR_CONF,
                            imgsz=1280, verbose=False)
        raw = []
        for r in det:
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].cpu().numpy())
                crop = img.crop((x1, y1, x2, y2))
                cls, conf = self._classify(crop)
                if conf < conf_filter:
                    continue
                code = _canon_la(cls) if self.script == "la" else cls
                reading = self.reading_map.get(cls) if self.script == "lb" else \
                    self.reading_map.get(code, None)
                raw.append((Sign([int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                                 code, reading, round(conf, 4)),
                           float(b.conf[0])))
        signs = _dedup([s for s, _ in raw], [dc for _, dc in raw])
        return signs


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _dedup(signs, det_confs):
    """Class-aware NMS: drop the lower-confidence of two same-code overlapping
    boxes (IoU >= DEDUP_IOU)."""
    order = sorted(range(len(signs)), key=lambda i: -det_confs[i])
    kept, dropped = [], set()
    for i in order:
        if i in dropped:
            continue
        kept.append(i)
        for j in order:
            if j == i or j in dropped:
                continue
            if signs[j].code == signs[i].code and \
               _iou(signs[i].box, signs[j].box) >= DEDUP_IOU:
                dropped.add(j)
    # preserve reading order (top-to-bottom, left-to-right)
    kept_signs = [signs[i] for i in kept]
    kept_signs.sort(key=lambda s: (s.box[1], s.box[0]))
    return kept_signs


def render_overlay(image_path, signs, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for s in signs:
        x, y, w, h = s.box
        draw.rectangle([x, y, x + w, y + h], outline=(0, 200, 0), width=2)
        label = s.code + (f"/{s.reading}" if s.reading else "")
        draw.text((x, max(0, y - 11)), label, fill=(0, 200, 0))
    img.save(str(out_path))


def signs_to_json(signs):
    return [asdict(s) for s in signs]
