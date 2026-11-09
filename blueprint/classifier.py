"""Blueprint type classification: elevation, floor_plan, section, site_plan.

Two implementations with the same `predict` contract, both reporting their `method` so the UI is honest:
- EfficientNetClassifier: torchvision EfficientNet-B0 ImageNet backbone (frozen) + a linear head trained by
  train.py on synthetic drawings (models/classifier.pt). Used when that file exists and torch loads.
- HeuristicClassifier: hand-written rules on line statistics (horizontal/vertical ratio, enclosed regions,
  circles, thick bands, aspect ratio). Always available; the fallback and a baseline to beat.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from blueprint import CLASSES


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    m = max(scores.values())
    exp = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exp.values())
    return {k: round(v / z, 4) for k, v in exp.items()}


def line_features(bgr: np.ndarray) -> dict[str, float]:
    """Cheap layout statistics used by the heuristic classifier (and shown in the report)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 110:
        gray = 255 - gray
    h, w = gray.shape
    scale = 800 / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = gray.shape
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    hlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    dark = ((gray < 128).astype(np.uint8)) * 255  # filled regions (slabs, cut walls) survive a global threshold
    thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 12)))
    walls = cv2.dilate(cv2.bitwise_or(hlines, vlines), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(cv2.bitwise_not(walls), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    total = h * w
    enclosed = 0
    for c in contours:
        a = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        if 0.004 * total < a < 0.5 * total and a / max(bw * bh, 1) > 0.6:
            enclosed += 1
    round_blobs = 0  # tree symbols on site plans: closed, near-circular outlines
    for c in cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
        a, per = cv2.contourArea(c), cv2.arcLength(c, True)
        if 150 < a < 6000 and per > 0 and 4 * math.pi * a / (per * per) > 0.82:
            round_blobs += 1
    hp, vp = float((hlines > 0).mean()), float((vlines > 0).mean())
    return {
        "aspect_ratio": round(w / h, 3),
        "ink_ratio": round(float((binary > 0).mean()), 4),
        "h_line_ratio": round(hp, 4),
        "v_line_ratio": round(vp, 4),
        "h_over_v": round(hp / (vp + 1e-6), 3),
        "thick_band_ratio": round(float((thick > 0).mean()), 4),
        "enclosed_regions": enclosed,
        "round_blobs": round_blobs,
    }


class HeuristicClassifier:
    method = "heuristic"
    note = "hand-set rules on line statistics (not a trained model)"

    def predict(self, bgr: np.ndarray) -> dict:
        f = line_features(bgr)
        # Hand-set rules from the synthetic drawings: trees are round blobs (site plan), cut slabs are thick dark
        # bands (section), floor plans have as many vertical as horizontal lines, elevations are mostly horizontal.
        scores = {
            "site_plan": 1.5 * min(f["round_blobs"], 10) - 2.0,
            "section": 150 * f["thick_band_ratio"] - 2.0,
            "floor_plan": 3.0 - 1.2 * min(f["h_over_v"], 8) + 100 * f["v_line_ratio"],
            "elevation": 0.8 * min(f["h_over_v"], 7) - 1.5 - 150 * f["thick_band_ratio"],
        }
        probs = _softmax(scores)
        best = max(probs, key=probs.get)
        return {"class": best, "confidence": probs[best], "probabilities": probs, "method": self.method,
                "note": self.note, "features": f}  # fmt: skip


class EfficientNetClassifier:
    method = "efficientnet_b0"

    def __init__(self, head_path: Path):
        import torch
        from torchvision import models, transforms

        self.torch = torch
        state = torch.load(head_path, map_location="cpu", weights_only=False)
        self.classes = state["classes"]
        self.metrics = state.get("metrics", {})
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)  # cached or downloaded once
        backbone.classifier = torch.nn.Identity()
        self.backbone = backbone.eval()
        self.head = torch.nn.Linear(1280, len(self.classes))
        self.head.load_state_dict(state["head"])
        self.head.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])  # fmt: skip
        acc = self.metrics.get("accuracy")
        self.note = "ImageNet backbone frozen, linear head trained on synthetic drawings" + (
            f" (held-out accuracy {acc:.0%})" if acc is not None else "")

    def embed(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self.torch.no_grad():
            return self.backbone(self.transform(rgb).unsqueeze(0))

    def predict(self, bgr: np.ndarray) -> dict:
        with self.torch.no_grad():
            logits = self.head(self.embed(bgr))
            probs = self.torch.softmax(logits, dim=1)[0].tolist()
        p = {c: round(v, 4) for c, v in zip(self.classes, probs, strict=True)}
        best = max(p, key=p.get)
        return {"class": best, "confidence": p[best], "probabilities": p, "method": self.method, "note": self.note,
                "features": line_features(bgr)}  # fmt: skip


def load_classifier(models_dir: Path):
    """EfficientNet head if trained and loadable, else the heuristic (with the reason in `.note`)."""
    head = models_dir / "classifier.pt"
    if head.exists():
        try:
            return EfficientNetClassifier(head)
        except Exception as e:  # noqa: BLE001 - torch/weights problems fall back to the heuristic
            h = HeuristicClassifier()
            h.note += f"; EfficientNet unavailable: {type(e).__name__}: {e}"
            return h
    h = HeuristicClassifier()
    h.note += "; no models/classifier.pt (run `just headstorm_blueprint-train`)"
    return h


__all__ = ["CLASSES", "EfficientNetClassifier", "HeuristicClassifier", "line_features", "load_classifier"]
