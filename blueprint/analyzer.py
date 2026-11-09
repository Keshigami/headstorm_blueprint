"""The pipeline: classification + OCR + segmentation (+ optional detection) -> JSON report and overlays."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from blueprint.classifier import load_classifier
from blueprint.detection import BlueprintDetection
from blueprint.ocr_module import BlueprintOCR
from blueprint.segmentation import BlueprintSegmentation

HERE = Path(__file__).resolve().parents[1]


def to_bgr(image) -> np.ndarray:
    if isinstance(image, str | Path):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(image)
        return img
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class BlueprintAnalyzer:
    def __init__(self, models_dir: Path | None = None, with_ocr: bool = True):
        self.classifier = load_classifier(models_dir or HERE / "models")
        self.ocr = BlueprintOCR() if with_ocr else None
        self.segmentation = BlueprintSegmentation()
        self.detection = BlueprintDetection()

    def status(self) -> dict:
        return {
            "classifier": {"method": self.classifier.method, "note": self.classifier.note},
            "ocr": {"available": bool(self.ocr and self.ocr.available),
                    "reason": self.ocr.unavailable_reason if self.ocr else "disabled"},  # fmt: skip
            "segmentation": "opencv",
            "detection": self.detection.note,
        }

    def analyze(self, image, include_visualization: bool = True) -> dict:
        bgr = to_bgr(image)
        results: dict = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                         "image_size": [int(bgr.shape[1]), int(bgr.shape[0])], "timings_ms": {}}  # fmt: skip

        t = time.perf_counter()
        results["classification"] = self.classifier.predict(bgr)
        results["timings_ms"]["classification"] = round((time.perf_counter() - t) * 1000, 1)

        t = time.perf_counter()
        text: list[dict] = []
        if self.ocr and self.ocr.available:
            text = self.ocr.extract_text(bgr)
            results["ocr"] = {
                "total_text_regions": len(text),
                "dimensions": [r["text"] for r in text if r["type"] == "dimension"],
                "room_labels": [r["text"] for r in text if r["type"] == "room_label"],
                "scale": [r["text"] for r in text if r["type"] == "scale"],
                "annotations": [r["text"] for r in text if r["type"] == "annotation"],
                "all_text": text,
            }
        else:
            results["ocr"] = {"skipped": self.ocr.unavailable_reason if self.ocr else "OCR disabled"}
        results["timings_ms"]["ocr"] = round((time.perf_counter() - t) * 1000, 1)

        t = time.perf_counter()
        seg = self.segmentation.segment(bgr)
        results["segmentation"] = self.segmentation.to_dict(seg)
        results["timings_ms"]["segmentation"] = round((time.perf_counter() - t) * 1000, 1)

        detections = self.detection.detect(bgr)
        results["detections"] = {"note": self.detection.note, "elements": detections}

        if include_visualization:
            seg_img = self.segmentation.visualize(bgr, seg, show_rooms=True, show_walls=True)
            ocr_img = self.ocr.visualize(bgr, text) if text else bgr.copy()
            if detections:
                ocr_img = self.detection.visualize(ocr_img, detections)
            label = f"{results['classification']['class']} ({results['classification']['confidence']:.0%}, " \
                    f"{results['classification']['method']})"  # fmt: skip
            for img in (seg_img, ocr_img):
                cv2.rectangle(img, (10, 10), (20 + 9 * len(label), 44), (0, 0, 0), -1)
                cv2.putText(img, label, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (197, 209, 79), 2, cv2.LINE_AA)
            results["_images"] = {"segmentation": to_pil(seg_img), "ocr": to_pil(ocr_img)}
        return results

    @staticmethod
    def report(results: dict) -> dict:
        """JSON-serialisable subset of `analyze` output."""
        return {k: v for k, v in results.items() if not k.startswith("_")}

    @staticmethod
    def summary(results: dict) -> str:
        c = results["classification"]
        s = results["segmentation"]
        o = results["ocr"]
        lines = [f"**Type: {c['class'].replace('_', ' ')}** at {c['confidence']:.0%} ({c['method']}: {c['note']})",
                 f"Segmentation: {s['num_rooms']} enclosed regions, {s['num_walls']} wall segments, "
                 f"mean thickness {s['mean_wall_thickness']} px"]  # fmt: skip
        if "skipped" in o:
            lines.append(f"OCR skipped: {o['skipped']}")
        else:
            lines.append(f"OCR: {o['total_text_regions']} text regions; dimensions {o['dimensions'][:6]}; "
                         f"room labels {o['room_labels'][:6]}; scale {o['scale'][:2]}")  # fmt: skip
        lines.append(f"Detection: {results['detections']['note']}" + (
            f"; {len(results['detections']['elements'])} elements" if results["detections"]["elements"] else ""))
        lines.append("Timings ms: " + ", ".join(f"{k} {v}" for k, v in results["timings_ms"].items()))
        return "\n\n".join(lines)
