"""EasyOCR text extraction with blueprint-specific tagging (dimension, room label, scale, annotation).
The reader is created once; if the model cannot be loaded or downloaded, `available` is False with the reason."""
from __future__ import annotations

import os
import re

import cv2
import numpy as np

ROOM_KEYWORDS = ["bedroom", "bath", "kitchen", "living", "dining", "garage", "office", "closet", "hall", "entry",
                 "porch"]  # fmt: skip
DIMENSION_PATTERNS = [r"\d+['\"]", r"\d+\.?\d*\s*(m|mm|cm|ft|in)\b", r"\d+\s*[xX]\s*\d+", r"\d+'-\d+\""]
COLORS = {"dimension": (75, 184, 242), "room_label": (197, 209, 79), "scale": (138, 113, 240),
          "annotation": (200, 200, 200)}  # fmt: skip


def classify_text(text: str) -> str:
    t = text.strip()
    for pat in DIMENSION_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return "dimension"
    if any(k in t.lower() for k in ROOM_KEYWORDS):
        return "room_label"
    if "scale" in t.lower() or re.search(r"\b1:\d+\b", t):
        return "scale"
    return "annotation"


class BlueprintOCR:
    _reader = None
    _error: str | None = None

    def __init__(self, allow_download: bool | None = None):
        if allow_download is None:
            allow_download = os.getenv("EASYOCR_DOWNLOAD", "1") != "0"
        if BlueprintOCR._reader is None and BlueprintOCR._error is None:
            try:
                import easyocr

                BlueprintOCR._reader = easyocr.Reader(["en"], gpu=False, download_enabled=allow_download, verbose=False)
            except Exception as e:  # noqa: BLE001 - surfaced as `available=False`
                BlueprintOCR._error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return BlueprintOCR._reader is not None

    @property
    def unavailable_reason(self) -> str | None:
        return BlueprintOCR._error

    @staticmethod
    def preprocess(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 110:  # blueprint-style (white on blue): invert so text is dark on light
            gray = 255 - gray
        return cv2.bilateralFilter(gray, 5, 40, 40)

    def extract_text(self, bgr: np.ndarray) -> list[dict]:
        if not self.available:
            raise RuntimeError(f"EasyOCR unavailable: {self._error}")
        results = BlueprintOCR._reader.readtext(self.preprocess(bgr))
        out = []
        for bbox, text, conf in results:
            out.append({"text": text, "bbox": [[int(x), int(y)] for x, y in bbox], "confidence": round(float(conf), 3),
                        "type": classify_text(text)})  # fmt: skip
        return out

    @staticmethod
    def visualize(bgr: np.ndarray, results: list[dict]) -> np.ndarray:
        out = bgr.copy()
        for r in results:
            pts = np.array(r["bbox"], np.int32)
            color = COLORS.get(r["type"], COLORS["annotation"])
            cv2.polylines(out, [pts], True, color, 2)
            x, y = pts[0]
            cv2.putText(out, r["text"][:24], (int(x), max(12, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)  # fmt: skip
        return out
