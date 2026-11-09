"""Optional Roboflow object detection (doors, windows, columns). Enabled only with ROBOFLOW_API_KEY and the
`roboflow` extra installed; otherwise `detect` returns [] and `note` says it was skipped. No mock detections."""
from __future__ import annotations

import os

import cv2
import numpy as np

COLORS = {"door": (48, 59, 255), "window": (255, 122, 0), "column": (100, 217, 76), "staircase": (0, 204, 255)}


class BlueprintDetection:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.project_id = os.getenv("ROBOFLOW_PROJECT_ID", "blueprint-elements")
        self.version = os.getenv("ROBOFLOW_VERSION", "1")
        self.model = None
        if not self.api_key:
            self.note = "Roboflow detection skipped: ROBOFLOW_API_KEY not set"
            return
        try:
            from roboflow import Roboflow

            project = Roboflow(api_key=self.api_key).workspace().project(self.project_id)
            self.model = project.version(self.version).model
            self.note = f"Roboflow detection enabled: {self.project_id} v{self.version}"
        except ImportError:
            self.note = "Roboflow detection skipped: install the `roboflow` extra (uv sync --extra roboflow)"
        except Exception as e:  # noqa: BLE001 - network/auth problems are reported, not fatal
            self.note = f"Roboflow initialisation failed: {type(e).__name__}: {e}"

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def detect(self, bgr: np.ndarray) -> list[dict]:
        if not self.enabled:
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prediction = self.model.predict(rgb, confidence=40, overlap=30).json()
        return prediction.get("predictions", [])

    @staticmethod
    def visualize(bgr: np.ndarray, predictions: list[dict]) -> np.ndarray:
        out = bgr.copy()
        for p in predictions:
            color = COLORS.get(p.get("class", ""), (0, 0, 255))
            x0, y0 = int(p["x"] - p["width"] / 2), int(p["y"] - p["height"] / 2)
            x1, y1 = int(p["x"] + p["width"] / 2), int(p["y"] + p["height"] / 2)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
            cv2.putText(out, f"{p['class']} {p['confidence']:.2f}", (x0 + 4, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)  # fmt: skip
        return out
