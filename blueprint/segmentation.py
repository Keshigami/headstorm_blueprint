"""OpenCV room and wall segmentation: adaptive threshold -> morphological wall mask -> enclosed regions (rooms)
and probabilistic Hough segments (walls). Taken from the original segmentation.py with JSON-friendly output."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PALETTE = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255), (100, 255, 255),
           (200, 150, 100), (100, 200, 150), (150, 100, 200)]  # fmt: skip


@dataclass
class Room:
    contour: np.ndarray
    area: float
    center: tuple[int, int]
    bounding_box: tuple[int, int, int, int]


@dataclass
class Wall:
    start: tuple[int, int]
    end: tuple[int, int]
    thickness: int


class BlueprintSegmentation:
    def __init__(self, min_room_area: int = 1500, wall_kernel: int = 25):
        self.min_room_area = min_room_area
        self.wall_kernel = wall_kernel

    def segment(self, bgr: np.ndarray) -> dict:
        binary = self.preprocess(bgr)
        wall_mask = self.detect_walls(binary)
        rooms = self.detect_rooms(wall_mask)
        walls = self.extract_wall_segments(wall_mask)
        return {"rooms": rooms, "walls": walls, "wall_mask": wall_mask, "processed": binary}

    @staticmethod
    def preprocess(bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 110:
            gray = 255 - gray
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    def detect_walls(self, binary: np.ndarray) -> np.ndarray:
        k = self.wall_kernel
        h_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        v_walls = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
        return cv2.dilate(cv2.bitwise_or(h_walls, v_walls), np.ones((3, 3), np.uint8), iterations=1)

    def detect_rooms(self, wall_mask: np.ndarray) -> list[Room]:
        room_mask = cv2.bitwise_not(wall_mask)
        contours, _ = cv2.findContours(room_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        total = wall_mask.shape[0] * wall_mask.shape[1]
        rooms = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_room_area or area > 0.6 * total:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) / max(w * h, 1) < 0.5:  # thin or ragged region, not a room
                continue
            m = cv2.moments(c)
            cx, cy = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) if m["m00"] else (x + w // 2, y + h // 2)
            rooms.append(Room(c, float(area), (cx, cy), (int(x), int(y), int(w), int(h))))
        rooms.sort(key=lambda r: -r.area)
        return rooms

    def extract_wall_segments(self, wall_mask: np.ndarray) -> list[Wall]:
        lines = cv2.HoughLinesP(wall_mask, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        walls = []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                walls.append(Wall((int(x1), int(y1)), (int(x2), int(y2)), self.estimate_thickness(wall_mask, mx, my)))
        return walls

    @staticmethod
    def estimate_thickness(mask: np.ndarray, x: int, y: int, max_search: int = 30) -> int:
        h, w = mask.shape
        left = right = 0
        for i in range(1, max_search):
            if x - i >= 0 and mask[y, x - i] > 0:
                left = i
            else:
                break
        for i in range(1, max_search):
            if x + i < w and mask[y, x + i] > 0:
                right = i
            else:
                break
        return left + right + 1

    @staticmethod
    def to_dict(results: dict, max_rooms: int = 12) -> dict:
        rooms = results["rooms"]
        walls = results["walls"]
        return {
            "num_rooms": len(rooms),
            "num_walls": len(walls),
            "wall_pixels_ratio": round(float((results["wall_mask"] > 0).mean()), 4),
            "mean_wall_thickness": round(float(np.mean([w.thickness for w in walls])), 2) if walls else 0.0,
            "rooms": [{"id": i + 1, "area_px": int(r.area), "center": list(r.center), "bbox": list(r.bounding_box)}
                      for i, r in enumerate(rooms[:max_rooms])],  # fmt: skip
        }

    @staticmethod
    def visualize(bgr: np.ndarray, results: dict, show_rooms: bool = True, show_walls: bool = True) -> np.ndarray:
        out = bgr.copy()
        if show_rooms:
            for i, room in enumerate(results["rooms"][:9]):
                color = PALETTE[i % len(PALETTE)]
                overlay = out.copy()
                cv2.drawContours(overlay, [room.contour], -1, color, -1)
                cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
                cv2.drawContours(out, [room.contour], -1, color, 2)
                cv2.putText(out, f"Room {i + 1}", room.center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, f"Room {i + 1}", room.center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)  # fmt: skip
        if show_walls:
            for wall in results["walls"][:150]:
                cv2.line(out, wall.start, wall.end, (0, 0, 255), 2)
        return out
