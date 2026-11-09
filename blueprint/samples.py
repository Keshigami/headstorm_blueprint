"""Draw synthetic architectural drawings with OpenCV: floor plans (rooms, walls, door gaps, labels, dimensions),
elevations (facade, roof, windows), sections (slabs, hatching, stairs) and site plans (boundary, footprint, trees,
north arrow). Used for the three demo samples and as the training set for the classifier head."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
ROOMS = ["KITCHEN", "LIVING ROOM", "BEDROOM 1", "BEDROOM 2", "BATH", "DINING", "OFFICE", "GARAGE", "HALL", "CLOSET"]


def _canvas(w: int, h: int, rng: random.Random, blueprint_style: bool = False) -> tuple[np.ndarray, tuple, tuple]:
    if blueprint_style:
        img = np.full((h, w, 3), (140, 70, 20), np.uint8)  # BGR blueprint blue
        ink, soft = (255, 255, 255), (220, 200, 180)
    else:
        img = np.full((h, w, 3), rng.choice([(255, 255, 255), (248, 248, 246), (245, 245, 250)]), np.uint8)
        ink, soft = (20, 20, 20), (110, 110, 110)
    return img, ink, soft


def _text(img, text, org, ink, scale=0.7, thick=2):
    cv2.putText(img, text, org, FONT, scale, ink, thick, cv2.LINE_AA)


def _dim(img, p0, p1, label, ink, scale=0.6):
    """Dimension line with end ticks and a label at the middle."""
    cv2.line(img, p0, p1, ink, 1, cv2.LINE_AA)
    for p in (p0, p1):
        if p0[1] == p1[1]:
            cv2.line(img, (p[0], p[1] - 8), (p[0], p[1] + 8), ink, 1)
        else:
            cv2.line(img, (p[0] - 8, p[1]), (p[0] + 8, p[1]), ink, 1)
    mx, my = (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, 2)
    if p0[1] == p1[1]:
        _text(img, label, (mx - tw // 2, my - 8), ink, scale)
    else:
        _text(img, label, (mx + 10, my + th // 2), ink, scale)


def _title_block(img, title, ink, rng, scale_text="SCALE 1:100"):
    h, w = img.shape[:2]
    x0, y0 = w - 330, h - 90
    cv2.rectangle(img, (x0, y0), (w - 20, h - 20), ink, 2)
    _text(img, title, (x0 + 12, y0 + 30), ink, 0.75, 2)
    _text(img, scale_text, (x0 + 12, y0 + 58), ink, 0.55, 1)
    _text(img, f"SHEET A-{rng.randint(100, 399)}", (x0 + 180, y0 + 58), ink, 0.55, 1)


def _ft(rng: random.Random) -> str:
    return rng.choice([f"{rng.randint(8, 30)}'-{rng.choice([0, 6])}\"", f"{rng.uniform(2.4, 9.0):.1f} m",
                       f"{rng.randint(2400, 9000)} mm"])  # fmt: skip


def draw_floor_plan(rng: random.Random, w: int = 1000, h: int = 750, labels: bool = True) -> np.ndarray:
    img, ink, soft = _canvas(w, h, rng, rng.random() < 0.15)
    m = 90
    x0, y0, x1, y1 = m, m, w - m, h - m - 40
    t = rng.randint(5, 9)
    cv2.rectangle(img, (x0, y0), (x1, y1), ink, t)
    # Recursive splits make rooms; each internal wall gets a door gap.
    rooms = [(x0, y0, x1, y1)]
    for _ in range(rng.randint(2, 4)):
        rooms.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
        rx0, ry0, rx1, ry1 = rooms.pop(0)
        if rx1 - rx0 > ry1 - ry0:
            cut = rng.randint(rx0 + (rx1 - rx0) // 3, rx1 - (rx1 - rx0) // 3)
            cv2.line(img, (cut, ry0), (cut, ry1), ink, t)
            g = rng.randint(ry0 + 30, ry1 - 80)
            cv2.line(img, (cut, g), (cut, g + 50), img[ry0 + t + 2, rx0 + t + 2].tolist(), t + 2)
            cv2.ellipse(img, (cut, g), (50, 50), 0, 0, 90, soft, 1)
            rooms += [(rx0, ry0, cut, ry1), (cut, ry0, rx1, ry1)]
        else:
            cut = rng.randint(ry0 + (ry1 - ry0) // 3, ry1 - (ry1 - ry0) // 3)
            cv2.line(img, (rx0, cut), (rx1, cut), ink, t)
            g = rng.randint(rx0 + 30, rx1 - 80)
            cv2.line(img, (g, cut), (g + 50, cut), img[ry0 + t + 2, rx0 + t + 2].tolist(), t + 2)
            cv2.ellipse(img, (g, cut), (50, 50), 0, 270, 360, soft, 1)
            rooms += [(rx0, ry0, rx1, cut), (rx0, cut, rx1, ry1)]
    names = rng.sample(ROOMS, k=len(rooms))
    for (rx0, ry0, rx1, ry1), name in zip(rooms, names, strict=True):
        if labels:
            (tw, th), _ = cv2.getTextSize(name, FONT, 0.6, 2)
            _text(img, name, ((rx0 + rx1) // 2 - tw // 2, (ry0 + ry1) // 2), ink, 0.6, 2)
            _text(img, _ft(rng), ((rx0 + rx1) // 2 - 30, (ry0 + ry1) // 2 + 28), soft, 0.5, 1)
        # a window on the outer wall
        if ry0 == y0:
            wx = rng.randint(rx0 + 20, max(rx0 + 21, rx1 - 60))
            cv2.rectangle(img, (wx, y0 - 3), (wx + 40, y0 + 3), ink, 1)
    _dim(img, (x0, y0 - 40), (x1, y0 - 40), _ft(rng), ink)
    _dim(img, (x1 + 40, y0), (x1 + 40, y1), _ft(rng), ink)
    if labels:
        _title_block(img, "FLOOR PLAN", ink, rng)
    return img


def draw_elevation(rng: random.Random, w: int = 1000, h: int = 750, labels: bool = True) -> np.ndarray:
    img, ink, soft = _canvas(w, h, rng, rng.random() < 0.15)
    ground = h - 150
    cv2.line(img, (40, ground), (w - 40, ground), ink, 3)
    for x in range(40, w - 40, 18):  # hatched ground
        cv2.line(img, (x, ground), (x - 10, ground + 10), soft, 1)
    bx0, bx1 = 150, w - 150
    storeys = rng.randint(1, 3)
    sh = rng.randint(130, 170)
    top = ground - storeys * sh
    cv2.rectangle(img, (bx0, top), (bx1, ground), ink, 2)
    roof = rng.choice(["gable", "flat", "hip"])
    if roof == "gable":
        pts = np.array([[bx0 - 30, top], [(bx0 + bx1) // 2, top - rng.randint(80, 140)], [bx1 + 30, top]])
        cv2.polylines(img, [pts], False, ink, 3)
    elif roof == "hip":
        pts = np.array([[bx0 - 30, top], [bx0 + 150, top - 80], [bx1 - 150, top - 80], [bx1 + 30, top]])
        cv2.polylines(img, [pts], False, ink, 3)
    else:
        cv2.rectangle(img, (bx0 - 20, top - 18), (bx1 + 20, top), ink, 2)
    for s in range(storeys):
        fy = ground - s * sh
        if s:
            cv2.line(img, (bx0, fy), (bx1, fy), soft, 1)
        n = rng.randint(3, 5)
        gap = (bx1 - bx0) // n
        for i in range(n):
            wx = bx0 + i * gap + gap // 4
            ww, wh = gap // 2, sh // 2
            wy = fy - sh + sh // 5
            if s == 0 and i == n // 2:
                cv2.rectangle(img, (wx + 5, fy - int(sh * 0.75)), (wx + ww - 5, fy), ink, 2)
                cv2.circle(img, (wx + ww - 15, fy - int(sh * 0.4)), 3, ink, -1)
                continue
            cv2.rectangle(img, (wx, wy), (wx + ww, wy + wh), ink, 2)
            cv2.line(img, (wx + ww // 2, wy), (wx + ww // 2, wy + wh), ink, 1)
            cv2.line(img, (wx, wy + wh // 2), (wx + ww, wy + wh // 2), ink, 1)
    if rng.random() < 0.6:  # siding lines
        for y in range(top + 12, ground, 14):
            cv2.line(img, (bx0 + 2, y), (bx1 - 2, y), soft, 1)
    _dim(img, (bx1 + 50, top), (bx1 + 50, ground), _ft(rng), ink)
    if labels:
        _text(img, rng.choice(["NORTH ELEVATION", "SOUTH ELEVATION", "EAST ELEVATION", "WEST ELEVATION"]),
              (bx0, ground + 60), ink, 0.8, 2)  # fmt: skip
        _text(img, "GRADE LEVEL 0.00", (50, ground - 10), soft, 0.5, 1)
        _title_block(img, "ELEVATION", ink, rng)
    return img


def draw_section(rng: random.Random, w: int = 1000, h: int = 750, labels: bool = True) -> np.ndarray:
    img, ink, soft = _canvas(w, h, rng, rng.random() < 0.15)
    ground = h - 140
    bx0, bx1 = 160, w - 160
    levels = rng.randint(2, 3)
    lh = rng.randint(140, 180)
    slab = rng.randint(14, 22)
    wall = rng.randint(14, 22)
    top = ground - levels * lh
    for lv in range(levels + 1):
        y = ground - lv * lh
        cv2.rectangle(img, (bx0 - wall, y - slab), (bx1 + wall, y), ink, -1)
        if labels:
            _text(img, f"LEVEL {lv}" if lv else "GROUND FLOOR", (bx1 + wall + 20, y - 4), ink, 0.55, 1)
        if lv:
            _dim(img, (bx0 - wall - 50, y), (bx0 - wall - 50, y + lh - slab), _ft(rng), ink, 0.5)
    for x in (bx0 - wall, bx1):  # cut walls, hatched
        cv2.rectangle(img, (x, top - slab), (x + wall, ground), ink, -1)
        for y in range(top, ground, 12):
            cv2.line(img, (x + 2, y), (x + wall - 2, y + 8), (160, 160, 160) if ink == (20, 20, 20) else (60, 40, 0), 1)
    # stairs on one level
    sx, sy = rng.randint(bx0 + 40, bx0 + 200), ground - slab
    for i in range(8):
        cv2.line(img, (sx + i * 22, sy - i * 14), (sx + i * 22 + 22, sy - i * 14), ink, 2)
        cv2.line(img, (sx + i * 22 + 22, sy - i * 14), (sx + i * 22 + 22, sy - (i + 1) * 14), ink, 2)
    # hatched ground
    cv2.line(img, (40, ground), (w - 40, ground), ink, 2)
    for x in range(40, w - 40, 14):
        cv2.line(img, (x, ground), (x - 12, ground + 14), soft, 1)
    # roof line
    cv2.line(img, (bx0 - wall - 20, top - slab), (bx1 + wall + 20, top - slab), ink, 3)
    if labels:
        _text(img, f"SECTION {rng.choice('ABC')}-{rng.choice('ABC')}", (bx0, h - 60), ink, 0.8, 2)
        _title_block(img, "BUILDING SECTION", ink, rng, "SCALE 1:50")
    return img


def draw_site_plan(rng: random.Random, w: int = 1000, h: int = 750, labels: bool = True) -> np.ndarray:
    img, ink, soft = _canvas(w, h, rng, rng.random() < 0.15)
    # property boundary (dashed polygon)
    pts = [(120, 100), (w - 140 + rng.randint(-40, 40), 110), (w - 120, h - 150), (140 + rng.randint(-30, 30), h - 130)]
    for i in range(4):
        p0, p1 = np.array(pts[i]), np.array(pts[(i + 1) % 4])
        for k in range(0, 100, 8):
            a = p0 + (p1 - p0) * k / 100
            b = p0 + (p1 - p0) * min(k + 4, 100) / 100
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), ink, 2)
    # building footprint, hatched
    fx0, fy0 = rng.randint(300, 380), rng.randint(240, 300)
    fx1, fy1 = fx0 + rng.randint(220, 320), fy0 + rng.randint(160, 240)
    cv2.rectangle(img, (fx0, fy0), (fx1, fy1), ink, 3)
    for d in range(-(fy1 - fy0), fx1 - fx0, 16):
        a = (max(fx0, fx0 + d), fy0 + max(0, -d))
        b = (min(fx1, fx0 + d + (fy1 - fy0)), fy1 - max(0, d + (fy1 - fy0) - (fx1 - fx0)))
        cv2.line(img, a, b, soft, 1)
    # driveway
    cv2.rectangle(img, (fx1, fy1 - 60), (w - 125, fy1), soft, 2)
    # trees
    for _ in range(rng.randint(6, 12)):
        cx, cy = rng.randint(150, w - 160), rng.randint(130, h - 170)
        if fx0 - 30 < cx < fx1 + 30 and fy0 - 30 < cy < fy1 + 30:
            continue
        r = rng.randint(14, 28)
        cv2.circle(img, (cx, cy), r, ink, 2)
        cv2.circle(img, (cx, cy), 3, ink, -1)
    # north arrow
    nx, ny = w - 80, 80
    cv2.circle(img, (nx, ny), 28, ink, 2)
    cv2.arrowedLine(img, (nx, ny + 20), (nx, ny - 20), ink, 2, tipLength=0.4)
    _text(img, "N", (nx - 8, ny - 34), ink, 0.6, 2)
    # setbacks
    _dim(img, (pts[0][0], fy0 - 30), (fx0, fy0 - 30), f"{rng.uniform(3, 9):.1f} m", ink, 0.5)
    _dim(img, (fx0 - 30, pts[0][1]), (fx0 - 30, fy0), f"{rng.uniform(3, 9):.1f} m", ink, 0.5)
    if labels:
        _text(img, f"LOT {rng.randint(1, 99)}", (pts[0][0] + 20, pts[0][1] + 40), ink, 0.7, 2)
        _text(img, rng.choice(["MAPLE STREET", "OAK AVENUE", "RIVER ROAD"]), (fx0, h - 80), soft, 0.6, 1)
        _title_block(img, "SITE PLAN", ink, rng, "SCALE 1:200")
    return img


DRAWERS = {"floor_plan": draw_floor_plan, "elevation": draw_elevation, "section": draw_section,
           "site_plan": draw_site_plan}  # fmt: skip


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Photo-like noise for training: slight rotation, blur, gaussian noise, scale."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), rng.uniform(0.9, 1.05))
    border = tuple(int(v) for v in img[2, 2])
    out = cv2.warpAffine(img, m, (w, h), borderValue=border)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(2, 10), out.shape)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def generate_samples(out_dir: Path, seed: int = 7) -> list[Path]:
    """The three demo inputs: a floor plan, an elevation and a site plan (plus a section for completeness)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths = []
    for name in ("floor_plan", "elevation", "site_plan"):
        p = out_dir / f"{name}.png"
        cv2.imwrite(str(p), DRAWERS[name](rng))
        paths.append(p)
    return paths


def generate_dataset(out_dir: Path, per_class: int, seed: int = 0) -> list[tuple[Path, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    items = []
    for cls, fn in DRAWERS.items():
        (out_dir / cls).mkdir(exist_ok=True)
        for i in range(per_class):
            img = fn(rng, labels=rng.random() < 0.85)
            if rng.random() < 0.7:
                img = augment(img, rng)
            p = out_dir / cls / f"{cls}_{i:03d}.png"
            cv2.imwrite(str(p), img)
            items.append((p, cls))
    return items
