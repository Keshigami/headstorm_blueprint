import json
import random

import cv2
import numpy as np

from blueprint import CLASSES
from blueprint.classifier import HeuristicClassifier, line_features, load_classifier
from blueprint.detection import BlueprintDetection
from blueprint.ocr_module import classify_text
from blueprint.samples import DRAWERS
from blueprint.segmentation import BlueprintSegmentation


def test_samples_exist_and_look_like_drawings(samples):
    for name in ("floor_plan", "elevation", "site_plan"):
        img = cv2.imread(str(samples / f"{name}.png"))
        assert img is not None and img.shape[1] >= 800
        assert 0.01 < line_features(img)["ink_ratio"] < 0.5


def test_segmentation_finds_rooms_and_walls(samples):
    seg = BlueprintSegmentation()
    res = seg.segment(cv2.imread(str(samples / "floor_plan.png")))
    d = seg.to_dict(res)
    assert d["num_rooms"] >= 3, d
    assert d["num_walls"] >= 10
    assert all(r["area_px"] > 0 and len(r["bbox"]) == 4 for r in d["rooms"])
    vis = seg.visualize(cv2.imread(str(samples / "floor_plan.png")), res)
    assert vis.shape[2] == 3


def test_heuristic_classifier_contract_and_floor_plan(samples):
    h = HeuristicClassifier()
    r = h.predict(cv2.imread(str(samples / "floor_plan.png")))
    assert r["class"] in CLASSES and r["method"] == "heuristic"
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
    assert r["class"] == "floor_plan", r
    assert h.predict(cv2.imread(str(samples / "site_plan.png")))["class"] == "site_plan"


def test_efficientnet_head_beats_chance_on_fresh_drawings(trained_head):
    clf = load_classifier(trained_head)
    assert clf.method == "efficientnet_b0", clf.note
    rng = random.Random(123)
    correct = total = 0
    for cls, draw in DRAWERS.items():
        for _ in range(3):
            total += 1
            correct += clf.predict(draw(rng))["class"] == cls
    assert correct / total >= 0.75, f"{correct}/{total}"
    m = json.loads((trained_head / "classifier_metrics.json").read_text())
    assert m["accuracy"] >= 0.8 and m["n_test"] >= 8


def test_classify_text_types():
    assert classify_text("12'-6\"") == "dimension"
    assert classify_text("3.6 m") == "dimension"
    assert classify_text("KITCHEN") == "room_label"
    assert classify_text("SCALE 1:100") == "scale"
    assert classify_text("SHEET A-201") == "annotation"


def test_ocr_reads_labels_and_dimensions(samples, ocr):
    text = ocr.extract_text(cv2.imread(str(samples / "floor_plan.png")))
    assert len(text) >= 5
    types = {t["type"] for t in text}
    assert "dimension" in types and "room_label" in types, [(t["text"], t["type"]) for t in text]
    assert any("FLOOR PLAN" in t["text"].upper() for t in text)


def test_detection_is_skipped_without_key(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    d = BlueprintDetection()
    assert not d.enabled and "skipped" in d.note and "ROBOFLOW_API_KEY" in d.note
    assert d.detect(np.zeros((100, 100, 3), np.uint8)) == []


def test_full_pipeline_report_is_json(samples, trained_head):
    from blueprint.analyzer import BlueprintAnalyzer

    a = BlueprintAnalyzer(models_dir=trained_head)
    r = a.analyze(str(samples / "floor_plan.png"))
    assert set(r) >= {"classification", "ocr", "segmentation", "detections", "timings_ms", "_images"}
    json.dumps(a.report(r))
    assert "Type:" in a.summary(r)
    assert r["_images"]["segmentation"].size == tuple(r["image_size"])


def test_gradio_app_builds(samples, trained_head):
    import app

    assert app.build_demo() is not None
