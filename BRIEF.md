---
title: Blueprint Analysis System
group: portfolio
role: Computer Vision
port: 8117
status: working
one_liner: Classifies an architectural drawing, reads its dimensions and room labels with EasyOCR, and segments rooms and walls with OpenCV, in a Gradio demo.
---
# Blueprint Analysis System

## Demo
<!-- demo:auto -->
![demo](demo/demo.gif)

Full walkthrough with captions: [demo/demo.mp4](demo/demo.mp4) (58 s, 1280x800, animated intro and end card). Run it yourself: `just setup && just up`, then open http://localhost:8117/.

Two-minute walkthrough:
1. Blueprint analysis: pick the first sample; one click runs the pipeline
2. Segmentation: rooms tinted, wall segments in red
3. OCR: dimensions, room labels and the scale, with confidence
4. The JSON report an integration would consume
5. Second sample: a floor plan
6. Its segmentation and OCR overlays
<!-- /demo:auto -->

## Brief
Construction and property teams get drawings as images and sort them by hand: which are floor plans, what are the
dimensions, how many rooms. This pipeline takes one image and returns the drawing type, the text with its role
(dimension, room label, scale), the enclosed regions and wall segments, and an annotated picture. "Solved" means an
engineer can drop a drawing and get a reviewable JSON report in a few seconds.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | torch and easyocr pin cleanly on 3.11 |
| Model or core logic | EfficientNet-B0 (torchvision, ImageNet, frozen) + linear head trained on synthetic drawings; EasyOCR; OpenCV morphology + Hough | Transfer learning gives a real classifier in a minute of CPU; OCR and segmentation need no training |
| Interface | Gradio 5 Blocks mounted on FastAPI (shared theme in `common/gradio_theme.py`) | Image in, images and JSON out, with `/health` |
| Data | `blueprint/samples.py` draws 4 drawing types; 240 for training, 3 demo samples in `samples/` | No real blueprint dataset is bundled |
| Tests | pytest (9 tests) | samples, segmentation, both classifiers, text tagging, OCR (skips with reason if unavailable), detection skip note, full pipeline, app builds |

## Libraries used
`torch` + `torchvision` (EfficientNet-B0 features, linear head), `easyocr` (text detection and recognition),
`opencv-python-headless` (drawing the samples, thresholds, morphology, contours, Hough lines), `numpy`, `pillow`
(image handoff to Gradio), `gradio` (UI), `fastapi` + `uvicorn` (health endpoint and server), `roboflow` (optional
extra, hosted detection), `ruff`, `pytest`.

## How it was built
1. Read the original five modules. Classifier, OCR and segmentation were sound designs but nothing ran: the package
   used relative imports from loose files, the classifier had ImageNet weights and an untrained 4-class head
   (`blueprint_classifier.pth` never existed), and detection returned random mock boxes without a key.
2. Kept the segmentation algorithm as it was (adaptive threshold, horizontal/vertical opening, contours, HoughLinesP),
   added a fill-ratio filter so ragged regions are not rooms, and JSON output. Kept the OCR tagging regexes and added
   inversion for white-on-blue drawings.
3. Wrote `blueprint/samples.py` to draw floor plans (recursive room splits, door gaps with swing arcs, labels,
   dimension lines, title block), elevations (storeys, windows, roofs, siding), sections (slabs, hatched cut walls,
   stairs) and site plans (dashed boundary, hatched footprint, trees, north arrow, setbacks), with augmentation.
4. `train.py`: 60 drawings per class -> frozen EfficientNet-B0 features -> linear head, 60 epochs, held-out
   accuracy, confusion matrix, and the heuristic scored on the same split. `load_classifier` picks the head when
   `models/classifier.pt` exists and the heuristic otherwise, and every prediction carries its `method`.
5. `HeuristicClassifier`: four hand-set rules on line statistics (round blobs = trees, thick dark bands = cut slabs,
   vertical/horizontal line ratio) so the pipeline runs with no model at all.
6. Detection now only runs with `ROBOFLOW_API_KEY`; otherwise the report says it was skipped.
7. Gradio app with one-click samples, two overlays, OCR table, JSON, and a "what is real" panel.

## Results
`models/classifier_metrics.json`, this machine, 48 s of CPU training, 192 train / 48 test synthetic drawings:

| Classifier | Held-out accuracy |
|---|---|
| EfficientNet-B0 + head | 1.00 (all four classes) |
| Heuristic rules | 0.875 |

A good number on real scanned drawings would be above 0.9 across the four types; this 1.00 is on drawings from the
same generator and only shows the head separates the synthetic styles. On the floor plan sample, OCR returns 14
regions including five dimensions in three notations and five room labels; segmentation returns 5 enclosed regions.

## Demo details
- Run: `just headstorm_blueprint-setup && just headstorm_blueprint-up`, open `http://127.0.0.1:8117/`.
- Two-minute walkthrough: (1) click the floor plan sample; (2) read the summary and the two overlays; (3) scroll to
  the OCR table and the JSON; (4) click the site plan sample and compare; (5) open "What is real and what is not".
- Video clip: `demo/clips/portfolio__headstorm_blueprint.webm` (recorded by `just demo-video`).
- Sample inputs: `samples/floor_plan.png` (5 rooms, 5 dimensions, title block), `samples/elevation.png`,
  `samples/site_plan.png` (trees, north arrow, two setbacks in metres).

## Limits and next steps
- Every image is synthetic. The next real step is a few hundred labelled scanned drawings and fine-tuning the backbone.
- Segmentation counts enclosed regions, not rooms: door gaps merge adjacent rooms, and elevations produce "rooms" from
  window panes. A wall-closing step or a learned segmentation model would fix both.
- OCR reads `SCALE 1:100` as `SCALE 1.100`; a blueprint-specific post-processor is easy and not done.
- Roboflow detection is untested here (no key); the integration code follows the SDK but was not exercised.
- First classification call on CPU is about 1 s and OCR 1-2 s per drawing; batch and GPU would change that.
