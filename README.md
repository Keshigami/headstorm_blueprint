<!-- public-top -->
# Blueprint Analysis System

Classifies an architectural drawing, reads its dimensions and room labels with EasyOCR, and segments rooms and walls with OpenCV, in a Gradio demo. Portfolio, Computer Vision.

**Watch it first:** [the 90-second film](demo/film.mp4) (what it does, how it is built, the proof, the answer) or the [60-second cut](demo/film-60s.mp4). The plain walkthrough with captions is [demo/demo.mp4](demo/demo.mp4).

## Try it in three commands

```bash
git clone https://github.com/Keshigami/headstorm_blueprint.git && cd headstorm_blueprint
just setup        # creates the venv with uv (Python 3.11)
just up           # http://127.0.0.1:8117/  (just down to stop)
```

You need [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) (`brew install uv just` on a Mac). Everything the app needs is in the repo (synthetic sample data where a dataset is used). No API key is needed anywhere in this project. `just test` runs the lint and the offline tests.

Step by step: [docs/Getting-Started.md](docs/Getting-Started.md), then [how it was built](docs/How-It-Was-Built.md), [results](docs/Results.md), [limits](docs/Limits-and-Next-Steps.md) and [the film](docs/The-Film.md). The same pages are on the [wiki](https://github.com/Keshigami/headstorm_blueprint/wiki).
<!-- /public-top -->
# Blueprint Analysis System

## Demo walkthrough
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

A computer-vision pipeline for architectural drawings: classify the drawing type, read dimensions and room labels
with OCR, segment rooms and walls, optionally detect doors and windows with Roboflow. Gradio demo on port **8117**.

## 1. Problem statement and business impact
- **The problem**: construction and real-estate teams receive drawings as images. Finding the floor plans among
  elevations and site plans, pulling out the dimensions and room names, and counting rooms is manual work.
- **Goal**: one call that returns the drawing type, the text with its role (dimension, room label, scale), the enclosed
  regions and wall segments, and a JSON report, with an annotated image for review.
- **Original claim**: "100% accuracy per blueprint type, trained on synthetic data". That number is reproduced here with
  the same caveat, and the heuristic baseline is reported next to it so the comparison is honest.

## 2. Technical solution
| Stage | What runs | Real or not |
|---|---|---|
| Classification | torchvision EfficientNet-B0 ImageNet backbone (frozen) + linear head trained by `train.py` on 240 synthetic drawings; falls back to `HeuristicClassifier` (hand-set rules on line statistics) when `models/classifier.pt` is missing | Real model, synthetic training data; the UI badge says which one is in use |
| OCR | EasyOCR (English), boxes tagged dimension / room_label / scale / annotation by regex | Real; weights download once (~100 MB) into `~/.EasyOCR` |
| Segmentation | OpenCV: adaptive threshold, morphological wall mask, contours for enclosed regions, probabilistic Hough for wall segments | Real, no learning |
| Detection | Roboflow hosted model, only with `ROBOFLOW_API_KEY` and `uv sync --extra roboflow` | Skipped with a note otherwise; no mock boxes |
| Samples | `blueprint/samples.py` draws floor plans, elevations, sections and site plans with OpenCV (rooms, walls, door arcs, labels, dimension lines, title blocks, north arrows, trees) | Synthetic |

Stack: Python 3.11 via uv, torch + torchvision, easyocr, opencv-python-headless, Gradio 5 (mounted on FastAPI for
`/health`), pytest.

## 3. Results on this machine
`models/classifier_metrics.json` after `just headstorm_blueprint-train` (60 images per class, 48 held out, 48 s):

| Classifier | Held-out accuracy | Per class |
|---|---|---|
| EfficientNet-B0 head | 100% | elevation 1.0, floor_plan 1.0, section 1.0, site_plan 1.0 |
| Heuristic rules (same split) | 87.5% | confuses some elevations with floor plans |

On the three demo samples: floor plan -> `floor_plan` 1.00, 14 OCR regions (dimensions `14'-6"`, `12'-6"`, `7566 mm`,
`7079 mm`, `10'-0"`; room labels CLOSET, KITCHEN, LIVING ROOM, OFFICE, BEDROOM 2; scale `SCALE 1.100`), 5 enclosed
regions, 224 wall segments. Elevation -> `elevation` 1.00; site plan -> `site_plan` 1.00 with setbacks `7.3m`, `5.7m`.
Timings: classification about 1 s on CPU (first call), OCR 1-2 s, segmentation 15-40 ms.

The 100% is on drawings from the same generator as the training set. It says the head separates these four synthetic
styles; it says nothing about scanned real blueprints.

## 4. How to run
```bash
just headstorm_blueprint-setup   # uv sync (torch, easyocr), draw samples/, train the head if models/ is empty
just headstorm_blueprint-train   # retrain the head (about 1 min CPU); writes models/classifier.pt + metrics json
just headstorm_blueprint-test    # ruff + pytest (9 tests; the OCR test skips with the reason if EasyOCR cannot load)
just headstorm_blueprint-up      # http://127.0.0.1:8117/
just headstorm_blueprint-down
ROBOFLOW_API_KEY=... uv sync --extra roboflow && uv run python app.py   # optional detection
```

## 5. Two-minute interviewer walkthrough
1. Open http://127.0.0.1:8117/. The header badges say `classifier: efficientnet_b0` and `EasyOCR en`.
2. Click the **floor plan** sample: the summary reads "Type: floor plan at 100%"; the left image tints each enclosed
   region and draws wall segments in red; the right image boxes every OCR hit (blue dimensions, teal room labels).
3. Scroll to the OCR table: five dimensions in three notations and five room names. The JSON report has the same plus
   timings and the layout features the heuristic uses.
4. Click the **site plan** sample: type flips to site plan, the trees are not rooms, the setbacks are read as metres.
5. Open "What is real and what is not" at the bottom.

## 6. What is real and what is simulated
Real: EfficientNet features and the trained head, EasyOCR, OpenCV segmentation, the timings. Synthetic: every image,
including the training set. Skipped unless configured: Roboflow detection. Removed from the original: the mock
"door/window/column" boxes that were generated at random when no API key was present.

## Standalone repo

Exported from Max's `caltech-capstone` workspace. `common/` is vendored (LLM door, UI tokens). Run: `just setup && just test && just up`.
LLM-backed features read `ANTHROPIC_API_KEY` (or `LLM_PROVIDER` + `OPENAI_API_KEY` / `GOOGLE_API_KEY`); with no key they run in a deterministic offline mode.
Film: `demo/film.mp4` (90 s, what it does, how it is built, the proof, the answer) and `demo/film-60s.mp4` (60 s cut); `demo/demo.mp4` is the plain walkthrough.
