# How it was built

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

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | torch and easyocr pin cleanly on 3.11 |
| Model or core logic | EfficientNet-B0 (torchvision, ImageNet, frozen) + linear head trained on synthetic drawings; EasyOCR; OpenCV morphology + Hough | Transfer learning gives a real classifier in a minute of CPU; OCR and segmentation need no training |
| Interface | Gradio 5 Blocks mounted on FastAPI (shared theme in `common/gradio_theme.py`) | Image in, images and JSON out, with `/health` |
| Data | `blueprint/samples.py` draws 4 drawing types; 240 for training, 3 demo samples in `samples/` | No real blueprint dataset is bundled |
| Tests | pytest (9 tests) | samples, segmentation, both classifiers, text tagging, OCR (skips with reason if unavailable), detection skip note, full pipeline, app builds |

## Libraries
`torch` + `torchvision` (EfficientNet-B0 features, linear head), `easyocr` (text detection and recognition),
`opencv-python-headless` (drawing the samples, thresholds, morphology, contours, Hough lines), `numpy`, `pillow`
(image handoff to Gradio), `gradio` (UI), `fastapi` + `uvicorn` (health endpoint and server), `roboflow` (optional
extra, hosted detection), `ruff`, `pytest`.
