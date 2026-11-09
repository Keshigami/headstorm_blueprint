# Blueprint analysis system · carousel

Spec: film/specs/portfolio__headstorm_blueprint.json · 8 slides, 1080x1350.

## 1/8 Hook

- One drawing in, a reviewable report out.

## 2/8 The question

- Rules get 87.5%. The head gets 100%.
- On synthetic drawings. Is that number honest?
- hand-set rules 87.5% → EfficientNet-B0 head 100%

## 3/8 What it does

- Blueprint analysis system
- Portfolio · Computer Vision
- Classify, read, segment: one drawing to one JSON report
- max@baluyot ~ $ curl -s :8117/gradio_api/call/analyze_1 -d '{"data": [{"path": "samples/floor_plan.png", "meta": {"_type": "gradio.FileData"}}]}'

## 4/8 How it is built

- 1. blueprint/samples.py draws 4 drawing types; 240  — torch
- 2. Hough — torch
- 3. EfficientNet-B0 (torchvision, ImageNet, frozen) — torch · torchvision · easyocr
- 4. Gradio 5 Blocks mounted on FastAPI (shar — Gradio
- 5. Dashboard — HTML + ui.css
- Heuristic scored on the same held-out split

## 5/8 The result

- 2 models on the same holdout
- Heuristic rules 87.5%
- EfficientNet-B0 head 100.0% (winner)

## 6/8 The proof

- First sample, an elevation: report, then walls in red
- screen: portfolio/headstorm_blueprint/demo/poster.png

## 7/8 The answer and the limit

- 100%
- held-out accuracy · synthetic drawings · rules 87.5%
- Every image is drawn, not scanned. Real scans untested.
- No mock boxes left. And it ships.

## 8/8 Try it

- Try it: click the floor plan sample.
- max@baluyot ~ $ git clone https://github.com/Keshigami/headstorm_blueprint.git
- max@baluyot ~ $ just headstorm_blueprint-setup
- max@baluyot ~ $ just headstorm_blueprint-up
- github.com/Keshigami/headstorm_blueprint
- Maximus Friedrich Baluyot
- AI & ML Specialist · Applied AI · ML Engineering · Automation

_Maximus Friedrich Baluyot · Caltech CTME · PGC AIML_
