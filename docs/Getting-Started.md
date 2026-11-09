# Getting started

## What you need
- [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). On a Mac: `brew install uv just`.
- Python 3.11 is installed by uv.
- No API key is needed anywhere in this project.

## Three commands
```bash
git clone https://github.com/Keshigami/headstorm_blueprint.git && cd headstorm_blueprint
just setup
just up           # http://127.0.0.1:8117/  (just down to stop)
```
Everything the app needs is in the repo (synthetic sample data where a dataset is used).

## Commands
`just setup`, `just train` where a model is trained, `just up` / `just down`, `just test`, `just demo`.

## What you see
- Run: `just headstorm_blueprint-setup && just headstorm_blueprint-up`, open `http://127.0.0.1:8117/`.
- Two-minute walkthrough: (1) click the floor plan sample; (2) read the summary and the two overlays; (3) scroll to
  the OCR table and the JSON; (4) click the site plan sample and compare; (5) open "What is real and what is not".
- Video clip: `demo/clips/portfolio__headstorm_blueprint.webm` (recorded by `just demo-video`).
- Sample inputs: `samples/floor_plan.png` (5 rooms, 5 dimensions, title block), `samples/elevation.png`,
  `samples/site_plan.png` (trees, north arrow, two setbacks in metres).

## Try this
Try it: click the floor plan sample.
