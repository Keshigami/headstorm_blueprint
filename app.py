"""Gradio UI for the blueprint analyzer on port 8117: upload or pick a sample; type, OCR boxes, segmentation, JSON."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

from blueprint.analyzer import BlueprintAnalyzer
from blueprint.samples import generate_samples

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
PORT = int(os.getenv("PORT", "8117"))
SERVICE = "headstorm_blueprint"
sys.path.insert(0, str(HERE.parents[1]))
from common.gradio_theme import CSS, DARK_JS, theme  # noqa: E402

if not any(SAMPLES.glob("*.png")):
    generate_samples(SAMPLES)
EXAMPLES = sorted(str(p) for p in SAMPLES.glob("*.png"))
analyzer = BlueprintAnalyzer()
STATUS = analyzer.status()
METRICS = HERE / "models" / "classifier_metrics.json"


def analyze(image):
    if image is None:
        raise gr.Error("Drop a blueprint or pick a sample below.")
    r = analyzer.analyze(image)
    text = r["ocr"].get("all_text", [])
    table = pd.DataFrame([{"text": t["text"], "type": t["type"], "confidence": t["confidence"]} for t in text])
    return (analyzer.summary(r), r["_images"]["segmentation"], r["_images"]["ocr"], table, analyzer.report(r))


def header_html() -> str:
    cls = STATUS["classifier"]
    badge = "on" if cls["method"] == "efficientnet_b0" else ""
    ocr = "EasyOCR en" if STATUS["ocr"]["available"] else "OCR unavailable"
    return (
        '<header class="top"><h1>Blueprint analysis</h1>'
        '<span class="sub">Classifies the drawing type, reads dimensions and room labels with EasyOCR, segments '
        "rooms and walls with OpenCV. Roboflow element detection is optional.</span>"
        f'<span class="badge {badge}">classifier: {cls["method"]}</span><span class="badge on">{ocr}</span>'
        '<span class="badge">port 8117</span></header>'
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Blueprint Analyzer") as demo:
        gr.HTML(header_html())
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Blueprint image", elem_id="input", height=380)
                run = gr.Button("Analyze blueprint", variant="primary", elem_id="run")
            with gr.Column(scale=1):
                summary = gr.Markdown("Drop a blueprint or pick a sample.", elem_id="result")
        with gr.Row():
            seg_img = gr.Image(label="Segmentation: rooms tinted, wall segments in red", height=420)
            ocr_img = gr.Image(label="OCR: blue dimensions, teal room labels, violet scale", height=420)
        with gr.Row():
            table = gr.Dataframe(label="OCR text", interactive=False, wrap=True)
            report = gr.JSON(label="JSON report")
        outputs = [summary, seg_img, ocr_img, table, report]
        gr.Examples(EXAMPLES, inputs=inp, outputs=outputs, fn=analyze, run_on_click=True, cache_examples=False,
                    label="Samples (one click runs the pipeline)")  # fmt: skip
        run.click(analyze, inputs=inp, outputs=outputs)
        inp.upload(analyze, inputs=inp, outputs=outputs)
        with gr.Accordion("What is real and what is not", open=False):
            m = json.loads(METRICS.read_text()) if METRICS.exists() else {}
            gr.Markdown(
                f"- Classifier: **{STATUS['classifier']['method']}**. {STATUS['classifier']['note']}.\n"
                + (f"- Held-out accuracy on synthetic drawings: {m['accuracy']:.0%} "
                   f"(heuristic on the same split: {m['heuristic_accuracy_same_split']:.0%}).\n" if m else "")
                + f"- OCR: EasyOCR, real. {STATUS['ocr']['reason'] or ''}\n"
                + "- Segmentation: OpenCV morphology + contours + Hough, real, no learning.\n"
                + f"- Detection: {STATUS['detection']}.\n"
                + "- Sample blueprints are drawn synthetically by blueprint/samples.py."
            )  # fmt: skip
    return demo


def build_app():
    from fastapi import FastAPI

    api = FastAPI(title=SERVICE)

    @api.get("/health")
    def health():
        return {"ok": True, "service": SERVICE, "model_loaded": STATUS["classifier"]["method"] == "efficientnet_b0",
                "status": STATUS}  # fmt: skip

    # Gradio 6: theme, css and js are mount/launch arguments, not Blocks arguments
    return gr.mount_gradio_app(api, build_demo(), path="/", theme=theme(), css=CSS, js=DARK_JS)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="127.0.0.1", port=PORT)
