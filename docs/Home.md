# Blueprint Analysis System

Classifies an architectural drawing, reads its dimensions and room labels with EasyOCR, and segments rooms and walls with OpenCV, in a Gradio demo. Portfolio, Computer Vision, by Maximus Friedrich Baluyot.

**Watch first:** [the 90-second film](https://github.com/Keshigami/headstorm_blueprint/blob/main/demo/film.mp4).

## Pages
- [Getting started](Getting-Started.md): run it in three commands, what you see, the API.
- [How it was built](How-It-Was-Built.md): from the brief to a running service, step by step.
- [Results](Results.md): the numbers and what they mean.
- [Limits and next steps](Limits-and-Next-Steps.md): what it cannot do yet.
- [The film](The-Film.md): what the four chapters show.

## The brief
Construction and property teams get drawings as images and sort them by hand: which are floor plans, what are the
dimensions, how many rooms. This pipeline takes one image and returns the drawing type, the text with its role
(dimension, room label, scale), the enclosed regions and wall segments, and an annotated picture. "Solved" means an
engineer can drop a drawing and get a reviewable JSON report in a few seconds.

## The one-line answer
100% · held-out accuracy · synthetic drawings · rules 87.5%

## One honest limit
Every image is drawn, not scanned. Real scans untested.
