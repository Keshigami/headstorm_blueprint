# Results

`models/classifier_metrics.json`, this machine, 48 s of CPU training, 192 train / 48 test synthetic drawings:

| Classifier | Held-out accuracy |
|---|---|
| EfficientNet-B0 + head | 1.00 (all four classes) |
| Heuristic rules | 0.875 |

A good number on real scanned drawings would be above 0.9 across the four types; this 1.00 is on drawings from the
same generator and only shows the head separates the synthetic styles. On the floor plan sample, OCR returns 14
regions including five dimensions in three notations and five room labels; segmentation returns 5 enclosed regions.
