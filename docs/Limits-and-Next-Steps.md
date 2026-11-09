# Limits and next steps

- Every image is synthetic. The next real step is a few hundred labelled scanned drawings and fine-tuning the backbone.
- Segmentation counts enclosed regions, not rooms: door gaps merge adjacent rooms, and elevations produce "rooms" from
  window panes. A wall-closing step or a learned segmentation model would fix both.
- OCR reads `SCALE 1:100` as `SCALE 1.100`; a blueprint-specific post-processor is easy and not done.
- Roboflow detection is untested here (no key); the integration code follows the SDK but was not exercised.
- First classification call on CPU is about 1 s and OCR 1-2 s per drawing; batch and GPU would change that.
