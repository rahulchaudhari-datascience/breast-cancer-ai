# Datasets Layout

This folder contains dataset organization guidance used by training pipelines.

Structure:

- `raw/` — original images (keep immutable)
- `processed/` — preprocessed images for training/validation
- `masks/` — segmentation masks aligned with `processed/`
- `annotations/` — CSV/JSON annotations and labels

Populate these folders with your dataset files before running `pipelines/training_pipeline.py`.
