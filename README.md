# Breast Cancer AI — Research Platform

An explainable AI system for mammogram analysis: segmentation, classification, BI-RADS prediction, and confidence scoring with PDF report generation.

## Architecture

```
Google Colab (Training)
    ↓
Trained Models (saved to Google Drive)
    ↓
Local Download or Docker Volume
    ↓
Inference Pipeline (Streamlit UI + FastAPI)
    ↓
Results, Reports, Heatmaps
```

---

## Quick Start

### Option A: Streamlit UI (Local)

1. **Setup environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Download trained models from Colab:**
   - See [Training in Google Colab](#training-in-google-colab) below
   - Download `classification_model.pth` and `segmentation_model.pth` to `./models/`

3. **Run Streamlit:**
   ```bash
   streamlit run app.py
   ```

4. **Upload a mammogram image** and see AI predictions, Grad-CAM heatmaps, and PDF report.

### Option B: FastAPI (Local)

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Test the API:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@mammogram.png" \
  -F "patient_id=Demo Patient" \
  -F "generate_report=false"
```

### Option C: Docker (Recommended for Production)

```bash
# Build and run both Streamlit UI and FastAPI with docker-compose
docker-compose up --build

# UI runs on http://localhost:8501
# API runs on http://localhost:8000
# Health check: curl http://localhost:8000/health
```

---

## Training in Google Colab

1. **Open the training notebook:**
   - See `notebooks/colab_training.ipynb` (template included in this repo)
   - Or download it and open in [Google Colab](https://colab.research.google.com)

2. **Steps in Colab:**
   ```
   - Upload your dataset (mammograms + masks + labels CSV)
   - Install dependencies (pytorch, timm, albumentations, etc.)
   - Train U-Net++ segmentation model
   - Train ConvNeXt-Tiny classification model
   - Save trained models to Google Drive
   ```

3. **Download models locally:**
   - Download `classification_model.pth` and `segmentation_model.pth` from Google Drive
   - Place in `./models/` folder

4. **Run inference locally** (see Quick Start above)

---

## Project Structure

```
breast-cancer-ai/
├── app.py                      # Streamlit UI
├── config.py                   # Configuration (device, paths, etc.)
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container image
├── docker-compose.yml          # Multi-container orchestration
│
├── api/
│   ├── main.py                # FastAPI server
│   ├── client_example.py       # Example HTTP client
│   └── tests/
│       └── test_api_health.py # Health check test
│
├── services/                   # Business logic
│   ├── model_builder.py       # Local model construction (no pretrained)
│   ├── model_downloader.py    # Download models from Colab
│   ├── preprocessing_service.py
│   ├── segmentation_service.py
│   ├── roi_service.py
│   ├── classification_service.py
│   ├── birads_service.py
│   ├── confidence_service.py
│   ├── explainability_service.py  # Grad-CAM++
│   └── report_service.py       # PDF generation
│
├── pipelines/
│   ├── inference_pipeline.py   # End-to-end inference
│   ├── training_pipeline.py    # Local training (optional)
│   └── evaluation_pipeline.py  # Metrics computation
│
├── utils/
│   ├── image_utils.py
│   ├── data_utils.py
│   ├── metrics_utils.py
│   └── visualization_utils.py
│
├── models/                     # Trained model storage
│   ├── classification_model.pth  # Download from Colab
│   └── segmentation_model.pth    # Download from Colab
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── masks/
│   └── annotations/
│
├── reports/                    # Generated PDF reports
│
└── notebooks/
    └── colab_training.ipynb    # Template for Google Colab training
```

---

## Models

### Architecture

- **Segmentation:** U-Net++ (encoder-decoder, no pretrained weights)
- **Classification:** ConvNeXt-Tiny (local construction, no pretrained weights)
- **Explainability:** Grad-CAM++
- **Confidence:** Uncertainty estimation + multi-source confidence fusion

### Training (Google Colab)

Models are **trained in Google Colab** with:
- GPU acceleration
- Tensorboard monitoring
- Checkpoints saved to Google Drive
- Batch size: 8 (RTX 3050 6GB optimization)
- Image size: 512×512
- Epochs: 50–100

### Local Inference

Downloaded models are loaded in `inference_pipeline.py` without requiring internet or pretrained weight downloads.

---

## API Endpoints

### GET `/health`
Health check.
```
curl http://127.0.0.1:8000/health
→ {"status": "ok"}
```

### POST `/predict`
Run full AI pipeline on an image.

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@test.png" \
  -F "patient_id=Patient001" \
  -F "generate_report=true"
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Malignant",
  "class_id": 1,
  "probability": 0.92,
  "confidence": 92.1,
  "birads": "BI-RADS 5",
  "birads_confidence": 88.5,
  "roi_status": "success",
  "report_path": "/app/reports/breast_cancer_report_20260625_120000.pdf"
}
```

---

## Configuration

Edit `config.py` to customize:
- `DEVICE` — "cpu" or "cuda"
- `IMAGE_SIZE` — 512 (default)
- `BATCH_SIZE` — 8 (training)
- Model checkpoint paths
- Output directories

---

## Hardware Requirements

- **Minimum:** CPU-only (inference works on 2GB RAM)
- **Recommended:** NVIDIA GPU with 6GB VRAM (RTX 3050+) for faster inference
- **Docker:** Any CPU (inference) or GPU passthrough

---

## Features

✅ Tumor Segmentation (U-Net++)  
✅ Malignant/Benign Classification (ConvNeXt)  
✅ BI-RADS Assessment  
✅ Confidence & Uncertainty Scoring  
✅ Grad-CAM++ Explainability  
✅ PDF Report Generation  
✅ Streamlit UI  
✅ FastAPI REST Server  
✅ Docker Deployment  

---

## Workflow Example

```bash
# 1. Train in Google Colab (see colab_training.ipynb)
# 2. Download models to ./models/
# 3. Run locally:

streamlit run app.py
# Upload mammogram → See predictions, heatmap, and PDF report

# Or via API:
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@mammogram.png" \
  -F "patient_id=Case123" \
  -F "generate_report=true"
```

---

## Disclaimer

This is a **research and educational prototype**. It is **not approved for clinical use**. Results must be reviewed by qualified radiologists and medical professionals before any clinical decision-making.

---

## License

Educational use only. See LICENSE file for details.
