# Quick Start Guide

## For Local Inference

This project is already completed and includes a finalized EfficientNet-B0 classifier.

Current best validation metrics:

- ROC-AUC: 0.7985
- Accuracy: 71.43%

## 1. Model Artifacts

The repository stores trained checkpoints under [models](models). The finalized classifier checkpoint is `effnet_best.pth`.

## 2. Setup Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 3. Verify Setup

```powershell
python setup.py check
```

The check command prints the environment summary, dependency availability, and any required model artifacts detected in the `models/` directory.

## 4. Run Streamlit UI

```powershell
streamlit run app.py
```

- Opens at http://127.0.0.1:8501
- Upload mammogram image
- View predictions, heatmap, PDF report

---

## 5. Run the API

```powershell
python setup.py serve
```

- API at http://127.0.0.1:8000
- Health: `curl http://127.0.0.1:8000/health`
- Predict: `curl -X POST http://127.0.0.1:8000/predict -F file=@image.png`

---

## 6. Docker Deployment

```bash
docker-compose up --build
```

- Streamlit UI: http://localhost:8501
- API: http://localhost:8000
- Reports & models mounted as volumes

---

## Troubleshooting

### Models Not Found
```
✗ required model artifact NOT FOUND
```
→ Check the files under `./models/` and confirm the checkpoint names expected by your runtime configuration.

### Missing Dependencies
```
✗ torch NOT INSTALLED
```
→ Run `python setup.py install`

### Port Already in Use
```
streamlit: Address already in use
```
→ Kill previous process or change port: `streamlit run app.py --server.port 8502`

---

## Next Steps

1. ✅ Review the model artifacts in [models](models)
2. ✅ Run `streamlit run app.py`
3. ✅ Upload a mammogram image and review predictions, heatmaps, and the PDF report

For more details, see `README.md` and `notebooks/COLAB_TRAINING.md`.
