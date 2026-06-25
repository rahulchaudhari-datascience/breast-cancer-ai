# Quick Start Guide

## For Local Inference (After Colab Training)

### 1. Download Models

Train models in Google Colab (see `notebooks/COLAB_TRAINING.md`), then:
- Download `classification_model.pth` and `segmentation_model.pth` from Google Drive
- Place in `./models/` folder

### 2. Setup Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Setup

```powershell
python setup.py check
```

Expected output:
```
✓ classification_model.pth (XXX MB)
✓ segmentation_model.pth (XXX MB)
Setup is complete! Run: streamlit run app.py
```

### 4. Run Streamlit UI

```powershell
streamlit run app.py
```

- Opens at http://127.0.0.1:8501
- Upload mammogram image
- View predictions, heatmap, PDF report

---

## For API Access

```powershell
python setup.py serve
```

- API at http://127.0.0.1:8000
- Health: `curl http://127.0.0.1:8000/health`
- Predict: `curl -X POST http://127.0.0.1:8000/predict -F file=@image.png`

---

## For Docker Deployment

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
✗ classification_model.pth NOT FOUND
```
→ Train in Google Colab first, then download to `./models/`

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

1. ✅ Train models in Google Colab
2. ✅ Download to `./models/`
3. ✅ Run `streamlit run app.py`
4. ✅ Upload mammogram → Get predictions + report

For more details, see `README.md` and `notebooks/COLAB_TRAINING.md`.
