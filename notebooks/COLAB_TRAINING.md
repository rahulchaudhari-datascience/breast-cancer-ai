# Google Colab Training Guide

## Overview

This guide explains how to train the U-Net++ segmentation and ConvNeXt classification models in Google Colab, then download them for local inference.

## Setup in Google Colab

### Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone the Repository

```bash
!cd /content/drive/MyDrive && git clone https://github.com/yourusername/breast-cancer-ai.git
%cd /content/drive/MyDrive/breast-cancer-ai
```

Or manually upload the project files.

### Step 3: Install Dependencies

```bash
!pip install -q torch torchvision timm albumentations segmentation-models-pytorch \
  scikit-learn pandas opencv-python grad-cam reportlab PyYAML tqdm
```

### Step 4: Upload Dataset

1. Create folders in Google Drive:
   - `/breast-cancer-ai/datasets/raw/` — original images
   - `/breast-cancer-ai/datasets/masks/` — segmentation masks (same names as images)
   - `/breast-cancer-ai/datasets/annotations/` — CSV with labels

2. CSV format (`datasets/annotations/labels.csv`):
   ```
   image_id,label,birads
   mammo_001.png,malignant,5
   mammo_002.png,benign,2
   ...
   ```

3. Upload via Colab UI or programmatically:
   ```python
   import shutil
   shutil.copytree("/path/to/your/dataset", "/content/breast-cancer-ai/datasets/")
   ```

---

## Training Workflow

### Segmentation Training (U-Net++)

```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/breast-cancer-ai')

from pipelines.training_pipeline import SegmentationTrainingPipeline
from config import DEVICE

# Initialize pipeline
pipeline = SegmentationTrainingPipeline(
    model_save_dir='/content/drive/MyDrive/breast-cancer-ai/models',
    batch_size=8,
    epochs=50,
)

# Train segmentation model
seg_metrics = pipeline.train(
    csv_path='datasets/annotations/labels.csv',
    images_dir='datasets/raw',
    masks_dir='datasets/masks',
    validation_split=0.2,
    early_stopping_patience=10,
)

print("Segmentation training complete!")
print(seg_metrics)
```

### Classification Training (ConvNeXt-Tiny)

```python
from pipelines.training_pipeline import ClassificationTrainingPipeline

# Initialize pipeline
cls_pipeline = ClassificationTrainingPipeline(
    model_save_dir='/content/drive/MyDrive/breast-cancer-ai/models',
    batch_size=8,
    epochs=100,
)

# Train classification model
cls_metrics = cls_pipeline.train(
    csv_path='datasets/annotations/labels.csv',
    images_dir='datasets/raw',
    validation_split=0.2,
    early_stopping_patience=15,
)

print("Classification training complete!")
print(cls_metrics)
```

---

## Post-Training

### Verify Models

```python
from pathlib import Path

models_dir = Path('/content/drive/MyDrive/breast-cancer-ai/models')
for model in models_dir.glob('*.pth'):
    print(f"✓ {model.name} ({model.stat().st_size / 1024**2:.1f} MB)")
```

### Download Models Locally

1. After training completes, go to Google Drive folder `/breast-cancer-ai/models/`
2. Download:
   - `classification_model.pth`
   - `segmentation_model.pth`
3. Place in your local `./models/` folder

### Tensorboard Monitoring (Optional)

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/breast-cancer-ai/logs
```

---

## Local Inference (After Downloading Models)

```bash
# 1. Activate local environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit
streamlit run app.py

# 4. Upload mammogram → See predictions, heatmap, PDF report
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` to 4 or 2 |
| `CSV not found` | Ensure CSV is in `datasets/annotations/` |
| `No images found` | Check image filenames match CSV `image_id` column |
| `Model not converging` | Increase `epochs` or reduce `learning_rate` |

---

## Notes

- Models are trained **without pretrained weights** (local construction)
- GPU acceleration is automatic in Colab
- Checkpoints are saved every epoch; best model is selected based on validation loss
- Training time: ~2-4 hours per model on Colab GPU
- Generated models are ~500MB each

---

## Next Steps

1. **Train in Colab** using the steps above
2. **Download models** to `./models/`
3. **Run inference locally** with `streamlit run app.py`
4. **Deploy with Docker** using `docker-compose up`
