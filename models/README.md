# Trained Models Storage

Place your trained models from Google Colab here:

- `classification_model.pth` — ConvNeXt-Tiny classification model (trained in Colab)
- `segmentation_model.pth` — U-Net++ segmentation model (trained in Colab)

## How to Get Models

1. **Train in Google Colab:**
   - See `../notebooks/COLAB_TRAINING.md`
   - Run the training notebook in Colab
   - Models will be saved to your Google Drive

2. **Download to Local:**
   - Access Google Drive: `breast-cancer-ai/models/`
   - Download both `.pth` files
   - Place them in this folder (`./models/`)

3. **Verify:**
   ```bash
   python setup.py check
   ```

Once models are in place, inference will work locally without internet access.
