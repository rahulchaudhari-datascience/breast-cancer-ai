from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io

# Lazy pipeline import to avoid heavy dependency errors during startup
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from pipelines.inference_pipeline import BreastCancerInferencePipeline
        except Exception as exc:
            raise RuntimeError(f"Failed to load inference pipeline: {exc}")

        _pipeline = BreastCancerInferencePipeline()

    return _pipeline

app = FastAPI(title="Breast Cancer AI API")

pipeline = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: str = Form("API Patient"),
    generate_report: bool = Form(True),
):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "error", "error": f"Invalid image: {exc}"})

    try:
        global pipeline
        if pipeline is None:
            pipeline = get_pipeline()
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "error": f"Failed to initialize pipeline: {exc}"})

    try:
        result = pipeline.predict(image=image_np, patient_id=patient_id, generate_report=generate_report)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(exc)})

    out = {
        "status": result.get("status"),
        "prediction": result.get("prediction"),
        "class_id": int(result.get("class_id")) if result.get("class_id") is not None else None,
        "probability": float(result.get("probability")) if result.get("probability") is not None else None,
        "confidence": float(result.get("confidence")) if result.get("confidence") is not None else None,
        "birads": result.get("birads"),
        "birads_confidence": float(result.get("birads_confidence")) if result.get("birads_confidence") is not None else None,
        "roi_status": result.get("roi_status"),
        "report_path": result.get("report_path"),
    }

    return JSONResponse(content=out)
