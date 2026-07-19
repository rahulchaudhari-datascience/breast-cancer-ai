from __future__ import annotations

import io
import logging
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

# Lazy pipeline import to avoid heavy dependency errors during startup
_pipeline = None


def get_pipeline():
    """Load the inference pipeline lazily so startup stays lightweight."""
    global _pipeline
    if _pipeline is None:
        try:
            from pipelines.inference_pipeline import BreastCancerInferencePipeline
        except Exception as exc:
            raise RuntimeError(f"Failed to load inference pipeline: {exc}") from exc

        _pipeline = BreastCancerInferencePipeline()

    return _pipeline


app = FastAPI(title="Breast Cancer AI API")
pipeline = None


@app.get("/health")
def health() -> Dict[str, str]:
    """Return a simple health-check payload."""
    return {"status": "ok"}


def _build_error_response(message: str, status_code: int, *, detail: str | None = None, request_info: Dict[str, Any] | None = None) -> JSONResponse:
    """Build a consistent error payload while preserving the existing endpoint contract."""
    payload: Dict[str, Any] = {"status": "error", "error": message}
    if detail is not None:
        payload["details"] = detail
    if request_info is not None:
        payload["request"] = request_info
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: str = Form("API Patient"),
    generate_report: bool = Form(True),
):
    """Run the full prediction pipeline for an uploaded mammogram image."""
    request_info = {"patient_id": patient_id, "generate_report": generate_report}
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
    except Exception as exc:
        LOGGER.exception("Invalid image received by API")
        return _build_error_response("Invalid image", 400, detail=str(exc), request_info=request_info)

    try:
        global pipeline
        if pipeline is None:
            pipeline = get_pipeline()
    except Exception as exc:
        LOGGER.exception("Failed to initialize inference pipeline")
        return _build_error_response("Failed to initialize pipeline", 500, detail=str(exc), request_info=request_info)

    try:
        result = pipeline.predict(image=image_np, patient_id=patient_id, generate_report=generate_report)
    except Exception as exc:
        LOGGER.exception("Inference request failed")
        return _build_error_response("Inference failed", 500, detail=str(exc), request_info=request_info)

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

    payload: Dict[str, Any] = {
        "status": "success",
        "message": "Prediction completed successfully",
        "data": out,
        "request": request_info,
    }
    payload.update(out)
    payload["status"] = "success"
    return JSONResponse(content=payload)
