"""Example client to test the FastAPI `/predict` endpoint locally."""

import requests

API_URL = "http://127.0.0.1:8000/predict"


def test_predict(image_path: str, patient_id: str = "Demo"):
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/png")}
        data = {"patient_id": patient_id, "generate_report": "false"}
        resp = requests.post(API_URL, files=files, data=data)
    print("Status:", resp.status_code)
    print(resp.json())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python api/client_example.py <image_path>")
    else:
        test_predict(sys.argv[1])
