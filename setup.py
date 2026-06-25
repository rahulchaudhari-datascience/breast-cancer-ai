"""
Setup and validation script for Breast Cancer AI inference environment.

Usage:
    python setup.py check       # Validate setup
    python setup.py install     # Install dependencies
    python setup.py serve       # Start API server
"""

import sys
import subprocess
from pathlib import Path


def check_models():
    """Check if trained models exist."""
    models_dir = Path("models")
    required = ["classification_model.pth", "segmentation_model.pth"]
    
    print("\n[Models Check]")
    missing = []
    for model in required:
        path = models_dir / model
        if path.exists():
            size_mb = path.stat().st_size / 1024**2
            print(f"  ✓ {model} ({size_mb:.1f} MB)")
        else:
            missing.append(model)
            print(f"  ✗ {model} NOT FOUND")
    
    if missing:
        print(f"\n  Action: Train models in Google Colab (see notebooks/COLAB_TRAINING.md)")
        print(f"         Then download {', '.join(missing)} to ./models/")
        return False
    
    return True


def check_environment():
    """Check Python environment."""
    print("\n[Environment Check]")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Location: {sys.executable}")


def check_dependencies():
    """Check if key packages are installed."""
    print("\n[Dependencies Check]")
    packages = [
        "torch",
        "torchvision",
        "timm",
        "opencv",
        "albumentations",
        "segmentation_models",
        "grad_cam",
        "fastapi",
        "streamlit",
    ]
    
    for pkg in packages:
        try:
            __import__(pkg.replace("_", "-").split("_")[0])
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} NOT INSTALLED")


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("\n[Installing Dependencies]")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def check_all():
    """Run all checks."""
    print("=" * 60)
    print("Breast Cancer AI — Setup Validation")
    print("=" * 60)
    
    check_environment()
    check_dependencies()
    has_models = check_models()
    
    print("\n" + "=" * 60)
    if has_models:
        print("✓ Setup is complete! Run: streamlit run app.py")
    else:
        print("⚠ Models are missing. Train in Google Colab first.")
    print("=" * 60)


def serve():
    """Start the FastAPI server."""
    print("\n[Starting API Server]")
    print("  Uvicorn: http://127.0.0.1:8000")
    print("  Health: http://127.0.0.1:8000/health")
    print("  Docs: http://127.0.0.1:8000/docs")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ])


def serve_ui():
    """Start the Streamlit UI."""
    print("\n[Starting Streamlit UI]")
    print("  http://127.0.0.1:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    
    if cmd == "check":
        check_all()
    elif cmd == "install":
        install_dependencies()
    elif cmd == "serve":
        serve()
    elif cmd == "ui":
        serve_ui()
    else:
        print(f"Unknown command: {cmd}")
        print("Available: check, install, serve, ui")
