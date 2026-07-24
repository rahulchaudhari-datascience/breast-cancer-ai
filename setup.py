"""Setup and validation script for the Breast Cancer AI project.

This script keeps the existing convenience commands used by the repository:
`check`, `install`, `serve`, and `ui`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_NAME = "Breast Cancer AI"
DEFAULT_MODELS = ("classification_model.pth",)
AVAILABLE_COMMANDS = ("check", "install", "serve", "ui")


def check_models() -> bool:
    """Check whether the expected trained model files exist."""
    models_dir = Path("models")

    print("\n[Models Check]")
    missing = []
    for model_name in DEFAULT_MODELS:
        path = models_dir / model_name
        if path.exists():
            size_mb = path.stat().st_size / 1024**2
            print(f"  ✓ {model_name} ({size_mb:.1f} MB)")
        else:
            missing.append(model_name)
            print(f"  ✗ {model_name} NOT FOUND")

    if missing:
        print("\n  Action: Train models in Google Colab (see notebooks/COLAB_TRAINING.md)")
        print(f"         Then download {', '.join(missing)} to ./models/")
        return False

    return True


def check_environment() -> None:
    """Print the active Python interpreter information."""
    print("\n[Environment Check]")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Location: {sys.executable}")


def check_dependencies() -> None:
    """Check whether the key runtime dependencies are importable."""
    print("\n[Dependencies Check]")
    packages = [
        "torch",
        "torchvision",
        "timm",
        "opencv",
        "albumentations",
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


def install_dependencies() -> None:
    """Install dependencies from requirements.txt."""
    print("\n[Installing Dependencies]")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def check_all() -> None:
    """Run the environment, dependency, and model checks."""
    print("=" * 60)
    print(f"{PROJECT_NAME} — Setup Validation")
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


def serve() -> None:
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


def serve_ui() -> None:
    """Start the Streamlit UI."""
    print("\n[Starting Streamlit UI]")
    print("  http://127.0.0.1:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser used by the setup helper."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} setup and validation helper.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="check",
        help="Command to run (default: check).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch the selected setup command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        check_all()
    elif args.command == "install":
        install_dependencies()
    elif args.command == "serve":
        serve()
    elif args.command == "ui":
        serve_ui()
    else:
        print(f"Unknown command: {args.command}")
        print(f"Available: {', '.join(AVAILABLE_COMMANDS)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
