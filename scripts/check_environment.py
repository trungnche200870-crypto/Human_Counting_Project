"""Fast preflight used by VS Code before starting either application."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("streamlit", "ultralytics", "aiohttp", "torch", "opencv-python", "pandas", "lap")


def main():
    errors = []
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    if sys.version_info[:2] != (3, 11):
        errors.append("Use the project's Python 3.11 virtual environment (.venv).")
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        errors.append("Wrong environment. Select .venv/Scripts/python.exe in VS Code.")
    for name in PACKAGES:
        try:
            print(f"  {name}: {version(name)}")
        except PackageNotFoundError:
            errors.append(f"Missing package: {name}")
    for name in ('yolo11n.pt', 'yolo11s.pt'):
        path = ROOT / name
        if not path.is_file() or path.stat().st_size < 1_000_000:
            errors.append(f"Missing or incomplete model: {name}")
    if not errors:
        try:
            import cv2
            import torch
            import aiohttp
            import pandas
            import streamlit
            import ultralytics
            gpu = torch.cuda.is_available()
            print("GPU: " + (torch.cuda.get_device_name(0) if gpu else "unavailable; CPU mode"))
        except Exception as exc:
            errors.append(f"Dependency import failed: {exc}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("Run Terminal > Run Task > Setup environment, then try again.")
        return 1
    print("Environment ready. No activation command is needed for the F5 configurations.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
