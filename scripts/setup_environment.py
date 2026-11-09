"""Create or reuse the Windows .venv without changing global Python or PowerShell policy.

Run with any installed Python 3: python scripts/setup_environment.py
If needed: python scripts/setup_environment.py --python C:/path/to/Python311/python.exe
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
MODELS = {
    name: f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}'
    for name in ('yolo11n.pt', 'yolo11s.pt')
}


def run(command):
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def is_python311(executable):
    try:
        result = subprocess.run([str(executable), '-c',
                                 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'],
                                capture_output=True, timeout=15)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def find_python311(explicit):
    if explicit:
        candidate = shutil.which(explicit) or explicit
        if is_python311(candidate):
            return candidate
        raise RuntimeError('--python must point to a working Python 3.11 interpreter.')
    candidates = [sys.executable]
    local_appdata = os.environ.get('LOCALAPPDATA')
    if local_appdata:
        candidates.append(Path(local_appdata) / 'Programs/Python/Python311/python.exe')
    for candidate in candidates:
        if is_python311(candidate):
            return candidate
    launcher = shutil.which('py')
    if launcher:
        result = subprocess.run([launcher, '-3.11', '-c', 'import sys; print(sys.executable)'],
                                capture_output=True, text=True, timeout=20)
        if result.returncode == 0 and is_python311(result.stdout.strip()):
            return result.stdout.strip()
    raise RuntimeError('Python 3.11 was not found. Install Python 3.11, then rerun with --python PATH_TO_PYTHON311.')


def download(url, destination):
    if destination.is_file() and destination.stat().st_size >= 1_000_000:
        print(f'Already present: {destination.name}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    print(f'Downloading official release: {destination.name}', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as handle:
        shutil.copyfileobj(response, handle)
    if partial.stat().st_size < 1_000_000:
        raise RuntimeError(f'Download is incomplete: {destination.name}')
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', help='Path to a Python 3.11 interpreter if creating .venv')
    parser.add_argument('--with-tunnel', action='store_true', help='Also download the official cloudflared tool')
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('This VS Code setup targets Windows.')
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    if VENV_PYTHON.is_file():
        if not is_python311(VENV_PYTHON):
            raise RuntimeError('Existing .venv does not run Python 3.11. It was not modified; select a compatible environment first.')
        print('Reusing .venv; installed CUDA/PyTorch builds are retained when requirements are already satisfied.', flush=True)
    else:
        if (ROOT / '.venv').exists():
            raise RuntimeError('An incomplete .venv already exists. Move it aside manually before creating a new environment.')
        python = find_python311(args.python)
        print(f'Creating .venv with {python}', flush=True)
        run([python, '-m', 'venv', ROOT / '.venv'])
    run([VENV_PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', ROOT / 'requirements.txt'])
    run([VENV_PYTHON, '-m', 'pip', 'check'])
    for name, url in MODELS.items():
        download(url, ROOT / name)
    if args.with_tunnel:
        download('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe',
                 ROOT / 'tools/cloudflared.exe')
    run([VENV_PYTHON, ROOT / 'scripts/check_environment.py'])
    print('Ready. Open this folder in VS Code, select a Run and Debug configuration, and press F5.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Setup failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
