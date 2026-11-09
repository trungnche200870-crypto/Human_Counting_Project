"""Download public test videos as data only; verify their recorded SHA-256."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads((ROOT / 'tests/public_samples.json').read_text())
    for source in sources:
        target = ROOT / source['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            urllib.request.urlretrieve(source['url'], target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != source['sha256']:
            raise RuntimeError(f"Sample changed: {target}. Check source before updating the manifest.")
        print(source['name'], 'verified', flush=True)
    (ROOT / 'data/public_samples/sources.json').write_text(json.dumps(sources, indent=2))


if __name__ == '__main__':
    main()
