"""Put the pipeline's script directories on sys.path so the tests can import
them the same way the scripts import each other when run as
`python src/<stage>/<script>.py` (Python puts a script's own directory on
sys.path, which is how e.g. `from _model_utils import ...` resolves)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for subdir in ("data", "eda", "features", "models", "output"):
    sys.path.insert(0, str(ROOT / "src" / subdir))
