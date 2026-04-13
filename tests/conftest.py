import sys
from pathlib import Path


# Allow running tests without installing the package.
_repo_root = Path(__file__).resolve().parents[1]
_src = _repo_root / "src"
if _src.exists():
    sys.path.insert(0, str(_src))
