"""Root conftest — ensures backend/ is on sys.path for all tests."""

import sys
from pathlib import Path

# Add the backend directory to sys.path so that `from agents.base import ...` works
backend_dir = str(Path(__file__).resolve().parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
