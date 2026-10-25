"""Make the backend package importable when a script is run from anywhere."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")
