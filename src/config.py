"""Shared model and dataset locations for clone-and-run demos."""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models/xgboost.joblib")))
CSV_PATH = Path(os.getenv("PAYSIM_PATH", str(ROOT / "data/raw/paysim.csv")))
if not CSV_PATH.exists():
    CSV_PATH = ROOT / "data/sample/paysim_demo.csv"
