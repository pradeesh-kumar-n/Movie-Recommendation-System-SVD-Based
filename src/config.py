

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RATINGS_FILE = DATA_DIR / "u.csv"
MOVIES_FILE = DATA_DIR / "mm.csv"
SVD_N_FACTORS = 50