from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PREPROCESSED_DATA_DIR = BASE_DIR / "data" / "preprocessed"
OUTPUT_DIR = BASE_DIR / "data" / "output"
