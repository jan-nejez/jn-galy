.PHONY: setup install download-model run-api run-main

setup:
	conda run -n jn-galy python -c "from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, OUTPUT_DIR; import os; [os.makedirs(d, exist_ok=True) for d in [RAW_DATA_DIR, PREPROCESSED_DATA_DIR, OUTPUT_DIR]]"

install:
	conda env update -f environment.yml --prune

download-model: setup
	conda run -n jn-galy python -m scripts.download_model

run-api:
	uvicorn app.api:app --reload --host 127.0.0.1 --port 8000

run-main:
	conda run -n jn-galy python -m app.main