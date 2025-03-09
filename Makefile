.PHONY: install download-model run-api run-main

install:
	conda env update -f environment.yml --prune

download-model:
	conda run -n jn-galy python -m scripts.download_model

run-api:
	uvicorn app.api:app --reload --host 127.0.0.1 --port 8000

run-main:
	conda run -n jn-galy python -m app.main