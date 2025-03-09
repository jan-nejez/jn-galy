# jn-galy

## Overview
This project is designed to preprocess phrases, vectorize them, and compute pairwise similarities using various metrics. It leverages FastAPI for API endpoints and uses a Conda environment for dependency management.

## Requirements
- Conda
- Python 3.10

## Installation
To set up the environment, run:
```sh
make install
```

## Download Model
To download the required model, run:
```sh
make download-model
```

## Running the API
To start the FastAPI server, run:
```sh
make run-api
```
The API will be available at `http://127.0.0.1:8000`.

Currently, the API has the following endpoints:
- POST '/phrase-similarity': Computes the pairwise similarities between posted phrase and precomputed vectors from phrases.csv or any other phrase list provided.
  - Request Body: 
      - "phrase": "your phrase here", 
      - "phrases": ".csv filename in data/raw" default='phrases.csv', 
      - "metric": "cosine|euclidean" default='cosine', 

## Running the Main Script
To run the main script for computing pairwise similarities, run:
```sh
make run-main
```

## Project Structure
- `environment.yml`: Conda environment configuration file.
- `Makefile`: Makefile with commands for installation, downloading models, and running the application.
- `app/main.py`: Main script for preprocessing phrases and computing similarities.
- `app/similarity.py`: Module for comparing phrases.
- `app/vectorizer.py`: Module for loading and vectorizing phrases.
- `data/raw/`: Directory for storing raw data.
- `data/output/`: Directory for storing processed similarity.
- `data/preprocessed/`: Directory for storing preprocessed vectors
- `config.py`: Configuration file with directory paths.

## Usage
The main script `app/main.py` can be executed to preprocess phrases and compute pairwise similarities. The results will be saved in the `OUTPUT_DIR` specified in `config.py`.

## Comments and future considerations
I don't really know final use case of this implementation so I've used different approaches to solve step 1 and step 2 of the exercise with batch processing as click implementation and on the fly as api endpoint. but as they share computation, I'm not sure if this was correct approach. Feels like both are suboptimal.

From phrases provided, it feels that goal of it could be some kind of NLP of query suggestions mapping. For that I can imagine a lot of phrases being provided so to speed up similarity calculation I would consider clustering precalculated vectors and compare similarity with cluster centroid instead of all vectors and then within cluster or even multiple layers of clustering.

From implementation perspective I've omitted a lot. No tests, basic and sporadic logging, minimal intput validation, no error handling. I've marked some TODOs, but a lot of them missing. Please don't try to break it, it's too easy.

I've started dockerizing the api part of app, but with vectors.csv being 3GB, I would have to create multi-stage build and there is a lot of traps to handle. So I won't even share it.

## License
This project is licensed under the MIT License.
