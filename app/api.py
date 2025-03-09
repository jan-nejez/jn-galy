from fastapi import FastAPI, HTTPException
import pandas as pd
import logging
from app.vectorizer import load_model, vectorise_phrase
from app.similarity import calculate_similarity
from app.main import load_phrases
from config import MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# TODO make sure that vectors exists
# TODO implement readiness check
logging.info("Loading model...")
MODEL = load_model(MODEL_PATH)
logging.info("Model loaded successfully")
app = FastAPI()


@app.post("/phrase-similarity")
async def phrase_similarity(request_data: dict):
    phrase = request_data.get("phrase")
    phrases_file = request_data.get("phrases", 'phrases.csv')
    save_vector = request_data.get("save_vector", False)  # Default to False
    metric = request_data.get("metric", "cosine")
    valid_metrics = ["cosine", "euclidean"]
    if metric not in valid_metrics:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid metric '{metric}'. Supported metrics are: {', '.join(valid_metrics)}"
        )
    logging.info(f"Vectorising phrase: {phrase}")
    phrase_vector = vectorise_phrase(phrase, MODEL)

    df_phrases = load_phrases(phrases_file, MODEL)
    pv = df_phrases.set_index("Phrases")["Vector"].to_dict()
    logging.info("Calculating similarity...")
    rows = [[phrase,  calculate_similarity(phrase_vector, pv[phrase], metric)] for phrase in pv.keys()]
    if save_vector:
        pass    # TODO to save vector to pickle so it can be compared next time with others

    if metric == 'cosine':
        df_results = pd.DataFrame(rows, columns=['Phrase', 'Cosine_similarity'])
        return df_results[df_results["Cosine_similarity"] == df_results["Cosine_similarity"].max()].to_dict()
    elif metric == 'euclidean':
        df_results = pd.DataFrame(rows, columns=['Phrase', 'Euclidean_distance'])
        return df_results[df_results["Euclidean_distance"] == df_results["Euclidean_distance"].min()].to_dict()



