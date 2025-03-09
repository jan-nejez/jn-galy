import os
import pandas as pd
from itertools import combinations
from app.similarity import compare_phrases
from app.vectorizer import load_model, vectorise_phrase
from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, OUTPUT_DIR, MODEL_PATH


def preprocess_phrases(df: pd.DataFrame, model) -> pd.DataFrame:
    # vectorise phrases dataframe so that we can calculate similarity and store it in a new column
    df['Vector'] = df['Phrases'].apply(lambda x: vectorise_phrase(x, model))
    df.to_pickle(PREPROCESSED_DATA_DIR / 'phrases.pkl')
    return df


def load_phrases(model=None) -> pd.DataFrame:
    phrases_pkl_path = PREPROCESSED_DATA_DIR / 'phrases.pkl'

    if os.path.exists(phrases_pkl_path):
        df_phrases = pd.read_pickle(phrases_pkl_path)
    else:
        if model is None:
            model = load_model(MODEL_PATH)
        df_phrases = preprocess_phrases(pd.read_csv(RAW_DATA_DIR / 'phrases.csv', encoding='latin1'), model)
    return df_phrases


def batch_execution():
    df_phrases = load_phrases()

    pv = df_phrases.set_index("Phrases")["Vector"].to_dict()
    combo = combinations(pv.keys(), 2)
    rows = [[phrase1, phrase2] + compare_phrases(pv[phrase1], pv[phrase2]) for phrase1, phrase2 in combo]

    df = pd.DataFrame(rows, columns=['Phrase1', 'Phrase2', 'L2_distance', 'Cosine_similarity'])
    df.to_csv(OUTPUT_DIR / 'similarity.csv')


if __name__ == '__main__':
    batch_execution()
