import os
import pandas as pd
from itertools import combinations
from similarity import compare_phrases
from vectorizer import load_model, vectorise_phrase
from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, OUTPUT_DIR


def preprocess_phrases(df: pd.DataFrame, model) -> pd.DataFrame:
    # vectorise phrases dataframe so that we can calculate similarity and store it in a new column
    df['Vector'] = df['Phrases'].apply(lambda x: vectorise_phrase(x, model))
    df.to_pickle(PREPROCESSED_DATA_DIR / 'phrases.pkl')
    return df


def batch_execution():
    phrases_pkl_path = PREPROCESSED_DATA_DIR / 'phrases.pkl'

    if os.path.exists(phrases_pkl_path):
        df_phrases = pd.read_pickle(phrases_pkl_path)
    else:
        model = load_model(PREPROCESSED_DATA_DIR / 'vectors.csv')
        df_phrases = preprocess_phrases(pd.read_csv(RAW_DATA_DIR / 'phrases.csv', encoding='latin1'), model)

    pv = df_phrases.set_index("Phrases")["Vector"].to_dict()
    combo = combinations(pv.keys(), 2)
    rows = [[phrase1, phrase2] + compare_phrases(pv[phrase1], pv[phrase2]) for phrase1, phrase2 in combo]

    df = pd.DataFrame(rows, columns=['Phrase1', 'Phrase2', 'L2_distance', 'Cosine_similarity'])
    df.to_csv(OUTPUT_DIR / 'similarity.csv')


if __name__ == '__main__':
    batch_execution()
