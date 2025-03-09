import os
import pandas as pd
import click
from itertools import combinations
from app.similarity import compare_phrases
from app.vectorizer import load_model, vectorise_phrase
from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, OUTPUT_DIR, MODEL_PATH


def preprocess_phrases(input_name: str, model) -> pd.DataFrame:
    # vectorise phrases dataframe so that we can calculate similarity and store it in a new column
    df = pd.read_csv(RAW_DATA_DIR / input_name, encoding='latin1')
    df['Vector'] = df['Phrases'].apply(lambda x: vectorise_phrase(x, model))
    df.to_pickle(PREPROCESSED_DATA_DIR / input_name.replace('.csv', '.pkl'))
    return df


def load_phrases(input_name: str, model=None) -> pd.DataFrame:
    phrases_pkl_path = PREPROCESSED_DATA_DIR / input_name.replace('.csv', '.pkl')

    if os.path.exists(phrases_pkl_path):
        df_phrases = pd.read_pickle(phrases_pkl_path)
    else:
        if model is None:
            model = load_model(MODEL_PATH)
        df_phrases = preprocess_phrases(input_name, model)
    return df_phrases


@click.command()
@click.option('--input_filename', type=str, default='phrases.csv', help='Name of the input csv file in data/raw '
                                                                        'directory. Default: phrases.csv')
def compute_pairwise_similarities(input_filename: str = 'phrases.csv'):
    df_phrases = load_phrases(input_filename)

    pv = df_phrases.set_index("Phrases")["Vector"].to_dict()
    combo = combinations(pv.keys(), 2)
    rows = [[phrase1, phrase2] + compare_phrases(pv[phrase1], pv[phrase2]) for phrase1, phrase2 in combo]

    df = pd.DataFrame(rows, columns=['Phrase1', 'Phrase2', 'L2_distance', 'Cosine_similarity'])
    df.to_csv(OUTPUT_DIR / 'similarity.csv')


if __name__ == '__main__':
    compute_pairwise_similarities()
