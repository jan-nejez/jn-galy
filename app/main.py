import pandas as pd
import numpy as np
from itertools import combinations
import pickle
from similarity import calculate_similarity
from vectorizer import load_model


def main():
    model = load_model('../data/vectors.csv')
    df_phrases = pd.read_csv('../data/phrases.csv', encoding='latin1')
    combo = combinations(df_phrases['Phrases'], 2)
    rows = [calculate_similarity(phrase1, phrase2, model) for phrase1, phrase2 in combo]

    df = pd.DataFrame(rows, columns=['Phrase1', 'Phrase2', 'L2_distance'])
    df.to_csv('../data/preprocessed/similarity.csv')


if __name__ == '__main__':
    main()
