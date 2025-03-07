import numpy as np
from vectorizer import vectorise_phrase


def l2_distance(vector1: np.array, vector2: np.array) -> float:
    return np.linalg.norm(vector1 - vector2)


def calculate_similarity(phrase1: str, phrase2: str, model) :
    vector1 = vectorise_phrase(phrase1, model)
    vector2 = vectorise_phrase(phrase2, model)
    if vector1 is None or vector2 is None:
        return None
    return [phrase1, phrase2, l2_distance(vector1, vector2)]
