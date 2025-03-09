import numpy as np


def calculate_similarity(vector1: np.array, vector2: np.array, metric: str) -> float:
    if metric == 'euclidean':
        return np.linalg.norm(vector1 - vector2)
    elif metric == 'cosine':
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    else:
        raise ValueError(f'Invalid metric: {metric}')


def compare_phrases(vector1: np.array, vector2: np.array) -> list:

    if vector1 is None or vector2 is None:
        return [None,
                None]
    return [calculate_similarity(vector1, vector2, 'euclidean'),
            calculate_similarity(vector1, vector2, 'cosine')]
