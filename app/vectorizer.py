from gensim.models import KeyedVectors
import numpy as np


def load_model(path):
    return KeyedVectors.load_word2vec_format(path)


def get_word_vector(word: str, model: KeyedVectors):
    try:
        return model[word]
    except KeyError:
        return None


def vectorise_phrase(phrase: str, model: KeyedVectors):
    words = phrase.split()
    vectors = [get_word_vector(word.lower(), model) for word in words]
    sum_vectors = np.sum([vector for vector in vectors if vector is not None], axis=0)
    return sum_vectors / np.linalg.norm(sum_vectors) if vectors else None
