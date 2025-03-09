
import gdown
from gensim.models import KeyedVectors
from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_google_word2vec(output_path):
    file_id = "0B7XkCwpI5KDYNlNUTTlSS21pQmM"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


if __name__ == '__main__':
    location = RAW_DATA_DIR / 'GoogleNews-vectors-negative300.bin.gz'
    download_google_word2vec(str(location))
    wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=1000000)

    wv.save_word2vec_format(PREPROCESSED_DATA_DIR / 'vectors.csv')
    logging.info("Model downloaded and saved successfully")