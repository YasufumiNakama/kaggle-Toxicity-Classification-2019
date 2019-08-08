import os
import sys
import time
import pickle
from contextlib import contextmanager
from gensim.models import word2vec, KeyedVectors
from keras.preprocessing.text import text_to_word_sequence

sys.path.append("../input/toxic-src")
from logger import setup_logger, LOGGER


# ===============
# Constants
# ===============
SAVE_DIR = "./"
DATA_DIR = "../input/jigsaw-unintended-bias-in-toxicity-classification"
LOGGER_PATH = os.path.join(SAVE_DIR, "log.txt")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")


# ===============
# Settings
# ===============
w2v_params = {
    "size": 300,
    "iter": 5,
    "seed": 0,
    "min_count": 1,
    "workers": 1
}
save_path = "exp3_w2v_selftrain_preprocess.model"
setup_logger(out_file=LOGGER_PATH)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


def train_w2v(train_text, w2v_params, save_path):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    model = word2vec.Word2Vec(train_corpus, **w2v_params)
    model.save(save_path)


if __name__ == '__main__':
    with open('cleaned_text_train.pkl', 'rb') as f:
        train_text = list(pickle.load(f))
    with open('cleaned_text_test.pkl', 'rb') as f:
        test_text = list(pickle.load(f))

    with timer('train embeddings'):
        train_w2v(train_text+test_text, w2v_params, save_path)
