import os
import sys
import time
import pandas as pd
from contextlib import contextmanager
from gensim.models import word2vec, KeyedVectors, FastText
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
save_path = "exp2_fasttext_selftrain_nopreprocess.model"
setup_logger(out_file=LOGGER_PATH)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


def train_w2v(train_text, w2v_params, save_path):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    model = FastText(train_corpus, **w2v_params)
    model.save(save_path)


if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df = train_df.append(test_df).reset_index(drop=True)

    with timer('train embeddings'):
        train_w2v(train_df['comment_text'], w2v_params, save_path)
