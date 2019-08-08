import os
import sys
import time
import nltk
import pickle
import numpy as np
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
FASTTEXT_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
GLOVE_PATH = '../input/glove.840B.300d.pkl'
ps = nltk.stem.PorterStemmer()
lc = nltk.stem.lancaster.LancasterStemmer()
sb = nltk.stem.snowball.SnowballStemmer('english')


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
save_path = "exp6_fasttext_finetune_preprocess"
setup_logger(out_file=LOGGER_PATH)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


def load_embeddings(embed_dir):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embed_dir, encoding="utf8"))
    return embedding_index

def load_embeddings_pickle(embed_dir):
    with open(embed_dir, 'rb') as f:
        embedding_index = pickle.load(f)
    return embedding_index


def load_embedding(embeddings_index, model, embedding_dim=300):
    words = model.wv.index2entity

    embedding_matrix = np.zeros((len(words), embedding_dim))

    for i, word in enumerate(words):
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
            continue
        word_ = word.upper()
        if word_ in embeddings_index:
            embedding_matrix[i] = embeddings_index[word_]
            continue
        word_ = word.capitalize()
        if word_ in embeddings_index:
            embedding_matrix[i] = embeddings_index[word_]
            continue
        word_ = ps.stem(word)
        if word_ in embeddings_index:
            embedding_matrix[i] = embeddings_index[word_]
            continue
        word_ = lc.stem(word)
        if word_ in embeddings_index:
            embedding_matrix[i] = embeddings_index[word_]
            continue
        word_ = sb.stem(word)
        if word_ in embeddings_index:
            embedding_matrix[i] = embeddings_index[word_]
            continue
        try:
            embedding_matrix[i] = embeddings_index["unkown"]
        except:
            continue

    return embedding_matrix


def train_w2v(train_text, w2v_params, save_path, embeddings_indexes):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    model = FastText(**w2v_params)
    model.build_vocab(train_corpus)

    embedding_matrix = [load_embedding(embeddings_index, model) for embeddings_index in embeddings_indexes]
    embedding_matrix = np.mean(embedding_matrix, axis=0)
    model.wv.vectors[:] = embedding_matrix
    model.trainables.syn1neg[:] = embedding_matrix
    model.train(train_corpus, total_examples=len(train_corpus), epochs=model.epochs)
    model.save("{}.model".format(save_path))

    dic = {}
    for word in model.wv.vocab:
        dic[word] = model.wv[word]
    with open("{}.pkl".format(save_path), 'wb') as f:
        pickle.dump(dic, f)


if __name__ == '__main__':
    with open('cleaned_text_train.pkl', 'rb') as f:
        train_text = list(pickle.load(f))
    with open('cleaned_text_test.pkl', 'rb') as f:
        test_text = list(pickle.load(f))

    with timer('train embeddings'):
        fasttext_index = load_embeddings(FASTTEXT_PATH)
        glove_index = load_embeddings_pickle(GLOVE_PATH)
        embeddings_indexes = [fasttext_index, glove_index]
        train_w2v(train_text+test_text, w2v_params, save_path, embeddings_indexes)
