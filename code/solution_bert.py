import os
import gc
import sys
import time
import numpy as np
import pandas as pd
from os.path import join
from contextlib import contextmanager
import torch
import torch.utils.data

sys.path.append("../input/toxic-src")
from util import seed_torch, convert_lines, setting
from logger import setup_logger, LOGGER

sys.path.append("../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT")
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass


# ===============
# Constants
# ===============
SAVE_DIR = "./"
DATA_DIR = "../input/jigsaw-unintended-bias-in-toxicity-classification"
LOGGER_PATH = os.path.join(SAVE_DIR, "log.txt")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
bert_config = BertConfig('../input/bert-output/bert_config.json')
model_path = "../input/bert-output/exp1_dic"

# ===============
# Settings
# ===============
seed = 0
device = "cuda:0"
# n_labels = len(AUX_COLUMNS) + 1
n_labels = 1
max_len = 220
batch_size = 32
exp = "exp1"
seed_torch(seed)
setup_logger(out_file=LOGGER_PATH)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


def inference(model, test_loader, device, n_labels):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    test_pred = []
    with torch.no_grad():
        for features in test_loader:
            features = features.to(device)
            logits = model(features, attention_mask=features > 0, labels=None)
            test_pred.append(torch.sigmoid(logits))

        test_pred = torch.cat(test_pred).float().cpu().numpy()

    return test_pred


def main():
    test_df = pd.read_csv(TEST_PATH)

    with timer('preprocessing text'):
        test_df['comment_text'] = test_df['comment_text'].astype(str)
        test_df = test_df.fillna(0)

    with timer('load embedding'):
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=True)
        X_text = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), max_len, tokenizer)

    with timer('train'):
        model = BertForSequenceClassification(bert_config, num_labels=n_labels)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_text, dtype=torch.long))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

        test_pred = inference(model, test_loader, device, n_labels)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    submission = pd.DataFrame.from_dict({
        'id': test_df['id'],
        'prediction': test_pred.reshape(-1)
    })
    submission.to_csv('submission.csv', index=False)
    LOGGER.info(submission.head())


if __name__ == '__main__':
    main()
