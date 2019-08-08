#! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a
import os
import gc
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apex import amp
from os.path import join
from contextlib import contextmanager
import torch
import torch.utils.data

sys.path.append("../input/toxic-src")
from util import seed_torch, convert_lines, setting
from trainer import train_one_epoch, validate
from logger import setup_logger, LOGGER
from toxic_metric import compute_bias_metrics_for_model, get_final_metric, calculate_overall_auc

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
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
bert_config = BertConfig('../input/bert-output/bert_config.json')
model_path = "../input/bert-output/exp1_dic"


# ===============
# Settings
# ===============
seed = 0
device = "cuda:0"
epochs = 1
n_labels = 1
max_len = 220
batch_size = 32
accumulation_steps = 2
train_size = 1200000
valid_size = 100000
gamma = 0.05
exp = "exp1"
seed_torch(seed)
setup_logger(out_file=LOGGER_PATH)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)


def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + identity_columns:
        convert_to_bool(bool_df, col)
    return bool_df


def main():
    # train_df = pd.read_csv(TRAIN_PATH).sample(frac=1.0, random_state=seed)
    # train_size = int(len(train_df) * 0.9)
    train_df = pd.read_csv(TRAIN_PATH).sample(train_size + valid_size, random_state=seed)
    LOGGER.info(f'data_size is {len(train_df)}')
    LOGGER.info(f'train_size is {train_size}')

    y = np.where(train_df['target'] >= 0.5, 1, 0)
    y_aux = train_df[AUX_COLUMNS].values

    identity_columns_new = []
    for column in identity_columns + ['target']:
        train_df[column + "_bin"] = np.where(train_df[column] >= 0.5, True, False)
        if column != "target":
            identity_columns_new.append(column + "_bin")

    sample_weights = np.ones(len(train_df), dtype=np.float32)
    sample_weights += train_df[identity_columns_new].sum(axis=1)
    sample_weights += train_df['target_bin'] * (~train_df[identity_columns_new]).sum(axis=1)
    sample_weights += (~train_df['target_bin']) * train_df[identity_columns_new].sum(axis=1) * 5
    sample_weights /= sample_weights.mean()

    with timer('preprocessing text'):
        # df["comment_text"] = [analyzer_embed(text) for text in df["comment_text"]]
        train_df['comment_text'] = train_df['comment_text'].astype(str)
        train_df = train_df.fillna(0)

    with timer('load embedding'):
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=True)
        X_text = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"), max_len, tokenizer)

    test_df = train_df[train_size:]

    with timer('train'):
        X_train, y_train, y_aux_train, w_train = X_text[:train_size], y[:train_size], y_aux[
                                                                                      :train_size], sample_weights[
                                                                                                    :train_size]
        X_val, y_val, y_aux_val, w_val = X_text[train_size:], y[train_size:], y_aux[train_size:], sample_weights[
                                                                                                  train_size:]
        model = BertForSequenceClassification(bert_config, num_labels=n_labels)
        model.load_state_dict(torch.load(model_path))
        model.zero_grad()
        model = model.to(device)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long),
                                                       torch.tensor(y_train, dtype=torch.float))
        valid = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long),
                                               torch.tensor(y_val, dtype=torch.float))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size * 2, shuffle=False)

        sample_weight_train = [w_train.values, np.ones_like(w_train)]
        sample_weight_val = [w_val.values, np.ones_like(w_val)]

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        num_train_optimization_steps = int(epochs * train_size / batch_size / accumulation_steps)
        total_step = int(epochs * train_size / batch_size)

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=2e-5*gamma,
                             warmup=0.05,
                             t_total=num_train_optimization_steps)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)

        LOGGER.info(f"Starting 1 epoch...")
        tr_loss, train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                accumulation_steps, total_step, n_labels)
        LOGGER.info(f'Mean train loss: {round(tr_loss,5)}')

        torch.save(model.state_dict(), '{}_dic'.format(exp))

        valid_loss, oof_pred = validate(model, valid_loader, criterion, device, n_labels)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    test_df["pred"] = oof_pred.reshape(-1)
    test_df = convert_dataframe_to_bool(test_df)
    bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns)
    LOGGER.info(bias_metrics_df)

    score = get_final_metric(bias_metrics_df, calculate_overall_auc(test_df))
    LOGGER.info(f'final score is {score}')

    test_df.to_csv("oof.csv", index=False)

    xs = list(range(1, len(train_losses) + 1))
    plt.plot(xs, train_losses, label='Train loss');
    plt.legend();
    plt.xticks(xs);
    plt.xlabel('Iter')
    plt.savefig("loss.png")


if __name__ == '__main__':
    main()