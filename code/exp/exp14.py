# ===============
# Bert cased
# ===============
# ! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a
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
from dataset import prepare_data_loader, LenMatchBatchSampler
from utils_gpt import seed_torch, convert_lines_head_tail, setting, trim_tensors
from logger import setup_logger, LOGGER
from losses import CustomLoss
from toxic_metric import compute_bias_metrics_for_model, get_final_metric, calculate_overall_auc

sys.path.append("../input/gpt2-pytorch/pytorch-pretrained-bert-master/pytorch-pretrained-BERT-master")
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig


def train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps, total_step, n_labels,
                    base_lr, steps_upd_logging=500, gamma=None):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    train_losses = []
    for step, (features, targets) in enumerate(train_loader):
        features = trim_tensors(features)
        features, targets = features.to(device), targets.to(device)
        logits = model(features, attention_mask=features > 0, labels=None)

        if n_labels == 1:
            loss = criterion(logits.view(-1, 1), targets.view(-1, 1))
        else:
            loss = criterion(logits, targets)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            optimizer.zero_grad()

        if gamma is not None and step == int(total_step / 2):
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * gamma

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            train_losses.append(total_loss / (step + 1))
            LOGGER.info(f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}')

    return total_loss / (step + 1), train_losses


def validate(model, valid_loader, criterion, device, n_labels):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    test_loss = 0.0
    oof_pred = []
    with torch.no_grad():

        for step, (features, targets) in enumerate(valid_loader):
            features = trim_tensors(features)
            features, targets = features.to(device), targets.to(device)

            logits = model(features, attention_mask=features > 0, labels=None)
            if n_labels == 1:
                loss = criterion(logits.view(-1, 1), targets.view(-1, 1))
            else:
                loss = criterion(logits, targets)

            test_loss += loss.item()
            oof_pred.append(torch.sigmoid(logits))

        oof_pred = torch.cat(oof_pred).float().cpu().numpy()

    for param in model.parameters():
        param.requires_grad = True

    LOGGER.info(f'Mean val loss: {round(test_loss / (step + 1), 5)}')
    return test_loss / (step + 1), oof_pred


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass


# ===============
# Constants
# ===============
SAVE_DIR = "./"
WORK_DIR = "../working/"
DATA_DIR = "../input/jigsaw-unintended-bias-in-toxicity-classification"
FOLD_PATH = "../input/toxic-folds/fold01.csv"
LOGGER_PATH = os.path.join(SAVE_DIR, "log.txt")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
BERT_MODEL_PATH = '../input/bert-pretrained-models/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12/'
bert_config = BertConfig(os.path.join(BERT_MODEL_PATH, 'bert_config.json'))


# ===============
# Settings
# ===============
fold_id = 0
seed = 0
device = "cuda:0"
epochs = 1
n_labels = len(AUX_COLUMNS) + 1
max_len = 220
head_len = 80
batch_size = 32
base_lr = 2e-5
gammas = [0.75, 0.5, 0.25]
accumulation_steps = 2
exp = "exp14_bert_base_cased"
seed_torch(seed)
setup_logger(out_file=LOGGER_PATH)
mkdir(WORK_DIR)
setting(BERT_MODEL_PATH, WORK_DIR)


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

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
def main():
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['male'] = np.load("../input/identity-column-data/male_labeled.npy")
    train_df['female'] = np.load("../input/identity-column-data/female_labeled.npy")
    train_df['homosexual_gay_or_lesbian'] = np.load("../input/identity-column-data/homosexual_gay_or_lesbian_labeled.npy")
    train_df['christian'] = np.load("../input/identity-column-data/christian_labeled.npy")
    train_df['jewish'] = np.load("../input/identity-column-data/jewish_labeled.npy")
    train_df['muslim'] = np.load("../input/identity-column-data/muslim_labeled.npy")
    train_df['black'] = np.load("../input/identity-column-data/black_labeled.npy")
    train_df['white'] = np.load("../input/identity-column-data/white_labeled.npy")
    train_df['psychiatric_or_mental_illness'] = np.load("../input/identity-column-data/psychiatric_or_mental_illness_labeled.npy")
    fold_df = pd.read_csv(FOLD_PATH)

    # y = np.where(train_df['target'] >= 0.5, 1, 0)
    y = train_df['target'].values
    y_aux = train_df[AUX_COLUMNS].values

    identity_columns_new = []
    for column in identity_columns + ['target']:
        train_df[column + "_bin"] = np.where(train_df[column] >= 0.5, True, False)
        if column != "target":
            identity_columns_new.append(column + "_bin")

    # Overall
    weights = np.ones((len(train_df),)) / 4
    # Subgroup
    weights += (train_df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (((train_df["target"].values >= 0.5).astype(bool).astype(np.int) +
                 (1 - (train_df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(
                     np.int))) > 1).astype(bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (((train_df["target"].values < 0.5).astype(bool).astype(np.int) +
                 (train_df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(
                     np.int)) > 1).astype(bool).astype(np.int) / 4
    loss_weight = 0.5

    with timer('preprocessing text'):
        # df["comment_text"] = [analyzer_embed(text) for text in df["comment_text"]]
        train_df['comment_text'] = train_df['comment_text'].astype(str)
        train_df = train_df.fillna(0)

    with timer('load embedding'):
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=False)
        X_text = convert_lines_head_tail(train_df["comment_text"].fillna("DUMMY_VALUE"), max_len,
                                         head_len, tokenizer)
        X_text = np.array(X_text).astype("int32")
        del tokenizer
        gc.collect()

    with timer('train'):
        train_index = fold_df.fold_id != fold_id
        valid_index = fold_df.fold_id == fold_id
        X_train, y_train, y_aux_train, w_train = X_text[train_index], y[train_index].astype("float32"), y_aux[
            train_index].astype("float32"), weights[train_index].astype("float32")
        X_val, y_val, y_aux_val, w_val = X_text[valid_index], y[valid_index].astype("float32"),\
                                         y_aux[valid_index].astype("float32"), weights[valid_index].astype("float32")
        test_df = train_df[valid_index]
        train_size = len(X_train)
        del X_text, y, y_aux, weights, train_index, valid_index, train_df, fold_df
        gc.collect()

        model = BertForSequenceClassification.from_pretrained(WORK_DIR, cache_dir=None, num_labels=n_labels)
        model.zero_grad()
        model = model.to(device)

        y_train = np.concatenate((y_train.reshape(-1, 1), w_train.reshape(-1, 1), y_aux_train), axis=1).astype(
            "float32")
        y_val = np.concatenate((y_val.reshape(-1, 1), w_val.reshape(-1, 1), y_aux_val), axis=1).astype("float32")
        del w_train, w_val, y_aux_train, y_aux_val
        gc.collect()

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long),
                                                       torch.tensor(y_train, dtype=torch.float32))
        valid = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long),
                                               torch.tensor(y_val, dtype=torch.float32))
        ran_sampler = torch.utils.data.RandomSampler(train_dataset)
        len_sampler = LenMatchBatchSampler(ran_sampler, batch_size=batch_size, drop_last=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=len_sampler)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size * 2, shuffle=False)
        del X_train, y_train, X_val, y_val
        gc.collect()
        LOGGER.info(f"done data loader setup")

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        num_train_optimization_steps = int(epochs * train_size / batch_size / accumulation_steps)
        total_step = int(epochs * train_size / batch_size)

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=base_lr,
                             warmup=0.005,
                             t_total=num_train_optimization_steps)
        LOGGER.info(f"done optimizer loader setup")

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        # criterion = torch.nn.BCEWithLogitsLoss().to(device)
        criterion = CustomLoss(loss_weight).to(device)
        LOGGER.info(f"done amp setup")

        for epoch in range(epochs):
            LOGGER.info(f"Starting {epoch} epoch...")
            LOGGER.info(f"length {train_size} train...")
            if epoch == 1:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr * gammas[1]
            tr_loss, train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                    accumulation_steps, total_step, n_labels, base_lr,
                                                    gamma=gammas[2 * epoch])
            LOGGER.info(f'Mean train loss: {round(tr_loss,5)}')

            torch.save(model.state_dict(), '{}_epoch{}_fold{}.pth'.format(exp, epoch, fold_id))

            valid_loss, oof_pred = validate(model, valid_loader, criterion, device, n_labels)
            LOGGER.info(f'Mean valid loss: {round(valid_loss,5)}')

        del model
        gc.collect()
        torch.cuda.empty_cache()

    test_df["pred"] = oof_pred[:, 0]
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
