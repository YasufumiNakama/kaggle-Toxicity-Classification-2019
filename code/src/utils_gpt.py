import os
import sys
import random
import shutil
import numpy as np
import torch

sys.path.append("../input/gpt2-pytorch/pytorch-pretrained-bert-master/pytorch-pretrained-BERT-master")
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch


def setting(BERT_MODEL_PATH, WORK_DIR):
    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
        BERT_MODEL_PATH + 'bert_model.ckpt',
        BERT_MODEL_PATH + 'bert_config.json',
        WORK_DIR + 'pytorch_model.bin')

    shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    text_length = []
    longer = 0
    for text in example:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
        text_length.append(len(tokens_a))
    return np.array(all_tokens), np.array(text_length)


def convert_lines_head_tail(example, max_seq_length, head_length, tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in example:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:head_length] + tokens_a[-(max_seq_length-head_length):]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


def convert_lines_gpt2(example, max_seq_length,tokenizer):
    all_tokens = []
    longer = 0
    for text in example:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(tokens_a)+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


def convert_lines_gpt2_head_tail(example, max_seq_length, head_length,tokenizer):
    all_tokens = []
    longer = 0
    for text in example:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:head_length] + tokens_a[-(max_seq_length-head_length):]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(tokens_a)+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)



def convert_lines_gpt2_(x, max_seq_length, tokenizer):
    x = x[0]
    tokens_a = tokenizer.tokenize(x)
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[-max_seq_length:]

    one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + \
                [0]*(max_seq_length-len(tokens_a))

    return np.array(one_token)


def convert_line_fast(example, max_seq_length,tokenizer):
    """
    https://www.kaggle.com/abhishek/convert-lines-faster-for-bert
    train_lines = zip(train_df['comment_text'].values.tolist(), train_df.target.values.tolist())
    res = Parallel(n_jobs=4, backend='multiprocessing')(delayed(convert_line)(i, 25, tokenizer) for i in train_lines)
    """
    example = example[0]
    max_seq_length -=2
    tokens_a = tokenizer.tokenize(example)
    if len(tokens_a)>max_seq_length:
      tokens_a = tokens_a[:max_seq_length]
    one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
    return one_token, len(tokens_a)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def trim_tensors(features):
    max_len = torch.max(torch.sum((features != 0 ), 1))
    if max_len > 2:
        features = features[:, :max_len]
    return features
