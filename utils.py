# coding: UTF-8
import os
from typing import List, Any, Tuple

import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train: List[Tuple[List[Any], int, int]] = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        # batches ([14,
        #    125,
        #    55,
        #    45,
        #    35,
        #    307,
        #    4,
        #    81,
        #    161,
        #    941,
        #    258,
        #    494,
        #    2,
        #    175,
        #    48,
        #    145,
        #    97,
        #    17,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761,
        #    4761],
        #   3,
        #   18)
        # batch_size = 128
        self.batch_size = batch_size  # 128
        self.batches = batches # len(self.batches)  180000
        self.n_batches = len(batches) // batch_size  # 1406.25
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)


# ((tensor([[  14,  125,   55,  ..., 4761, 4761, 4761],
#           [ 135,   80,   33,  ..., 4761, 4761, 4761],
#           [ 152,   13,  469,  ..., 4761, 4761, 4761],
#           ...,
#           [ 160, 1667, 1147,  ..., 4761, 4761, 4761],
#           [  31,   75,    4,  ..., 4761, 4761, 4761],
#           [ 321,  566,  130,  ..., 4761, 4761, 4761]]),
#   tensor([18, 22, 25, 25, 23, 20, 17, 22, 16, 11, 23, 23, 22, 15,  7, 23, 20, 25,
#           15,  9, 17, 15, 24, 20, 17, 17, 13, 20, 19, 20, 22, 22, 21, 22, 23, 19,
#           12, 20, 23, 18, 22, 25, 23, 20, 19, 17, 17, 15, 17, 26, 16, 22, 21, 18,
#           16, 12, 23, 19, 20, 21, 12, 24, 18, 14, 25, 16, 24, 24, 23, 20, 20, 20,
#           18, 16, 23, 14, 23, 21, 19, 17, 24, 21, 23, 23, 19, 15, 12, 22, 25, 14,
#           21, 20, 22, 15, 22, 18, 16, 17, 13, 21, 21, 18, 21, 11, 19, 22, 14, 22,
#           15, 22, 15, 22, 22, 15, 25, 16, 18, 18, 14, 19, 13, 29, 20, 18, 22, 16,
#           18, 22])),
#  tensor([3, 4, 1, 7, 5, 5, 9, 1, 8, 4, 3, 7, 5, 2, 1, 8, 1, 1, 8, 4, 4, 6, 7, 1,
#          9, 4, 2, 9, 4, 2, 2, 9, 8, 9, 1, 3, 9, 5, 9, 6, 7, 2, 9, 5, 9, 4, 5, 6,
#          8, 1, 2, 1, 4, 0, 5, 4, 9, 6, 5, 5, 2, 4, 5, 5, 7, 8, 6, 7, 7, 2, 9, 0,
#          4, 6, 7, 2, 9, 7, 9, 0, 2, 9, 9, 4, 9, 0, 0, 4, 1, 2, 5, 5, 7, 0, 5, 9,
#          5, 3, 4, 6, 8, 3, 5, 9, 3, 9, 4, 9, 5, 4, 6, 2, 3, 6, 7, 4, 6, 2, 2, 2,
#          0, 1, 6, 4, 4, 2, 2, 3]))