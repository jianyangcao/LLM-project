# coding: UTF-8
import os
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
        for line in tqdm(f): #tqdm 给循环加进度条
            lin = line.strip()#去掉行首尾空白符（含换行）
            if not lin: #如果这一行是空行，跳过
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content): #把文本切成token序列
                vocab_dic[word] = vocab_dic.get(word, 0) + 1 #对每个token计数
        #1.变成（token，count）列表 2.过滤掉出现次数太少的token和选取前max_size token
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        # 给列表元素编号，从0开始
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        #加入UNK和PAD toekn，编号分别为N和N+1
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    #有就直接加载，没有就建立
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def _collect_class_list(paths):
        class_set = []
        seen = set()
        for p in paths:
            if not os.path.exists(p):
                continue
            with open(p, 'r', encoding='UTF-8') as f:
                for line in f:
                    lin = line.strip()
                    if not lin:
                        continue
                    parts = lin.split('\t')
                    if len(parts) < 2:
                        continue
                    label_field = parts[1]
                    for raw in label_field.split():
                        if not raw:
                            continue
                        if '#' in raw:
                            label_name = raw.split('#', 1)[0]
                        else:
                            label_name = raw
                        if label_name and label_name not in seen:
                            seen.add(label_name)
                            class_set.append(label_name)
        return class_set

    def _parse_labels(label_field, class_to_id):
        type_vec = np.zeros(len(class_to_id), dtype=np.int32)
        score_sum = 0
        for raw in label_field.split():
            if not raw:
                continue
            if '#' in raw:
                label_name, score_str = raw.split('#', 1)
                try:
                    score = int(score_str)
                except ValueError:
                    score = 0
            else:
                label_name = raw
                score = 0
            if label_name in class_to_id:
                type_vec[class_to_id[label_name]] = 1
            score_sum += score
        if score_sum > 0:
            sent_label = 2  # positive
        elif score_sum < 0:
            sent_label = 0  # negative
        else:
            sent_label = 1  # neutral
        return type_vec, sent_label

    if getattr(config, 'class_list', None) is None or len(getattr(config, 'class_list', [])) == 0:
        config.class_list = _collect_class_list([config.train_path, config.dev_path, config.test_path])
    config.num_classes = len(config.class_list)
    config.sentiment_class_list = getattr(config, 'sentiment_class_list', ['neg', 'neu', 'pos'])
    config.num_sentiment_classes = len(config.sentiment_class_list)
    class_to_id = {c: i for i, c in enumerate(config.class_list)}

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                parts = lin.split('\t')
                if len(parts) < 2:
                    continue
                content, label_field = parts[0], parts[1]
                words_line = []
                token = tokenizer(content) #token列表
                seq_len = len(token)
                #统一长度，在需要的pad的时候，seq_len是不变的，截断的时候才会等于pad_size
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    #如果word在vocab里：用它的id，没有的话就用UNK的id
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                #把一个样本打包成四元组加入contents，就是样本列表
                #words_line：输入序列（id列表，固定长度）
                #type_vec：多标签类别向量
                #sent_label：情感类别(0/1/2)
                #seq_len：有效长度
                type_vec, sent_label = _parse_labels(label_field, class_to_id)
                contents.append((words_line, type_vec, sent_label, seq_len))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        # 这里的batches通常为实际传进来的样本列表
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0 #迭代器的指针/游标，表示当前取到第几个 batch 了
        self.device = device

    def _to_tensor(self, datas): #data一般为样本列表
        #得到一个二维张量，形状为【batch_size，pad_size】
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        #得到一个二维张量，形状为【batch_size, num_classes】
        y_type = torch.FloatTensor([_[1] for _ in datas]).to(self.device)
        #得到一个一维张量，形状为【batch_size】
        y_sent = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        #得到一个一维张量，形状为【batch_size】
        seq_len = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len), (y_type, y_sent)

    #定义了“每次迭代返回一个 batch”，并处理最后一个不满 batch（residue）以及迭代结束
    #self.index 表示当前要取第几个 batch（从 0 开始）；self.n_batches 是“满 batch 的个数”（整除得到的）
    #self.residue 表示是否存在“最后一个不满 batch”
    def __next__(self):
        #处理最后一个不满batch
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        #迭代结束
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        #正常情况，返回一个满batch
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
    #迭代器就是我自己；后续每次取数据请调用我的 __next__()
    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config): #
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
