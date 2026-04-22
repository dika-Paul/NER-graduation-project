import torch
from torch.utils.data import Dataset


def read_conll_4(path):
    """
    读取 CoNLL-2003 四列格式文件

    参数：
        path

    返回：
        sentences
        tags
    """
    sentences = []
    tags = []

    words = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 空行表示一句话结束
            if line == "":
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue

            # 跳过文档起始标记
            if line.startswith("-DOCSTART-"):
                continue

            parts = line.split()

            # CoNLL-2003 有 4 列：word POS chunk NER
            # 做一个稳妥判断，只要列数 >= 4 就取第 1 列和最后 1 列
            if len(parts) < 4:
                continue

            word = parts[0]
            label = parts[-1]

            words.append(word)
            labels.append(label)

    # 防止最后一句后面没有空行
    if words:
        sentences.append(words)
        tags.append(tags)

    return sentences, tags


def read_conll_2(path):
    """
    读取 CoNLL 风格的两列数据文件

    参数：
        path

    返回：
        sentences
        tags
    """
    sentences = []   # 保存所有句子
    tags = []        # 保存所有句子的标签序列

    words = []       # 当前句子的词
    labels = []      # 当前句子的标签

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 去掉首尾空白字符

            # 如果遇到空行，说明一个句子结束了
            if not line:
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue

            # 按空格切分，一般应得到 [单词, 标签]
            parts = line.split()

            # 如果这一行不是“词 标签”结构，就跳过
            if len(parts) != 2:
                continue

            word, label = parts
            words.append(word)
            labels.append(label)

    # 防止文件最后一句后面没有空行，导致最后一句漏掉
    if words:
        sentences.append(words)
        tags.append(labels)

    return sentences, tags


def build_vocab(sentences, min_freq=1):
    """
    根据训练集构建词表 word2idx

    参数：
        sentences
        min_freq: 最小词频

    返回：
        word2idx
    """
    word_count = {}

    # 统计每个词出现的次数
    for sent in sentences:
        for w in sent:
            word_count[w] = word_count.get(w, 0) + 1

    # 预留两个特殊符号
    # <PAD> 用于后续补齐
    # <UNK> 用于测试时遇到训练集里没有的词
    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    # 将满足词频要求的词加入词表
    for w, c in word_count.items():
        if c >= min_freq:
            word2idx[w] = len(word2idx)

    return word2idx


def build_tag2idx(tags_list):
    """
    根据标签序列构建 tag2idx

    参数：
        tags_list

    返回：
        tag2idx
    """
    tag2idx = {}

    # 遍历所有标签，加入字典
    for tags in tags_list:
        for t in tags:
            if t not in tag2idx:
                tag2idx[t] = len(tag2idx)

    ind2tag = {v: k for k, v in tag2idx.items()}
    return tag2idx, ind2tag


def encode_sentence(sentence, word2idx):
    """
    将一句话中的词转换成索引序列

    参数：
        sentence
        word2idx

    返回：
        索引列表
    """
    return [word2idx.get(w, word2idx["<UNK>"]) for w in sentence]


def encode_tags(tags, tag2idx):
    """
    将标签序列转换成索引序列

    参数：
        tags
        tag2idx

    返回：
        标签索引列表
    """
    return [tag2idx[t] for t in tags]


class NERDataset(Dataset):
    """
    NER 数据集类

    作用：
    1. 保存原始句子和标签
    2. 在 __getitem__ 中把词和标签转成 id
    """

    def __init__(self, sentences, tags, word2idx, tag2idx):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        取出第 idx 条样本，并转成 id 序列

        返回：
            sentence_ids: 词 id 列表
            tag_ids:      标签 id 列表
            length:       句子真实长度
        """
        sentence = self.sentences[idx]
        tags = self.tags[idx]

        sentence_ids = encode_sentence(sentence, self.word2idx)
        tag_ids = encode_tags(tags, self.tag2idx)
        length = len(sentence_ids)

        return sentence_ids, tag_ids, length


def collate_fn(batch):
    """
    自定义 batch 拼接函数

    因为每句话长度不同，DataLoader 不能直接堆叠，
    所以我们要手动做 padding。

    参数：
        batch: 一个 batch 的样本列表
               每个元素是 (sentence_ids, tag_ids, length)

    返回：
        sentences_padded: [batch_size, max_len]
        tags_padded:      [batch_size, max_len]
        lengths:          [batch_size]
    """
    sentence_ids_list, tag_ids_list, lengths = zip(*batch)

    batch_size = len(sentence_ids_list)
    max_len = max(lengths)

    # 词 padding 用 0，对应 <PAD>
    sentences_padded = torch.full(
        size=(batch_size, max_len),
        fill_value=0,
        dtype=torch.long
    )

    # 标签 padding 这里也用 0
    # 注意：
    # 这些 padding 位置后面会被 mask 掉，不会参与 CRF 计算
    # 所以填什么标签都无所谓，但通常填 0 最稳妥
    tags_padded = torch.full(
        size=(batch_size, max_len),
        fill_value=0,
        dtype=torch.long
    )

    # 把每条句子的有效部分拷贝进去
    for i, (sentence_ids, tag_ids, length) in enumerate(zip(sentence_ids_list, tag_ids_list, lengths)):
        sentences_padded[i, :length] = torch.tensor(sentence_ids, dtype=torch.long)
        tags_padded[i, :length] = torch.tensor(tag_ids, dtype=torch.long)

    # 句长也转成张量，后面 pack 和构造 mask 时会用到
    lengths = torch.tensor(lengths, dtype=torch.long)

    return sentences_padded, tags_padded, lengths
