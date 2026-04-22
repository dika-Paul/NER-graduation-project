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
        tags.append(label)

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


class NERDataset(Dataset):
    """
    保存原始句子和标签。
    注意：这里不做编码，编码放到 collate_fn 里统一完成。
    """

    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]


def build_collate_fn(tokenizer, label2id, max_length=128):
    """
    返回一个适用于 DataLoader 的 collate_fn

    作用：
    1. 批量调用 tokenizer
    2. 使用 word_ids() 做标签对齐
    3. 只保留每个词第一个 subword 的标签
    4. 其余位置标签设为 -100，训练时自动忽略
    """

    def collate_fn(batch):
        batch_sentences, batch_tags = zip(*batch)

        # tokenizer 输入的是“按词切好”的句子
        encodings = tokenizer(
            list(batch_sentences),
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        all_labels = []

        for i, tags in enumerate(batch_tags):
            word_ids = encodings.word_ids(batch_index=i)

            label_ids = []
            previous_word_id = None

            for word_id in word_ids:
                # 特殊 token: [CLS], [SEP], [PAD]
                if word_id is None:
                    label_ids.append(-100)

                # 一个词的第一个 subword，保留原标签
                elif word_id != previous_word_id:
                    label_ids.append(label2id[tags[word_id]])

                # 同一个词后续 subword，不参与 loss
                else:
                    label_ids.append(-100)

                previous_word_id = word_id

            all_labels.append(label_ids)

        batch_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(all_labels, dtype=torch.long)
        }

        # 有些模型（如 BERT）会返回 token_type_ids
        if "token_type_ids" in encodings:
            batch_dict["token_type_ids"] = encodings["token_type_ids"]

        return batch_dict

    return collate_fn