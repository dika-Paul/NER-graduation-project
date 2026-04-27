import torch
from torch.utils.data import Dataset


def read_conll_4(path):
    """
    读取至少包含四列的 CoNLL 风格文件。

    默认取第一列作为 token，最后一列作为标签。
    """
    sentences = []
    tags = []

    words = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue

            if line.startswith("-DOCSTART-"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            words.append(parts[0])
            labels.append(parts[-1])

    if words:
        sentences.append(words)
        tags.append(labels)

    return sentences, tags


def read_conll_2(path):
    """
    读取两列格式的 CoNLL 风格文件：token 和 label。
    """
    sentences = []
    tags = []

    words = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            word, label = parts
            words.append(word)
            labels.append(label)

    if words:
        sentences.append(words)
        tags.append(labels)

    return sentences, tags


def build_vocab(sentences, min_freq=1):
    """
    为 BiLSTM 分支构建词表。
    """
    word_count = {}
    for sentence in sentences:
        for word in sentence:
            word_count[word] = word_count.get(word, 0) + 1

    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, count in word_count.items():
        if count >= min_freq:
            word2idx[word] = len(word2idx)

    return word2idx


def build_tag2idx(tags_list):
    """
    根据标签序列构建标签词表。
    """
    tag2idx = {}

    for tags in tags_list:
        for tag in tags:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)

    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return tag2idx, idx2tag


def encode_sentence(sentence, word2idx):
    """
    将 token 序列转换为 BiLSTM 分支使用的词 id 序列。
    """
    unk_id = word2idx["<UNK>"]
    return [word2idx.get(word, unk_id) for word in sentence]


class NERDataset(Dataset):
    """
    保存原始 token 序列和标签序列。

    这里不提前编码，统一放到 collate_fn 中处理，
    因为 BERT 分支需要做词级标签与 subword 分词结果的对齐。
    """

    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx]


def build_collate_fn(tokenizer, label2id, word2idx, max_length=128):
    """
    为并行 BERT + BiLSTM + CRF 模型构造 collate_fn。

    返回的主要张量包括：
        input_ids / attention_mask / token_type_ids：
            BERT 分支使用的 subword 级输入
        word_input_ids：
            BiLSTM 分支使用的词级 id
        first_subword_positions：
            每个保留词在 input_ids 中首个 subword 的位置
        word_attention_mask：
            用于 BiLSTM padding 和 CRF 解码的词级 mask
        labels：
            与融合后词级输出对齐的标签
    """

    def collate_fn(batch):
        batch_sentences, batch_tags = zip(*batch)

        encodings = tokenizer(
            list(batch_sentences),
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded_examples = []
        max_words = 0

        for i, (sentence, tags) in enumerate(zip(batch_sentences, batch_tags)):
            word_ids = encodings.word_ids(batch_index=i)

            first_subword_positions = []
            previous_word_id = None

            for token_position, word_id in enumerate(word_ids):
                if word_id is None:
                    continue

                if word_id != previous_word_id:
                    first_subword_positions.append(token_position)

                previous_word_id = word_id

            valid_word_count = len(first_subword_positions)

            if valid_word_count == 0:
                raise ValueError(
                    "当前 max_length 截断后，句子里没有保留下任何词。"
                    "请增大并行模型的 max_length。"
                )

            kept_sentence = list(sentence[:valid_word_count])
            kept_tags = list(tags[:valid_word_count])

            encoded_examples.append(
                {
                    "word_input_ids": encode_sentence(kept_sentence, word2idx),
                    "labels": [label2id[tag] for tag in kept_tags],
                    "first_subword_positions": first_subword_positions,
                    "word_count": valid_word_count,
                }
            )
            max_words = max(max_words, valid_word_count)

        batch_size = len(encoded_examples)

        word_input_ids = torch.zeros((batch_size, max_words), dtype=torch.long)
        labels = torch.zeros((batch_size, max_words), dtype=torch.long)
        first_subword_positions = torch.zeros((batch_size, max_words), dtype=torch.long)
        word_attention_mask = torch.zeros((batch_size, max_words), dtype=torch.long)

        for i, example in enumerate(encoded_examples):
            word_count = example["word_count"]

            word_input_ids[i, :word_count] = torch.tensor(
                example["word_input_ids"], dtype=torch.long
            )
            labels[i, :word_count] = torch.tensor(example["labels"], dtype=torch.long)
            first_subword_positions[i, :word_count] = torch.tensor(
                example["first_subword_positions"], dtype=torch.long
            )
            word_attention_mask[i, :word_count] = 1

        batch_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "word_input_ids": word_input_ids,
            "word_attention_mask": word_attention_mask,
            "first_subword_positions": first_subword_positions,
            "labels": labels,
        }

        if "token_type_ids" in encodings:
            batch_dict["token_type_ids"] = encodings["token_type_ids"]

        return batch_dict

    return collate_fn
