import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM_CRF(nn.Module):
    """
    结构：
        输入词索引
            -> Embedding
            -> BiLSTM
            -> Linear
            -> CRF
    """

    def __init__(self, vocab_size, tag_to_ix, embedding_dim=128, hidden_dim=256):
        super().__init__()

        # 基础属性
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 词嵌入层：把词 ID 映射成词向量
        self.word_embeds = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Dropout(p=0.25)
        )

        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.lstm_dropout = nn.Dropout(p=0.25)

        # 线性层：把 LSTM 输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF 层：直接调用 torchcrf
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_lstm_features(self, sentences, lengths):
        """
        计算一个 batch 的发射分数（emission scores）
        """
        # 1. 词嵌入
        embeds = self.word_embeds(sentences)

        # 2. pack，避免 padding 部分影响双向 LSTM 的有效计算
        packed = pack_padded_sequence(
            embeds,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # 3. 过双向 LSTM
        packed_out, _ = self.lstm(packed)

        # 4. pad 回来，保证输出长度和输入 batch 的最大长度一致
        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=sentences.size(1)
        )

        lstm_out = self.lstm_dropout(lstm_out)

        # 5. 全连接映射
        feats = self.hidden2tag(lstm_out)

        return feats

    def _make_mask(self, lengths, max_len, device):
        """
        根据真实长度构造 mask

        参数：
            lengths: [batch_size]
            max_len: 当前 batch 的最大句长
            device:  所在设备

        返回：
            mask: [batch_size, max_len]
                  有效 token 位置为 True，padding 位置为 False
        """
        # 例如 lengths=[5,3]，max_len=5 时，mask 为：
        # [[1,1,1,1,1],
        #  [1,1,1,0,0]]
        range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
        mask = range_tensor < lengths.unsqueeze(1)
        return mask

    def neg_log_likelihood_single(self, feats, tags):
        """
        单句负对数似然损失

        参数：
            feats: [seq_len, tagset_size]
            tags:  [seq_len]

        返回：
            loss
        """
        # torchcrf 期望的输入是 batch 形式
        # 所以这里给单句补一个 batch 维度
        feats = feats.unsqueeze(0)  # [1, seq_len, tagset_size]
        tags = tags.unsqueeze(0)    # [1, seq_len]

        # 单句没有 padding，因此 mask 全为 True
        mask = torch.ones_like(tags, dtype=torch.bool)

        # CRF.forward 返回的是对数似然 log-likelihood
        # 训练时通常取负号作为 loss
        loss = -self.crf(feats, tags, mask=mask, reduction="mean")
        return loss

    def neg_log_likelihood(self, sentences, tags, lengths):
        """
        batch 版负对数似然损失

        参数：
            sentences: [batch_size, seq_len]
            tags:      [batch_size, seq_len]
            lengths:   [batch_size]

        返回：
            loss: 标量张量
        """
        # 1. 先拿到整个 batch 的发射分数
        feats = self._get_lstm_features(sentences, lengths)

        # 2. 根据真实长度生成 mask
        mask = self._make_mask(lengths, sentences.size(1), sentences.device)

        # 3. 调用 torchcrf 计算对数似然，并取负号作为损失
        loss = -self.crf(feats, tags, mask=mask, reduction="mean")
        return loss

    def predict_single(self, feats):
        """
        单句预测

        参数：
            feats: [seq_len, tagset_size]

        返回：
            score:   这里返回 None，仅为了兼容你之前的调用习惯
            tag_seq: 预测标签 id 序列
        """
        # 补 batch 维度
        feats = feats.unsqueeze(0)  # [1, seq_len, tagset_size]

        # 单句全是有效位置
        mask = torch.ones(feats.size(0), feats.size(1), dtype=torch.bool, device=feats.device)

        # decode 返回 List[List[int]]
        best_paths = self.crf.decode(feats, mask=mask)
        tag_seq = best_paths[0]

        # 为了兼容你之前的写法：_, pred_ids = model.predict_single(feats_i)
        return None, tag_seq

    def forward(self, sentences, lengths):
        """
        batch 版 forward，用于预测。

        参数：
            sentences: [batch_size, seq_len]
            lengths:   [batch_size]

        返回：
            batch_paths: 一个列表，长度为 batch_size
                         每个元素是该句子的预测标签 id 序列
        """
        # 1. 先得到整个 batch 的发射分数
        feats = self._get_lstm_features(sentences, lengths)

        # 2. 构造 mask，告诉 CRF 哪些位置是有效 token
        mask = self._make_mask(lengths, sentences.size(1), sentences.device)

        # 3. 调用 CRF 的 decode，得到 batch 中每个句子的最优标签路径
        batch_paths = self.crf.decode(feats, mask=mask)

        return batch_paths
