import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from transformers import AutoConfig, AutoModel


class BertBiLstmCrfNER(nn.Module):
    """
    用于命名实体识别的并行 BERT + BiLSTM + CRF 模型。

    结构：
        分词后的句子
            -> 分支1：BERT 处理 subword，再抽取每个词首个 subword 的表示
            -> 分支2：词向量 + BiLSTM
            -> 在词级别拼接两条分支的特征
            -> 线性映射
            -> CRF 解码
    """

    def __init__(
        self,
        model_name,
        num_labels,
        word_vocab_size,
        word_embedding_dim=128,
        lstm_hidden_size=256,
        lstm_num_layers=1,
        dropout=0.25,
        word_pad_idx=0,
        id2label=None,
        label2id=None,
    ):
        super().__init__()

        # 将标签映射写入 transformer 配置，便于后续保存和加载。
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # subword 编码分支。
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        # 词级编码分支。
        self.word_embeddings = nn.Embedding(
            num_embeddings=word_vocab_size,
            embedding_dim=word_embedding_dim,
            padding_idx=word_pad_idx,
        )
        self.word_dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=word_embedding_dim,
            hidden_size=lstm_hidden_size // 2,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # 词级融合后的分类层。
        self.fusion_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size + lstm_hidden_size, num_labels)

        # CRF 用于最终的词级标签序列解码。
        self.crf = CRF(num_labels, batch_first=True)

    def _gather_first_subword_states(self, sequence_output, first_subword_positions):
        """
        将 subword 级别的 BERT 表示转换为词级表示。

        first_subword_positions 中保存了每个词在 BERT 输入里
        对应首个 subword 的位置。
        """
        hidden_size = sequence_output.size(-1)
        gather_index = first_subword_positions.unsqueeze(-1).expand(-1, -1, hidden_size)
        return torch.gather(sequence_output, dim=1, index=gather_index)

    def _encode_word_branch(self, word_input_ids, word_attention_mask):
        """
        使用 BiLSTM 对词 id 序列进行编码。
        """
        lengths = word_attention_mask.sum(dim=1).to(dtype=torch.long)
        embeddings = self.word_embeddings(word_input_ids)
        embeddings = self.word_dropout(embeddings)

        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.bilstm(packed)

        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=word_input_ids.size(1),
        )
        return lstm_output

    def _get_emissions(
        self,
        input_ids,
        attention_mask,
        word_input_ids,
        word_attention_mask,
        first_subword_positions,
        token_type_ids=None,
    ):
        """
        构造每个有效词位置对应的 CRF 发射分数。
        """
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = bert_outputs.last_hidden_state
        bert_word_features = self._gather_first_subword_states(
            sequence_output, first_subword_positions
        )

        lstm_word_features = self._encode_word_branch(word_input_ids, word_attention_mask)

        fused_features = torch.cat([bert_word_features, lstm_word_features], dim=-1)
        fused_features = self.fusion_dropout(fused_features)

        emissions = self.classifier(fused_features)
        return emissions

    def neg_log_likelihood(
        self,
        input_ids,
        attention_mask,
        word_input_ids,
        word_attention_mask,
        first_subword_positions,
        labels,
        token_type_ids=None,
    ):
        """
        计算一个 batch 的 CRF 负对数似然损失。
        """
        mask = word_attention_mask.bool()
        emissions = self._get_emissions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_input_ids=word_input_ids,
            word_attention_mask=word_attention_mask,
            first_subword_positions=first_subword_positions,
            token_type_ids=token_type_ids,
        )
        return -self.crf(emissions, labels, mask=mask, reduction="mean")

    def forward(
        self,
        input_ids,
        attention_mask,
        word_input_ids,
        word_attention_mask,
        first_subword_positions,
        labels=None,
        token_type_ids=None,
    ):
        """
        执行并行编码，并通过 CRF 解码最优标签路径。
        """
        mask = word_attention_mask.bool()
        emissions = self._get_emissions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_input_ids=word_input_ids,
            word_attention_mask=word_attention_mask,
            first_subword_positions=first_subword_positions,
            token_type_ids=token_type_ids,
        )

        output_dict = {
            "emissions": emissions,
            "predictions": self.crf.decode(emissions, mask=mask),
        }

        if labels is not None:
            output_dict["loss"] = -self.crf(emissions, labels, mask=mask, reduction="mean")

        return output_dict
