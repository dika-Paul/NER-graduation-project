import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertSoftmaxNER(nn.Module):
    """
    BERT + Softmax 的 NER baseline

    结构：
        input_ids
            -> BERT
            -> Dropout
            -> Linear
            -> CrossEntropyLoss
    """

    def __init__(self, model_name, num_labels, id2label=None, label2id=None):
        super().__init__()

        # 配置文件中写入标签映射，便于后续保存/加载
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )

        # 加载预训练 BERT 编码器
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        # dropout 概率
        dropout_prob = (
            self.config.classifier_dropout
            if getattr(self.config, "classifier_dropout", None) is not None
            else self.config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # 忽略 -100 的位置（特殊 token、padding、非首 subword）
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        """
        参数：
            input_ids:      [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels:         [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]，对 BERT 有用，RoBERTa 可为空

        返回：
            dict:
                {
                    "logits": logits,
                    "loss": loss(如果 labels 不为空)
                }
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state

        # dropout + 分类
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]

        output_dict = {"logits": logits}

        if labels is not None:
            # CrossEntropyLoss 要求形状：
            # logits: [N, C]
            # labels: [N]
            loss = self.loss_fn(
                logits.view(-1, self.config.num_labels),
                labels.view(-1)
            )
            output_dict["loss"] = loss

        return output_dict