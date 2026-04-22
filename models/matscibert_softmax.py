from models.bert_softmax import BertSoftmaxNER

class MatSciBertSoftmaxNER(BertSoftmaxNER):
    """
    MatSciBERT + Softmax NER model.

    This keeps the same training/evaluation interface as the existing
    BertSoftmaxNER baseline, but defaults to the materials-domain
    pretrained encoder.
    """

    def __init__(self, num_labels, id2label=None, label2id=None, model_name="m3rg-iitd/matscibert"):
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
