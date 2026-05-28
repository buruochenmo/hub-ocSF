from transformers import BertModel
import torch
import torch.nn as nn

class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-chinese", num_classes=15):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # 分类头：把BERT的  输出映射到类别数
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 获取BERT的  输出（句子级表示）
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # shape: [batch_size, hidden_size]
        logits = self.classifier(cls_output)
        return logits
