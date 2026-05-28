import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import TnewsDataset
from model import BertClassifier
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
max_length = 128
num_classes = 15

# 1. 初始化tokenizer和测试集
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
test_dataset = TnewsDataset("data/test.json", tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 2. 加载训练好的模型
model = BertClassifier(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("bert_classifier.pth"))
model.eval()

# 3. 评估
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 输出评估结果
acc = accuracy_score(all_labels, all_preds)
print(f"测试集准确率: {acc:.4f}")
print(classification_report(all_labels, all_preds))
