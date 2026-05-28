import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import TnewsDataset
from model import BertClassifier
from tqdm import tqdm

# 超参数配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
lr = 2e-5
epochs = 3
max_length = 128
num_classes = 15

# 1. 初始化tokenizer和数据集
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_dataset = TnewsDataset("data/train.json", tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 2. 初始化模型、损失函数、优化器
model = BertClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)

# 3. 训练循环
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "bert_classifier.pth")
print("训练完成，模型已保存！")
