# -*- coding: utf-8 -*-
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pickle
import torch
import os
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel
import torch.nn as nn
import json
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("nanda-rani/TTPXHunter")

import torch.nn as nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomRobertaClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.roberta = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss=None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
    )

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Training function
def train_model(model, train_loader, optimizer):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # # Check the classifier head
        # print("Classifier weight shape:", model.classifier.out_proj.weight.shape)

        # # Check the batch labels
        # print("Labels:", batch["labels"])
        # print("Max label in batch:", batch["labels"].max().item())

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds += torch.argmax(logits, dim=1).cpu().tolist()
            labels += batch["labels"].cpu().tolist()

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    # Calculate confusion matrix and extract false positives
    cm = confusion_matrix(labels, preds)
    fp = cm.sum(axis=0) - np.diag(cm)  # false positives per class
    total_fp = fp.sum()  # total false positives

    return acc, precision, recall, f1, total_fp, preds

# K-Fold Cross Validation
def cross_validate(texts, labels, k=5, epochs=20, batch_size=8):
    # print(labels)
    # print(len(labels))
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        print(f"\nFold {fold+1}/{k}")
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print("Train labels:", set(train_labels))
        print("New data labels:", set(val_labels))

        train_dataset = TextDataset(train_texts, train_labels)
        val_dataset = TextDataset(val_texts, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        config = RobertaConfig.from_pretrained("nanda-rani/TTPXHunter")
        config.num_labels = max(labels)+1
        #print(max(labels))
        # model = RobertaForSequenceClassification.from_pretrained("nanda-rani/TTPXHunter", ignore_mismatched_sizes=True)
        # model.classifier.out_proj = nn.Linear(model.config.hidden_size, len(set(labels)))
        # model.config.num_labels = num_labels  # update config as well
        base_model = RobertaModel.from_pretrained("nanda-rani/TTPXHunter")
        model = CustomRobertaClassifier(base_model, hidden_size=768, num_labels=max(labels)+1)
        model.to(device)
        #print(model.classifier.out_proj)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_model(model, train_loader, optimizer)

        acc, precision, recall, f1, total_fp, preds = evaluate_model(model, val_loader)
        print(f"✅ Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FP: {total_fp}, pred: {preds}")
        fold_results.append((acc, precision, recall, f1))

        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.savefig(f"my_plot_{fold}.png")

    return fold_results

with open('label_dict.pkl', 'rb') as file:
    labels_dic = pickle.load(file)

df_train = pd.DataFrame(pd.read_csv('../unique_train_df.csv'))
df_test = pd.DataFrame(pd.read_csv('../unique_train_df.csv'))

sentences = df_train["text"].tolist() + df_test["text"].tolist()
labels = []
for _,va in df_train["cats"].items():
    for k,v in va.items():
        if v == 1.0:
            labels.append(k)
for _,va in df_test["cats"].items():
    for k,v in va.items():
        if v == 1.0:
            labels.append(k)

# print(labels)
labels_id = []
for lb in labels:
    # print(labels_dic[lb])
    labels_id.append(labels_dic[lb])

num_labels = len(set(labels_id))
config = RobertaConfig.from_pretrained("nanda-rani/TTPXHunter")
config.num_labels = num_labels  # Explicitly override

model = RobertaForSequenceClassification.from_pretrained(
    "nanda-rani/TTPXHunter",
    config=config,
    ignore_mismatched_sizes=True
)

# Sanity check:
print(model.classifier.out_proj)

# Run 5-fold CV
results = cross_validate(sentences, labels_id, k=5)

# 📊 Average metrics
results = np.array(results)
avg_metrics = results.mean(axis=0)
print("\n📈 Average Results:")
print(f"Accuracy: {avg_metrics[0]:.4f}")
print(f"Precision: {avg_metrics[1]:.4f}")
print(f"Recall: {avg_metrics[2]:.4f}")
print(f"F1 Score: {avg_metrics[3]:.4f}")




