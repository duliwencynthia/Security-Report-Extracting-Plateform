from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
import pickle
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from imblearn.over_sampling import SMOTE
from collections import Counter
import time
import ast
import logging
import os
from datetime import datetime

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"roberta_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will print to console as well
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting TTPhunter model training")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load standard RoBERTa tokenizer and model instead of custom model
model_name = "nanda-rani/TTPXHunter"
logger.info(f"Loading {model_name} tokenizer and model")
tokenizer = RobertaTokenizer.from_pretrained(model_name)


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

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def get_embeddings(self, input_ids=None, attention_mask=None):
        """Extract embeddings from the model for SMOTE processing"""
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
        return pooled_output


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


# Modified training function that applies SMOTE before each epoch with time measurement
def train_model_with_smote(model, train_texts, train_labels, batch_size, optimizer, epochs=50):
    model.train()

    # Create initial dataset
    train_dataset = TextDataset(train_texts, train_labels)

    total_embedding_time = 0
    total_smote_time = 0
    total_training_time = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # Get embeddings for all training examples
        embeddings = []
        labels = []

        # Time embedding extraction
        embedding_start = time.time()

        # Process in batches to avoid memory issues
        embed_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        for batch in tqdm(embed_loader, desc="Extracting embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}
            embedding = model.get_embeddings(input_ids=batch["input_ids"],
                                             attention_mask=batch["attention_mask"])
            embeddings.append(embedding.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

        # Concatenate all embeddings
        X_embed = np.vstack(embeddings)
        y = np.array(labels)

        embedding_end = time.time()
        embedding_time = embedding_end - embedding_start
        total_embedding_time += embedding_time
        logger.info(f"Embedding extraction time: {embedding_time:.2f} seconds")

        # Apply SMOTE - measure time
        smote_start = time.time()

        logger.info(f"Class distribution before SMOTE: {dict(Counter(y))}")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_embed, y)
        logger.info(f"Class distribution after SMOTE: {dict(Counter(y_resampled))}")

        smote_end = time.time()
        smote_time = smote_end - smote_start
        total_smote_time += smote_time
        logger.info(f"SMOTE processing time: {smote_time:.2f} seconds")

        # Create PyTorch tensors from resampled data
        X_tensor = torch.tensor(X_resampled, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_resampled, dtype=torch.long).to(device)

        # Create simple dataset and dataloader for the resampled data
        class EmbeddingDataset(Dataset):
            def __init__(self, embeddings, labels):
                self.embeddings = embeddings
                self.labels = labels

            def __getitem__(self, idx):
                return {"embeddings": self.embeddings[idx], "labels": self.labels[idx]}

            def __len__(self):
                return len(self.labels)

        embed_dataset = EmbeddingDataset(X_tensor, y_tensor)
        embed_loader = DataLoader(embed_dataset, batch_size=batch_size, shuffle=True)

        # Train with SMOTE balanced data - measure time
        training_start = time.time()

        for batch in tqdm(embed_loader, desc="Training"):
            embeddings = batch["embeddings"]
            labels = batch["labels"]

            # Forward pass using embeddings
            logits = model.classifier(model.dropout(embeddings))
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        training_end = time.time()
        training_time = training_end - training_start
        total_training_time += training_time
        logger.info(f"Training time: {training_time:.2f} seconds")

    # Report total times
    logger.info("\n===== TIMING SUMMARY (SMOTE) =====")
    logger.info(f"Total embedding extraction time: {total_embedding_time:.2f} seconds")
    logger.info(f"Total SMOTE processing time: {total_smote_time:.2f} seconds")
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    logger.info(f"Total time: {total_embedding_time + total_smote_time + total_training_time:.2f} seconds")

    return {
        "embedding_time": total_embedding_time,
        "smote_time": total_smote_time,
        "training_time": total_training_time,
        "total_time": total_embedding_time + total_smote_time + total_training_time
    }


# Evaluation function with time measurement
def evaluate_model(model, val_loader):
    model.eval()
    preds, labels = [], []

    eval_start = time.time()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds += torch.argmax(logits, dim=1).cpu().tolist()
            labels += batch["labels"].cpu().tolist()

    eval_end = time.time()
    eval_time = eval_end - eval_start
    logger.info(f"Evaluation time: {eval_time:.2f} seconds")

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    # Calculate confusion matrix and extract false positives
    cm = confusion_matrix(labels, preds)
    fp = cm.sum(axis=0) - np.diag(cm)  # false positives per class
    total_fp = fp.sum()  # total false positives

    return acc, precision, recall, f1, total_fp, preds, eval_time


# K-Fold Cross Validation with SMOTE and time measurement
def cross_validate_with_smote(texts, labels, k=5, epochs=50, batch_size=32):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    fold_times = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        logger.info(f"\n🧪 Fold {fold + 1}/{k}")
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        val_dataset = TextDataset(val_texts, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Time model initialization
        init_start = time.time()

        # Initialize model with standard RoBERTa
        base_model = RobertaModel.from_pretrained(model_name)
        model = CustomRobertaClassifier(base_model, hidden_size=768, num_labels=max(labels) + 1)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=5e-5)

        init_end = time.time()
        init_time = init_end - init_start
        logger.info(f"Model initialization time: {init_time:.2f} seconds")

        # Train with SMOTE and measure time
        training_times = train_model_with_smote(model, train_texts, train_labels, batch_size, optimizer, epochs)

        # Evaluate and measure time
        acc, precision, recall, f1, total_fp, preds, eval_time = evaluate_model(model, val_loader)
        logger.info(
            f"✅ Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FP: {total_fp}")

        fold_results.append((acc, precision, recall, f1))
        fold_times.append({
            "init_time": init_time,
            "embedding_time": training_times["embedding_time"],
            "smote_time": training_times["smote_time"],
            "training_time": training_times["training_time"],
            "eval_time": eval_time,
            "total_time": init_time + training_times["total_time"] + eval_time
        })

    # Calculate average times across folds
    avg_times = {key: sum(fold[key] for fold in fold_times) / len(fold_times) for key in fold_times[0].keys()}

    return fold_results, avg_times


if __name__ == "__main__":
    try:
        # Log system info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Using model: {model_name}")

        # Load data
        logger.info("Loading data...")
        with open('label_dict.pkl', 'rb') as file:
            labels_dic = pickle.load(file)

        df_train = pd.DataFrame(pd.read_csv('./unique_train_df.csv'))
        df_test = pd.DataFrame(pd.read_csv('./unique_train_df.csv'))
        df_train["cats"] = df_train["cats"].apply(ast.literal_eval)
        df_test["cats"] = df_test["cats"].apply(ast.literal_eval)
        df_train["text"] = df_train["text"].apply(
            lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))
        df_test["text"] = df_test["text"].apply(
            lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))

        sentences = df_train["text"].tolist() + df_test["text"].tolist()
        labels = []
        for _, va in df_train["cats"].items():
            for k, v in va.items():
                if v == 1.0:
                    labels.append(k)
        for _, va in df_test["cats"].items():
            for k, v in va.items():
                if v == 1.0:
                    labels.append(k)

        labels_id = []
        for lb in labels:
            labels_id.append(labels_dic[lb])

        # Print initial class distribution
        logger.info(f"Initial class distribution: {dict(Counter(labels_id))}")

        # Run 5-fold CV with SMOTE and time measurements
        logger.info("Starting 5-fold cross-validation with SMOTE")
        start_time = time.time()
        results, timing = cross_validate_with_smote(sentences, labels_id, k=5)
        total_time = time.time() - start_time

        # Average metrics
        results = np.array(results)
        avg_metrics = results.mean(axis=0)
        logger.info("\n📈 Average Results with SMOTE:")
        logger.info(f"Accuracy: {avg_metrics[0]:.4f}")
        logger.info(f"Precision: {avg_metrics[1]:.4f}")
        logger.info(f"Recall: {avg_metrics[2]:.4f}")
        logger.info(f"F1 Score: {avg_metrics[3]:.4f}")

        # Print timing summary
        logger.info("\n⏱️ TIMING SUMMARY FOR STANDARD ROBERTA IMPLEMENTATION:")
        logger.info(f"Average initialization time: {timing['init_time']:.2f} seconds")
        logger.info(f"Average embedding extraction time: {timing['embedding_time']:.2f} seconds")
        logger.info(f"Average SMOTE processing time: {timing['smote_time']:.2f} seconds")
        logger.info(f"Average training time: {timing['training_time']:.2f} seconds")
        logger.info(f"Average evaluation time: {timing['eval_time']:.2f} seconds")
        logger.info(f"Average total time per fold: {timing['total_time']:.2f} seconds")
        logger.info(f"Total execution time: {total_time:.2f} seconds")

        logger.info(f"Log file saved to: {log_file}")

    except Exception as e:
        logger.exception(f"An error occurred during execution: {str(e)}")