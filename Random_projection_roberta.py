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
from sklearn.random_projection import GaussianRandomProjection
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("nanda-rani/TTPXHunter")


class RandomProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Initialize with Gaussian random weights
        self.projection_matrix = nn.Parameter(
            torch.randn(input_dim, output_dim) / np.sqrt(output_dim),
            requires_grad=False  # Typically random projection is not trainable
        )

    def forward(self, x):
        return torch.matmul(x, self.projection_matrix)


class CustomRobertaWithProjection(nn.Module):
    def __init__(self, base_model, input_dim=768, projection_dim=256, num_labels=2):
        super().__init__()
        self.roberta = base_model
        self.projection = RandomProjectionLayer(input_dim, projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(projection_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Apply random projection
        projected_output = self.projection(pooled_output)

        # Continue with standard classification pipeline
        projected_output = self.dropout(projected_output)
        logits = self.classifier(projected_output)

        loss = None
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


# Training function with time measurement
def train_model(model, train_loader, optimizer):
    model.train()
    batch_times = []

    for batch in tqdm(train_loader, desc="Training"):
        batch_start = time.time()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_end = time.time()
        batch_times.append(batch_end - batch_start)

    avg_batch_time = sum(batch_times) / len(batch_times)
    total_train_time = sum(batch_times)

    print(f"Average batch processing time: {avg_batch_time:.4f} seconds")
    print(f"Total training time: {total_train_time:.2f} seconds")

    return total_train_time


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
    print(f"Evaluation time: {eval_time:.2f} seconds")

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    # Calculate confusion matrix and extract false positives
    cm = confusion_matrix(labels, preds)
    fp = cm.sum(axis=0) - np.diag(cm)  # false positives per class
    total_fp = fp.sum()  # total false positives

    return acc, precision, recall, f1, total_fp, preds, eval_time


# Helper function to find optimal projection dim with time measurement
def find_optimal_projection_dim(texts, labels, projection_dims=[64, 128, 256, 384], test_fold=0):
    """Find optimal projection dimension by testing on a single fold"""
    dim_times = {}

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kfold.split(texts))

    # Use one fold for testing different projection dimensions
    train_idx, val_idx = folds[test_fold]
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Time for feature extraction
    feature_start = time.time()

    # Extract features from RoBERTa
    base_model = RobertaModel.from_pretrained("nanda-rani/TTPXHunter")
    base_model.to(device)
    base_model.eval()

    # Get features
    train_features = []
    train_label_list = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting train features"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pooled_output = outputs.pooler_output
            train_features.append(pooled_output.cpu().numpy())
            train_label_list.extend(batch["labels"].cpu().numpy())

    train_features = np.vstack(train_features)
    train_label_array = np.array(train_label_list)

    feature_end = time.time()
    feature_time = feature_end - feature_start
    print(f"Feature extraction time: {feature_time:.2f} seconds")

    # Test different projection dimensions
    results = {}
    for dim in projection_dims:
        print(f"Testing projection dimension: {dim}")
        dim_start = time.time()

        # Create and fit projection
        random_projection = GaussianRandomProjection(n_components=dim, random_state=42)
        train_features_projected = random_projection.fit_transform(train_features)

        projection_time = time.time() - dim_start
        print(f"Projection time for dim {dim}: {projection_time:.4f} seconds")

        # Train a simple classifier on projected features
        clf_start = time.time()
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_features_projected, train_label_array)

        clf_time = time.time() - clf_start
        print(f"Classifier training time for dim {dim}: {clf_time:.4f} seconds")

        # Test on validation set
        val_features = []
        val_label_list = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Extracting val features"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                pooled_output = outputs.pooler_output
                val_features.append(pooled_output.cpu().numpy())
                val_label_list.extend(batch["labels"].cpu().numpy())

        val_features = np.vstack(val_features)
        val_label_array = np.array(val_label_list)

        # Project and predict
        eval_start = time.time()
        val_features_projected = random_projection.transform(val_features)
        preds = clf.predict(val_features_projected)

        eval_time = time.time() - eval_start
        print(f"Evaluation time for dim {dim}: {eval_time:.4f} seconds")

        # Calculate metrics
        acc = accuracy_score(val_label_array, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_label_array, preds, average="weighted")

        total_dim_time = projection_time + clf_time + eval_time
        print(f"Dim {dim}: Accuracy={acc:.4f}, F1={f1:.4f}, Total time={total_dim_time:.4f}s")

        results[dim] = (acc, precision, recall, f1)
        dim_times[dim] = total_dim_time

    # Find optimal dimension based on F1 score
    optimal_dim = max(results.keys(), key=lambda x: results[x][3])
    print(f"Optimal projection dimension: {optimal_dim}")
    print(f"Time for dimension selection: {sum(dim_times.values()):.2f} seconds")

    return optimal_dim, dim_times


# K-Fold Cross Validation with Random Projection
def cross_validate_with_projection(texts, labels, projection_dim=256, k=5, epochs=3, batch_size=8):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    fold_times = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        print(f"\n🧪 Fold {fold + 1}/{k}")
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = TextDataset(train_texts, train_labels)
        val_dataset = TextDataset(val_texts, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Time model initialization
        init_start = time.time()

        # Initialize model with random projection
        base_model = RobertaModel.from_pretrained("nanda-rani/TTPXHunter")
        model = CustomRobertaWithProjection(
            base_model,
            input_dim=768,
            projection_dim=projection_dim,
            num_labels=max(labels) + 1
        )
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=5e-5)

        init_end = time.time()
        init_time = init_end - init_start
        print(f"Model initialization time: {init_time:.2f} seconds")

        # Train the model with time tracking
        fold_train_times = []
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_time = train_model(model, train_loader, optimizer)
            fold_train_times.append(epoch_time)

        total_train_time = sum(fold_train_times)

        # Evaluate with time tracking
        acc, precision, recall, f1, total_fp, preds, eval_time = evaluate_model(model, val_loader)
        print(f"✅ Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FP: {total_fp}")

        fold_results.append((acc, precision, recall, f1))
        fold_times.append({
            "init_time": init_time,
            "train_time": total_train_time,
            "eval_time": eval_time,
            "total_time": init_time + total_train_time + eval_time
        })

    # Calculate average times across folds
    avg_times = {key: sum(fold[key] for fold in fold_times) / len(fold_times) for key in fold_times[0].keys()}

    return fold_results, avg_times


if __name__ == "__main__":
    # Load data
    with open('label_dict.pkl', 'rb') as file:
        labels_dic = pickle.load(file)

    df_train = pd.DataFrame(pd.read_csv('./unique_train_df.csv'))
    df_test = pd.DataFrame(pd.read_csv('./unique_train_df.csv'))
    df_train["cats"] = df_train["cats"].apply(ast.literal_eval)
    df_test["cats"] = df_test["cats"].apply(ast.literal_eval)
    df_train["text"] = df_train["text"].apply(lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))
    df_test["text"] = df_test["text"].apply(lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))

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

    # Find optimal projection dimension (optional, can be time-consuming)
    # Comment out for faster execution if you want to use a fixed dimension
    dim_selection_start = time.time()
    optimal_dim, dim_times = find_optimal_projection_dim(sentences[:100],
                                                         labels_id[:100])  # Using a subset for faster execution
    dim_selection_time = time.time() - dim_selection_start
    print(f"Dimension selection time: {dim_selection_time:.2f} seconds")
    # optimal_dim = 256  # Default reasonable value if not running dimension selection

    # Run 5-fold CV with Random Projection
    start_time = time.time()
    results, timing = cross_validate_with_projection(sentences, labels_id, projection_dim=optimal_dim, k=5)
    total_time = time.time() - start_time

    # Average metrics
    results = np.array(results)
    avg_metrics = results.mean(axis=0)
    print("\n📈 Average Results with Random Projection:")
    print(f"Projection Dimension: {optimal_dim}")
    print(f"Accuracy: {avg_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f}")
    print(f"Recall: {avg_metrics[2]:.4f}")
    print(f"F1 Score: {avg_metrics[3]:.4f}")

    # Print timing summary
    print("\n⏱️ TIMING SUMMARY FOR RANDOM PROJECTION IMPLEMENTATION:")
    if 'dim_selection_time' in locals():
        print(f"Dimension selection time: {dim_selection_time:.2f} seconds")
    print(f"Average initialization time: {timing['init_time']:.2f} seconds")
    print(f"Average training time: {timing['train_time']:.2f} seconds")
    print(f"Average evaluation time: {timing['eval_time']:.2f} seconds")
    print(f"Average total time per fold: {timing['total_time']:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")
