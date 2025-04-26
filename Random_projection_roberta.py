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
import time
import ast
import os
import json
from datetime import datetime
from sklearn.random_projection import GaussianRandomProjection

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
log_dir = "logs"
results_dir = "results"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
results_file = os.path.join(results_dir, f"results_{timestamp}.json")
loss_file = os.path.join(results_dir, f"loss_values_{timestamp}.json")


def log_message(message):
    """Write message to log file and print to console"""
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")


# Load tokenizer
log_message(f"Using device: {device}")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
log_message("Loaded RoBERTa tokenizer")


class ProjectedFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)


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


# Simple classifier model for projected features
class ProjectedFeatureClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_labels)
        )

    def forward(self, features, labels=None):
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return logits, loss


# Helper function to find optimal projection dim with time measurement
def find_optimal_projection_dim(texts, labels, projection_dims=[64, 128, 256, 384], test_fold=0):
    """Find optimal projection dimension by testing on a single fold"""
    log_message("\n🔍 Finding optimal projection dimension")
    dim_times = {}
    dim_losses = {}

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kfold.split(texts))

    # Use one fold for testing different projection dimensions
    train_idx, val_idx = folds[test_fold]
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Create datasets for encoding
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)

    # DataLoaders for encoding
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Time for feature extraction
    feature_start = time.time()
    log_message("Extracting features from RoBERTa model")

    # Extract features from RoBERTa
    base_model = RobertaModel.from_pretrained("roberta-base")
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

    feature_end = time.time()
    feature_time = feature_end - feature_start
    log_message(f"Feature extraction time: {feature_time:.2f} seconds")

    # Test different projection dimensions
    results = {}
    for dim in projection_dims:
        log_message(f"Testing projection dimension: {dim}")
        dim_start = time.time()

        # Create and fit projection
        random_projection = GaussianRandomProjection(n_components=dim, random_state=42)
        train_features_projected = random_projection.fit_transform(train_features)

        projection_time = time.time() - dim_start
        log_message(f"Projection time for dim {dim}: {projection_time:.4f} seconds")

        # Train a simple classifier on projected features
        clf_start = time.time()
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_features_projected, train_label_array)

        clf_time = time.time() - clf_start
        log_message(f"Classifier training time for dim {dim}: {clf_time:.4f} seconds")

        # Project validation features and predict
        eval_start = time.time()
        val_features_projected = random_projection.transform(val_features)
        preds = clf.predict(val_features_projected)

        # Calculate loss using log loss
        from sklearn.metrics import log_loss
        try:
            loss_value = log_loss(val_label_array, clf.predict_proba(val_features_projected))
            log_message(f"Validation loss for dim {dim}: {loss_value:.4f}")
            dim_losses[dim] = loss_value
        except:
            loss_value = None
            log_message(f"Could not calculate loss for dim {dim}")
            dim_losses[dim] = None

        eval_time = time.time() - eval_start
        log_message(f"Evaluation time for dim {dim}: {eval_time:.4f} seconds")

        # Calculate metrics
        acc = accuracy_score(val_label_array, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_label_array, preds, average="weighted")

        total_dim_time = projection_time + clf_time + eval_time
        log_message(
            f"Dim {dim}: Accuracy={acc:.4f}, F1={f1:.4f}, Loss={loss_value if loss_value else 'N/A'}, Total time={total_dim_time:.4f}s")

        results[dim] = (acc, precision, recall, f1)
        dim_times[dim] = total_dim_time

    # Find optimal dimension based on F1 score
    optimal_dim = max(results.keys(), key=lambda x: results[x][3])
    log_message(f"Optimal projection dimension: {optimal_dim}")
    log_message(f"Time for dimension selection: {sum(dim_times.values()):.2f} seconds")

    return optimal_dim, dim_times, dim_losses


# Function to extract and project features for all data
def extract_and_project_features(texts, labels, projection_dim):
    log_message(f"\n🔄 Extracting and projecting features with dimension {projection_dim}")

    # Create dataset for encoding
    text_dataset = TextDataset(texts, labels)
    loader = DataLoader(text_dataset, batch_size=8, shuffle=False)

    # Extract features from RoBERTa
    base_model = RobertaModel.from_pretrained("roberta-base")
    base_model.to(device)
    base_model.eval()

    # Get features
    features = []
    label_list = []

    extract_start = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pooled_output = outputs.pooler_output
            features.append(pooled_output.cpu().numpy())
            label_list.extend(batch["labels"].cpu().numpy())

    features = np.vstack(features)
    extract_time = time.time() - extract_start
    log_message(f"Feature extraction time: {extract_time:.2f} seconds")

    # Project features
    project_start = time.time()
    random_projection = GaussianRandomProjection(n_components=projection_dim, random_state=42)
    projected_features = random_projection.fit_transform(features)
    project_time = time.time() - project_start
    log_message(f"Projection time: {project_time:.2f} seconds")

    return projected_features, label_list, extract_time + project_time


# Training function for projected features with detailed loss logging
def train_projected_model(model, train_loader, optimizer, epoch):
    model.train()
    batch_times = []
    total_loss = 0
    batch_losses = []

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        batch_start = time.time()

        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        logits, loss = model(features, labels)
        loss_value = loss.item()
        total_loss += loss_value
        batch_losses.append(loss_value)

        # Log batch loss every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == len(train_loader) - 1:
            log_message(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss_value:.6f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_end = time.time()
        batch_times.append(batch_end - batch_start)

    avg_batch_time = sum(batch_times) / len(batch_times)
    total_train_time = sum(batch_times)
    avg_loss = total_loss / len(train_loader)

    log_message(f"Average batch processing time: {avg_batch_time:.4f} seconds")
    log_message(f"Total training time: {total_train_time:.2f} seconds")
    log_message(f"Average training loss: {avg_loss:.6f}")

    return total_train_time, avg_loss, batch_losses


# Evaluation function for projected features with detailed loss logging
def evaluate_projected_model(model, val_loader, epoch=None):
    model.eval()
    preds, labels_list = [], []
    total_loss = 0
    batch_losses = []

    eval_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)

            logits, loss = model(features, labels)
            loss_value = loss.item()
            total_loss += loss_value
            batch_losses.append(loss_value)

            # Log batch loss every 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == len(val_loader) - 1:
                if epoch is not None:
                    log_message(
                        f"Evaluation - Epoch {epoch}, Batch {batch_idx + 1}/{len(val_loader)}, Loss: {loss_value:.6f}")
                else:
                    log_message(f"Evaluation - Batch {batch_idx + 1}/{len(val_loader)}, Loss: {loss_value:.6f}")

            preds += torch.argmax(logits, dim=1).cpu().tolist()
            labels_list += labels.cpu().tolist()

    eval_end = time.time()
    eval_time = eval_end - eval_start
    avg_loss = total_loss / len(val_loader)

    log_message(f"Evaluation time: {eval_time:.2f} seconds")
    log_message(f"Validation loss: {avg_loss:.6f}")

    acc = accuracy_score(labels_list, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, preds, average="weighted")

    # Calculate confusion matrix and extract false positives
    cm = confusion_matrix(labels_list, preds)
    fp = cm.sum(axis=0) - np.diag(cm)  # false positives per class
    total_fp = fp.sum()  # total false positives

    return acc, precision, recall, f1, total_fp, preds, eval_time, avg_loss, batch_losses


# K-Fold Cross Validation with Random Projection Preprocessing
def cross_validate_with_projection_preprocessing(texts, labels, projection_dim=256, k=5, epochs=10, batch_size=8):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    fold_times = []
    fold_losses = []

    all_fold_losses = {
        "folds": []
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        log_message(f"\n🧪 Fold {fold + 1}/{k}")
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Extract and project features
        log_message("Extracting and projecting training features")
        train_features, train_labels_array, train_prep_time = extract_and_project_features(
            train_texts, train_labels, projection_dim
        )

        log_message("Extracting and projecting validation features")
        val_features, val_labels_array, val_prep_time = extract_and_project_features(
            val_texts, val_labels, projection_dim
        )

        # Create datasets for projected features
        train_dataset = ProjectedFeatureDataset(train_features, train_labels_array)
        val_dataset = ProjectedFeatureDataset(val_features, val_labels_array)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Time model initialization
        init_start = time.time()

        # Initialize classifier for projected features
        num_labels = max(labels) + 1
        model = ProjectedFeatureClassifier(projection_dim, num_labels)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=5e-4)

        init_end = time.time()
        init_time = init_end - init_start
        log_message(f"Model initialization time: {init_time:.2f} seconds")

        # Train the model with time tracking
        fold_train_times = []
        fold_train_losses = []
        fold_val_losses = []
        fold_batch_losses = {
            "train": [],
            "val": []
        }

        for epoch in range(epochs):
            log_message(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_time, train_loss, train_batch_losses = train_projected_model(model, train_loader, optimizer,
                                                                               epoch + 1)
            fold_train_times.append(epoch_time)
            fold_train_losses.append(train_loss)
            fold_batch_losses["train"].extend([(epoch, i, loss) for i, loss in enumerate(train_batch_losses)])

            # Evaluate after each epoch
            acc, precision, recall, f1, total_fp, _, _, val_loss, val_batch_losses = evaluate_projected_model(model,
                                                                                                              val_loader,
                                                                                                              epoch + 1)
            fold_val_losses.append(val_loss)
            fold_batch_losses["val"].extend([(epoch, i, loss) for i, loss in enumerate(val_batch_losses)])

            log_message(f"Epoch {epoch + 1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            log_message(f"Epoch {epoch + 1} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        total_train_time = sum(fold_train_times)

        # Final evaluation with time tracking
        acc, precision, recall, f1, total_fp, preds, eval_time, final_val_loss, _ = evaluate_projected_model(model,
                                                                                                             val_loader)
        log_message(
            f"\n✅ Final Metrics - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FP: {total_fp}")
        log_message(f"Final Train Loss: {fold_train_losses[-1]:.6f}, Final Val Loss: {final_val_loss:.6f}")

        fold_results.append((acc, precision, recall, f1))
        fold_times.append({
            "prep_time": train_prep_time + val_prep_time,
            "init_time": init_time,
            "train_time": total_train_time,
            "eval_time": eval_time,
            "total_time": train_prep_time + val_prep_time + init_time + total_train_time + eval_time
        })
        fold_losses.append({
            "train_losses": fold_train_losses,
            "val_losses": fold_val_losses,
            "final_val_loss": final_val_loss
        })

        # Add detailed loss tracking for this fold
        all_fold_losses["folds"].append({
            "fold": fold + 1,
            "train_losses": fold_train_losses,
            "val_losses": fold_val_losses,
            "batch_losses": fold_batch_losses
        })

        # Save loss data after each fold to prevent data loss in case of interruption
        with open(loss_file, 'w') as f:
            json.dump(all_fold_losses, f, indent=4, cls=NumpyEncoder)
        log_message(f"Loss data updated in {loss_file}")

    # Calculate average times across folds
    avg_times = {key: sum(fold[key] for fold in fold_times) / len(fold_times) for key in fold_times[0].keys()}

    return fold_results, avg_times, fold_losses, all_fold_losses


# Helper class to handle numpy serialization for JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    log_message("Starting script execution")

    # Load data
    try:
        with open('label_dict.pkl', 'rb') as file:
            labels_dic = pickle.load(file)
        log_message("Successfully loaded label dictionary")

        df_train = pd.DataFrame(pd.read_csv('./unique_train_df.csv'))
        df_test = pd.DataFrame(pd.read_csv('./unique_train_df.csv'))
        log_message(f"Loaded training data with {len(df_train)} samples and test data with {len(df_test)} samples")

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

        log_message(f"Prepared {len(sentences)} sentences with {len(labels_id)} labels")
        log_message(f"Number of unique labels: {len(set(labels_id))}")
    except Exception as e:
        log_message(f"Error loading data: {str(e)}")
        raise

    # For faster execution during testing, you might want to reduce data size
    # Uncomment the lines below during development/testing
    # sentences = sentences[:100]
    # labels_id = labels_id[:100]
    # epochs = 3

    epochs = 10  # Set to original value for full training

    # Find optimal projection dimension
    # Comment out for faster execution if you want to use a fixed dimension
    dim_selection_start = time.time()
    # Using a subset for faster dimension selection
    sample_size = min(500, len(sentences))
    optimal_dim, dim_times, dim_losses = find_optimal_projection_dim(
        sentences[:sample_size],
        labels_id[:sample_size]
    )
    dim_selection_time = time.time() - dim_selection_start
    log_message(f"Dimension selection time: {dim_selection_time:.2f} seconds")
    # optimal_dim = 256  # Uncomment and use this instead if skipping dimension selection

    # Save dimension selection results
    dim_selection_results = {
        "optimal_dimension": int(optimal_dim),
        "dimensions_tested": list(dim_times.keys()),
        "times": {str(k): v for k, v in dim_times.items()},
        "losses": {str(k): v for k, v in dim_losses.items()}
    }

    dim_selection_file = os.path.join(results_dir, f"dimension_selection_{timestamp}.json")
    with open(dim_selection_file, 'w') as f:
        json.dump(dim_selection_results, f, indent=4)
    log_message(f"Dimension selection results saved to {dim_selection_file}")

    # Run 5-fold CV with Random Projection Preprocessing
    log_message(f"\n🔄 Running 5-fold cross-validation with projection preprocessing (dim={optimal_dim})")
    start_time = time.time()

    results, timing, losses, all_fold_losses = cross_validate_with_projection_preprocessing(
        sentences, labels_id, projection_dim=optimal_dim, k=5, epochs=epochs
    )

    total_time = time.time() - start_time

    # Average metrics
    results = np.array(results)
    avg_metrics = results.mean(axis=0)
    log_message("\n📈 Average Results with Random Projection Preprocessing:")
    log_message(f"Projection Dimension: {optimal_dim}")
    log_message(f"Accuracy: {avg_metrics[0]:.4f}")
    log_message(f"Precision: {avg_metrics[1]:.4f}")
    log_message(f"Recall: {avg_metrics[2]:.4f}")
    log_message(f"F1 Score: {avg_metrics[3]:.4f}")

    # Print timing summary
    log_message("\n⏱️ TIMING SUMMARY FOR RANDOM PROJECTION PREPROCESSING:")
    if 'dim_selection_time' in locals():
        log_message(f"Dimension selection time: {dim_selection_time:.2f} seconds")
    log_message(f"Average feature extraction and projection time: {timing['prep_time']:.2f} seconds")
    log_message(f"Average initialization time: {timing['init_time']:.2f} seconds")
    log_message(f"Average training time: {timing['train_time']:.2f} seconds")
    log_message(f"Average evaluation time: {timing['eval_time']:.2f} seconds")
    log_message(f"Average total time per fold: {timing['total_time']:.2f} seconds")
    log_message(f"Total execution time: {total_time:.2f} seconds")

    # Calculate average losses across folds
    avg_train_losses = np.mean([fold["train_losses"] for fold in losses], axis=0)
    avg_val_losses = np.mean([fold["val_losses"] for fold in losses], axis=0)

    log_message("\n📉 LOSS SUMMARY:")
    log_message(f"Initial avg train loss: {avg_train_losses[0]:.6f}")
    log_message(f"Final avg train loss: {avg_train_losses[-1]:.6f}")
    log_message(f"Initial avg val loss: {avg_val_losses[0]:.6f}")
    log_message(f"Final avg val loss: {avg_val_losses[-1]:.6f}")

    # Print loss progression
    log_message("\nLoss progression by epoch (average across folds):")
    for epoch in range(len(avg_train_losses)):
        log_message(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_losses[epoch]:.6f}, Val Loss = {avg_val_losses[epoch]:.6f}")

    # Save results to JSON file
    results_dict = {
        "model": "roberta-base with random projection preprocessing",
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "projection_dimension": int(optimal_dim),
        "metrics": {
            "accuracy": float(avg_metrics[0]),
            "precision": float(avg_metrics[1]),
            "recall": float(avg_metrics[2]),
            "f1_score": float(avg_metrics[3])
        },
        "timing": {
            "dimension_selection_time": float(dim_selection_time) if 'dim_selection_time' in locals() else None,
            "avg_prep_time": timing["prep_time"],
            "avg_init_time": timing["init_time"],
            "avg_train_time": timing["train_time"],
            "avg_eval_time": timing["eval_time"],
            "avg_total_time": timing["total_time"],
            "total_execution_time": float(total_time)
        },
        "losses": {
            "avg_train_losses": avg_train_losses.tolist(),
            "avg_val_losses": avg_val_losses.tolist(),
            "final_avg_train_loss": float(avg_train_losses[-1]),
            "final_avg_val_loss": float(avg_val_losses[-1])
        },
        "fold_results": [
            {
                "fold": i + 1,
                "accuracy": float(results[i][0]),
                "precision": float(results[i][1]),
                "recall": float(results[i][2]),
                "f1_score": float(results[i][3]),
                "train_loss": losses[i]["train_losses"][-1],
                "val_loss": losses[i]["final_val_loss"]
            }
            for i in range(len(results))
        ],
        "config": {
            "epochs": epochs,
            "data_size": len(sentences),
            "num_labels": len(set(labels_id)),
            "n_folds": 5
        }
    }

    # Save final results
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=4, cls=NumpyEncoder)

    # Make sure final loss data is saved
    with open(loss_file, 'w') as f:
        json.dump(all_fold_losses, f, indent=4, cls=NumpyEncoder)

    log_message(f"Results saved to {results_file}")
    log_message(f"Detailed loss data saved to {loss_file}")
    log_message(f"Log saved to {log_file}")
    log_message("Script execution completed")