from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pickle
import torch
import os
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel
import torch.nn as nn
import json
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import time
from datetime import timedelta
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import get_cosine_schedule_with_warmup
import torch.backends.cudnn as cudnn

cudnn.benchmark = False
cudnn.deterministic = True
torch.cuda.empty_cache()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load RoBERTa base tokenizer instead of TTPXHunter
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


class CustomRobertaClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels):
        super().__init__()
        self.roberta = base_model
        self.dropout = nn.Dropout(0.1)
        # Store num_labels for easy access
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Add class_weights parameter with default None

    def forward(self, input_ids=None, attention_mask=None, labels=None, class_weights=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Correct pooling: mean pooling over non-masked tokens
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)

        # Apply attention mask
        masked_hidden_state = last_hidden_state * attention_mask
        sum_hidden = masked_hidden_state.sum(dim=1)  # Sum over tokens
        sum_mask = attention_mask.sum(dim=1)  # Number of valid tokens

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        pooled_output = sum_hidden / sum_mask  # Mean pooling

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

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


# Training function with detailed loss tracking
def train_model(model, train_loader, optimizer, scheduler=None, fold=0, epoch=0, class_weights=None):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    step_losses = []
    global_step = epoch * num_batches

    if class_weights is not None:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        # Move data to device BEFORE passing to model
        batch_labels = batch["labels"]
        batch_input_ids = batch["input_ids"]
        batch_attention_mask = batch["attention_mask"]
        #outputs = model(**batch)
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits
        # Manually calculate loss
        loss = loss_fn(logits.view(-1, model.num_labels), batch_labels.view(-1))
        batch_loss = loss.item()
        total_loss += batch_loss

        # Track step loss
        current_step = global_step + batch_idx + 1
        step_losses.append({
            "fold": fold,
            "epoch": epoch + 1,
            "step": current_step,
            "batch": batch_idx + 1,
            "loss": batch_loss
        })

        if (batch_idx + 1) % 10 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch + 1}, Step {current_step}: Loss = {batch_loss:.6f}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    return total_loss / num_batches, step_losses

    # Save step losses to file
    step_loss_file = f"fold_{fold + 1}_epoch_{epoch + 1}_step_losses.json"
    with open(step_loss_file, "w") as f:
        json.dump(step_losses, f, indent=2)

    print(f"Saved step losses to {step_loss_file}")

    # Create step loss plot
    plt.figure(figsize=(10, 6))
    steps = [s["step"] for s in step_losses]
    losses = [s["loss"] for s in step_losses]
    plt.plot(steps, losses)
    plt.title(f"Fold {fold + 1}, Epoch {epoch + 1} - Loss by Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"fold_{fold + 1}_epoch_{epoch + 1}_step_loss.png")
    plt.close()

    return total_loss / num_batches, step_losses  # Return average loss and step losses


# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    preds, labels = [], []
    total_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.item()

            preds += torch.argmax(logits, dim=1).cpu().tolist()
            labels += batch["labels"].cpu().tolist()

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    # Calculate confusion matrix and extract false positives
    cm = confusion_matrix(labels, preds)
    fp = cm.sum(axis=0) - np.diag(cm)  # false positives per class
    total_fp = fp.sum()  # total false positives

    return acc, precision, recall, f1, total_fp, preds, total_loss / num_batches


# Save model function
def save_model(model, tokenizer, fold, metrics, output_dir="saved_models"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create fold-specific directory
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }, os.path.join(fold_dir, "model.pt"))

    # Save tokenizer
    tokenizer.save_pretrained(fold_dir)

    # Save metrics
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print(f"Model and metrics saved to {fold_dir}")


# K-Fold Cross Validation with time tracking and learning rate scheduler
def cross_validate(texts, labels, k=5, epochs=30, batch_size=16, learning_rate=2e-5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    fold_times = []
    fold_metrics = []

    # For storing detailed training metrics
    training_history = {}
    all_step_losses = []  # Store all step losses across folds

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        print(f"\n{'=' * 50}\nFold {fold + 1}/{k}\n{'=' * 50}")
        fold_start_time = time.time()

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        num_labels = len(set(labels))
        print(f"number_classes: {num_labels}")

        train_dataset = TextDataset(train_texts, train_labels)
        val_dataset = TextDataset(val_texts, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=num_labels,
            hidden_dropout_prob=0.3,  # Increased dropout
            attention_probs_dropout_prob=0.3
        )
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Learning rate scheduler: ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        fold_history = {"train_loss": [], "val_loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [],
                        "epoch_times": []}

        best_val_f1 = 0
        best_model_state = None
        early_stopping_patience = 5
        no_improve_epochs = 0

        # Loss function (with optional class weights)
        class_counts = np.bincount(train_labels, minlength=num_labels)
        class_weights = torch.tensor(len(train_labels) / (num_labels * class_counts + 1e-9), dtype=torch.float32).to(
            device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            model.train()
            total_train_loss = 0
            for batch in tqdm(train_loader, desc="Training"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits
                loss = loss_fn(logits, batch["labels"])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            model.eval()
            preds, labels_true = [], []
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    logits = outputs.logits
                    loss = loss_fn(logits, batch["labels"])
                    total_val_loss += loss.item()

                    preds += torch.argmax(logits, dim=1).cpu().tolist()
                    labels_true += batch["labels"].cpu().tolist()

            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            acc = accuracy_score(labels_true, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(labels_true, preds, average="weighted")

            fold_history["train_loss"].append(avg_train_loss)
            fold_history["val_loss"].append(avg_val_loss)
            fold_history["accuracy"].append(acc)
            fold_history["precision"].append(precision)
            fold_history["recall"].append(recall)
            fold_history["f1"].append(f1)
            fold_history["epoch_times"].append(time.time() - epoch_start_time)

            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_model_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": {
                        "accuracy": acc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1
                    }
                }
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Save best model
        fold_dir = f"saved_models/fold_{fold}"
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(best_model_state, os.path.join(fold_dir, "model.pt"))

        training_history[f"fold_{fold + 1}"] = fold_history
        fold_results.append((acc, precision, recall, f1))
        fold_times.append(time.time() - fold_start_time)
        fold_metrics.append(best_model_state["metrics"])

    return fold_results, fold_times, fold_metrics, training_history


# Main execution
if __name__ == "__main__":
    print("Loading data...")

    # Load label dictionary
    with open('../label_dict.pkl', 'rb') as file:
        labels_dic = pickle.load(file)

    # Load datasets
    df_train = pd.DataFrame(pd.read_csv('../unique_train_df.csv'))
    df_test = pd.DataFrame(pd.read_csv('../unique_train_df.csv'))
    df_train["cats"] = df_train["cats"].apply(ast.literal_eval)
    df_test["cats"] = df_test["cats"].apply(ast.literal_eval)
    df_train["text"] = df_train["text"].apply(
        lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))
    df_test["text"] = df_test["text"].apply(
        lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))

    # Prepare data
    #sentences = df_train["text"].tolist() + df_test["text"].tolist()
    sentences = df_train["text"].tolist()
    labels = []
    for _, va in df_train["cats"].items():
        for k, v in va.items():
            if v == 1.0:
                labels.append(k)
    # for _, va in df_test["cats"].items():
    #     for k, v in va.items():
    #         if v == 1.0:
    #             labels.append(k)

    # Convert labels to IDs
    labels_id = []
    for lb in labels:
        labels_id.append(labels_dic[lb])

    #print("number of labels:", len(labels_id))
    print("Min label id:", min(labels_id))
    print("Max label id:", max(labels_id))
    print("Unique labels:", set(labels_id))

    print(f"Original labels (before remapping): {sorted(set(labels_id))}")
    print(f"Number of original classes: {len(set(labels_id))}")
    # Create mapping old label ➔ new label
    old_labels = sorted(list(set(labels_id)))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(old_labels)}

    # Apply mapping
    labels_id = [label_mapping[label] for label in labels_id]

    # (Optional) Save label mapping for future decoding
    # with open("label_mapping.json", "w") as f:
    #     json.dump(label_mapping, f, indent=2)

    print(f"📢 Final labels after remapping: {sorted(set(labels_id))}")
    print(f"✅ Number of final classes: {len(set(labels_id))}")
    print(f"✅ Total number of samples: {len(sentences)}")

    # Print dataset statistics
    print(f"Total samples: {len(sentences)}")
    print(f"Number of unique classes: {len(set(labels_id))}")
    print(f"Class distribution: {np.bincount(labels_id)}")

    # Set training parameters
    params = {
        "k": 5,  # Number of folds
        "epochs": 30,  # Number of training epochs
        "batch_size": 16,  # Batch size
        "learning_rate": 2e-5  # Learning rate (slightly reduced for RoBERTa base)
    }

    # Start time measurement for the entire process
    total_start_time = time.time()

    # Run cross-validation
    print(f"\nStarting {params['k']}-fold cross-validation with {params['epochs']} epochs using RoBERTa-base...")
    results, fold_times, fold_metrics, training_history, all_step_losses = cross_validate(
        sentences,
        labels_id,
        k=params["k"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"]
    )

    # Calculate total time
    total_time = time.time() - total_start_time

    # Calculate average metrics
    results_array = np.array(results)
    avg_metrics = results_array.mean(axis=0)
    std_metrics = results_array.std(axis=0)

    # Print overall results
    print("\n" + "=" * 50)
    print("🔍 CROSS-VALIDATION RESULTS (RoBERTa)")
    print("=" * 50)
    print(f"Total training time: {timedelta(seconds=total_time)}")
    print(f"Average fold time: {timedelta(seconds=np.mean(fold_times))}")

    print("\n📊 PER-FOLD METRICS:")
    for i, ((acc, prec, rec, f1, fp), time_taken) in enumerate(zip(results, fold_times)):
        print(
            f"Fold {i + 1}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, Time={timedelta(seconds=time_taken)}")

    print("\n📈 AVERAGE RESULTS:")
    print(f"Accuracy: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"Recall: {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"F1 Score: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
    print(f"False Positives: {avg_metrics[4]:.1f} ± {std_metrics[4]:.1f}")

    # Plot overall metrics across folds
    plt.figure(figsize=(12, 8))
    metrics = ["Accuracy", "Precision", "Recall", "F1"]

    for i, metric_name in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        values = [result[i] for result in results]
        plt.bar(range(1, len(values) + 1), values)
        plt.axhline(y=np.mean(values), color='r', linestyle='-', label=f'Mean: {np.mean(values):.4f}')
        plt.title(f"{metric_name} across folds")
        plt.xlabel("Fold")
        plt.ylabel(metric_name)
        plt.ylim(0, 1)
        plt.legend()

    plt.tight_layout()
    plt.savefig("roberta_base_cross_validation_metrics.png")

    # Save final results to JSON
    final_results = {
        "model": "roberta",
        "params": params,
        "avg_metrics": {
            "accuracy": float(avg_metrics[0]),
            "precision": float(avg_metrics[1]),
            "recall": float(avg_metrics[2]),
            "f1": float(avg_metrics[3]),
            "false_positives": float(avg_metrics[4])
        },
        "std_metrics": {
            "accuracy": float(std_metrics[0]),
            "precision": float(std_metrics[1]),
            "recall": float(std_metrics[2]),
            "f1": float(std_metrics[3]),
            "false_positives": float(std_metrics[4])
        },
        "fold_results": [
            {
                "fold": i + 1,
                "metrics": {
                    "accuracy": float(results[i][0]),
                    "precision": float(results[i][1]),
                    "recall": float(results[i][2]),
                    "f1": float(results[i][3]),
                    "false_positives": int(results[i][4])
                },
                "time_seconds": float(fold_times[i])
            }
            for i in range(len(results))
        ],
        "total_time_seconds": float(total_time),
        "average_fold_time_seconds": float(np.mean(fold_times))
    }

    with open("roberta_base_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n✅ Results saved to roberta_base_results.json")
    print("✅ Step-level losses saved to all_step_losses.json")
    print("\n🎉 Training and evaluation completed!")