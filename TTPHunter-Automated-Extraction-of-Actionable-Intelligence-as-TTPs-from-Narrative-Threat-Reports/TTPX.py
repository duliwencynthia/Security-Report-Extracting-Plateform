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

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        # Move data to device BEFORE passing to model
        batch_labels = batch["labels"].to(device)
        batch_input_ids = batch["input_ids"].to(device)
        batch_attention_mask = batch["attention_mask"].to(device)
        #outputs = model(**batch)
        outputs = model(input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        labels=batch_labels,
                        class_weights=class_weights)  # <-- Pass weights here
        loss = outputs.loss
        batch_loss = loss.item()
        total_loss += batch_loss

        # Track step-level loss
        current_step = global_step + batch_idx + 1
        step_losses.append({
            "fold": fold,
            "epoch": epoch + 1,
            "step": current_step,
            "batch": batch_idx + 1,
            "loss": batch_loss
        })

        # Print loss every 10 steps
        if (batch_idx + 1) % 10 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch + 1}, Step {current_step}: Loss = {batch_loss:.6f}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #added
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

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
def cross_validate(texts, labels, k=5, epochs=10, batch_size=16, learning_rate=5e-6):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    fold_times = []
    fold_metrics = []

    # For storing detailed training metrics
    training_history = {}
    all_step_losses = []  # Store all step losses across folds

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{k}")
        print(f"{'=' * 50}")

        fold_start_time = time.time()

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"Training set size: {len(train_texts)}")
        print(f"Validation set size: {len(val_texts)}")
        print("Train labels distribution:", np.bincount(train_labels))
        print("Validation labels distribution:", np.bincount(val_labels))

        print("Calculating class weights for weighted loss...")
        num_labels = max(labels) + 1  # Or determine dynamically: len(np.unique(labels_id))
        print("number_classes:", num_labels)
        class_counts = np.bincount(train_labels, minlength=num_labels)

        print(f"Train labels unique: {sorted(set(train_labels))}")
        print(f"Validation labels unique: {sorted(set(val_labels))}")
        print(f"Max label ID: {max(train_labels)}")
        print(f"Number of classes (num_labels): {max(labels) + 1}")

        # Handle potential zero counts if a class is missing in a fold's train set (rare with shuffle/stratify)
        if np.any(class_counts == 0):
            print(
                f"Warning: Class counts for fold {fold + 1}: {class_counts}. Some classes might be missing in the training split.")
            # Assign a very high weight or use a floor? Using inverse frequency might be unstable.
            # A simple approach is to use 1 / (count + epsilon)
            epsilon = 1e-6  # Small value to prevent division by zero
            weights = 1.0 / (class_counts + epsilon)
            # Normalize weights (optional, but can help)
            weights = weights / np.sum(weights) * num_labels

        else:
            # Standard inverse frequency weighting
            # Formula: weight = total_samples / (num_classes * count_per_class)
            total_samples = len(train_labels)
            weights = total_samples / (num_labels * class_counts)

        print(f"Calculated weights: {weights}")
        # Convert weights to a tensor and move to the device
        class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

        train_dataset = TextDataset(train_texts, train_labels)
        val_dataset = TextDataset(val_texts, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Model initialization - using RoBERTa base
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=num_labels
        )
        model.to(device)

        # Create optimizer with weight decay
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Learning rate scheduler
        total_steps = len(train_loader) * epochs
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.05 * total_steps)  # 5% warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # For tracking metrics
        fold_history = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "epoch_times": [],
            "step_losses": []  # New tracking for step losses
        }

        # Training loop
        best_val_f1 = 0
        best_model_state = None

        # Create fold loss directory
        fold_loss_dir = f"fold_{fold + 1}_losses"
        if not os.path.exists(fold_loss_dir):
            os.makedirs(fold_loss_dir)

        #early stop
        patience = 3
        best_f1 = 0
        epochs_no_improve = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train with detailed loss tracking
            # train_loss, epoch_step_losses = train_model(model, train_loader, optimizer, scheduler, fold, epoch)
            train_loss, epoch_step_losses = train_model(
                model,
                train_loader,
                optimizer,
                scheduler,
                fold,
                epoch,
                class_weights=class_weights_tensor  # <-- Pass the tensor here
            )

            # Track all step losses
            all_step_losses.extend(epoch_step_losses)
            fold_history["step_losses"].extend(epoch_step_losses)

            # Save all step losses for this epoch
            with open(f"{fold_loss_dir}/epoch_{epoch + 1}_losses.json", "w") as f:
                json.dump(epoch_step_losses, f, indent=2)

            # Evaluate
            acc, precision, recall, f1, total_fp, preds, val_loss = evaluate_model(model, val_loader)

            # Record metrics
            epoch_time = time.time() - epoch_start
            fold_history["train_loss"].append(train_loss)
            fold_history["val_loss"].append(val_loss)
            fold_history["accuracy"].append(acc)
            fold_history["precision"].append(precision)
            fold_history["recall"].append(recall)
            fold_history["f1"].append(f1)
            fold_history["epoch_times"].append(epoch_time)

            # Save best model state
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
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Print epoch results with time
            print(f"Epoch completed in {timedelta(seconds=epoch_time)}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Save complete fold history with step losses
        with open(f"{fold_loss_dir}/complete_fold_history.json", "w") as f:
            json.dump(fold_history, f, indent=2)

        # Load the best model for final evaluation
        if best_model_state:
            model.load_state_dict(best_model_state["model_state_dict"])
            print(f"\nLoaded best model from epoch {best_model_state['epoch'] + 1}")

        # Final evaluation
        acc, precision, recall, f1, total_fp, preds, val_loss = evaluate_model(model, val_loader)
        print(f"\n✅ Final Evaluation:")
        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FP: {total_fp}")

        # Save best model checkpoint
        save_model(model, tokenizer, fold, best_model_state["metrics"])

        # Calculate fold time
        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)
        print(f"Fold completed in {timedelta(seconds=fold_time)}")

        # Store fold results
        fold_results.append((acc, precision, recall, f1, total_fp))
        fold_metrics.append(best_model_state["metrics"])
        training_history[f"fold_{fold + 1}"] = fold_history

        # Plot training curves for this fold
        plt.figure(figsize=(12, 10))

        # Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(fold_history["train_loss"], label="Train Loss")
        plt.plot(fold_history["val_loss"], label="Validation Loss")
        plt.title(f"Fold {fold + 1} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy curve
        plt.subplot(2, 2, 2)
        plt.plot(fold_history["accuracy"])
        plt.title(f"Fold {fold + 1} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        # F1 curve
        plt.subplot(2, 2, 3)
        plt.plot(fold_history["f1"])
        plt.title(f"Fold {fold + 1} F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1")

        # Precision-Recall curves
        plt.subplot(2, 2, 4)
        plt.plot(fold_history["precision"], label="Precision")
        plt.plot(fold_history["recall"], label="Recall")
        plt.title(f"Fold {fold + 1} Precision-Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"fold_{fold + 1}_training_curves.png")
        plt.close()

        # Generate and save confusion matrix
        cm = confusion_matrix(val_labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Fold {fold + 1} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"fold_{fold + 1}_confusion_matrix.png")
        plt.close()

    # Save overall training history
    with open("training_history.json", "w") as f:
        json.dump(training_history, f)

    # Save all step losses across all folds
    with open("all_step_losses.json", "w") as f:
        json.dump(all_step_losses, f, indent=2)

    # Create global step loss visualization
    plt.figure(figsize=(15, 8))

    # Group by fold
    for fold in range(k):
        fold_steps = [s for s in all_step_losses if s["fold"] == fold]
        if fold_steps:
            steps = [s["step"] for s in fold_steps]
            losses = [s["loss"] for s in fold_steps]
            plt.plot(steps, losses, label=f"Fold {fold + 1}")

    plt.title("Training Loss by Step Across All Folds")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("global_step_loss.png")
    plt.close()

    # Return complete results
    return fold_results, fold_times, fold_metrics, training_history, all_step_losses


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
        "epochs": 20,  # Number of training epochs
        "batch_size": 16,  # Batch size
        "learning_rate": 2e-4  # Learning rate (slightly reduced for RoBERTa base)
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