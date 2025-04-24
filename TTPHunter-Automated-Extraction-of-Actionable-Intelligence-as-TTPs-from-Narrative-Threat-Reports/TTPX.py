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
import time  # Added for time measurement

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("nanda-rani/TTPXHunter")


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
    total_loss = 0
    start_time = time.time()

    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    elapsed_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    return avg_loss, elapsed_time


# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    preds, labels = [], []
    total_loss = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            if outputs.loss is not None:
                total_loss += outputs.loss.item()

            preds += torch.argmax(logits, dim=1).cpu().tolist()
            labels += batch["labels"].cpu().tolist()

    elapsed_time = time.time() - start_time
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    # Calculate confusion matrix and extract false positives
    cm = confusion_matrix(labels, preds)
    fp = cm.sum(axis=0) - np.diag(cm)  # false positives per class
    total_fp = fp.sum()  # total false positives

    return acc, precision, recall, f1, total_fp, preds, total_loss / len(val_loader), elapsed_time


# Function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, fold, metrics, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_fold{fold}_epoch{epoch}.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


# K-Fold Cross Validation with fine-tuning and time tracking
def cross_validate(texts, labels, k=5, epochs=50, batch_size=32, patience=3, checkpoint_dir="checkpoints"):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    training_times = []

    # Create a DataFrame to store all training metrics
    training_metrics = pd.DataFrame(columns=[
        'fold', 'epoch', 'train_loss', 'val_loss', 'accuracy',
        'precision', 'recall', 'f1', 'false_positives',
        'training_time', 'eval_time'
    ])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{k}")
        print(f"{'=' * 50}")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"Train set size: {len(train_texts)}, Validation set size: {len(val_texts)}")
        print(f"Train labels distribution: {np.bincount(train_labels)}")
        print(f"Validation labels distribution: {np.bincount(val_labels)}")

        train_dataset = TextDataset(train_texts, train_labels)
        val_dataset = TextDataset(val_texts, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        config = RobertaConfig.from_pretrained("nanda-rani/TTPXHunter")
        num_labels = max(labels) + 1
        config.num_labels = num_labels

        base_model = RobertaModel.from_pretrained("nanda-rani/TTPXHunter")
        model = CustomRobertaClassifier(base_model, hidden_size=768, num_labels=num_labels)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=5e-5)

        # For early stopping
        best_f1 = 0
        no_improvement = 0
        best_checkpoint_path = None
        fold_start_time = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss, train_time = train_model(model, train_loader, optimizer)
            print(f"Training loss: {train_loss:.4f}, Time: {train_time:.2f}s")

            # Evaluate
            acc, precision, recall, f1, total_fp, preds, val_loss, eval_time = evaluate_model(model, val_loader)
            print(
                f"Validation - Loss: {val_loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FP: {total_fp}")
            print(f"Evaluation time: {eval_time:.2f}s")

            # Save metrics
            metrics_row = {
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'false_positives': total_fp,
                'training_time': train_time,
                'eval_time': eval_time
            }
            training_metrics = pd.concat([training_metrics, pd.DataFrame([metrics_row])], ignore_index=True)

            # Save checkpoint if this is the best model so far
            if f1 > best_f1:
                best_f1 = f1
                no_improvement = 0
                checkpoint_path = save_checkpoint(
                    model, optimizer, epoch, fold + 1,
                    {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'fp': total_fp},
                    checkpoint_dir
                )
                best_checkpoint_path = checkpoint_path
            else:
                no_improvement += 1
                print(f"No improvement for {no_improvement} epochs")

            # Early stopping
            if no_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Record the fold's total training time
        fold_time = time.time() - fold_start_time
        training_times.append(fold_time)
        print(f"\nFold {fold + 1} completed in {fold_time:.2f} seconds")

        # Load the best model for final evaluation
        if best_checkpoint_path:
            checkpoint = torch.load(best_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {best_checkpoint_path}")

        # Final evaluation
        final_acc, final_precision, final_recall, final_f1, final_fp, final_preds, _, _ = evaluate_model(model,
                                                                                                         val_loader)
        print(f"\n📊 Final fold {fold + 1} results:")
        print(
            f"Accuracy: {final_acc:.4f}, Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1: {final_f1:.4f}, FP: {final_fp}")

        fold_results.append((final_acc, final_precision, final_recall, final_f1, final_fp))

        # Save confusion matrix
        cm = confusion_matrix(val_labels, final_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold + 1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"confusion_matrix_fold{fold + 1}.png")
        plt.close()

    # Save training metrics to CSV
    training_metrics.to_csv("training_metrics.csv", index=False)

    # Calculate average training time
    avg_train_time = sum(training_times) / len(training_times)
    print(f"\n⏱️ Average training time per fold: {avg_train_time:.2f} seconds")

    # Plot training metrics
    plot_training_metrics(training_metrics)

    return fold_results, training_metrics, training_times


# Plot training metrics
def plot_training_metrics(metrics_df):
    # Create a directory for plots
    os.makedirs("plots", exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    for fold in metrics_df['fold'].unique():
        fold_data = metrics_df[metrics_df['fold'] == fold]
        plt.plot(fold_data['epoch'], fold_data['train_loss'], label=f'Fold {fold} - Train Loss')
        plt.plot(fold_data['epoch'], fold_data['val_loss'], label=f'Fold {fold} - Val Loss', linestyle='--')

    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/loss_curves.png")
    plt.close()

    # Plot F1 score
    plt.figure(figsize=(12, 6))
    for fold in metrics_df['fold'].unique():
        fold_data = metrics_df[metrics_df['fold'] == fold]
        plt.plot(fold_data['epoch'], fold_data['f1'], label=f'Fold {fold} - F1 Score')

    plt.title('F1 Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/f1_curves.png")
    plt.close()

    # Plot training time per epoch
    plt.figure(figsize=(12, 6))
    for fold in metrics_df['fold'].unique():
        fold_data = metrics_df[metrics_df['fold'] == fold]
        plt.plot(fold_data['epoch'], fold_data['training_time'], label=f'Fold {fold}')

    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/training_time.png")
    plt.close()


# Add SequenceClassifierOutput if not imported
from transformers.modeling_outputs import SequenceClassifierOutput

# Main code
if __name__ == "__main__":
    # Start timing the entire process
    total_start_time = time.time()

    # Load label dictionary
    with open('../label_dict.pkl', 'rb') as file:
        labels_dic = pickle.load(file)

    # Load data
    df_train = pd.DataFrame(pd.read_csv('../unique_train_df.csv'))
    df_test = pd.DataFrame(pd.read_csv('../unique_train_df.csv'))

    # Process data
    df_train["cats"] = df_train["cats"].apply(ast.literal_eval)
    df_test["cats"] = df_test["cats"].apply(ast.literal_eval)
    df_train["text"] = df_train["text"].apply(
        lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))
    df_test["text"] = df_test["text"].apply(
        lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))

    # Combine sentences and labels
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

    # Map labels to IDs
    labels_id = []
    for lb in labels:
        labels_id.append(labels_dic[lb])

    # Print dataset statistics
    print(f"Total samples: {len(sentences)}")
    print(f"Label distribution: {np.bincount(labels_id)}")

    # Run cross-validation with fine-tuning and time tracking
    results, metrics_df, training_times = cross_validate(
        sentences,
        labels_id,
        k=5,
        epochs=50,
        batch_size=32,
        patience=3,
        checkpoint_dir="model_checkpoints"
    )

    # Calculate and print average metrics
    results_array = np.array(results)
    avg_metrics = results_array.mean(axis=0)

    print("\n📈 Average Results:")
    print(f"Accuracy: {avg_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f}")
    print(f"Recall: {avg_metrics[2]:.4f}")
    print(f"F1 Score: {avg_metrics[3]:.4f}")
    print(f"Average False Positives: {avg_metrics[4]:.2f}")

    # Calculate total runtime
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n⏱️ Total Runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Average time per fold: {sum(training_times) / len(training_times):.2f}s")

    # Save results to file
    with open("model_results.json", "w") as f:
        json.dump({
            "accuracy": float(avg_metrics[0]),
            "precision": float(avg_metrics[1]),
            "recall": float(avg_metrics[2]),
            "f1_score": float(avg_metrics[3]),
            "false_positives": float(avg_metrics[4]),
            "total_runtime_seconds": total_time,
            "training_times_per_fold": training_times
        }, f, indent=4)

    print("\n✅ Results saved to model_results.json")


