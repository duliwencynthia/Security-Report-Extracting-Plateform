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
import sys
import logging
from datetime import timedelta, datetime
from transformers.modeling_outputs import SequenceClassifierOutput

# Configure logging
log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load RoBERTa base tokenizer
logger.info("Loading RoBERTa base tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


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


# Training function with batch-level loss logging
def train_model(model, train_loader, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    # For logging batch-level loss
    log_interval = max(1, num_batches // 10)  # Log approximately 10 times per epoch

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        batch_loss = loss.item()
        total_loss += batch_loss

        # Log batch-level loss
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0 or batch_idx == num_batches - 1:
            logger.info(f"  Batch {batch_idx + 1}/{num_batches} - Loss: {batch_loss:.6f}")

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    logger.info(f"Average training loss: {avg_loss:.6f}")
    return avg_loss


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
            batch_loss = loss.item()
            total_loss += batch_loss

            preds += torch.argmax(logits, dim=1).cpu().tolist()
            labels += batch["labels"].cpu().tolist()

    avg_loss = total_loss / num_batches
    logger.info(f"Average validation loss: {avg_loss:.6f}")

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    # Calculate confusion matrix and extract false positives
    cm = confusion_matrix(labels, preds)
    fp = cm.sum(axis=0) - np.diag(cm)  # false positives per class
    total_fp = fp.sum()  # total false positives

    return acc, precision, recall, f1, total_fp, preds, avg_loss


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

    logger.info(f"Model and metrics saved to {fold_dir}")


# K-Fold Cross Validation with time tracking and learning rate scheduler
def cross_validate(texts, labels, k=5, epochs=20, batch_size=8, learning_rate=5e-5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    fold_times = []
    fold_metrics = []

    # For storing detailed training metrics
    training_history = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Fold {fold + 1}/{k}")
        logger.info(f"{'=' * 50}")

        fold_start_time = time.time()

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        logger.info(f"Training set size: {len(train_texts)}")
        logger.info(f"Validation set size: {len(val_texts)}")
        logger.info(f"Train labels distribution: {np.bincount(train_labels)}")
        logger.info(f"Validation labels distribution: {np.bincount(val_labels)}")

        train_dataset = TextDataset(train_texts, train_labels)
        val_dataset = TextDataset(val_texts, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Model initialization - using RoBERTa base
        logger.info("Initializing RoBERTa base model...")
        base_model = RobertaModel.from_pretrained("roberta-base")
        model = CustomRobertaClassifier(base_model, hidden_size=768, num_labels=max(labels) + 1)
        model.to(device)

        # Create optimizer with weight decay
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Learning rate scheduler
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # For tracking metrics
        fold_history = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "epoch_times": []
        }

        # Training loop
        best_val_f1 = 0
        best_model_state = None

        for epoch in range(epochs):
            epoch_start = time.time()
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss = train_model(model, train_loader, optimizer, scheduler)

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
                logger.info(f"New best model found! F1: {f1:.4f}")

            # Print epoch results with time
            logger.info(f"Epoch completed in {timedelta(seconds=epoch_time)}")
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            logger.info(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Load the best model for final evaluation
        if best_model_state:
            model.load_state_dict(best_model_state["model_state_dict"])
            logger.info(f"\nLoaded best model from epoch {best_model_state['epoch'] + 1}")

        # Final evaluation
        logger.info("\nPerforming final evaluation...")
        acc, precision, recall, f1, total_fp, preds, val_loss = evaluate_model(model, val_loader)
        logger.info(f"\n✅ Final Evaluation:")
        logger.info(
            f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FP: {total_fp}")

        # Save best model checkpoint
        save_model(model, tokenizer, fold, best_model_state["metrics"])

        # Calculate fold time
        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)
        logger.info(f"Fold completed in {timedelta(seconds=fold_time)}")

        # Store fold results
        fold_results.append((acc, precision, recall, f1, total_fp))
        fold_metrics.append(best_model_state["metrics"])
        training_history[f"fold_{fold + 1}"] = fold_history

        # Plot training curves for this fold
        logger.info(f"Generating training curves for fold {fold + 1}...")
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
        logger.info(f"Generating confusion matrix for fold {fold + 1}...")
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
    logger.info("Training history saved to training_history.json")

    # Return complete results
    return fold_results, fold_times, fold_metrics, training_history


# Main execution
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("STARTING ROBERTA-BASE TRAINING PROCESS")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_filename}")

    logger.info("Loading data...")

    try:
        # Load label dictionary
        with open('../label_dict.pkl', 'rb') as file:
            labels_dic = pickle.load(file)
        logger.info("Label dictionary loaded successfully")

        # Load datasets
        df_train = pd.DataFrame(pd.read_csv('../unique_train_df.csv'))
        df_test = pd.DataFrame(pd.read_csv('../unique_train_df.csv'))
        logger.info("CSV data loaded successfully")

        df_train["cats"] = df_train["cats"].apply(ast.literal_eval)
        df_test["cats"] = df_test["cats"].apply(ast.literal_eval)
        df_train["text"] = df_train["text"].apply(
            lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))
        df_test["text"] = df_test["text"].apply(
            lambda x: ' '.join(map(str, x.tolist())) if isinstance(x, pd.Series) else str(x))
        logger.info("Data preprocessing completed")

        # Prepare data
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

        # Convert labels to IDs
        labels_id = []
        for lb in labels:
            labels_id.append(labels_dic[lb])

        # Print dataset statistics
        logger.info(f"Total samples: {len(sentences)}")
        logger.info(f"Number of unique classes: {len(set(labels_id))}")
        logger.info(f"Class distribution: {np.bincount(labels_id)}")

        # Set training parameters
        params = {
            "k": 5,  # Number of folds
            "epochs": 20,  # Number of training epochs
            "batch_size": 8,  # Batch size
            "learning_rate": 2e-5  # Learning rate (slightly reduced for RoBERTa base)
        }
        logger.info(f"Training parameters: {params}")

        # Start time measurement for the entire process
        total_start_time = time.time()

        # Run cross-validation
        logger.info(
            f"\nStarting {params['k']}-fold cross-validation with {params['epochs']} epochs using RoBERTa-base...")
        results, fold_times, fold_metrics, training_history = cross_validate(
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
        logger.info("\n" + "=" * 50)
        logger.info("🔍 CROSS-VALIDATION RESULTS (RoBERTa-base)")
        logger.info("=" * 50)
        logger.info(f"Total training time: {timedelta(seconds=total_time)}")
        logger.info(f"Average fold time: {timedelta(seconds=np.mean(fold_times))}")

        logger.info("\n📊 PER-FOLD METRICS:")
        for i, ((acc, prec, rec, f1, fp), time_taken) in enumerate(zip(results, fold_times)):
            logger.info(
                f"Fold {i + 1}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, Time={timedelta(seconds=time_taken)}")

        logger.info("\n📈 AVERAGE RESULTS:")
        logger.info(f"Accuracy: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
        logger.info(f"Precision: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
        logger.info(f"Recall: {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
        logger.info(f"F1 Score: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
        logger.info(f"False Positives: {avg_metrics[4]:.1f} ± {std_metrics[4]:.1f}")

        # Plot overall metrics across folds
        logger.info("Generating final cross-validation metrics visualization...")
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
        logger.info("Saving final results...")
        final_results = {
            "model": "roberta-base",
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

        logger.info("\n✅ Results saved to roberta_base_results.json")
        logger.info("\n🎉 Training and evaluation completed!")

    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")
        raise

    finally:
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)