#!/bin/bash
set -e  # Exit immediately on error

echo "[1/5] Creating virtual environment..."
python3 -m venv venv

echo "[2/5] Activating virtual environment..."
source venv/bin/activate

echo "[3/5] Upgrading pip and installing dependencies..."
pip install --upgrade pip

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers scikit-learn pandas tqdm numpy

echo "[4/5] Downloading RoBERTa tokenizer and model (optional)..."
python3 -c "
from transformers import RobertaTokenizer, RobertaForSequenceClassification
RobertaTokenizer.from_pretrained('roberta-base')
RobertaForSequenceClassification.from_pretrained('roberta-base')
"

echo "[5/5] Environment setup complete!"
echo "Activate your environment with: source venv/bin/activate"