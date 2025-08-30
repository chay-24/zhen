#!/bin/bash

echo "Setting up Chinese to English Translator..."
echo "-------------------------------------------"

echo "Updating system packages"
sudp apt-get update

# Install Tesseract OCR and Chinese language support
sudp apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra

# Install py dependencies
pip install -r requirements.txt

# Download the translation model
python3 -c "
from transformers import MarianMTModel, MarianTokenizer
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
print("Model downloaded successfully!")
"

echo "Setup completed successfully!"
