#!/bin/bash

echo "Setting up Chinese to English Translator..."
echo "-------------------------------------------"

# Check if running as root for apt commands
if [[ $EUID -ne 0 ]]; then
   echo "This script needs sudo privileges for system package installation."
   echo "Please run with sudo or ensure you have sudo access."
fi

echo "Updating system packages..."
sudo apt-get update

# Install Tesseract OCR and Chinese language support
echo "Installing Tesseract OCR and Chinese language packs..."
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra

# Install system dependencies for OpenCV and other packages
echo "Installing system dependencies..."
sudo apt-get install -y python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Install SentencePiece (required for MarianTokenizer)
echo "Installing SentencePiece..."
sudo apt-get install -y cmake build-essential pkg-config libgoogle-perftools-dev
pip install sentencepiece

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "Creating requirements.txt..."
    cat > requirements.txt << EOF
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
sentencepiece>=0.1.95
opencv-python>=4.5.0
pillow>=8.0.0
pytesseract>=0.3.8
PyMuPDF>=1.20.0
numpy>=1.21.0
argparse
dataclasses
EOF
fi

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Test if SentencePiece is properly installed
echo "Testing SentencePiece installation..."
python3 -c "import sentencepiece; print('SentencePiece installed successfully')" || {
    echo "SentencePiece installation failed. Trying alternative installation..."
    pip install --upgrade --force-reinstall sentencepiece
}

# Download and cache the translation model
echo "Downloading translation model (this may take a few minutes)..."
python3 -c "
try:
    from transformers import MarianMTModel, MarianTokenizer
    print('Loading tokenizer...')
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
    print('Loading model...')
    model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
    print('Model downloaded and cached successfully!')
except Exception as e:
    print(f'Error downloading model: {e}')
    print('You may need to download the model manually later.')
"

# Test Tesseract installation
echo "Testing Tesseract installation..."
tesseract --version || {
    echo "Warning: Tesseract installation may have issues"
}

# Test if Chinese language packs are available
echo "Testing Chinese language support..."
tesseract --list-langs | grep -E "(chi_sim|chi_tra)" || {
    echo "Warning: Chinese language packs may not be properly installed"
    echo "You may need to install them manually:"
    echo "sudo apt-get install tesseract-ocr-chi-sim tesseract-ocr-chi-tra"
}

echo ""
echo "Setup completed!"
echo "-------------------------------------------"
