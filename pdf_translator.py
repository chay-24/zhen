#!/usr/bin/env python3

import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pytesseract
import fitz
from transformers import MarianMTModel, MarianTokenizer
import torch
from dataclasses import dataclass

@dataclass
class TextRegion:
  """Represents a text region with its position and content."""
  x: int
  y: int
  width: int
  height: int
  text: str
  confidence: float


class PDFTranslator:
  def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-zh-en"):
    print(f"Loading translation model: {model_name}")
    self.tokenizer = MarianTokenizer.from_pretrained(model_name)
    self.model = MarianMTModel.from_pretrained(model_name)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    print(f"model loaded on device: {self.device}")

    self.tesseract_config = r'--oem 3 --psm 6 -l chi_sim+chi_tra'
  

  def preprocess_image(self, image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR results
    """
    # Convert to grayscale if colored
    if len(image.shape) == 3:
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
      gray = image.copy()

    # Apply Gaussian blur to reduce noice
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(
      blurred, 255, dv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11
    )

    # Morphological operations to clean up the image.
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morpholoyEx(threshold, cv2.MORPH_CLOSE, kernel)

    return cleaned


  def extract_text_regions(self, image: np.ndarray) -> List[TextRegion]:
    """
    Extracts the text regions from image using OCR.
    """
    processed_image = self.preprocess_image(image)

    data = pytesseract.image_to_data(
      processed_image,
      config=self.tesseract_config,
      output_type=pytesseract.Output.DICT
    )

    text_regions = []
    n_boxes = len(data['text'])

    for i in range(n_boxes):
      text = data['text'][i].strip()
      confidence = int(data['conf'][i])

      # Filter out low confidence and empty text
      if confidence > 30 and text and len(text) > 0:
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

        # check if text contains Chinese characters
        if self.contains_chinese(text):
          text_regions.append(TextRegion(x, y, w, h, text, confidence))

    return text_regions


  def contains_chinese(self, text: str) -> bool:
    for char in text:
      if '\u4e00' <= char <= '\u9fff':
        return True
    return False

  
  def translate_text(self, text: str) -> str:
    if not text.strip():
      return ""

    # Tokenize input
    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    # Generate translation
    with torch.no_grad():
      outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
      return translated
    
  
  def get_font_for_text(self, text: sr, box_width: int, box_height: int) -> Tuple[ImageFont.ImageFont, int]:
    font_paths = [
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
      "/System/Library/Fonts/Arial.ttf"
      "arial.ttf"
    ]

    font_path = None
    for path in font_paths:
      if os.path.exists(path):
        font_path = path
        break
    
    font_size = min(box_height - 4, 20):

    while font_size > 8:
      try:
        if font_path:
          font = ImageFont.truetype(font_path, font_size)
        else:
          font = ImageFont.load_default()

        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if text_width <= box_width - 4 and text_height <= box_height - 4:
          return font, font_size
      
      except Exception:
        pass
      font_size -= 1

    return ImageFont.load_default(), font_size
  

  def create_translated_image(self, original_image: np.ndarray, text_regions: List[TextRegion]) -> np.ndarray:
        """
        Create a new image with translated text overlaying the original positions.
        """
        # Convert to PIL Image
        if len(original_image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(original_image)
        
        # Create a drawing context
        draw = ImageDraw.Draw(pil_image)
        
        for region in text_regions:
            # Translate the text
            translated_text = self.translate_text(region.text)
            print(f"Original: {region.text}")
            print(f"Translated: {translated_text}")
            
            # Clear the original text area (fill with white)
            draw.rectangle(
                [region.x, region.y, region.x + region.width, region.y + region.height],
                fill="white"
            )
            
            # Get appropriate font
            font, font_size = self.get_font_for_text(translated_text, region.width, region.height)
            
            # Draw the translated text
            draw.text(
                (region.x + 2, region.y + 2),
                translated_text,
                fill="black",
                font=font
            )
        
        # Convert back to numpy array
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
  def process_image_file(self, input_path: str, output_path: str) -> None:
      """
      Process a single image file.
      """
      print(f"Processing image: {input_path}")
      
      # Read image
      image = cv2.imread(input_path)
      if image is None:
          raise ValueError(f"Could not read image: {input_path}")
      
      # Extract text regions
      print("Extracting text regions...")
      text_regions = self.extract_text_regions(image)
      print(f"Found {len(text_regions)} text regions")
      
      if not text_regions:
          print("No Chinese text found in the image")
          # Just copy the original image
          cv2.imwrite(output_path, image)
          return
      
      # Create translated image
      print("Creating translated image...")
      translated_image = self.create_translated_image(image, text_regions)
      
      # Save result
      cv2.imwrite(output_path, translated_image)
      print(f"Translated image saved to: {output_path}")
  
  def process_pdf_file(self, input_path: str, output_path: str) -> None:
      """
      Process a PDF file, extracting and translating text from each page.
      """
      print(f"Processing PDF: {input_path}")
      
      # Open the PDF
      pdf_document = fitz.open(input_path)
      
      # Create a new PDF for output
      output_pdf = fitz.open()
      
      for page_num in range(pdf_document.page_count):
          print(f"Processing page {page_num + 1}/{pdf_document.page_count}")
          
          page = pdf_document[page_num]
          
          # Convert page to image
          mat = fitz.Matrix(2.0, 2.0)  # Higher resolution
          pix = page.get_pixmap(matrix=mat)
          img_data = pix.tobytes("png")
          
          # Convert to numpy array
          nparr = np.frombuffer(img_data, np.uint8)
          image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
          
          # Extract and translate text
          text_regions = self.extract_text_regions(image)
          
          if text_regions:
              # Create translated image
              translated_image = self.create_translated_image(image, text_regions)
              
              # Convert back to PDF page
              _, img_encoded = cv2.imencode('.png', translated_image)
              img_bytes = img_encoded.tobytes()
              
              # Create new page from translated image
              img_pdf = fitz.open("png", img_bytes)
              output_pdf.insert_pdf(img_pdf)
              img_pdf.close()
          else:
              # No Chinese text found, keep original page
              output_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
      
      # Save the output PDF
      output_pdf.save(output_path)
      output_pdf.close()
      pdf_document.close()
      
      print(f"Translated PDF saved to: {output_path}")
  
  def process_file(self, input_path: str, output_path: Optional[str] = None) -> None:
      """
      Process an image or PDF file.
      """
      if not os.path.exists(input_path):
          raise FileNotFoundError(f"File not found: {input_path}")
      
      # Determine output path if not provided
      if not output_path:
          base_name, ext = os.path.splitext(input_path)
          output_path = f"{base_name}_translated{ext}"
      
      # Determine file type and process accordingly
      _, ext = os.path.splitext(input_path.lower())
      
      if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
          self.process_image_file(input_path, output_path)
      elif ext == '.pdf':
          self.process_pdf_file(input_path, output_path)
      else:
          raise ValueError(f"Unsupported file format: {ext}")
  


def main():
  parser = argparse.ArgumentParser(description="Translate Chinese text in images/PDFs to English")
  parser.add_argument("input_file", help="Input PDF file path")
  parser.add_argument("-o", "--output", help="Path to save the translated file")
  parser.add_argument("-m", "--model", default="Helsinki-NLP/opus-mt-zh-en",
                      help="Model name for trasnlation")

  args = parser.parse_args()

  try:
    translator = ImagePDFTranslator(args.model)

    translator.process_file(args.input_file, args.output)

    print("\nTranslation completed successfully!")
  
  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

if __name__=="__main__":
  main()