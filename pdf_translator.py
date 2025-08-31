#!/usr/bin/env python3

import os
import sys
import argparse
import json
from typing import List

import fitz
from transformers import MarianMTModel, MarianTokenizer
import torch


class ChineseTextTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-zh-en"):
        print(f"Loading translation model: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

    def contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def extract_chinese_text(self, text: str) -> List[str]:
        """Extract Chinese text segments from given text."""
        chinese_texts = []
        
        # Split text into lines and filter for Chinese content
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and self.contains_chinese(line):
                chinese_texts.append(line)
        
        return chinese_texts

    def translate_text(self, text: str) -> str:
        """Translate Chinese text to English."""
        if not text.strip():
            return ""

        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=512, num_beams=4)
                translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return translated
        except Exception as e:
            print(f"Translation error for '{text}': {e}")
            return ""  # Return empty string if translation fails

    def process_pdf_file(self, input_path: str) -> List[dict]:
        """
        Process a PDF file, extracting and translating Chinese text from each page.
        """
        print(f"Processing PDF: {input_path}")

        # Open the PDF
        pdf_document = fitz.open(input_path)
        all_translations = []

        for page_num in range(pdf_document.page_count):
            print(f"Processing page {page_num + 1}/{pdf_document.page_count}")

            page = pdf_document[page_num]
            
            # Extract text directly from PDF
            page_text = page.get_text()

            # Extract Chinese text
            chinese_texts = self.extract_chinese_text(page_text)

            if chinese_texts:
                print(f"Found {len(chinese_texts)} Chinese text segments on page {page_num + 1}")
                
                # Translate each text
                for text in chinese_texts:
                    translation = self.translate_text(text)
                    if translation:  # Only add if translation is successful
                        all_translations.append({
                            "chinese": text,
                            "english": translation
                        })
                        print(f"Page {page_num + 1}: '{text}' → '{translation}'")
            else:
                print(f"No Chinese text found on page {page_num + 1}")

        pdf_document.close()
        return all_translations

    def save_translations(self, translations: List[dict], output_path: str) -> str:
        """
        Save translations to JSON file and return JSON string.
        """
        output_data = {
            "total_translations": len(translations),
            "translation_model": "Helsinki-NLP/opus-mt-zh-en",
            "translations": translations
        }

        json_string = json.dumps(output_data, ensure_ascii=False, indent=2)

        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_string)
        
        print(f"Translations saved to: {output_path}")
        return json_string

    def process_file(self, input_path: str, output_path: str = None) -> str:
        """
        Process a PDF file and return translations as JSON string.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        # Determine output path if not provided
        if not output_path:
            base_name, _ = os.path.splitext(input_path)
            output_path = f"{base_name}_translations.json"

        # Check if it's a PDF file
        _, ext = os.path.splitext(input_path.lower())
        if ext != '.pdf':
            raise ValueError(f"Only PDF files are supported. Got: {ext}")

        translations = self.process_pdf_file(input_path)

        # Save and return JSON
        if translations:
            print(f"\nExtraction and translation completed!")
            print(f"Found and translated {len(translations)} Chinese text segments.")
            return self.save_translations(translations, output_path)
        else:
            print("\nNo Chinese text found in the PDF file.")
            empty_data = {
                "total_translations": 0,
                "translation_model": "Helsinki-NLP/opus-mt-zh-en",
                "translations": []
            }
            json_string = json.dumps(empty_data, ensure_ascii=False, indent=2)
            
            # Save empty JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_string)
            print(f"Empty translation file saved to: {output_path}")
            
            return json_string


def main():
    parser = argparse.ArgumentParser(description="Extract and translate Chinese text from PDFs to JSON")
    parser.add_argument("input_file", help="Input PDF file path")
    parser.add_argument("-o", "--output", help="Path to save the JSON translation file")
    parser.add_argument("-m", "--model", default="Helsinki-NLP/opus-mt-zh-en",
                        help="Model name for translation")

    args = parser.parse_args()

    try:
        translator = ChineseTextTranslator(args.model)
        json_output = translator.process_file(args.input_file, args.output)
        
        # Print the JSON output
        print("\n" + "="*50)
        print("TRANSLATION RESULTS (JSON):")
        print("="*50)
        print(json_output)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()