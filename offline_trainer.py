#!/usr/bin/env python3
"""
Offline trainer - internet bağlantısı olmadan çalışır
"""
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import yaml

def main():
    # Basit GPT-2 tokenizer kullan (çoğu sistemde cache'de var)
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True)
        print("✓ GPT-2 model yüklendi (cache'den)")
    except:
        print("✗ GPT-2 model cache'de yok")
        return
    
    # Dataset yükle
    with open('Zogoria_QA_clean.json', 'r') as f:
        dataset = json.load(f)
    
    print(f"✓ {len(dataset)} örnek yüklendi")
    
    # Örnek tokenization
    sample = dataset[0]
    text = f"Soru: {sample['input']}\nCevap: {sample['output']}"
    tokens = tokenizer.encode(text, max_length=256, truncation=True)
    
    print(f"✓ Tokenization test: {len(tokens)} token")
    print(f"Örnek: {text[:100]}...")

if __name__ == "__main__":
    main()
