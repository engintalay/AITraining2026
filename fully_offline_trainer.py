#!/usr/bin/env python3
"""
Tamamen offline çalışan basit trainer
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path

class SimpleTokenizer:
    def __init__(self):
        # Basit karakter bazlı tokenizer
        self.vocab = {}
        self.reverse_vocab = {}
        self.pad_token = '<PAD>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        
    def build_vocab(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Özel tokenlar ekle
        vocab_list = [self.pad_token, self.eos_token, self.unk_token] + sorted(list(chars))
        self.vocab = {char: i for i, char in enumerate(vocab_list)}
        self.reverse_vocab = {i: char for char, i in self.vocab.items()}
        
    def encode(self, text, max_length=256):
        tokens = [self.vocab.get(char, self.vocab[self.unk_token]) for char in text]
        if len(tokens) > max_length - 1:
            tokens = tokens[:max_length-1]
        tokens.append(self.vocab[self.eos_token])
        
        # Padding
        while len(tokens) < max_length:
            tokens.append(self.vocab[self.pad_token])
            
        return tokens
    
    def decode(self, tokens):
        return ''.join([self.reverse_vocab.get(token, self.unk_token) for token in tokens])

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, self.max_length)
        return torch.tensor(tokens, dtype=torch.long)

def main():
    print("🚀 Offline Trainer başlatılıyor...")
    
    # Dataset yükle
    with open('Zogoria_QA_clean.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Metinleri hazırla
    texts = []
    for item in dataset:
        text = f"Soru: {item['input']}\nCevap: {item['output']}"
        texts.append(text)
    
    print(f"✓ {len(texts)} metin yüklendi")
    
    # Tokenizer oluştur
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    
    print(f"✓ Vocabulary oluşturuldu: {len(tokenizer.vocab)} token")
    
    # Dataset oluştur
    train_dataset = TextDataset(texts, tokenizer, max_length=256)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    print(f"✓ DataLoader hazır: {len(train_loader)} batch")
    
    # Test
    sample_batch = next(iter(train_loader))
    print(f"✓ Sample batch shape: {sample_batch.shape}")
    print(f"✓ Sample text: {texts[0][:100]}...")
    
    print("\n🎯 Eğitim için hazır! (Gerçek eğitim kodu eklenebilir)")

if __name__ == "__main__":
    main()
