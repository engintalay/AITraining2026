#!/usr/bin/env python3
"""
Basit test scripti - Gradio olmadan
"""
import os
from src.config import AppConfig
from src.evaluator import Evaluator

def main():
    config_path = os.getenv("CONFIG_PATH", "config_gtx1650.yaml")
    
    try:
        print("🔄 Model yükleniyor...")
        config = AppConfig.load_from_yaml(config_path)
        evaluator = Evaluator(config)
        print("✅ Model yüklendi!")
        
        while True:
            question = input("\n❓ Soru (çıkmak için 'q'): ")
            if question.lower() == 'q':
                break
                
            print("🔄 Cevaplar üretiliyor...")
            result = evaluator.compare(question)
            
            print(f"\n🤖 Base Model ({result['base_model']['response_time_ms']}ms):")
            print(result['base_model']['answer'])
            
            print(f"\n🎯 Fine-tuned Model ({result['finetuned_model']['response_time_ms']}ms):")
            print(result['finetuned_model']['answer'])
            
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    main()
