# Config-Based LLM Fine-Tuning & Test Platform

Bu proje, NVIDIA GPU'lar üzerinde LLM (Large Language Model) fine-tuning (ince ayar) işlemleri yapmak, modelleri test etmek ve karşılaştırmak için geliştirilmiş konfigürasyon tabanlı bir platformdur.

## 🚀 Özellikler

*   **Tamamen Konfigüre Edilebilir:** Tüm eğitim ve model ayarları tek bir `config.yaml` dosyasından yönetilir.
*   **Düşük VRAM Dostu:** 4-bit ve 8-bit quantization (QLoRA) desteği ile 12-16GB VRAM'de çalışabilir.
*   **Kesintiye Dayanıklı (Resume):** Eğitim yarıda kalırsa, otomatik olarak son checkpoint'ten devam eder.
*   **Modüler Mimari:** Eğitim, Test ve API katmanları birbirinden ayrılmıştır.
*   **Karşılaştırma API'si:** Base model ile Fine-tuned modeli yan yana karşılaştıran REST API.

## 🛠️ Kurulum

### Gereksinimler
*   Linux İşletim Sistemi
*   NVIDIA GPU (Min. 12GB VRAM önerilir)
*   Python 3.10+
*   CUDA Toolkit

### Adımlar

1.  **Projeyi Klonlayın:** (Eğer git kullanıyorsanız)
    ```bash
    git clone <repo-url>
    cd AITraining2026
    ```

2.  **Sanal Ortam Oluşturun (Opsiyonel ama Önerilir):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Konfigürasyon

Projenin kalbi `config.yaml` dosyasıdır. Örnek bir konfigürasyon aşağıdaki gibidir:

```yaml
model:
  name_or_path: "mistralai/Mistral-7B-v0.1" # HuggingFace model ID veya yerel yol
  quantization_bit: 4                       # 4 veya 8 (VRAM tasarrufu için 4 önerilir)
  use_gradient_checkpointing: true          # VRAM tasarrufu sağlar

peft:
  r: 16                                     # LoRA rank
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]      # Hangi modüllere uygulanacağı

training:
  batch_size: 1
  gradient_accumulation_steps: 4            # Düşük batch size'ı telafi etmek için artırın
  num_train_epochs: 1
  learning_rate: 2.0e-4
  output_dir: "experiments/my_finetune"
  resume_from_checkpoint: true              # Otomatik devam etme özelliği

data:
  dataset_path: "dataset.json"              # Eğitim verisi yolu
  max_seq_length: 512
```

## 🏋️‍♂️ Eğitim (Fine-Tuning)

Eğitimi başlatmak için aşağıdaki komutu çalıştırın:

```bash
python main.py --config config.yaml
```

*   **İpucu:** Eğer `config.yaml` dosyasında `resume_from_checkpoint: true` ise ve `output_dir` içinde daha önce alınmış bir kayıt varsa, eğitim kaldığı yerden devam eder.
*   Eğitim sırasında loglar ekrana ve `training.log` dosyasına basılır.

## 🧪 Test ve Karşılaştırma

### 1. API Sunucusu (FastAPI)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Gradio Test Arayüzü
Kolay test için Gradio web arayüzü:
```bash
python gradio_app.py
```
Tarayıcıda `http://localhost:7002` adresini açın.

### 3. Manuel API Testi
```bash
curl -X POST "http://localhost:8000/compare" \
     -H "Content-Type: application/json" \
     -d '{"question": "Python nedir?"}'
```

## 📂 Klasör Yapısı

*   `src/`: Kaynak kodlar (Trainer, Config, Utils, vb.)
*   `main.py`: Eğitim başlatma dosyası.
*   `config.yaml`: Ayar dosyası.
*   `dataset.json`: Eğitim veriseti.
*   `requirements.txt`: Python kütüphaneleri.
