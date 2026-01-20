#!/bin/bash
deactivate
source venv/bin/activate
# GPU kontrolünü yap ve sonucu bir değişkene ata
GPU_CHECK=$(lspci | grep -i 'GeForce GTX 1650 Mobile' | wc -l)
# Eğer sonuç 1 ise GTX 1650 konfigürasyonuyla çalıştır
if [ "$GPU_CHECK" -eq 1 ]; then
    echo "GTX 1650 Mobile tespit edildi. Özel konfigürasyon yükleniyor..."
    python main.py --config config_gtx1650.yaml
else
    echo "GTX 1650 bulunamadı. Varsayılan konfigürasyon yükleniyor..."
    python main.py --config config.yaml
fi
