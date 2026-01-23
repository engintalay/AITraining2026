#!/bin/bash
deactivate
source venv/bin/activate

echo "🔍 GPU kontrolü yapılıyor..."

# GPU VRAM kontrolü (MB cinsinden)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

if [ -z "$GPU_VRAM" ]; then
    echo "❌ NVIDIA GPU bulunamadı!"
    exit 1
fi

echo "📊 GPU VRAM: ${GPU_VRAM}MB"

# 4GB = 4096MB threshold
if [ "$GPU_VRAM" -le 4096 ]; then
    echo "⚡ Düşük VRAM tespit edildi (≤4GB). GTX1650 config kullanılıyor..."
    CONFIG_FILE="config_gtx1650.yaml"
else
    echo "🚀 Yeterli VRAM tespit edildi (>4GB). Normal config kullanılıyor..."
    CONFIG_FILE="config.yaml"
fi

echo "🎯 Kullanılan config: $CONFIG_FILE"
python main.py --config $CONFIG_FILE
