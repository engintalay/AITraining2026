#!/bin/bash
source venv/bin/activate
export CONFIG_PATH="config_gtx1650.yaml"
echo "🚀 API Sunucusu başlatılıyor..."
echo "📱 Web arayüzü: http://localhost:8000/web"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

