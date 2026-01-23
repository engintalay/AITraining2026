import gradio as gr
import os
from src.config import AppConfig
from src.evaluator import Evaluator

# Global evaluator
evaluator = None

def initialize_evaluator():
    global evaluator
    config_path = os.getenv("CONFIG_PATH", "config_gtx1650.yaml")
    if os.path.exists(config_path):
        try:
            config = AppConfig.load_from_yaml(config_path)
            evaluator = Evaluator(config)
            return "✅ Model yüklendi!"
        except Exception as e:
            return f"❌ Model yüklenemedi: {str(e)}"
    else:
        return f"❌ Config dosyası bulunamadı: {config_path}"

def compare_models(question):
    if not evaluator:
        return "❌ Model yüklenmedi", "❌ Model yüklenmedi"
    
    try:
        result = evaluator.compare(question)
        base_answer = result["base_model"]["answer"]
        finetuned_answer = result["finetuned_model"]["answer"]
        base_time = result["base_model"]["response_time_ms"]
        finetuned_time = result["finetuned_model"]["response_time_ms"]
        
        return (
            f"**Base Model** ({base_time}ms):\n{base_answer}",
            f"**Fine-tuned Model** ({finetuned_time}ms):\n{finetuned_answer}"
        )
    except Exception as e:
        error_msg = f"❌ Hata: {str(e)}"
        return error_msg, error_msg

with gr.Blocks(title="LLM Model Comparison") as demo:
    gr.Markdown("# 🤖 LLM Model Comparison Tool")
    
    # Model initialization
    with gr.Row():
        init_btn = gr.Button("Model Yükle", variant="secondary")
        status_text = gr.Textbox(label="Durum", value="Model yüklenmedi", interactive=False)
    
    init_btn.click(fn=initialize_evaluator, outputs=[status_text])
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Soru", 
            placeholder="Sorunuzu buraya yazın...",
            lines=3
        )
    
    compare_btn = gr.Button("Modelleri Karşılaştır", variant="primary")
    
    with gr.Row():
        base_output = gr.Textbox(label="Base Model Cevabı", lines=10)
        finetuned_output = gr.Textbox(label="Fine-tuned Model Cevabı", lines=10)
    
    compare_btn.click(
        fn=compare_models,
        inputs=[question_input],
        outputs=[base_output, finetuned_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7003, share=False)
