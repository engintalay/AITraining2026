import gradio as gr
import requests
import json

def compare_models(question):
    try:
        response = requests.post(
            "http://localhost:8000/compare",
            json={"question": question},
            timeout=120
        )
        if response.status_code == 200:
            data = response.json()
            base_answer = data["base_model"]["answer"]
            finetuned_answer = data["finetuned_model"]["answer"]
            base_time = data["base_model"]["response_time_ms"]
            finetuned_time = data["finetuned_model"]["response_time_ms"]
            
            return (
                f"**Base Model** ({base_time}ms):\n{base_answer}",
                f"**Fine-tuned Model** ({finetuned_time}ms):\n{finetuned_answer}"
            )
        else:
            error_msg = f"API Error: {response.status_code}"
            return error_msg, error_msg
    except Exception as e:
        error_msg = f"Connection Error: {str(e)}"
        return error_msg, error_msg

with gr.Blocks(title="LLM Model Comparison") as demo:
    gr.Markdown("# 🤖 LLM Model Comparison Tool")
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Question", 
            placeholder="Enter your question here...",
            lines=3
        )
    
    compare_btn = gr.Button("Compare Models", variant="primary")
    
    with gr.Row():
        base_output = gr.Textbox(label="Base Model Response", lines=10)
        finetuned_output = gr.Textbox(label="Fine-tuned Model Response", lines=10)
    
    compare_btn.click(
        fn=compare_models,
        inputs=[question_input],
        outputs=[base_output, finetuned_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7002, share=False)
