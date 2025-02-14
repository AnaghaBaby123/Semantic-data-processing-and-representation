import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json

# Initialize model and tokenizer
MODEL_PATH = "gokul-pv/Llama-3.2-1B-Instruct-16bit-TeSO"

def load_model():
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_PATH,
    #     torch_dtype=torch.float16,
    #     device_map="cuda"
    # )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
    )

    
    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    
    # Optimize for inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

class CustomTextStreamer:
    """Custom streamer that captures only the model's response"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_text = []
        self.next_tokens_are_prompt = True
        
    def put(self, value):
        if isinstance(value, torch.Tensor):
            if len(value.shape) > 1:
                value = value[0]
            decoded_text = self.tokenizer.decode(value.tolist(), skip_special_tokens=True)
        else:
            decoded_text = value
            
        if self.next_tokens_are_prompt:
            # Skip the prompt tokens
            self.next_tokens_are_prompt = False
        else:
            self.generated_text.append(decoded_text)
            print(decoded_text, end="", flush=True)
    
    def end(self):
        self.next_tokens_are_prompt = True
        print("")
    
    def get_generated_text(self):
        return "".join(self.generated_text)

def analyze_architecture(code_input, temperature=1.5, max_tokens=512):
    """
    Analyze architecture code using the loaded model
    """
    model, tokenizer = load_model()
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert in analyzing system architecture written using code. "
                      "You check the architecture and provide clear and detailed explanations "
                      "regarding how the architecture can improved for better performance, "
                      "scalability, maintainability and cost effectiveness. You also check "
                      "for possible cyber security issues and if the components can be "
                      "replaced with newer and better components."
        },
        {
            "role": "user",
            "content": code_input
        }
    ]
    
    # Tokenize input with chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Initialize text streamer for real-time output
    # text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    text_streamer = CustomTextStreamer(tokenizer)
    
    # Generate response
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=temperature,
            min_p=0.1
        )
    
    # Return only the generated response
    return text_streamer.get_generated_text()

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Tech stack Optimizer - TeSO")
        
        with gr.Row():
            with gr.Column():
                code_input = gr.Code(
                    label="Input Architecture Code",
                    language="python",
                    lines=10
                )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.5,
                        label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=512,
                        step=64,
                        label="Max Tokens"
                    )
                
                submit_btn = gr.Button("Analyze Architecture")
            
            with gr.Column():
                output = gr.Markdown(label="Analysis Results")
        
        submit_btn.click(
            fn=analyze_architecture,
            inputs=[code_input, temperature, max_tokens],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        share=True,  # Enable sharing
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860  # Default Gradio port
    )