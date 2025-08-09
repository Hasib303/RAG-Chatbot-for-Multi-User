import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Optional


class ModelHandler:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device: str = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_model(self):
        try:
            print(f"Loading model {self.model_name} on {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.models_dir,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.models_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        if not self.generator:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            outputs = self.generator(
                text,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            response = outputs[0]['generated_text'].strip()
            return self._clean_response(response)
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response."
    
    def _clean_response(self, response: str) -> str:
        response = response.replace("<|im_end|>", "").strip()
        if response.startswith("assistant\n"):
            response = response[10:].strip()
        return response
    
    def generate_with_context(self, context: str, question: str, max_length: int = 512) -> str:
        prompt = f"""Based on the following context, please answer the question.

Context: {context}

Question: {question}

Answer:"""
        
        return self.generate_response(prompt, max_length)
    
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None