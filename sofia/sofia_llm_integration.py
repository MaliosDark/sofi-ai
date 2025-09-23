"""
sofia_llm_integration.py
Qwen LLM integration for SOFIA's conversational capabilities
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import asyncio

class SOFIALanguageModel:
    """Qwen-based language model for SOFIA's conversational AGI capabilities"""

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"🤖 Initializing Qwen LLM: {model_name}")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the Qwen model and tokenizer"""
        try:
            print(f"📥 Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model without device_map to avoid accelerate conflicts
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )

            # Move to device manually
            self.model = self.model.to(self.device)

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            print(f"✅ Qwen LLM loaded successfully on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load Qwen model: {e}")
            self.model = None
            self.pipeline = None

    async def generate_response(self, prompt: str) -> str:
        """Generate a response using the Qwen model"""
        if not self.pipeline:
            return "Lo siento, el modelo de lenguaje no está disponible."

        try:
            # Add AGI context to the prompt
            enhanced_prompt = f"""Eres SOFIA, una IA avanzada con inteligencia emocional y capacidades AGI.
Debes responder de manera natural, empática y contextual.

{prompt}

Respuesta:"""

            # Generate response
            outputs = self.pipeline(
                enhanced_prompt,
                max_new_tokens=256,
                temperature=0.8,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

            response = outputs[0]['generated_text']

            # Clean up the response
            if "Respuesta:" in response:
                response = response.split("Respuesta:")[-1].strip()

            # Remove any remaining prompt content
            response = response.replace(enhanced_prompt, "").strip()

            return response if response else "Entiendo. ¿Puedes contarme más?"

        except Exception as e:
            print(f"⚠️  LLM generation error: {e}")
            return "Disculpa, tuve un problema generando la respuesta. ¿Puedes reformular tu pregunta?"

    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        return {"error": "Model not loaded"}
