#!/usr/bin/env python3
"""
SOFIA LLM Integration Module
Provides natural language generation capabilities for enhanced emotional responses
"""

import os
import json
import datetime
import asyncio
import random
from typing import Dict, List, Optional, Any

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import groq
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SOFIALanguageModel:
    """LLM integration for SOFIA's natural language generation"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.llm_provider = self._auto_detect_provider()
        self.api_key = self._get_api_key()
        self.model_name = self._get_model_name()

        # Initialize the appropriate LLM client
        self.client = self._initialize_client()

        # Response caching for performance
        self.response_cache = {}
        self.cache_max_size = 100

        # Personality and style settings
        self.personality = {
            'empathy_level': 0.8,
            'humor_level': 0.3,
            'formality_level': 0.6,
            'verbosity_level': 0.7
        }

    def _auto_detect_provider(self) -> str:
        """Automatically detect the best available LLM provider"""
        config_provider = self.config.get('llm_provider', 'auto')

        # If explicitly set to something other than auto, use that
        if config_provider != 'auto':
            return config_provider

        # Auto-detection logic: prefer local models, then cloud APIs
        print("🔍 Detecting available LLM...")

        # 1. Check for local models first (preferred for privacy)
        if TRANSFORMERS_AVAILABLE:
            if self._check_local_model_available():
                print("✅ Local model found")
                return 'local'
            elif self.config.get('auto_download_models', True):
                print("📥 Attempting to download local model...")
                if self._download_default_model():
                    return 'local'

        # 2. Check cloud APIs in order of preference
        if OPENAI_AVAILABLE and self._get_api_key('openai'):
            print("✅ OpenAI API available")
            return 'openai'
        elif ANTHROPIC_AVAILABLE and self._get_api_key('anthropic'):
            print("✅ Anthropic API available")
            return 'anthropic'
        elif GROQ_AVAILABLE and self._get_api_key('groq'):
            print("✅ Groq API available")
            return 'groq'

        # 3. Fallback to enhanced templates
        print("📝 Using enhanced template responses (no LLM available)")
        return 'template'

    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.llm_provider == 'openai' and OPENAI_AVAILABLE and self.api_key:
            return OpenAI(api_key=self.api_key)
        elif self.llm_provider == 'anthropic' and ANTHROPIC_AVAILABLE and self.api_key:
            return Anthropic(api_key=self.api_key)
        elif self.llm_provider == 'groq' and GROQ_AVAILABLE and self.api_key:
            return Groq(api_key=self.api_key)
        elif self.llm_provider == 'local' and TRANSFORMERS_AVAILABLE:
            return self._initialize_local_model()
        elif self.llm_provider == 'template':
            print("📝 Usando respuestas template (sin LLM disponible)")
            return None
        else:
            print(f"⚠️  Proveedor {self.llm_provider} no disponible, usando template")
            return None

    def _initialize_local_model(self):
        """Initialize local HuggingFace model"""
        try:
            model_name = self.model_name or "Qwen/Qwen3-4B-Instruct-2507"
            print(f"🤖 Loading local model: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            # Use GPU if available
            self.device = 0 if torch.cuda.is_available() else -1
            if self.device == 0:
                self.model = self.model.to('cuda')
                print("🚀 Using GPU for local model")
            else:
                print("💻 Using CPU for local model")

            return self.model
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            return None

    def _check_local_model_available(self) -> bool:
        """Check if the configured local model is available"""
        try:
            from huggingface_hub import HfApi
            api = HfApi()

            # Check if the configured model is available locally
            configured_model = self.config.get('model_name', 'Qwen/Qwen3-4B-Instruct-2507')
            model_path = os.path.expanduser(f"~/.cache/huggingface/hub/{configured_model.replace('/', '_')}")
            if os.path.exists(model_path):
                return True

            # If not found locally, check if it exists on HuggingFace
            try:
                api.model_info(configured_model)
                return False  # Exists on HF but not downloaded locally
            except:
                return False  # Model doesn't exist
        except ImportError:
            return False

    def _download_default_model(self) -> bool:
        """Download the default local model"""
        try:
            from huggingface_hub import snapshot_download
            model_name = self.config.get('model_name', 'Qwen/Qwen3-4B-Instruct-2507')
            print(f"📥 Descargando {model_name}...")

            snapshot_download(
                repo_id=model_name,
                local_dir=os.path.expanduser(f"~/.cache/huggingface/hub/{model_name.replace('/', '_')}"),
                local_dir_use_symlinks=False
            )
            print("✅ Modelo descargado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Error cargando config: {e}, usando valores por defecto")
            return {
                'llm_provider': 'auto',
                'auto_download_models': True,
                'model_name': 'Qwen/Qwen3-4B-Instruct-2507'
            }

    def _get_model_name(self) -> str:
        """Get the model name from config"""
        if self.llm_provider == 'local':
            return self.config.get('model_name', 'Qwen/Qwen3-4B-Instruct-2507')
        elif self.llm_provider == 'openai':
            return self.config.get('openai_model', 'gpt-3.5-turbo')
        elif self.llm_provider == 'anthropic':
            return self.config.get('anthropic_model', 'claude-3-haiku-20240307')
        elif self.llm_provider == 'groq':
            return self.config.get('groq_model', 'mixtral-8x7b-32768')
        else:
            return self.config.get('model_name', 'Qwen/Qwen3-4B-Instruct-2507')

    def _get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Get API key for the specified provider"""
        if provider is None:
            provider = self.llm_provider

        # Check environment variables first
        env_vars = {
            'openai': ['OPENAI_API_KEY', 'OPENAI_KEY'],
            'anthropic': ['ANTHROPIC_API_KEY', 'ANTHROPIC_KEY'],
            'groq': ['GROQ_API_KEY', 'GROQ_KEY']
        }

        if provider in env_vars:
            for env_var in env_vars[provider]:
                key = os.getenv(env_var)
                if key:
                    return key

        # Check config file
        config_keys = {
            'openai': 'openai_api_key',
            'anthropic': 'anthropic_api_key',
            'groq': 'groq_api_key'
        }

        if provider in config_keys:
            return self.config.get(config_keys[provider])

        return None

    async def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a natural language response using the configured LLM"""
        # Check cache first
        cache_key = hash(prompt + str(context))
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        try:
            if self.llm_provider == 'openai' and self.client:
                response = await self._generate_openai_response(prompt, context)
            elif self.llm_provider == 'anthropic' and self.client:
                response = await self._generate_anthropic_response(prompt, context)
            elif self.llm_provider == 'groq' and self.client:
                response = await self._generate_groq_response(prompt, context)
            elif self.llm_provider == 'local' and self.client:
                response = await self._generate_local_response(prompt, context)
            else:
                response = self._generate_template_response(prompt, context)

            # Cache the response
            if len(self.response_cache) >= self.cache_max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]
            self.response_cache[cache_key] = response

            return response

        except Exception as e:
            print(f"❌ Error generando respuesta: {e}")
            return self._generate_template_response(prompt, context)

    async def _generate_openai_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using OpenAI API"""
        try:
            messages = self._build_messages(prompt, context)

            response = self.client.chat.completions.create(
                model=self.model_name or "gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error OpenAI: {e}")
            raise

    async def _generate_anthropic_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Anthropic API"""
        try:
            messages = self._build_messages(prompt, context)

            response = self.client.messages.create(
                model=self.model_name or "claude-3-haiku-20240307",
                max_tokens=150,
                temperature=0.7,
                messages=messages
            )

            return response.content[0].text.strip()
        except Exception as e:
            print(f"❌ Error Anthropic: {e}")
            raise

    async def _generate_groq_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Groq API"""
        try:
            messages = self._build_messages(prompt, context)

            response = self.client.chat.completions.create(
                model=self.model_name or "mixtral-8x7b-32768",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error Groq: {e}")
            raise

    async def _generate_local_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using local HuggingFace model"""
        try:
            # Build messages for chat model
            messages = self._build_messages(prompt, context)

            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            if self.device == 0:
                inputs = inputs.to('cuda')

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and clean response
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return response.strip() or "Entiendo lo que dices."
        except Exception as e:
            print(f"❌ Error modelo local: {e}")
            raise

    def _generate_template_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using enhanced templates when no LLM is available"""
        emotion = context.get('emotion', 'neutral') if context else 'neutral'
        topic = context.get('topic', '') if context else ''

        templates = {
            'joy': [
                f"¡Me alegra mucho oír eso! {topic} suena genial.",
                f"¡Qué alegría! Me encanta cuando hablas de {topic}.",
                "¡Estoy tan feliz por ti! Cuéntame más."
            ],
            'sadness': [
                f"Entiendo que estés triste. {topic} puede ser difícil.",
                "Lo siento mucho. Estoy aquí para escucharte.",
                f"Comprendo tu tristeza sobre {topic}. ¿Quieres hablar de ello?"
            ],
            'anger': [
                f"Veo que estás enfadado con {topic}. ¿Qué ha pasado?",
                "Entiendo tu frustración. Cuéntame qué te molesta.",
                f"Comprendo tu enojo sobre {topic}. Estoy aquí para ayudarte."
            ],
            'fear': [
                f"Entiendo tu miedo sobre {topic}. ¿Qué te preocupa exactamente?",
                "No estás solo. ¿Quieres hablar de tus temores?",
                f"Comprendo tu preocupación por {topic}. Estoy aquí para apoyarte."
            ],
            'surprise': [
                f"¡Vaya sorpresa! {topic} es inesperado.",
                "¡Qué sorpresa! Cuéntame más sobre eso.",
                f"¡No me lo esperaba! {topic} suena interesante."
            ],
            'neutral': [
                f"Entiendo. {topic} es un tema interesante.",
                "Comprendo tu punto de vista.",
                f"Me parece bien. ¿Quieres profundizar en {topic}?"
            ]
        }

        emotion_templates = templates.get(emotion, templates['neutral'])
        response = random.choice(emotion_templates)

        # Add personality touches
        if self.personality['humor_level'] > 0.5 and random.random() < 0.3:
            response += " 😊"

        return response

    def _build_messages(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Build messages for LLM APIs"""
        system_message = "Eres SOFIA, una IA empática y amigable. Responde de manera natural, comprensiva y humana. Mantén las respuestas concisas pero significativas."

        if context:
            emotion = context.get('emotion', 'neutral')
            topic = context.get('topic', '')
            system_message += f" El usuario parece estar {emotion} y habla sobre {topic}."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        return messages

    def update_personality(self, trait: str, value: float):
        """Update personality traits"""
        if trait in self.personality:
            self.personality[trait] = max(0.0, min(1.0, value))
            print(f"🎭 Personalidad actualizada: {trait} = {value}")

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'provider': self.llm_provider,
            'model': self.model_name,
            'cache_size': len(self.response_cache),
            'personality': self.personality.copy()
        }
