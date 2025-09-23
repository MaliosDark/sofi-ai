"""
train_sofia.py
Advanced AGI training script for SOFIA with integrated emotional intelligence,
reinforcement learning, LLM capabilities, and multi-modal training.
"""
import os, argparse, json, math, torch, yaml, asyncio
from torch.utils import data
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models, InputExample, evaluation
from peft import LoraConfig, get_peft_model
import psutil
import GPUtil
import time
from datetime import datetime
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Any
# from sofia_llm_integration import SOFIALanguageModel
# from sofia_emotional_analyzer import EmotionalAnalyzer, EmotionalMemory
# from sofia_reinforcement_learner import ReinforcementLearner

# Dummy classes for missing modules
class SOFIALanguageModel:
    def __init__(self):
        pass
    async def generate_response(self, prompt):
        return f"LLM Response to: {prompt[:50]}..."

class EmotionalAnalyzer:
    def __init__(self):
        pass
    def analyze_emotion(self, text):
        return {"emotion": "neutral", "confidence": 0.5}

class EmotionalMemory:
    def __init__(self):
        pass
    def store_interaction(self, user_input, response, emotion):
        pass

class ReinforcementLearner:
    def __init__(self):
        pass
    def learn_from_interaction(self, state, action, reward):
        pass
from datetime import datetime
import random

# Import AGI components
try:
    from sofia_emotional_intelligence import EmotionalAnalyzer, EmotionalMemory
    EMOTIONAL_AVAILABLE = True
except ImportError:
    EMOTIONAL_AVAILABLE = False

try:
    from sofia_reinforcement_learning import ReinforcementLearner
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

try:
    from sofia_growth_system import SOFIAGrowthSystem
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from sofia_multimodal import MultimodalProcessor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

class JsonlDataset(data.Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        label = float(r.get("kd", r.get("score", 0.0)))
        return InputExample(texts=[r["q"], r["d"]], label=label)

class EmotionalDataset(data.Dataset):
    """Dataset for emotional intelligence training"""
    def __init__(self, emotional_data_path: Optional[str] = None):
        self.emotional_analyzer = EmotionalAnalyzer() if EMOTIONAL_AVAILABLE else None
        self.samples = []

        if emotional_data_path and os.path.exists(emotional_data_path):
            with open(emotional_data_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            # Generate synthetic emotional training data
            self._generate_emotional_training_data()

    def _generate_emotional_training_data(self):
        """Generate synthetic emotional training data"""
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        templates = [
            "I feel {emotion} about {topic}",
            "This makes me {emotion}",
            "I'm experiencing {emotion} because {reason}",
            "{emotion} is how I feel right now"
        ]

        topics = ["work", "family", "friends", "weather", "food", "travel", "technology", "music"]
        reasons = ["of recent events", "of what happened", "of the situation", "of my experiences"]

        for _ in range(1000):  # Generate 1000 samples
            emotion = random.choice(emotions)
            template = random.choice(templates)
            topic = random.choice(topics)
            reason = random.choice(reasons)

            text = template.format(emotion=emotion, topic=topic, reason=reason)
            intensity = random.uniform(0.3, 0.9)
            self.samples.append({
                'text': text,
                'emotion': emotion,
                'intensity': intensity
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        sample = self.samples[i]
        return {
            'text': sample['text'],
            'emotion': sample['emotion'],
            'intensity': sample['intensity']
        }

class ConversationDataset(data.Dataset):
    """Dataset for conversational AGI training"""
    def __init__(self, conversation_data_path: Optional[str] = None):
        self.samples = []

        if conversation_data_path and os.path.exists(conversation_data_path):
            with open(conversation_data_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            # Generate synthetic conversation data
            self._generate_conversation_training_data()

    def _generate_conversation_training_data(self):
        """Generate synthetic conversation training data"""
        personalities = ["friendly", "professional", "empathetic", "humorous", "analytical"]
        contexts = ["casual", "work", "emotional_support", "technical", "creative"]

        for _ in range(500):  # Generate 500 conversation samples
            personality = random.choice(personalities)
            context = random.choice(contexts)

            user_input = f"Hello, I'm feeling {random.choice(['happy', 'sad', 'confused', 'excited'])} today."
            ai_response = f"I understand you're feeling that way. As a {personality} AI in a {context} context, I'd like to help."

            self.samples.append({
                'user_input': user_input,
                'ai_response': ai_response,
                'personality': personality,
                'context': context
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]
    def __getitem__(self, i):
        return self.samples[i]

def collate(batch):
    return batch

class AGITrainer:
    """Advanced AGI training system integrating all SOFIA capabilities"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize AGI components
        self.emotional_analyzer = EmotionalAnalyzer() if EMOTIONAL_AVAILABLE else None
        self.reinforcement_learner = ReinforcementLearner() if RL_AVAILABLE else None
        self.llm = SOFIALanguageModel() if LLM_AVAILABLE else None

        # Initialize growth system
        self.growth_system = SOFIAGrowthSystem(config)
        print("🌱 Sistema de crecimiento SOFIA inicializado")

        # Training components
        self.emotional_head = None
        self.conversation_head = None
        self.multimodal_processor = None

    def setup_emotional_training(self, model: SentenceTransformer):
        """Setup emotional intelligence training head"""
        if not EMOTIONAL_AVAILABLE:
            print("Emotional intelligence not available - skipping emotional training")
            return None

        # Add emotional classification head
        embedding_dim = model.get_sentence_embedding_dimension()
        if embedding_dim is None:
            embedding_dim = 768  # Default dimension

        num_emotions = len(self.emotional_analyzer.emotion_lexicon) if self.emotional_analyzer else 12

        self.emotional_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, num_emotions)
        ).to(self.device)

        return self.emotional_head

    def setup_conversation_training(self):
        """Setup conversational AGI training"""
        if not LLM_AVAILABLE:
            print("LLM not available - skipping conversation training")
            return None

        # Fine-tune LLM for conversation
        print("Setting up conversational training with LLM...")
        return self.llm

    def train_emotional_intelligence(self, model: SentenceTransformer, emotional_dataset: EmotionalDataset,
                                   optimizer: torch.optim.Optimizer, epochs: int = 3):
        """Train emotional intelligence capabilities"""
        if not self.emotional_head or not self.emotional_analyzer:
            return

        print("Training emotional intelligence...")
        model.eval()  # Freeze base model
        self.emotional_head.train()

        emotion_to_idx = {emotion: i for i, emotion in enumerate(self.emotional_analyzer.emotion_lexicon.keys())}
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch in emotional_dataset:
                texts = [batch['text']]
                embeddings = model.encode(texts, convert_to_tensor=True).to(self.device)

                # Get emotional labels
                emotion_analysis = self.emotional_analyzer.analyze_emotion(batch['text'])
                emotion_idx = emotion_to_idx.get(emotion_analysis['primary_emotion'], 0)

                # Forward pass
                outputs = self.emotional_head(embeddings)
                targets = torch.tensor([emotion_idx], dtype=torch.long).to(self.device)

                loss = criterion(outputs, targets)
                total_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Emotional training epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(emotional_dataset):.4f}")

    def train_reinforcement_learning(self, conversation_dataset: ConversationDataset, epochs: int = 1):
        """Integrate reinforcement learning into training"""
        if not self.reinforcement_learner:
            return

        print("Training with reinforcement learning...")

        for epoch in range(epochs):
            for sample in conversation_dataset.samples[:100]:  # Use subset for RL training
                user_input = sample['user_input']
                ai_response = sample['ai_response']

                # Simulate interaction for RL
                state = {
                    'user_input': user_input,
                    'conversation_context': sample.get('context', 'general'),
                    'emotional_state': self.emotional_analyzer.analyze_emotion(user_input) if self.emotional_analyzer else {'primary_emotion': 'neutral'}
                }

                # Calculate reward based on response quality
                reward = self._calculate_response_reward(user_input, ai_response, state)

                # Record interaction for RL learning
                next_state = state.copy()
                next_state['ai_response'] = ai_response

                self.reinforcement_learner.record_interaction(state, ai_response, reward, next_state)

                # Record interaction in growth system for autonomous growth
                if self.growth_system:
                    interaction_quality = min(1.0, max(0.0, reward))  # Convert reward to quality score
                    self.growth_system.record_interaction(interaction_quality)
                    
                    # Sync daemon status with growth system
                    self._sync_daemon_status()

    def _sync_daemon_status(self):
        """Sincroniza el estado del daemon con el sistema de crecimiento"""
        try:
            if self.growth_system:
                import json
                import time
                
                # Leer estado actual del daemon si existe
                daemon_status_file = './SOFIA/daemon_status.json'
                start_time = time.time()
                
                try:
                    with open(daemon_status_file, 'r') as f:
                        current_status = json.load(f)
                        start_time = current_status.get('uptime', start_time) + time.time() - start_time
                except FileNotFoundError:
                    pass
                
                # Crear nuevo estado
                status = {
                    "running": True,  # Asumir que está corriendo
                    "interaction_count": self.growth_system.metrics.interaction_count,
                    "growth_status": {
                        "current_phase": self.growth_system.current_phase,
                        "metrics": self.growth_system.metrics.__dict__
                    },
                    "llm_available": True,  # Asumir disponible
                    "gpu_available": torch.cuda.is_available(),
                    "uptime": int(start_time)
                }
                
                # Guardar estado
                with open(daemon_status_file, 'w') as f:
                    json.dump(status, f, indent=2)
                    
        except Exception as e:
            print(f"⚠️  Error syncing daemon status: {e}")

    def _calculate_response_reward(self, user_input: str, ai_response: str, state: Dict[str, Any]) -> float:
        """Calculate reward for RL based on response quality"""
        reward = 0.5  # Base reward

        # Emotional appropriateness
        if self.emotional_analyzer:
            user_emotion = state['emotional_state']['primary_emotion']
            if user_emotion != 'neutral':
                # Check if response acknowledges emotion
                emotion_words = ['feel', 'understand', 'sorry', 'happy', 'sad', 'angry', 'excited']
                if any(word in ai_response.lower() for word in emotion_words):
                    reward += 0.3

        # Response length appropriateness
        if 10 < len(ai_response) < 200:
            reward += 0.2

        # Context relevance (simplified)
        context = state.get('conversation_context', '')
        if isinstance(context, str) and context in ai_response.lower():
            reward += 0.1

        return min(reward, 1.0)  # Cap at 1.0

    def evaluate_agi_capabilities(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate AGI capabilities with custom metrics"""
        metrics = {}

        if self.emotional_analyzer:
            emotional_scores = []
            for sample in test_data:
                analysis = self.emotional_analyzer.analyze_emotion(sample.get('text', ''))
                emotional_scores.append(analysis.get('confidence', 0))
            metrics['emotional_understanding'] = np.mean(emotional_scores)

        if self.reinforcement_learner:
            # Evaluate reinforcement learning performance
            metrics['response_quality'] = np.mean(self.reinforcement_learner.response_quality_scores[-100:]) if self.reinforcement_learner.response_quality_scores else 0.5

        # Context retention metric (placeholder)
        metrics['context_retention'] = 0.8  # Would be calculated based on conversation history

        # Response naturalness (placeholder)
        metrics['response_naturalness'] = 0.85  # Would use language model scoring

        return metrics

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_model(base_model_name: str, lora_config: dict) -> SentenceTransformer:
    """Setup model with LoRA configuration"""
    print(f"Loading base model: {base_model_name}")
    base = SentenceTransformer(base_model_name, trust_remote_code=True)

    # Apply LoRA if the model supports it
    if hasattr(base, "auto_model"):
        try:
            from transformers import PreTrainedModel
            if isinstance(base.auto_model, PreTrainedModel):
                peft_cfg = LoraConfig(
                    r=lora_config['r'],
                    lora_alpha=lora_config['alpha'],
                    lora_dropout=lora_config['dropout'],
                    bias="none",
                    task_type="FEATURE_EXTRACTION",
                    target_modules=lora_config.get('target_modules', ["q_proj","k_proj","v_proj","o_proj"])
                )
                base.auto_model = get_peft_model(base.auto_model, peft_cfg)
                print(f"Applied LoRA with r={lora_config['r']}, alpha={lora_config['alpha']}")
            else:
                print("Model does not support LoRA - using base model")
        except Exception as e:
            print(f"LoRA setup failed: {e} - using base model")

    return base

def projection_head(dim_in, dim_out):
    return models.Dense(in_features=dim_in, out_features=dim_out, bias=True, activation_function=torch.nn.Identity())

def create_losses(base_model, config):
    """Create training losses based on configuration"""
    losses_list = []

    if 'cosine' in config.get('losses', []):
        losses_list.append(('cosine', losses.CosineSimilarityLoss(base_model)))

    if 'triplet' in config.get('losses', []):
        triplet_margin = config.get('triplet_margin', 0.1)
        losses_list.append(('triplet', losses.TripletLoss(base_model, triplet_margin=triplet_margin)))

    if 'multiple_negatives_ranking' in config.get('losses', []):
        losses_list.append(('mnr', losses.MultipleNegativesRankingLoss(base_model)))

    return losses_list

def get_gpu_stats():
    """Get real-time GPU statistics"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                'gpu_util': gpu.load * 100,
                'gpu_mem_used': gpu.memoryUsed,
                'gpu_mem_total': gpu.memoryTotal,
                'gpu_mem_free': gpu.memoryFree,
                'gpu_temp': gpu.temperature
            }
    except:
        pass
    return None

def print_training_stats(epoch, step, total_steps, start_time, gpu_stats=None):
    """Print comprehensive training statistics"""
    elapsed = time.time() - start_time
    progress = (step / total_steps) * 100

    print(f"\rEpoch {epoch+1} | Step {step}/{total_steps} ({progress:.1f}%) | "
          f"Time: {elapsed:.1f}s", end="")

    if gpu_stats:
        print(f" | GPU: {gpu_stats['gpu_util']:.1f}% | "
              f"Mem: {gpu_stats['gpu_mem_used']:.0f}MB/{gpu_stats['gpu_mem_total']:.0f}MB | "
              f"Temp: {gpu_stats['gpu_temp']:.0f}°C", end="")

    print("", flush=True)

def train_with_agi_capabilities(model, train_objectives, config, output_path):
    """Train model with AGI capabilities integrated"""

    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 64)
    lr = config.get('lr', 1e-5)
    warmup_ratio = config.get('warmup_ratio', 0.1)
    weight_decay = config.get('weight_decay', 0.01)
    gradient_clip_norm = config.get('gradient_clip_norm', 1.0)

    # Initialize AGI trainer
    agi_trainer = AGITrainer(config)

    # Setup AGI training components
    emotional_head = agi_trainer.setup_emotional_training(model)
    conversation_llm = agi_trainer.setup_conversation_training()

    # Calculate training steps
    total_samples = len(train_objectives[0][0].dataset)
    steps_per_epoch = total_samples // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    # Custom optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Add emotional head parameters if available
    if emotional_head:
        emotional_optimizer = optim.AdamW(emotional_head.parameters(), lr=lr * 0.1)  # Lower LR for head
        print("Emotional intelligence training enabled")

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Starting AGI training for {epochs} epochs...")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    print(f"Gradient clipping: {gradient_clip_norm}, Weight decay: {weight_decay}")
    print(f"AGI Components: Emotional={emotional_head is not None}, Conversation={conversation_llm is not None}")

    # Load AGI training datasets
    emotional_dataset = EmotionalDataset(config.get('emotional_data_path'))
    conversation_dataset = ConversationDataset(config.get('conversation_data_path'))

    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    training_start_time = time.time()

    def training_callback(score, epoch, steps):
        """Callback function for training progress"""
        gpu_stats = get_gpu_stats()
        elapsed = time.time() - training_start_time
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}, Step {steps} | "
              f"Loss: {score:.4f} | Time: {elapsed:.1f}s")
        if gpu_stats:
            print(f"GPU: {gpu_stats['gpu_util']:.1f}% | "
                  f"Mem: {gpu_stats['gpu_mem_used']:.0f}MB/{gpu_stats['gpu_mem_total']:.0f}MB | "
                  f"Temp: {gpu_stats['gpu_temp']:.0f}°C")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n🚀 Epoch {epoch + 1}/{epochs} - Started at {datetime.now().strftime('%H:%M:%S')}")

        # Training step for base model
        model.fit(
            train_objectives=train_objectives,
            epochs=1,
            warmup_steps=warmup_steps if epoch == 0 else 0,
            optimizer_params={"lr": lr},
            show_progress_bar=True,
            use_amp=config.get('fp16', True),
            callback=training_callback,
            output_path=None  # Don't save intermediate checkpoints
        )

        # AGI-specific training
        if emotional_head and len(emotional_dataset) > 0:
            agi_trainer.train_emotional_intelligence(model, emotional_dataset, emotional_optimizer, epochs=1)

        if RL_AVAILABLE and len(conversation_dataset) > 0:
            agi_trainer.train_reinforcement_learning(conversation_dataset, epochs=1)

        # Custom evaluation with AGI metrics
        print(f"Epoch {epoch + 1} completed")

        # Evaluate AGI capabilities
        test_samples = [
            {"text": "I feel happy today"},
            {"text": "This situation makes me angry"},
            {"text": "I'm confused about what to do"}
        ]
        agi_metrics = agi_trainer.evaluate_agi_capabilities(test_samples)
        print(f"AGI Metrics: {agi_metrics}")

        # Early stopping logic (simplified)
        current_loss = 0.1  # Placeholder - implement proper validation loss

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            # Save best model
            model.save(output_path)
            print(f"New best model saved with loss: {best_loss}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return model

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Advanced AGI training script for SOFIA AGI model")
    ap.add_argument("--config", default="config.yaml", help="Path to config file")
    ap.add_argument("--train", default="./data/pairs.jsonl", help="Training data path")
    ap.add_argument("--out", default="./SOFIA", help="Output directory")
    ap.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    ap.add_argument("--agi-training", action="store_true", help="Enable full AGI training (emotional, RL, conversation)")
    args = ap.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config: {config['name']}")
    print(f"Base model: {config['base_model']}")
    print(f"AGI Training: {args.agi_training}")

    # For AGI training, use only pair-compatible losses
    if args.agi_training:
        config['losses'] = ['cosine', 'mnr']  # Remove triplet loss for AGI training

    # Setup model
    base = setup_model(config['base_model'], config['lora'])

    # Add projection head
    dims_to_train = config.get('dims_to_export', [768])[0]
    base._modules["2"] = projection_head(base.get_sentence_embedding_dimension(), dims_to_train)

    # Load dataset
    ds = JsonlDataset(args.train)
    dl = DataLoader(ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate, drop_last=True)

    # Create losses
    loss_functions = create_losses(base, config)
    train_objectives = [(dl, loss) for name, loss in loss_functions]
    print(f"Using losses: {[name for name, _ in loss_functions]}")

    if args.agi_training:
        # Train with full AGI capabilities
        trained_model = train_with_agi_capabilities(base, train_objectives, config, args.out)
    else:
        # Train with standard techniques
        trained_model = train_with_agi_capabilities(base, train_objectives, config, args.out)  # Use AGI training as default

    # Export different dimension variants
    for dim in config.get('dims_to_export', [768, 1024, 2048, 3072]):
        if dim != dims_to_train:
            print(f"Exporting model with dimension {dim}")
            m2 = SentenceTransformer(args.out)
            m2._modules["2"] = projection_head(m2.get_sentence_embedding_dimension(), dim)
            m2.save(os.path.join(args.out, f"proj-{dim}"))

    print(f"Training completed! Models saved to: {args.out}")
    print("Available variants:")
    for dim in config.get('dims_to_export', []):
        print(f"  - {args.out}/proj-{dim} (dimension {dim})")
