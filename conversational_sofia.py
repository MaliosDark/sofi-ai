#!/usr/bin/env python3
"""
conversational_sofia.py
Conversational SOFIA with memory and context awareness.
Towards AGI: Maintains conversation history, adapts responses, learns from interactions.
"""
import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

class ConversationalSOFIA:
    def __init__(self, model_path="./SOFIA-v2-lora", prompts_path="prompts.json"):
        print("🧠 Initializing Conversational SOFIA...")
        self.model = SentenceTransformer(model_path)
        self.prompts = json.load(open(prompts_path))
        self.conversation_history = []
        self.memory_file = "sofia_memory.json"
        self.load_memory()
        print("✅ SOFIA ready for conversation!")

    def load_memory(self):
        """Load conversation memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.conversation_history = data.get('history', [])
                    print(f"📚 Loaded {len(self.conversation_history)} previous interactions")
            except:
                print("⚠️  Could not load memory, starting fresh")

    def save_memory(self):
        """Save conversation memory to file"""
        try:
            data = {
                'history': self.conversation_history[-50:],  # Keep last 50 interactions
                'last_updated': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            print("⚠️  Could not save memory")

    def add_to_history(self, user_input, sofia_response, embedding):
        """Add interaction to conversation history"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'sofia': sofia_response,
            'embedding': embedding.tolist()[:10],  # Store first 10 dims for memory
            'context_length': len(self.conversation_history)
        }
        self.conversation_history.append(interaction)

    def get_conversation_context(self, max_turns=3):
        """Get recent conversation context"""
        recent = self.conversation_history[-max_turns:]
        if not recent:
            return ""

        context = "\n".join([
            f"User: {turn['user']}\nSOFIA: {turn['sofia']}"
            for turn in recent
        ])
        return f"Previous conversation:\n{context}\n\n"

    def generate_embedding(self, text, mode="query"):
        """Generate embedding with conversation context"""
        context = self.get_conversation_context()

        # Create contextual prompt
        if context:
            full_text = f"{context}Current query: {text}"
        else:
            full_text = text

        # Add mode-specific prefix
        pref = self.prompts.get(mode, "")
        contextual_text = f"{pref}{full_text}"

        print(f"🧠 Processing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        if context:
            print(f"📝 With {len(self.conversation_history)} turns of context")

        embedding = self.model.encode([contextual_text], normalize_embeddings=self.prompts["normalize"])[0]
        return embedding

    def analyze_conversation_patterns(self):
        """Analyze conversation patterns for AGI insights"""
        if len(self.conversation_history) < 3:
            return "Need more conversations for analysis"

        # Simple pattern analysis
        user_lengths = [len(turn['user'].split()) for turn in self.conversation_history]
        avg_user_length = np.mean(user_lengths)

        embeddings = np.array([turn['embedding'] for turn in self.conversation_history])
        embedding_variance = np.var(embeddings, axis=0).mean()

        return {
            'total_interactions': len(self.conversation_history),
            'avg_user_query_length': avg_user_length,
            'embedding_variance': embedding_variance,
            'conversation_depth': len(self.conversation_history) / max(1, len(set([turn['user'] for turn in self.conversation_history])))
        }

    def chat(self, user_input):
        """Main chat function with AGI features"""
        # Generate contextual embedding
        embedding = self.generate_embedding(user_input)

        # Simple response generation (could be enhanced with LLM)
        response = self.generate_response(user_input, embedding)

        # Add to history
        self.add_to_history(user_input, response, embedding)
        self.save_memory()

        # AGI insights
        if len(self.conversation_history) % 5 == 0:
            insights = self.analyze_conversation_patterns()
            print(f"\n🧠 AGI Insights: {insights}")

        return response, embedding

    def generate_response(self, user_input, embedding):
        """Generate response (placeholder - could integrate with LLM)"""
        # For now, just acknowledge and show embedding info
        emb_stats = {
            'magnitude': np.linalg.norm(embedding),
            'sparsity': np.mean(np.abs(embedding) < 0.01),
            'max_val': np.max(np.abs(embedding))
        }

        response = f"Understood: '{user_input}'\n"
        response += f"Embedding stats: magnitude={emb_stats['magnitude']:.3f}, "
        response += f"sparsity={emb_stats['sparsity']:.3f}, "
        response += f"max={emb_stats['max_val']:.3f}"

        if len(self.conversation_history) > 0:
            response += f"\nConversation turn #{len(self.conversation_history) + 1}"

        return response

def main():
    # Get input from args or stdin
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
    else:
        user_input = sys.stdin.read().strip()

    if not user_input:
        print("Usage: python conversational_sofia.py 'your message'")
        print("Or: echo 'message' | python conversational_sofia.py")
        sys.exit(1)

    # Initialize and chat
    sofia = ConversationalSOFIA()
    response, embedding = sofia.chat(user_input)

    print(f"\n🤖 SOFIA: {response}")
    print(f"\n🔢 Embedding (first 10): {embedding[:10]}")

if __name__ == "__main__":
    main()
