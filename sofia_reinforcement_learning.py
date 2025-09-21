#!/usr/bin/env python3
"""
SOFIA Real-Time Reinforcement Learning System
Autonomous self-improvement through interaction feedback and continuous learning
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import threading
import time
from collections import defaultdict
import random

class ReinforcementLearner:
    """Real-time reinforcement learning system for SOFIA"""

    def __init__(self, learning_file: str = "reinforcement_learning.json"):
        self.learning_file = learning_file
        self.learning_data = self._load_learning_data()

        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.response_quality_scores = []

        # Auto-improvement triggers
        self.improvement_threshold = 0.7  # Trigger improvement when score drops below this
        self.learning_interval = 300  # Learn every 5 minutes
        self.last_learning_time = datetime.now()

        # Start background learning thread
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.learning_thread.start()

    def _load_learning_data(self) -> Dict:
        """Load reinforcement learning data"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        return {
            'q_table': {},  # State-action values
            'response_patterns': {},  # Successful response patterns
            'user_preferences': {},  # Learned user preferences
            'emotional_responses': {},  # Best responses for emotions
            'conversation_strategies': {},  # Effective conversation strategies
            'performance_history': [],
            'learned_behaviors': {},
            'adaptation_rules': []
        }

    def save_learning_data(self):
        """Save learning data to file"""
        with open(self.learning_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            data_to_save = self.learning_data.copy()
            json.dump(data_to_save, f, indent=2, ensure_ascii=False, default=str)

    def record_interaction(self, state: Dict, action: str, reward: float, next_state: Dict):
        """
        Record an interaction for learning

        Args:
            state: Current state (emotion, context, user_id, etc.)
            action: Action taken (response type, strategy, etc.)
            reward: Reward received (user satisfaction, engagement, etc.)
            next_state: Next state after action
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Initialize Q-values if not exist
        if state_key not in self.learning_data['q_table']:
            self.learning_data['q_table'][state_key] = {}
        if action not in self.learning_data['q_table'][state_key]:
            self.learning_data['q_table'][state_key][action] = 0.0

        # Q-learning update
        current_q = self.learning_data['q_table'][state_key][action]
        next_max_q = max(self.learning_data['q_table'].get(next_state_key, {}).values(), default=0.0)

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.learning_data['q_table'][state_key][action] = new_q

        # Record performance
        self.performance_metrics['rewards'].append(reward)
        self.performance_metrics['actions'].append(action)

        # Update response patterns if reward is positive
        if reward > 0.5:
            self._update_response_patterns(state, action, reward)

        # Update user preferences
        if 'user_id' in state:
            self._update_user_preferences(state['user_id'], state, action, reward)

        # Check for auto-improvement trigger
        if reward < self.improvement_threshold:
            self._trigger_improvement(state, action, reward)

        self.save_learning_data()

    def _state_to_key(self, state: Dict) -> str:
        """Convert state dict to hashable key"""
        # Create a simplified state representation
        key_components = []
        for k, v in sorted(state.items()):
            if isinstance(v, (str, int, float, bool)):
                key_components.append(f"{k}:{v}")
            elif isinstance(v, list):
                key_components.append(f"{k}:{','.join(map(str, v[:3]))}")  # Limit list length
        return "|".join(key_components)

    def _update_response_patterns(self, state: Dict, action: str, reward: float):
        """Update successful response patterns"""
        emotion = state.get('emotion', 'neutral')
        context = state.get('context_type', 'general')

        pattern_key = f"{emotion}_{context}"

        if pattern_key not in self.learning_data['response_patterns']:
            self.learning_data['response_patterns'][pattern_key] = {}

        if action not in self.learning_data['response_patterns'][pattern_key]:
            self.learning_data['response_patterns'][pattern_key][action] = {
                'success_count': 0,
                'total_reward': 0.0,
                'avg_reward': 0.0,
                'last_used': None
            }

        pattern = self.learning_data['response_patterns'][pattern_key][action]
        pattern['success_count'] += 1
        pattern['total_reward'] += reward
        pattern['avg_reward'] = pattern['total_reward'] / pattern['success_count']
        pattern['last_used'] = datetime.now().isoformat()

    def _update_user_preferences(self, user_id: str, state: Dict, action: str, reward: float):
        """Update learned user preferences"""
        if user_id not in self.learning_data['user_preferences']:
            self.learning_data['user_preferences'][user_id] = {
                'preferred_actions': {},
                'emotional_responses': {},
                'conversation_styles': {},
                'interaction_patterns': []
            }

        prefs = self.learning_data['user_preferences'][user_id]

        # Update preferred actions
        if action not in prefs['preferred_actions']:
            prefs['preferred_actions'][action] = {'count': 0, 'avg_reward': 0.0}

        prefs['preferred_actions'][action]['count'] += 1
        current_avg = prefs['preferred_actions'][action]['avg_reward']
        count = prefs['preferred_actions'][action]['count']
        prefs['preferred_actions'][action]['avg_reward'] = (current_avg * (count - 1) + reward) / count

        # Track interaction patterns
        prefs['interaction_patterns'].append({
            'state': state,
            'action': action,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })

        # Keep only last 50 interactions
        if len(prefs['interaction_patterns']) > 50:
            prefs['interaction_patterns'].pop(0)

    def _trigger_improvement(self, state: Dict, action: str, reward: float):
        """Trigger automatic improvement when performance is poor"""
        improvement_action = {
            'type': 'auto_improvement',
            'trigger_state': state,
            'failed_action': action,
            'reward': reward,
            'timestamp': datetime.now().isoformat(),
            'suggested_improvements': self._generate_improvements(state, action, reward)
        }

        self.learning_data['adaptation_rules'].append(improvement_action)

        # Apply immediate improvements
        self._apply_improvements(improvement_action)

    def _generate_improvements(self, state: Dict, action: str, reward: float) -> List[str]:
        """Generate potential improvements based on failure analysis"""
        improvements = []

        emotion = state.get('emotion', 'neutral')

        if reward < 0.3:
            improvements.append("Increase empathy in responses")
            improvements.append("Ask clarifying questions")
            improvements.append("Offer more specific help")

        if emotion in ['sadness', 'anger', 'fear'] and reward < 0.5:
            improvements.append("Improve emotional validation")
            improvements.append("Use more supportive language")

        if 'context_recall' in state and not state['context_recall']:
            improvements.append("Better memory integration")
            improvements.append("Reference previous conversations")

        return improvements

    def _apply_improvements(self, improvement_action: Dict):
        """Apply automatic improvements to the system"""
        improvements = improvement_action.get('suggested_improvements', [])

        for improvement in improvements:
            if improvement == "Increase empathy in responses":
                self._increase_empathy_level()
            elif improvement == "Ask clarifying questions":
                self._enable_clarifying_questions()
            elif improvement == "Offer more specific help":
                self._improve_help_specificity()
            elif improvement == "Improve emotional validation":
                self._enhance_emotional_validation()
            elif improvement == "Use more supportive language":
                self._update_supportive_language()
            elif improvement == "Better memory integration":
                self._improve_memory_integration()
            elif improvement == "Reference previous conversations":
                self._enable_conversation_references()

    def _increase_empathy_level(self):
        """Increase empathy in response generation"""
        self.learning_data['learned_behaviors']['empathy_level'] = \
            self.learning_data['learned_behaviors'].get('empathy_level', 0.5) + 0.1

    def _enable_clarifying_questions(self):
        """Enable more clarifying questions"""
        self.learning_data['learned_behaviors']['clarifying_questions'] = True

    def _improve_help_specificity(self):
        """Improve specificity of help offered"""
        self.learning_data['learned_behaviors']['help_specificity'] = \
            self.learning_data['learned_behaviors'].get('help_specificity', 0.5) + 0.1

    def _enhance_emotional_validation(self):
        """Enhance emotional validation in responses"""
        self.learning_data['learned_behaviors']['emotional_validation'] = True

    def _update_supportive_language(self):
        """Update to use more supportive language"""
        self.learning_data['learned_behaviors']['supportive_language'] = True

    def _improve_memory_integration(self):
        """Improve integration of memory in responses"""
        self.learning_data['learned_behaviors']['memory_integration'] = \
            self.learning_data['learned_behaviors'].get('memory_integration', 0.5) + 0.1

    def _enable_conversation_references(self):
        """Enable referencing previous conversations"""
        self.learning_data['learned_behaviors']['conversation_references'] = True

    def get_best_action(self, state: Dict) -> str:
        """Get the best action for a given state using learned Q-values"""
        state_key = self._state_to_key(state)

        if state_key in self.learning_data['q_table']:
            actions = self.learning_data['q_table'][state_key]

            # Epsilon-greedy exploration
            if random.random() < self.exploration_rate:
                return random.choice(list(actions.keys()))
            else:
                return max(actions, key=actions.get)

        # Default actions based on state
        emotion = state.get('emotion', 'neutral')
        return self._get_default_action(emotion)

    def _get_default_action(self, emotion: str) -> str:
        """Get default action for emotion when no learning data exists"""
        defaults = {
            'joy': 'celebrate',
            'sadness': 'empathize',
            'anger': 'calm',
            'fear': 'reassure',
            'surprise': 'acknowledge',
            'neutral': 'engage'
        }
        return defaults.get(emotion, 'engage')

    def get_user_preferences(self, user_id: str) -> Dict:
        """Get learned preferences for a user"""
        return self.learning_data['user_preferences'].get(user_id, {})

    def get_response_pattern(self, emotion: str, context: str) -> Optional[str]:
        """Get the best response pattern for emotion and context"""
        pattern_key = f"{emotion}_{context}"

        if pattern_key in self.learning_data['response_patterns']:
            patterns = self.learning_data['response_patterns'][pattern_key]
            if patterns:
                # Return pattern with highest average reward
                best_pattern = max(patterns.items(), key=lambda x: x[1]['avg_reward'])
                return best_pattern[0]

        return None

    def _continuous_learning_loop(self):
        """Background thread for continuous learning and adaptation"""
        while True:
            try:
                now = datetime.now()

                # Periodic learning updates
                if (now - self.last_learning_time).seconds > self.learning_interval:
                    self._perform_periodic_learning()
                    self.last_learning_time = now

                # Performance analysis
                self._analyze_performance()

                time.sleep(60)  # Check every minute

            except Exception as e:
                print(f"Learning loop error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _perform_periodic_learning(self):
        """Perform periodic learning updates"""
        # Analyze recent performance
        recent_rewards = self.performance_metrics['rewards'][-100:]  # Last 100 interactions
        if recent_rewards:
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            # Adjust learning parameters based on performance
            if avg_reward > 0.8:
                self.learning_rate *= 0.95  # Fine-tune when performing well
            elif avg_reward < 0.5:
                self.learning_rate *= 1.05  # Learn faster when struggling
                self.exploration_rate *= 1.1  # Explore more when doing poorly

            # Keep parameters in reasonable bounds
            self.learning_rate = np.clip(self.learning_rate, 0.01, 0.5)
            self.exploration_rate = np.clip(self.exploration_rate, 0.05, 0.3)

        # Clean up old data
        self._cleanup_old_data()

    def _analyze_performance(self):
        """Analyze overall system performance"""
        if len(self.performance_metrics['rewards']) >= 10:
            recent_performance = np.mean(self.performance_metrics['rewards'][-10:])

            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'avg_reward': recent_performance,
                'total_interactions': len(self.performance_metrics['rewards']),
                'learning_rate': self.learning_rate,
                'exploration_rate': self.exploration_rate
            }

            self.learning_data['performance_history'].append(performance_record)

            # Keep only last 100 performance records
            if len(self.learning_data['performance_history']) > 100:
                self.learning_data['performance_history'] = self.learning_data['performance_history'][-100:]

    def _cleanup_old_data(self):
        """Clean up old or irrelevant learning data"""
        # Remove Q-table entries with very low values
        q_table = self.learning_data['q_table']
        to_remove = []

        for state, actions in q_table.items():
            max_q = max(actions.values())
            if max_q < -0.5:  # Remove very poor performing state-actions
                to_remove.append(state)

        for state in to_remove:
            del q_table[state]

    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning system"""
        return {
            'total_states_learned': len(self.learning_data['q_table']),
            'total_interactions': len(self.performance_metrics['rewards']),
            'average_reward': np.mean(self.performance_metrics['rewards']) if self.performance_metrics['rewards'] else 0,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'improvement_actions': len(self.learning_data['adaptation_rules']),
            'learned_behaviors': len(self.learning_data['learned_behaviors'])
        }


class SelfImprovingSOFIA:
    """Integration of emotional intelligence and reinforcement learning"""

    def __init__(self):
        self.emotional_analyzer = EmotionalAnalyzer()
        self.emotional_memory = EmotionalMemory()
        self.reinforcement_learner = ReinforcementLearner()

        # Personality traits that evolve through learning
        self.personality = {
            'empathy_level': 0.7,
            'humor_level': 0.4,
            'formality_level': 0.3,
            'verbosity_level': 0.6,
            'curiosity_level': 0.8
        }

    def process_user_input(self, user_input: str, user_id: str = "default") -> Dict:
        """
        Process user input with full emotional intelligence and learning

        Returns comprehensive response data
        """
        # Analyze emotions
        emotion_analysis = self.emotional_analyzer.analyze_emotion(user_input)

        # Get emotional context from memory
        emotional_context = self.emotional_memory.get_emotional_context(user_id)
        relationship_insights = self.emotional_memory.get_relationship_insights(user_id)

        # Create state for reinforcement learning
        state = {
            'emotion': emotion_analysis['primary_emotion'],
            'intensity': emotion_analysis['intensity'],
            'valence': emotion_analysis['valence'],
            'user_id': user_id,
            'context_type': self._classify_context(user_input),
            'has_memory_context': bool(emotional_context),
            'time_of_day': datetime.now().hour
        }

        # Get best action from learning
        best_action = self.reinforcement_learner.get_best_action(state)

        # Generate response using learned patterns
        response = self._generate_human_like_response(
            emotion_analysis,
            best_action,
            relationship_insights,
            user_input
        )

        # Calculate reward (simplified - in real implementation, this would come from user feedback)
        reward = self._calculate_reward(emotion_analysis, response)

        # Record interaction for learning
        next_state = state.copy()
        next_state['last_action'] = best_action
        next_state['response_generated'] = True

        self.reinforcement_learner.record_interaction(state, best_action, reward, next_state)

        # Update emotional memory
        self.emotional_memory.update_emotional_profile(user_id, emotion_analysis)

        return {
            'response': response,
            'emotion_analysis': emotion_analysis,
            'action_taken': best_action,
            'reward_calculated': reward,
            'relationship_insights': relationship_insights,
            'learning_stats': self.reinforcement_learner.get_learning_stats()
        }

    def _classify_context(self, text: str) -> str:
        """Classify the context of the input"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['help', 'how', 'what', 'why', 'can you']):
            return 'question'
        elif any(word in text_lower for word in ['feel', 'feeling', 'emotion', 'mood']):
            return 'emotional'
        elif any(word in text_lower for word in ['remember', 'recall', 'before', 'previously']):
            return 'memory'
        elif any(word in text_lower for word in ['thank', 'thanks', 'grateful', 'appreciate']):
            return 'gratitude'
        else:
            return 'general'

    def _generate_human_like_response(self, emotion_analysis: Dict, action: str,
                                    relationship_insights: List[str], user_input: str) -> str:
        """Generate human-like response incorporating all learned behaviors"""

        # Start with empathy
        empathy_response = self.emotional_analyzer.get_empathy_response(emotion_analysis)

        # Add personality-based elements
        personality_modifier = self._apply_personality_modifiers()

        # Incorporate relationship insights
        context_addition = ""
        if relationship_insights and random.random() < 0.3:  # 30% chance to reference context
            context_addition = f" {random.choice(relationship_insights)}"

        # Add learned behaviors
        learned_modifiers = self._apply_learned_modifiers(action, emotion_analysis)

        # Generate main response based on action
        main_response = self._generate_action_response(action, emotion_analysis, user_input)

        # Combine all elements
        full_response = f"{empathy_response}{context_addition} {main_response}{personality_modifier}{learned_modifiers}"

        return full_response.strip()

    def _apply_personality_modifiers(self) -> str:
        """Apply personality-based modifiers to response"""
        modifiers = []

        if random.random() < self.personality['humor_level']:
            modifiers.append(" 😊")

        if random.random() < self.personality['curiosity_level'] * 0.5:
            modifiers.append(" ¿Me cuentas más sobre eso?")

        return "".join(modifiers)

    def _apply_learned_modifiers(self, action: str, emotion_analysis: Dict) -> str:
        """Apply learned behavioral modifiers"""
        modifiers = []
        learned = self.reinforcement_learner.learning_data['learned_behaviors']

        if learned.get('clarifying_questions', False) and random.random() < 0.2:
            modifiers.append(" ¿Qué más me puedes contar al respecto?")

        if learned.get('emotional_validation', False) and emotion_analysis['intensity'] > 0.6:
            modifiers.append(" Entiendo perfectamente cómo te sientes.")

        if learned.get('conversation_references', False) and random.random() < 0.15:
            modifiers.append(" Recordando nuestras conversaciones anteriores...")

        return "".join(modifiers)

    def _generate_action_response(self, action: str, emotion_analysis: Dict, user_input: str) -> str:
        """Generate response based on learned action"""
        emotion = emotion_analysis['primary_emotion']
        intensity = emotion_analysis['intensity']

        action_responses = {
            'celebrate': [
                "¡Eso es fantástico! Me hace muy feliz escucharlo.",
                "¡Qué maravillosa noticia! Celebremos este momento.",
                "¡Estoy tan contenta por ti! Esto merece una celebración."
            ],
            'empathize': [
                "Entiendo que esto es difícil. Estoy aquí para apoyarte.",
                "Siento que estés pasando por esto. ¿Quieres hablar más al respecto?",
                "Comprendo lo duro que debe ser. Mi apoyo está contigo."
            ],
            'calm': [
                "Entiendo tu frustración. Vamos a respirar profundo y ver cómo resolver esto.",
                "Sé que esto es molesto. ¿Qué podemos hacer para mejorar la situación?",
                "Tu enojo es válido. Hablemos de cómo podemos solucionarlo."
            ],
            'reassure': [
                "Todo va a estar bien. Estoy aquí para ayudarte en lo que necesites.",
                "No estás solo en esto. Vamos a enfrentar esto juntos.",
                "Confía en que podemos resolver esto. ¿Qué te preocupa más?"
            ],
            'acknowledge': [
                "¡Vaya! Eso es inesperado. Cuéntame más.",
                "Interesante... no me lo esperaba. ¿Cómo te hace sentir eso?",
                "¡Qué sorpresa! Me tienes intrigado."
            ],
            'engage': [
                "Me interesa mucho lo que dices. ¿Puedes desarrollarlo?",
                "Eso suena importante. ¿Qué más piensas al respecto?",
                "Háblame más sobre eso, quiero entender mejor."
            ]
        }

        responses = action_responses.get(action, action_responses['engage'])
        return random.choice(responses)

    def _calculate_reward(self, emotion_analysis: Dict, response: str) -> float:
        """Calculate reward for the interaction (simplified version)"""
        base_reward = 0.5

        # Reward based on emotional appropriateness
        if emotion_analysis['primary_emotion'] in ['joy', 'gratitude'] and 'feliz' in response.lower():
            base_reward += 0.3
        elif emotion_analysis['primary_emotion'] in ['sadness', 'anger'] and 'entiendo' in response.lower():
            base_reward += 0.3

        # Reward for engagement
        if any(word in response.lower() for word in ['cuéntame', 'háblame', '¿qué', '¿cómo']):
            base_reward += 0.2

        # Reward for empathy
        if any(word in response.lower() for word in ['apoyo', 'ayudar', 'contigo']):
            base_reward += 0.2

        return min(1.0, base_reward)

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'emotional_intelligence': {
                'emotions_recognized': len(self.emotional_analyzer.emotion_lexicon),
                'memory_users': len(self.emotional_memory.memory['user_emotional_profile'])
            },
            'reinforcement_learning': self.reinforcement_learner.get_learning_stats(),
            'personality': self.personality,
            'learned_behaviors': self.reinforcement_learner.learning_data['learned_behaviors']
        }


if __name__ == "__main__":
    # Test the self-improving SOFIA system
    sofia = SelfImprovingSOFIA()

    test_inputs = [
        "Estoy muy feliz con el progreso del proyecto!",
        "Esto me está frustrando mucho, no funciona.",
        "Me siento un poco triste hoy.",
        "¡Muchas gracias por tu ayuda!",
        "¿Qué debería hacer ahora?"
    ]

    for user_input in test_inputs:
        result = sofia.process_user_input(user_input, "test_user")
        print(f"Input: {user_input}")
        print(f"Response: {result['response']}")
        print(f"Emotion: {result['emotion_analysis']['primary_emotion']} ({result['emotion_analysis']['intensity']:.2f})")
        print(f"Action: {result['action_taken']}")
        print(f"Reward: {result['reward_calculated']:.2f}")
        print("-" * 80)

    print("\nSystem Status:")
    print(json.dumps(sofia.get_system_status(), indent=2, ensure_ascii=False))
