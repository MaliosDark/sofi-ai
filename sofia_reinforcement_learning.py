#!/usr/bin/env python3
"""
SOFIA Real-Time Reinforcement Learning System
Autonomous self-improvement through interaction feedback and continuous learning
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import threading
import time
from collections import defaultdict
import random
import math

# Import emotional intelligence components
from sofia_emotional_intelligence import EmotionalAnalyzer, EmotionalMemory

# Import LLM integration (optional)
try:
    from sofia_llm_integration import SOFIALanguageModel
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    SOFIALanguageModel = None

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
            self.learning_rate = max(0.01, min(0.5, self.learning_rate))
            self.exploration_rate = max(0.05, min(0.3, self.exploration_rate))

        # Clean up old data
        self._cleanup_old_data()

    def _analyze_performance(self):
        """Analyze overall system performance"""
        if len(self.performance_metrics['rewards']) >= 10:
            recent_performance = sum(self.performance_metrics['rewards'][-10:]) / len(self.performance_metrics['rewards'][-10:])

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
            'average_reward': sum(self.performance_metrics['rewards']) / len(self.performance_metrics['rewards']) if self.performance_metrics['rewards'] else 0,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'improvement_actions': len(self.learning_data['adaptation_rules']),
            'learned_behaviors': len(self.learning_data['learned_behaviors'])
        }


class SelfImprovingSOFIA:
    """Integration of emotional intelligence and reinforcement learning"""

    def __init__(self, use_llm: bool = True):
        self.emotional_analyzer = EmotionalAnalyzer()
        self.emotional_memory = EmotionalMemory()
        self.reinforcement_learner = ReinforcementLearner()

        # Initialize LLM if available and requested
        self.use_llm = use_llm and LLM_AVAILABLE
        if self.use_llm:
            try:
                self.llm = SOFIALanguageModel()
                print("🧠 LLM integration enabled for enhanced responses")
            except Exception as e:
                print(f"⚠️  LLM initialization failed: {e}, falling back to template responses")
                self.use_llm = False
                self.llm = None
        else:
            self.llm = None
            if use_llm and not LLM_AVAILABLE:
                print("⚠️  LLM module not available, install sofia_llm_integration.py or required packages")

        # Personality traits that evolve through learning
        self.personality = {
            'empathy_level': 0.7,
            'humor_level': 0.4,
            'formality_level': 0.3,
            'verbosity_level': 0.6,
            'curiosity_level': 0.8
        }

    async def process_user_input(self, user_input: str, user_id: str = "default") -> Dict:
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

        # Generate response using LLM or learned patterns
        if self.use_llm and self.llm:
            response = await self._generate_llm_response(
                user_input, emotion_analysis, relationship_insights, user_id
            )
        else:
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

    async def _generate_llm_response(self, user_input: str, emotion_analysis: Dict,
                                   relationship_insights: List[str], user_id: str) -> str:
        """Generate response using LLM for enhanced natural language generation"""

        if not self.llm:
            # Fallback to template response if LLM not available
            return self._generate_human_like_response(emotion_analysis, 'engage', relationship_insights, user_input)

        # Get conversation history for context (simplified - using available methods)
        conversation_history = []  # TODO: Implement proper conversation history retrieval

        # Create context dictionary
        context = {
            'user_id': user_id,
            'relationship_insights': relationship_insights,
            'personality': self.personality,
            'learned_behaviors': self.reinforcement_learner.learning_data.get('learned_behaviors', {})
        }

        try:
            # Generate response using LLM
            llm_response = await self.llm.generate_emotional_response(
                user_input, emotion_analysis, context, conversation_history
            )

            # Apply personality modifiers and learned behaviors to LLM response
            personality_modifier = self._apply_personality_modifiers()
            learned_modifiers = self._apply_learned_modifiers('llm_generated', emotion_analysis)

            # Add relationship insights if appropriate
            context_addition = ""
            if relationship_insights and random.random() < 0.2:  # Lower frequency for LLM responses
                context_addition = f" {random.choice(relationship_insights)}"

            # Combine LLM response with learned modifiers
            full_response = f"{llm_response}{context_addition}{personality_modifier}{learned_modifiers}"

            return full_response.strip()

        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to template response")
            # Fallback to template response on LLM failure
            return self._generate_human_like_response(emotion_analysis, 'engage', relationship_insights, user_input)

    def _apply_personality_modifiers(self) -> str:
        """Apply personality-based modifiers to response with more variety"""
        modifiers = []

        if random.random() < self.personality['humor_level']:
            modifiers.append(random.choice([" 😊", " 😉", " 😄"]))

        if random.random() < self.personality['curiosity_level'] * 0.2:  # Further reduced frequency
            curiosity_phrases = [
                " I'd love to hear more about that.",
                " That sounds fascinating.",
                " I'm curious to know more.",
                " Tell me your thoughts on that."
            ]
            modifiers.append(random.choice(curiosity_phrases))

        return "".join(modifiers)

    def _apply_learned_modifiers(self, action: str, emotion_analysis: Dict) -> str:
        """Apply learned behavioral modifiers with more variety"""
        modifiers = []
        learned = self.reinforcement_learner.learning_data['learned_behaviors']

        if learned.get('clarifying_questions', False) and random.random() < 0.15:  # Reduced frequency
            clarifying_questions = [
                " What else comes to mind?",
                " Can you share more details?",
                " How did that make you feel?",
                " What happened next?"
            ]
            modifiers.append(random.choice(clarifying_questions))

        if learned.get('emotional_validation', False) and emotion_analysis['intensity'] > 0.6:
            validation_phrases = [
                " I completely understand how you feel.",
                " Your feelings are completely valid.",
                " It's okay to feel this way.",
                " I can sense how strongly you feel about this."
            ]
            modifiers.append(random.choice(validation_phrases))

        if learned.get('conversation_references', False) and random.random() < 0.1:  # Reduced frequency
            reference_phrases = [
                " This reminds me of our earlier conversation.",
                " Building on what you said before...",
                " Connecting this to what we discussed...",
                " Remembering our previous talks..."
            ]
            modifiers.append(random.choice(reference_phrases))

        return "".join(modifiers)

    def _generate_action_response(self, action: str, emotion_analysis: Dict, user_input: str) -> str:
        """Generate response based on learned action with more context and specificity"""
        emotion = emotion_analysis['primary_emotion']
        intensity = emotion_analysis['intensity']
        user_words = user_input.lower().split()

        # Extract key topics/words from user input for more specific responses
        key_topics = self._extract_key_topics(user_input)

        action_responses = {
            'celebrate': [
                f"That's fantastic about {key_topics[0] if key_topics else 'that'}! It really warms my heart to hear it. You must be feeling great.",
                f"What wonderful news about {key_topics[0] if key_topics else 'your progress'}! This calls for a celebration. Tell me more about how it happened.",
                f"I'm genuinely thrilled for you with {key_topics[0] if key_topics else 'this achievement'}! You deserve all the happiness in the world.",
                f"That's absolutely amazing! I'm so happy things are going well with {key_topics[0] if key_topics else 'everything'}.",
                f"Wow, that's fantastic news! I can feel your excitement about {key_topics[0] if key_topics else 'this'} from here."
            ],
            'empathize': [
                f"I can sense how challenging {key_topics[0] if key_topics else 'this situation'} must be for you. I'm here whenever you need to talk.",
                f"I'm truly sorry you're dealing with {key_topics[0] if key_topics else 'this difficulty'}. It sounds really tough right now.",
                f"I understand how tough {key_topics[0] if key_topics else 'this'} can be. You don't have to go through it alone.",
                f"That sounds really difficult with {key_topics[0] if key_topics else 'everything that is happening'}. My thoughts are with you.",
                f"I can hear how much {key_topics[0] if key_topics else 'this'} is affecting you. Take all the time you need."
            ],
            'calm': [
                f"I hear your frustration with {key_topics[0] if key_topics else 'this situation'}. Let's take a deep breath together and see what we can do.",
                f"That sounds really annoying about {key_topics[0] if key_topics else 'what happened'}. I understand why you'd feel this way.",
                f"Your feelings about {key_topics[0] if key_topics else 'this'} are completely valid. Sometimes these things just build up.",
                f"I get why {key_topics[0] if key_topics else 'this'} would be frustrating. Let's think about what might help.",
                f"That does sound irritating with {key_topics[0] if key_topics else 'the situation'}. You have every right to feel upset about it."
            ],
            'reassure': [
                f"Everything will work out with {key_topics[0] if key_topics else 'this'}. You've got this, and I'm here to help.",
                f"You're not alone in dealing with {key_topics[0] if key_topics else 'this challenge'}. We'll figure it out together.",
                f"I believe we can overcome {key_topics[0] if key_topics else 'this obstacle'}. You've shown strength before.",
                f"Things have a way of working out, especially with {key_topics[0] if key_topics else 'situations like this'}. Stay positive.",
                f"You've handled difficult things before with {key_topics[0] if key_topics else 'challenges'}. This is no different."
            ],
            'acknowledge': [
                f"Wow, {key_topics[0] if key_topics else 'that'} is really surprising! I wasn't expecting to hear that.",
                f"That's quite unexpected about {key_topics[0] if key_topics else 'the situation'}. How did that come about?",
                f"I'm genuinely intrigued by {key_topics[0] if key_topics else 'what you just said'}. That's not something I hear every day.",
                f"That's really interesting about {key_topics[0] if key_topics else 'your experience'}. I can see why you'd mention it.",
                f"Wow, I didn't see {key_topics[0] if key_topics else 'that'} coming! That's quite something."
            ],
            'engage': [
                f"I'm really interested in your thoughts about {key_topics[0] if key_topics else 'this'}. You always have such unique perspectives.",
                f"That sounds significant regarding {key_topics[0] if key_topics else 'your experience'}. I'd love to hear your take on it.",
                f"Your perspective on {key_topics[0] if key_topics else 'this topic'} is always so thoughtful. What made you think about it?",
                f"That's really fascinating about {key_topics[0] if key_topics else 'what you mentioned'}. I always learn something new from you.",
                f"I appreciate you sharing about {key_topics[0] if key_topics else 'this'}. It gives me a lot to think about."
            ]
        }

        responses = action_responses.get(action, action_responses['engage'])
        return random.choice(responses)

    def _extract_key_topics(self, user_input: str) -> List[str]:
        """Extract key topics and meaningful words from user input for contextual responses"""
        # Filter out negative/insulting words that shouldn't be referenced
        negative_words = {
            'stupid', 'dumb', 'idiot', 'fuck', 'shit', 'damn', 'asshole', 'bitch',
            'kill', 'die', 'death', 'suicide', 'hate', 'suck', 'crap', 'bullshit',
            'fuck you', 'fuk', 'fucking', 'bitch', 'asshole', 'bastard', 'shithead'
        }

        # Skip common words and Spanish articles/pronouns
        skip_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'this', 'that', 'these', 'those', 'what', 'when', 'where', 'why', 'how', 'who',
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might',
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'en',
            'que', 'qué', 'como', 'cuando', 'donde', 'porque', 'como', 'yo', 'tu', 'usted',
            'nosotros', 'ellos', 'esto', 'eso', 'esta', 'esa', 'estos', 'esos', 'estas', 'esas',
            'hola', 'gracias', 'por', 'favor', 'si', 'no', 'muy', 'mucho', 'poco', 'todo',
            'nada', 'algo', 'alguien', 'nadie', 'siempre', 'nunca', 'ahora', 'después', 'antes',
            'hoy', 'ayer', 'mañana', 'bueno', 'malo', 'grande', 'pequeño', 'alto', 'bajo',
            'soy', 'eres', 'es', 'somos', 'son', 'estoy', 'estas', 'esta', 'estamos', 'estan',
            'fui', 'fuiste', 'fue', 'fuimos', 'fueron', 'ser', 'estar', 'tener', 'hacer',
            'decir', 'ir', 'ver', 'dar', 'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner',
            'parecer', 'quedar', 'creer', 'hablar', 'llevar', 'dejar', 'seguir', 'encontrar',
            'llamar', 'venir', 'pensar', 'salir', 'volver', 'tomar', 'conocer', 'vivir', 'sentir',
            'tratar', 'mirar', 'contar', 'empezar', 'esperar', 'buscar', 'existir', 'entrar',
            'trabajar', 'escribir', 'perder', 'producir', 'ocurrir', 'entender', 'pedir', 'recibir',
            'recordar', 'terminar', 'permitir', 'aparecer', 'conseguir', 'comenzar', 'servir',
            'sacar', 'necesitar', 'mantener', 'resultar', 'leer', 'caer', 'cambiar', 'presentar',
            'crear', 'abrir', 'considerar', 'oír', 'acabar', 'convertir', 'ganar', 'formar',
            'traer', 'partir', 'morir', 'aceptar', 'realizar', 'suponer', 'comprender', 'lograr',
            'explicar', 'preguntar', 'tocar', 'reconocer', 'estudiar', 'alcanzar', 'nacer', 'dirigir',
            'correr', 'utilizar', 'pagar', 'ayudar', 'gustar', 'jugar', 'escuchar', 'cumplir',
            'sofia', 'escucharme', 'tengo', 'está', 'aprecio', '¡todo', '¡qué', '¿qué', 'hello', 'he;llo'
        }

        # Common meaningful topic categories (English + Spanish)
        topic_keywords = {
            'work': ['work', 'job', 'project', 'task', 'deadline', 'meeting', 'office', 'career', 'business', 'trabajo', 'empleo', 'proyecto', 'tarea', 'plazo', 'reunión', 'oficina', 'carrera', 'negocio'],
            'health': ['sick', 'ill', 'health', 'doctor', 'pain', 'tired', 'sleep', 'medicine', 'hospital', 'enfermo', 'salud', 'doctor', 'dolor', 'cansado', 'sueño', 'medicina', 'hospital'],
            'relationships': ['friend', 'family', 'partner', 'love', 'relationship', 'alone', 'lonely', 'together', 'amigo', 'familia', 'pareja', 'amor', 'relación', 'solo', 'solitario', 'juntos'],
            'emotions': ['happy', 'sad', 'angry', 'frustrated', 'worried', 'anxious', 'excited', 'scared', 'confused', 'feliz', 'triste', 'enojado', 'frustrado', 'preocupado', 'ansioso', 'emocionado', 'asustado', 'confundido'],
            'future': ['future', 'tomorrow', 'next', 'plan', 'goal', 'dream', 'hope', 'worry', 'planning', 'futuro', 'mañana', 'próximo', 'plan', 'meta', 'sueño', 'esperanza', 'preocupación', 'planificación'],
            'past': ['yesterday', 'before', 'memory', 'remember', 'used', 'previously', 'ago', 'ayer', 'antes', 'memoria', 'recordar', 'usado', 'previamente', 'hace'],
            'problems': ['problem', 'issue', 'trouble', 'difficult', 'hard', 'challenge', 'stuck', 'help', 'problema', 'asunto', 'problema', 'difícil', 'duro', 'desafío', 'atascado', 'ayuda'],
            'success': ['success', 'achievement', 'accomplished', 'proud', 'won', 'finished', 'completed', 'great', 'éxito', 'logro', 'conseguido', 'orgulloso', 'ganado', 'terminado', 'completado', 'genial'],
            'learning': ['learn', 'study', 'school', 'class', 'teacher', 'book', 'knowledge', 'understand', 'aprender', 'estudiar', 'escuela', 'clase', 'profesor', 'libro', 'conocimiento', 'entender'],
            'technology': ['computer', 'phone', 'internet', 'software', 'app', 'code', 'programming', 'computadora', 'teléfono', 'internet', 'software', 'aplicación', 'código', 'programación']
        }

        words = [w.lower().strip('.,!?') for w in user_input.split()]
        found_topics = []

        # Check for negative words first - if found, don't extract specific topics
        has_negative = any(word in negative_words for word in words)
        if has_negative:
            return ['this situation']  # Generic fallback for negative inputs

        # Look for meaningful topic matches
        for category, keywords in topic_keywords.items():
            for word in words:
                if word in keywords:
                    found_topics.append(word)
                    if len(found_topics) >= 2:  # Limit to 2 topics
                        break
            if len(found_topics) >= 2:
                break

        # If no specific topics found, look for nouns that aren't in skip list
        if not found_topics:
            for word in words:
                # Skip short words, numbers, and words in skip list
                if (len(word) > 3 and
                    word not in skip_words and
                    not word.isdigit() and
                    not any(char.isdigit() for char in word) and
                    word not in negative_words):
                    found_topics.append(word)
                    if len(found_topics) >= 1:
                        break

        # Final fallback - use generic terms instead of random words
        if not found_topics:
            found_topics = ['what you mentioned', 'your experience', 'this topic']

        return found_topics[:2]

    def _calculate_reward(self, emotion_analysis: Dict, response: str) -> float:
        """Calculate reward for the interaction (simplified version)"""
        base_reward = 0.5

        # Reward based on emotional appropriateness
        if emotion_analysis['primary_emotion'] in ['joy', 'gratitude'] and 'happy' in response.lower():
            base_reward += 0.3
        elif emotion_analysis['primary_emotion'] in ['sadness', 'anger'] and 'understand' in response.lower():
            base_reward += 0.3

        # Reward for engagement
        if any(word in response.lower() for word in ['tell me', 'what', 'how']):
            base_reward += 0.2

        # Reward for empathy
        if any(word in response.lower() for word in ['support', 'help', 'with you']):
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
