#!/usr/bin/env python3
"""
SOFIA Advanced Chat Simulator
Complete integration of LLM, Emotional Intelligence, and Reinforcement Learning
"""

import asyncio
import datetime
import signal
import sys
from typing import Dict, Any

# Import SOFIA components
from sofia_llm_integration import SOFIALanguageModel
from sofia_emotional_intelligence import EmotionalAnalyzer, EmotionalMemory
from sofia_reinforcement_learning import ReinforcementLearner

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    shutdown_flag = True
    print("\n👋 See you later!")
    sys.exit(0)

class SOFIABrain:
    """SOFIA's complete brain integrating all components"""

    def __init__(self):
        """Initialize SOFIA's brain with all components"""
        self.emotional_analyzer = EmotionalAnalyzer()
        self.emotional_memory = EmotionalMemory()
        self.reinforcement_learner = ReinforcementLearner()
        
        # Initialize LLM
        self.llm = SOFIALanguageModel()
        
        # Initialize conversation context
        self.conversation_history = []
        self.current_user_profile = {
            'conversation_count': 0,
            'emotional_state': 'neutral',
            'preferred_topics': []
        }
        self.user_id = "default"
    
    def _calculate_reward(self, user_input: str, response: str, emotional_context: dict) -> float:
        """Calculate reward for reinforcement learning based on response quality"""
        reward = 0.5  # Base reward
        
        # Bonus for emotional relevance - but penalize inappropriate empathy
        emotion = emotional_context['primary_emotion']
        intensity = emotional_context['intensity']
        confidence = emotional_context['confidence']
        
        if emotion != 'neutral' and intensity > 0.4 and confidence > 0.6:
            # Appropriate to show empathy for genuine emotions
            reward += 0.2
        elif emotion == 'neutral' or intensity < 0.3:
            # For neutral conversations, direct helpful responses are better
            reward += 0.1
        
        # Bonus for response length (not too short, not too long)
        response_length = len(response.split())
        if 8 <= response_length <= 45:
            reward += 0.2
        elif response_length < 5:
            reward -= 0.1  # Penalize too short responses
        elif response_length > 60:
            reward -= 0.1  # Penalize too long responses
        
        # Bonus for confidence in emotional analysis
        reward += confidence * 0.1
        
        # Check for response quality indicators
        response_lower = response.lower()
        if any(word in response_lower for word in ['help', 'assist', 'tell me more', 'explain']):
            reward += 0.1  # Good engagement words
            
        return min(1.0, max(0.0, reward))
    
    def _generate_fallback_response(self, user_input: str, emotional_context: dict) -> str:
        """Generate a fallback response when LLM fails"""
        emotion = emotional_context.get('primary_emotion', 'neutral')
        
        fallback_responses = {
            'joy': "That's wonderful! I can sense your happiness. Tell me more about what's making you feel so good!",
            'sadness': "I understand you're going through a difficult time. I'm here to listen and support you.",
            'anger': "I can see you're feeling frustrated. That must be really challenging. What's bothering you?",
            'fear': "It sounds like you're feeling anxious or worried. Would you like to talk about what's concerning you?",
            'surprise': "That sounds unexpected! How are you feeling about this situation?",
            'neutral': "I hear you. Could you tell me more about what's on your mind?"
        }
        
        return fallback_responses.get(emotion, "I understand. Could you tell me more about that.")

    def _clean_repetitive_greetings(self, response: str, user_input: str) -> str:
        """Clean repetitive greetings from responses to make conversation flow more naturally"""
        response_lower = response.lower()
        
        # Check if this is a follow-up message (not the first interaction)
        is_follow_up = len(self.conversation_history) > 0
        
        if is_follow_up:
            # Remove repetitive greetings in follow-up responses
            greeting_patterns = [
                "hello vladimir!",
                "hi vladimir!", 
                "hey vladimir!",
                f"hello {self.user_id}!",
                f"hi {self.user_id}!",
                f"hey {self.user_id}!"
            ]
            
            for pattern in greeting_patterns:
                if response_lower.startswith(pattern.lower()):
                    # Remove the greeting and clean up the response
                    response = response[len(pattern):].strip()
                    # Remove leading punctuation if present
                    if response.startswith(",") or response.startswith("!"):
                        response = response[1:].strip()
                    # Capitalize first letter if needed
                    if response and not response[0].isupper():
                        response = response[0].upper() + response[1:]
                    break
        
        return response

    async def process_message(self, user_input: str) -> str:
        """Process user message using all SOFIA components"""
        
        try:
            # 1. Emotional analysis of input
            emotional_context = self.emotional_analyzer.analyze_emotion(user_input)
            
            # 2. Prepare context for LLM
            context = {
                'user_profile': self.current_user_profile,
                'emotional_context': emotional_context,
                'conversation_history': self.conversation_history[-5:],  # Last 5 interactions
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # 3. Create state for reinforcement learning
            state = {
                'emotion': emotional_context['primary_emotion'],
                'intensity': emotional_context['intensity'],
                'valence': emotional_context['valence'],
                'user_id': self.user_id,
                'context_type': 'chat',
                'time_of_day': datetime.datetime.now().hour
            }
            
            # 4. Get best action from RL
            best_action = self.reinforcement_learner.get_best_action(state)
            
            # 5. Generate response with LLM
            enhanced_prompt = self._create_enhanced_prompt(user_input, context)
            
            try:
                llm_response = await self.llm.generate_response(enhanced_prompt)
            except Exception as e:
                print(f"⚠️  LLM generation error: {e}")
                llm_response = self._generate_fallback_response(user_input, emotional_context)
            
            # 6. Enhance with emotional empathy (only when appropriate)
            try:
                empathy_response = self.emotional_analyzer.get_empathy_response(emotional_context)
            except Exception as e:
                print(f"⚠️  Empathy analysis error: {e}")
                empathy_response = ""
            
            # Smart empathy integration - only add empathy for genuinely emotional situations
            should_add_empathy = (
                emotional_context['primary_emotion'] in ['sadness', 'anger', 'fear', 'guilt'] and  # Only negative emotions
                emotional_context['intensity'] > 0.6 and  # Only strong emotions
                emotional_context['confidence'] > 0.7 and  # Only when very confident
                len(empathy_response.strip()) > 0
            )
            
            if should_add_empathy and llm_response:
                # Check if LLM response already shows empathy
                llm_lower = llm_response.lower()
                empathy_indicators = ['sorry', 'understand', 'feel', 'help', 'support', 'there for you']
                
                if not any(indicator in llm_lower for indicator in empathy_indicators):
                    # Add empathy only if LLM response doesn't already show empathy
                    enhanced_response = f"{empathy_response} {llm_response}"
                else:
                    enhanced_response = llm_response
            else:
                enhanced_response = llm_response or "I understand. Could you tell me more?"
            
            # Post-process response to avoid repetitive greetings
            enhanced_response = self._clean_repetitive_greetings(enhanced_response, user_input)
            
            # 7. Calculate reward and learn from interaction
            reward = self._calculate_reward(user_input, enhanced_response, emotional_context)
            
            next_state = state.copy()
            next_state['response_generated'] = True
            next_state['last_action'] = best_action
            
            try:
                self.reinforcement_learner.record_interaction(state, best_action, reward, next_state)
            except Exception as e:
                print(f"⚠️  RL learning error: {e}")
            
            # 8. Update emotional memory
            try:
                self.emotional_memory.update_emotional_profile(self.user_id, emotional_context)
            except Exception as e:
                print(f"⚠️  Emotional memory error: {e}")
            
            # 9. Update conversation history and user profile
            self._update_conversation_history(user_input, enhanced_response, emotional_context)
            self._update_user_profile(user_input, emotional_context)
            
            return enhanced_response
            
        except Exception as e:
            print(f"⚠️  Error processing message: {e}")
            return "I apologize, but I encountered an error processing your message. Could you please try again?"

    def _create_enhanced_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create enhanced prompt with emotional and historical context"""
        
        emotional_state = context['emotional_context'].get('primary_emotion', 'neutral')
        emotional_intensity = context['emotional_context'].get('intensity', 0.5)
        conversation_context = ""
        
        if self.conversation_history:
            recent_exchanges = self.conversation_history[-2:]
            conversation_context = "Recent conversation:\n" + "\n".join([
                f"User: {exchange['user_input']}\nSOFIA: {exchange['response']}"
                for exchange in recent_exchanges
            ]) + "\n\n"
        
        # Create more natural emotion description
        emotion_description = emotional_state
        if emotional_intensity > 0.7:
            emotion_description = f"strongly {emotional_state}"
        elif emotional_intensity > 0.5:
            emotion_description = f"moderately {emotional_state}"
        elif emotional_intensity < 0.3:
            emotion_description = f"slightly {emotional_state}"
        
        enhanced_prompt = f"""You are SOFIA, an advanced AI assistant with emotional intelligence. You understand context, emotions, and provide thoughtful, helpful responses.

{conversation_context}Current user message: {user_input}
Detected emotional state: {emotion_description}

Guidelines:
- Respond naturally and conversationally in English
- Show empathy ONLY when the user clearly expresses negative emotions like sadness, anger, fear, or disappointment
- For neutral statements, questions, or positive topics, respond directly and helpfully without adding unnecessary emotional context
- Provide helpful and specific information when asked
- Ask clarifying questions when needed
- Keep responses concise but complete
- Don't assume emotional situations that aren't clearly stated
- Focus on being helpful and engaging

Response:"""
        
        return enhanced_prompt

    def _update_conversation_history(self, user_input: str, response: str, emotional_context: Dict[str, Any]):
        """Update conversation history"""
        
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'emotional_context': emotional_context
        }
        
        self.conversation_history.append(entry)
        
        # Keep only the last 50 interactions
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def _update_user_profile(self, user_input: str, emotional_context: Dict[str, Any]):
        """Update user profile based on interaction"""
        
        self.current_user_profile['conversation_count'] += 1
        self.current_user_profile['emotional_state'] = emotional_context.get('primary_emotion', 'neutral')
        
        # Detect interest topics (basic)
        keywords = user_input.lower().split()
        topic_keywords = ['technology', 'science', 'art', 'music', 'sports', 'food', 'travel']
        
        for keyword in topic_keywords:
            if keyword in user_input.lower():
                if keyword not in self.current_user_profile['preferred_topics']:
                    self.current_user_profile['preferred_topics'].append(keyword)

    def get_stats(self) -> Dict[str, Any]:
        """Get SOFIA brain statistics"""
        
        return {
            'conversation_count': len(self.conversation_history),
            'user_profile': self.current_user_profile,
            'rl_stats': self.reinforcement_learner.get_learning_stats(),
            'emotional_stats': {
                'analyzer_loaded': True,
                'memory_loaded': True
            }
        }

async def main():
    """Main function for the chat simulator"""
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🤖 SOFIA Advanced Chat Simulator")
    print("=" * 50)
    print("SOFIA now integrates:")
    print("  🧠 LLM (Qwen2.5-1.5B)")
    print("  ❤️  Emotional Intelligence")
    print("  🎯 Reinforcement Learning")
    print("=" * 50)
    print("Type 'quit' to exit, 'stats' to see statistics")
    print()    
    try:
        # Initialize SOFIA's brain
        sofia_brain = SOFIABrain()
        # LLM initializes automatically in __init__
        
        print("✅ SOFIA is ready! Starting conversation...")
        print()        
        while not shutdown_flag:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 See you later!")
                    break
                
                if user_input.lower() == 'stats':
                    stats = sofia_brain.get_stats()
                    print("\n📊 SOFIA Statistics:")
                    print(f"  Conversations: {stats['conversation_count']}")
                    print(f"  User emotional state: {stats['user_profile']['emotional_state']}")
                    print(f"  Preferred topics: {stats['user_profile']['preferred_topics']}")
                    print(f"  RL learned states: {stats['rl_stats']['total_states_learned']}")
                    print(f"  Average reward: {stats['rl_stats']['average_reward']:.2f}")
                    print()                    
                    continue
                
                if not user_input:
                    continue
                
                # Process user input through SOFIA's brain
                print("🧠 SOFIA is thinking...")
                response = await sofia_brain.process_message(user_input)
                print(f"SOFIA: {response}")
                print()                
            except KeyboardInterrupt:
                print("\n👋 See you later!")
                break
            except EOFError:
                print("\n👋 See you later!")
                break
            except Exception as e:
                print(f"⚠️  Error processing message: {e}")
                print("Please try again.")
                
    except KeyboardInterrupt:
        print("\n👋 See you later!")
    except Exception as e:
        print(f"❌ Error initializing SOFIA: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 See you later!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
from sofia_emotional_intelligence import EmotionalAnalyzer, EmotionalMemory
from sofia_reinforcement_learning import ReinforcementLearner

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    shutdown_flag = True
    print("\n👋 See you later!")
    sys.exit(0)
"""
SOFIA Advanced Chat Simulator
Integra LLM, Inteligencia Emocional y Aprendizaje por Refuerzo
"""

import asyncio
import json
import datetime
from typing import Dict, Any, Optional

# SOFIA modules
from sofia_llm_integration import SOFIALanguageModel
from sofia_emotional_intelligence import EmotionalAnalyzer, EmotionalMemory
from sofia_reinforcement_learning import ReinforcementLearner

class SOFIABrain:
    """Cerebro principal de SOFIA que integra todos los componentes"""
    
    def __init__(self):
        """Initialize SOFIA's brain with all components"""
        self.emotional_analyzer = EmotionalAnalyzer()
        self.emotional_memory = EmotionalMemory()
        self.reinforcement_learner = ReinforcementLearner()
        
        # Initialize LLM
        self.llm = SOFIALanguageModel()
        
        # Initialize conversation context
        self.conversation_history = []
        self.current_user_profile = {
            'conversation_count': 0,
            'emotional_state': 'neutral',
            'preferred_topics': []
        }
        self.user_id = "default"
    
    def _calculate_reward(self, user_input: str, response: str, emotional_context: dict) -> float:
        """Calculate reward for reinforcement learning based on response quality"""
        reward = 0.5  # Base reward
        
        # Bonus for emotional relevance - but penalize inappropriate empathy
        emotion = emotional_context['primary_emotion']
        intensity = emotional_context['intensity']
        confidence = emotional_context['confidence']
        
        if emotion != 'neutral' and intensity > 0.4 and confidence > 0.6:
            # Appropriate to show empathy for genuine emotions
            reward += 0.2
        elif emotion == 'neutral' or intensity < 0.3:
            # For neutral conversations, direct helpful responses are better
            reward += 0.1
        
        # Bonus for response length (not too short, not too long)
        response_length = len(response.split())
        if 8 <= response_length <= 45:
            reward += 0.2
        elif response_length < 5:
            reward -= 0.1  # Penalize too short responses
        elif response_length > 60:
            reward -= 0.1  # Penalize too long responses
        
        # Bonus for confidence in emotional analysis
        reward += confidence * 0.1
        
        # Check for response quality indicators
        response_lower = response.lower()
        if any(word in response_lower for word in ['help', 'assist', 'tell me more', 'explain']):
            reward += 0.1  # Good engagement words
            
        return min(1.0, max(0.0, reward))
    
    def _generate_fallback_response(self, user_input: str, emotional_context: dict) -> str:
        """Generate a fallback response when LLM fails"""
        emotion = emotional_context.get('primary_emotion', 'neutral')
        
        fallback_responses = {
            'joy': "That's wonderful! I can sense your happiness. Tell me more about what's making you feel so good!",
            'sadness': "I understand you're going through a difficult time. I'm here to listen and support you.",
            'anger': "I can see you're feeling frustrated. That must be really challenging. What's bothering you?",
            'fear': "It sounds like you're feeling anxious or worried. Would you like to talk about what's concerning you?",
            'surprise': "That sounds unexpected! How are you feeling about this situation?",
            'neutral': "I hear you. Could you tell me more about what's on your mind?"
        }
        
        return fallback_responses.get(emotion, "I understand. Please tell me more about that.")

    async def process_message(self, user_input: str) -> str:
        """Process user message using all SOFIA components"""
        
        try:
            # 1. Emotional analysis of input
            emotional_context = self.emotional_analyzer.analyze_emotion(user_input)
            
            # 2. Prepare context for LLM
            context = {
                'user_profile': self.current_user_profile,
                'emotional_context': emotional_context,
                'conversation_history': self.conversation_history[-5:],  # Last 5 interactions
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # 3. Create state for reinforcement learning
            state = {
                'emotion': emotional_context['primary_emotion'],
                'intensity': emotional_context['intensity'],
                'valence': emotional_context['valence'],
                'user_id': self.user_id,
                'context_type': 'chat',
                'time_of_day': datetime.datetime.now().hour
            }
            
            # 4. Get best action from RL
            best_action = self.reinforcement_learner.get_best_action(state)
            
            # 5. Generate response with LLM
            enhanced_prompt = self._create_enhanced_prompt(user_input, context)
            
            try:
                llm_response = await self.llm.generate_response(enhanced_prompt)
            except Exception as e:
                print(f"⚠️  LLM generation error: {e}")
                llm_response = self._generate_fallback_response(user_input, emotional_context)
            
            # 6. Enhance with emotional empathy (only when appropriate)
            try:
                empathy_response = self.emotional_analyzer.get_empathy_response(emotional_context)
            except Exception as e:
                print(f"⚠️  Empathy analysis error: {e}")
                empathy_response = ""
            
            # Smart empathy integration - only add empathy for genuinely emotional situations
            should_add_empathy = (
                emotional_context['primary_emotion'] != 'neutral' and
                emotional_context['intensity'] > 0.4 and  # Only for moderate to strong emotions
                emotional_context['confidence'] > 0.6 and  # Only when we're confident about the emotion
                len(empathy_response.strip()) > 0  # Only if we have a meaningful empathy response
            )
            
            if should_add_empathy and llm_response:
                # Check if LLM response already shows empathy
                llm_lower = llm_response.lower()
                empathy_indicators = ['sorry', 'understand', 'feel', 'help', 'support', 'there for you']
                
                if not any(indicator in llm_lower for indicator in empathy_indicators):
                    # Add empathy only if LLM response doesn't already show empathy
                    enhanced_response = f"{empathy_response} {llm_response}"
                else:
                    enhanced_response = llm_response
            else:
                enhanced_response = llm_response or "I understand. Could you tell me more?"
            
            # 7. Calculate reward and learn from interaction
            reward = self._calculate_reward(user_input, enhanced_response, emotional_context)
            
            next_state = state.copy()
            next_state['response_generated'] = True
            next_state['last_action'] = best_action
            
            try:
                self.reinforcement_learner.record_interaction(state, best_action, reward, next_state)
            except Exception as e:
                print(f"⚠️  RL learning error: {e}")
            
            # 8. Update emotional memory
            try:
                self.emotional_memory.update_emotional_profile(self.user_id, emotional_context)
            except Exception as e:
                print(f"⚠️  Emotional memory error: {e}")
            
            # 9. Update conversation history and user profile
            self._update_conversation_history(user_input, enhanced_response, emotional_context)
            self._update_user_profile(user_input, emotional_context)
            
            return enhanced_response
            
        except Exception as e:
            print(f"⚠️  Error processing message: {e}")
            return "I apologize, but I encountered an error processing your message. Could you please try again?"

    def _create_enhanced_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create enhanced prompt with emotional and historical context"""
        
        emotional_state = context['emotional_context'].get('primary_emotion', 'neutral')
        emotional_intensity = context['emotional_context'].get('intensity', 0.5)
        conversation_context = ""
        
        if self.conversation_history:
            recent_exchanges = self.conversation_history[-2:]
            conversation_context = "Recent conversation:\n" + "\n".join([
                f"User: {exchange['user_input']}\nSOFIA: {exchange['response']}"
                for exchange in recent_exchanges
            ]) + "\n\n"
        
        # Create more natural emotion description
        emotion_description = emotional_state
        if emotional_intensity > 0.7:
            emotion_description = f"strongly {emotional_state}"
        elif emotional_intensity > 0.5:
            emotion_description = f"moderately {emotional_state}"
        elif emotional_intensity < 0.3:
            emotion_description = f"slightly {emotional_state}"
        
        enhanced_prompt = f"""You are SOFIA, an advanced AI assistant with emotional intelligence. You understand context, emotions, and provide thoughtful, helpful responses.

{conversation_context}Current user message: {user_input}
Detected emotional state: {emotion_description}

Guidelines:
- Respond naturally and conversationally in English
- Show empathy ONLY when the user clearly expresses negative emotions like sadness, anger, fear, or disappointment
- For neutral statements, questions, or positive topics, respond directly and helpfully without adding unnecessary emotional context
- Provide helpful and specific information when asked
- Ask clarifying questions when needed
- Keep responses concise but complete
- Don't assume emotional situations that aren't clearly stated
- Focus on being helpful and engaging

Response:"""
        
        return enhanced_prompt

    def _update_conversation_history(self, user_input: str, response: str, emotional_context: Dict[str, Any]):
        """Update conversation history"""
        
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'emotional_context': emotional_context
        }
        
        self.conversation_history.append(entry)
        
        # Keep only the last 50 interactions
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def _update_user_profile(self, user_input: str, emotional_context: Dict[str, Any]):
        """Update user profile based on interaction"""
        
        self.current_user_profile['conversation_count'] += 1
        self.current_user_profile['emotional_state'] = emotional_context.get('primary_emotion', 'neutral')
        
        # Detect interest topics (basic)
        keywords = user_input.lower().split()
        topic_keywords = ['technology', 'science', 'art', 'music', 'sports', 'food', 'travel']
        
        for keyword in topic_keywords:
            if keyword in user_input.lower():
                if keyword not in self.current_user_profile['preferred_topics']:
                    self.current_user_profile['preferred_topics'].append(keyword)

    def get_stats(self) -> Dict[str, Any]:
        """Get SOFIA brain statistics"""
        
        return {
            'conversation_count': len(self.conversation_history),
            'user_profile': self.current_user_profile,
            'rl_stats': self.reinforcement_learner.get_learning_stats(),
            'emotional_stats': {
                'analyzer_loaded': True,
                'memory_loaded': True
            }
        }

async def main():
    """Main function for the chat simulator"""
    
    print("🤖 SOFIA Advanced Chat Simulator")
    print("=" * 50)
    print("SOFIA now integrates:")
    print("  🧠 LLM (Qwen2.5-1.5B)")
    print("  ❤️  Emotional Intelligence")
    print("  🎯 Reinforcement Learning")
    print("=" * 50)
    print("Type 'quit' to exit, 'stats' to see statistics")
    print()
    
    try:
        # Initialize SOFIA's brain
        sofia = SOFIABrain()
        
        print("✅ SOFIA is ready! Starting conversation...")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 See you later!")
                    break
                elif user_input.lower() == 'stats':
                    stats = sofia.get_stats()
                    print("\n📊 SOFIA Statistics:")
                    print(f"  Conversations: {stats['conversation_count']}")
                    print(f"  User emotional state: {stats['user_profile']['emotional_state']}")
                    print(f"  Preferred topics: {stats['user_profile']['preferred_topics']}")
                    print(f"  RL learned states: {stats['rl_stats']['total_states_learned']}")
                    print(f"  Average reward: {stats['rl_stats']['average_reward']:.2f}")
                    print()
                    continue
                elif not user_input:
                    continue
                
                # Process user input through SOFIA's brain
                print("🧠 SOFIA is thinking...")
                response = await sofia.process_message(user_input)
                print(f"SOFIA: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 See you later!")
                break
            except Exception as e:
                print(f"⚠️  Error processing message: {e}")
                print("Please try again.")
                
    except KeyboardInterrupt:
        print("\n👋 See you later!")
    except Exception as e:
        print(f"❌ Error initializing SOFIA: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 See you later!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
