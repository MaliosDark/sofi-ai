#!/usr/bin/env python3
"""
SOFIA Emotional Intelligence Core
Advanced sentiment analysis and emotional understanding for human-like interactions
"""

import re
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import os

class EmotionalAnalyzer:
    """Advanced emotional intelligence analyzer for SOFIA"""

    def __init__(self):
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.context_patterns = self._load_context_patterns()
        self.intensity_modifiers = self._load_intensity_modifiers()

    def _load_emotion_lexicon(self) -> Dict[str, Dict]:
        """Load comprehensive emotion lexicon with intensities"""
        return {
            # Positive emotions
            'joy': {'words': ['happy', 'excited', 'thrilled', 'delighted', 'ecstatic', 'overjoyed', 'blissful', 'cheerful', 'gleeful', 'jubilant'], 'intensity': 0.8, 'valence': 1.0},
            'love': {'words': ['love', 'adore', 'cherish', 'fond', 'affectionate', 'devoted', 'passionate', 'tender'], 'intensity': 0.9, 'valence': 1.0},
            'gratitude': {'words': ['thankful', 'grateful', 'appreciative', 'indebted', 'obliged'], 'intensity': 0.7, 'valence': 0.8},
            'pride': {'words': ['proud', 'accomplished', 'achieved', 'successful', 'triumphant'], 'intensity': 0.8, 'valence': 0.9},
            'hope': {'words': ['hopeful', 'optimistic', 'confident', 'encouraged', 'positive'], 'intensity': 0.6, 'valence': 0.8},

            # Negative emotions
            'sadness': {'words': ['sad', 'unhappy', 'depressed', 'down', 'blue', 'melancholy', 'sorrowful', 'heartbroken'], 'intensity': 0.8, 'valence': -1.0},
            'anger': {'words': ['angry', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'outraged', 'hostile'], 'intensity': 0.9, 'valence': -0.9},
            'fear': {'words': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'panicked', 'frightened'], 'intensity': 0.8, 'valence': -0.8},
            'disgust': {'words': ['disgusted', 'repulsed', 'revolted', 'nauseated', 'appalled', 'offended'], 'intensity': 0.7, 'valence': -0.7},
            'guilt': {'words': ['guilty', 'remorseful', 'ashamed', 'regretful', 'sorry'], 'intensity': 0.7, 'valence': -0.6},

            # Neutral/Complex emotions
            'surprise': {'words': ['surprised', 'shocked', 'amazed', 'astonished', 'startled', 'unexpected'], 'intensity': 0.6, 'valence': 0.0},
            'confusion': {'words': ['confused', 'puzzled', 'bewildered', 'perplexed', 'lost', 'uncertain'], 'intensity': 0.5, 'valence': -0.2},
            'curiosity': {'words': ['curious', 'interested', 'intrigued', 'fascinated', 'inquisitive'], 'intensity': 0.6, 'valence': 0.4},
            'trust': {'words': ['trust', 'confident', 'reliable', 'faithful', 'loyal'], 'intensity': 0.7, 'valence': 0.8},
            'anticipation': {'words': ['excited', 'eager', 'expectant', 'anxious', 'waiting'], 'intensity': 0.5, 'valence': 0.3}
        }

    def _load_context_patterns(self) -> Dict[str, List[str]]:
        """Load contextual patterns that modify emotional interpretation"""
        return {
            'intensifiers': ['very', 'extremely', 'really', 'so', 'absolutely', 'totally', 'completely', 'utterly'],
            'diminishers': ['a little', 'somewhat', 'slightly', 'kind of', 'sort of', 'barely'],
            'negations': ['not', "n't", 'never', 'no', 'none'],
            'questions': ['how', 'what', 'why', 'when', 'where', 'who'],
            'emphasis': ['!', '!!', '...', '???']
        }

    def _load_intensity_modifiers(self) -> Dict[str, float]:
        """Load modifiers that affect emotional intensity"""
        return {
            '!': 1.2,      # Single exclamation increases intensity
            '!!': 1.4,     # Double exclamation increases more
            '...': 0.8,    # Ellipsis decreases intensity
            '???': 1.1,    # Question marks show confusion/emphasis
            'CAPS': 1.3    # All caps indicates strong emotion
        }

    def analyze_emotion(self, text: str) -> Dict:
        """
        Comprehensive emotional analysis of input text

        Returns:
            dict: {
                'primary_emotion': str,
                'intensity': float (0-1),
                'valence': float (-1 to 1),
                'confidence': float (0-1),
                'secondary_emotions': list,
                'context_clues': list,
                'emotional_state': str
            }
        """
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)

        # Initialize emotion scores
        emotion_scores = {}
        context_modifiers = []

        # Analyze each word for emotional content
        for word in words:
            for emotion, data in self.emotion_lexicon.items():
                if word in data['words']:
                    base_intensity = data['intensity']
                    modified_intensity = self._apply_context_modifiers(text, base_intensity, word)

                    if emotion not in emotion_scores:
                        emotion_scores[emotion] = {
                            'score': 0,
                            'valence': data['valence'],
                            'occurrences': 0
                        }

                    emotion_scores[emotion]['score'] += modified_intensity
                    emotion_scores[emotion]['occurrences'] += 1

        # Apply punctuation and style modifiers
        punctuation_modifier = self._analyze_punctuation(text)
        caps_modifier = self._analyze_capitalization(text)

        for emotion in emotion_scores:
            emotion_scores[emotion]['score'] *= punctuation_modifier * caps_modifier

        # Determine primary and secondary emotions
        if emotion_scores:
            sorted_emotions = sorted(emotion_scores.items(),
                                   key=lambda x: x[1]['score'] * x[1]['occurrences'],
                                   reverse=True)

            primary_emotion = sorted_emotions[0][0]
            primary_score = sorted_emotions[0][1]['score']
            primary_valence = sorted_emotions[0][1]['valence']

            secondary_emotions = [e[0] for e in sorted_emotions[1:3]] if len(sorted_emotions) > 1 else []

            # Calculate confidence based on score consistency and context
            confidence = min(1.0, primary_score / 2.0)

            # Determine overall emotional state
            emotional_state = self._classify_emotional_state(primary_emotion, primary_valence, primary_score)

            return {
                'primary_emotion': primary_emotion,
                'intensity': min(1.0, primary_score),
                'valence': primary_valence,
                'confidence': confidence,
                'secondary_emotions': secondary_emotions,
                'context_clues': context_modifiers,
                'emotional_state': emotional_state,
                'raw_scores': emotion_scores
            }
        else:
            return {
                'primary_emotion': 'neutral',
                'intensity': 0.1,
                'valence': 0.0,
                'confidence': 0.5,
                'secondary_emotions': [],
                'context_clues': [],
                'emotional_state': 'neutral',
                'raw_scores': {}
            }

    def _apply_context_modifiers(self, text: str, base_intensity: float, target_word: str) -> float:
        """Apply contextual modifiers to emotional intensity"""
        modified_intensity = base_intensity

        # Check for intensifiers before the emotion word
        words = text.split()
        try:
            word_index = words.index(target_word)
            # Look at previous 2 words for intensifiers
            for i in range(max(0, word_index-2), word_index):
                if words[i] in self.context_patterns['intensifiers']:
                    modified_intensity *= 1.5
                elif words[i] in self.context_patterns['diminishers']:
                    modified_intensity *= 0.7
        except ValueError:
            pass

        # Check for negations
        if any(neg in text for neg in self.context_patterns['negations']):
            modified_intensity *= -0.8  # Negation flips or reduces emotion

        return modified_intensity

    def _analyze_punctuation(self, text: str) -> float:
        """Analyze punctuation for emotional intensity"""
        modifier = 1.0

        if '!!!' in text:
            modifier *= 1.6
        elif '!!' in text:
            modifier *= 1.4
        elif '!' in text:
            modifier *= 1.2

        if '???' in text:
            modifier *= 1.2
        elif '??' in text:
            modifier *= 1.1

        if '...' in text:
            modifier *= 0.9

        return modifier

    def _analyze_capitalization(self, text: str) -> float:
        """Analyze capitalization patterns"""
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)

        if caps_words > len(words) * 0.3:  # More than 30% caps
            return 1.3
        elif caps_words > 0:
            return 1.1

        return 1.0

    def _classify_emotional_state(self, emotion: str, valence: float, intensity: float) -> str:
        """Classify overall emotional state"""
        if intensity < 0.3:
            return 'neutral'
        elif valence > 0.5:
            if intensity > 0.7:
                return 'highly_positive'
            else:
                return 'positive'
        elif valence < -0.5:
            if intensity > 0.7:
                return 'highly_negative'
            else:
                return 'negative'
        else:
            return 'mixed'

    def get_empathy_response(self, emotion_analysis: Dict) -> str:
        """Generate empathetic response based on emotional analysis"""
        emotion = emotion_analysis['primary_emotion']
        intensity = emotion_analysis['intensity']
        state = emotion_analysis['emotional_state']

        empathy_templates = {
            'highly_positive': [
                "¡Qué maravilla! Me alegra tanto escuchar eso.",
                "¡Estoy tan feliz por ti! Eso suena increíble.",
                "¡Qué noticia tan buena! Me hace sonreír."
            ],
            'positive': [
                "Me alegra escuchar eso.",
                "Eso suena bien, ¿no?",
                "Qué bonito momento."
            ],
            'highly_negative': [
                "Lo siento mucho, eso debe ser realmente difícil.",
                "Estoy aquí para escucharte, cuéntame más si quieres.",
                "Entiendo que esto es duro, ¿hay algo que pueda hacer?"
            ],
            'negative': [
                "Entiendo que no estás pasando por un buen momento.",
                "A veces las cosas se ponen difíciles, estoy aquí.",
                "Siento que te sientas así."
            ],
            'neutral': [
                "Entiendo.",
                "Ajá, continúa.",
                "Te escucho."
            ],
            'mixed': [
                "Parece que hay sentimientos encontrados aquí.",
                "Entiendo que esto es complejo.",
                "Hay mucho que procesar en esto."
            ]
        }

        import random
        templates = empathy_templates.get(state, empathy_templates['neutral'])
        return random.choice(templates)


class EmotionalMemory:
    """Memory system that stores emotional context and relationships"""

    def __init__(self, memory_file: str = "emotional_memory.json"):
        self.memory_file = memory_file
        self.memory = self._load_memory()

    def _load_memory(self) -> Dict:
        """Load emotional memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            'user_emotional_profile': {},
            'conversation_history': [],
            'emotional_patterns': {},
            'relationship_context': {}
        }

    def save_memory(self):
        """Save emotional memory to file"""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def update_emotional_profile(self, user_id: str, emotion_analysis: Dict):
        """Update user's emotional profile"""
        if user_id not in self.memory['user_emotional_profile']:
            self.memory['user_emotional_profile'][user_id] = {
                'emotion_counts': {},
                'average_valence': 0,
                'average_intensity': 0,
                'interaction_count': 0,
                'last_emotion': None,
                'emotional_trends': []
            }

        profile = self.memory['user_emotional_profile'][user_id]

        # Update emotion counts
        emotion = emotion_analysis['primary_emotion']
        profile['emotion_counts'][emotion] = profile['emotion_counts'].get(emotion, 0) + 1

        # Update averages
        profile['interaction_count'] += 1
        profile['average_valence'] = (
            (profile['average_valence'] * (profile['interaction_count'] - 1)) +
            emotion_analysis['valence']
        ) / profile['interaction_count']

        profile['average_intensity'] = (
            (profile['average_intensity'] * (profile['interaction_count'] - 1)) +
            emotion_analysis['intensity']
        ) / profile['interaction_count']

        profile['last_emotion'] = emotion

        # Track emotional trends (last 10 interactions)
        profile['emotional_trends'].append({
            'emotion': emotion,
            'intensity': emotion_analysis['intensity'],
            'valence': emotion_analysis['valence'],
            'timestamp': datetime.now().isoformat()
        })
        if len(profile['emotional_trends']) > 10:
            profile['emotional_trends'].pop(0)

        self.save_memory()

    def get_emotional_context(self, user_id: str) -> Dict:
        """Get emotional context for user"""
        if user_id in self.memory['user_emotional_profile']:
            return self.memory['user_emotional_profile'][user_id]
        return {}

    def get_relationship_insights(self, user_id: str) -> List[str]:
        """Get insights about user's emotional patterns"""
        profile = self.get_emotional_context(user_id)
        insights = []

        if not profile:
            return ["Aún estoy conociéndote mejor."]

        # Analyze emotional patterns
        emotion_counts = profile['emotion_counts']
        if emotion_counts:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            insights.append(f"Veo que a menudo te sientes {dominant_emotion}.")

        # Analyze trends
        trends = profile['emotional_trends']
        if len(trends) >= 3:
            recent_emotions = [t['emotion'] for t in trends[-3:]]
            if len(set(recent_emotions)) == 1:
                insights.append(f"Últimamente has estado sintiéndote {recent_emotions[0]} consistentemente.")
            elif recent_emotions[0] != recent_emotions[-1]:
                insights.append("He notado cambios en tu estado de ánimo últimamente.")

        # Valence analysis
        avg_valence = profile['average_valence']
        if avg_valence > 0.3:
            insights.append("Generalmente tienes una perspectiva positiva.")
        elif avg_valence < -0.3:
            insights.append("A veces pareces llevar una carga emocional pesada.")

        return insights


if __name__ == "__main__":
    # Test the emotional analyzer
    analyzer = EmotionalAnalyzer()

    test_texts = [
        "I'm so excited about this new project!",
        "This is really frustrating and annoying.",
        "I'm feeling a bit sad today.",
        "Thank you so much! I really appreciate your help.",
        "I'm confused about what to do next.",
        "This makes me incredibly happy!!!"
    ]

    for text in test_texts:
        analysis = analyzer.analyze_emotion(text)
        empathy = analyzer.get_empathy_response(analysis)
        print(f"Text: {text}")
        print(f"Analysis: {analysis}")
        print(f"Empathy: {empathy}")
        print("-" * 50)
