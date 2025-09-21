#!/usr/bin/env python3
"""
SOFIA Meta-Cognition System
Provides self-awareness, error detection, and decision analysis capabilities
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfidenceEstimator(nn.Module):
    """
    Estimates confidence scores for SOFIA's predictions
    """

    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """Estimate confidence for similarity prediction"""
        combined = torch.cat([embedding1, embedding2], dim=1)
        confidence = self.confidence_head(combined)
        return confidence.squeeze()

class ErrorDetector:
    """
    Detects and analyzes errors in SOFIA's predictions
    """

    def __init__(self, error_threshold: float = 0.3):
        self.error_threshold = error_threshold
        self.error_history = deque(maxlen=1000)
        self.error_patterns = defaultdict(int)

    def detect_error(self, prediction: float, ground_truth: float,
                    confidence: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if a prediction contains an error

        Args:
            prediction: Model's similarity prediction (0-1)
            ground_truth: Actual similarity score (0-1)
            confidence: Model's confidence in prediction (0-1)
            context: Additional context information

        Returns:
            Error analysis dictionary
        """

        error_magnitude = abs(prediction - ground_truth)
        is_error = error_magnitude > self.error_threshold

        # Low confidence + high error = likely error
        confidence_weighted_error = error_magnitude * (1 - confidence)

        error_info = {
            'is_error': is_error,
            'error_magnitude': error_magnitude,
            'confidence_weighted_error': confidence_weighted_error,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'confidence': confidence,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'error_type': self._classify_error(prediction, ground_truth, confidence)
        }

        if is_error:
            self.error_history.append(error_info)
            self._update_error_patterns(error_info)

        return error_info

    def _classify_error(self, prediction: float, ground_truth: float, confidence: float) -> str:
        """Classify the type of error"""
        error_mag = abs(prediction - ground_truth)

        if confidence < 0.3 and error_mag > 0.5:
            return "low_confidence_high_error"
        elif abs(prediction - 0.5) < 0.1 and abs(ground_truth - 0.5) > 0.3:
            return "neutral_prediction_bias"
        elif (prediction > 0.8 and ground_truth < 0.3) or (prediction < 0.2 and ground_truth > 0.7):
            return "extreme_misclassification"
        elif error_mag > 0.4:
            return "large_error"
        else:
            return "moderate_error"

    def _update_error_patterns(self, error_info: Dict[str, Any]):
        """Update error pattern statistics"""
        error_type = error_info['error_type']
        self.error_patterns[error_type] += 1

        # Analyze context patterns
        context = error_info.get('context', {})
        if 'text1_length' in context and 'text2_length' in context:
            length_ratio = context['text1_length'] / max(context['text2_length'], 1)
            if length_ratio > 3 or length_ratio < 0.33:
                self.error_patterns['length_mismatch'] += 1

        if 'domain' in context:
            domain = context['domain']
            self.error_patterns[f'domain_{domain}'] += 1

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        if not self.error_history:
            return {'total_errors': 0, 'error_rate': 0.0}

        total_predictions = len(self.error_history)
        errors = sum(1 for e in self.error_history if e['is_error'])
        error_rate = errors / total_predictions

        # Error magnitude statistics
        error_magnitudes = [e['error_magnitude'] for e in self.error_history if e['is_error']]
        avg_error_magnitude = statistics.mean(error_magnitudes) if error_magnitudes else 0

        # Confidence analysis
        confidence_when_wrong = [e['confidence'] for e in self.error_history if e['is_error']]
        avg_confidence_wrong = statistics.mean(confidence_when_wrong) if confidence_when_wrong else 0

        return {
            'total_errors': errors,
            'total_predictions': total_predictions,
            'error_rate': error_rate,
            'average_error_magnitude': avg_error_magnitude,
            'average_confidence_when_wrong': avg_confidence_wrong,
            'error_patterns': dict(self.error_patterns),
            'recent_errors': list(self.error_history)[-10:]  # Last 10 errors
        }

class DecisionAnalyzer:
    """
    Analyzes SOFIA's decision-making process and provides insights
    """

    def __init__(self):
        self.decision_history = deque(maxlen=2000)
        self.decision_patterns = defaultdict(lambda: defaultdict(int))

    def analyze_decision(self, query: str, results: List[Tuple[int, float]],
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a decision-making process

        Args:
            query: The input query
            results: List of (index, score) tuples
            context: Additional context

        Returns:
            Decision analysis
        """

        analysis = {
            'query': query,
            'top_result_score': results[0][1] if results else 0,
            'result_distribution': self._analyze_score_distribution(results),
            'query_characteristics': self._analyze_query(query),
            'decision_confidence': self._calculate_decision_confidence(results),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }

        self.decision_history.append(analysis)
        self._update_decision_patterns(analysis)

        return analysis

    def _analyze_score_distribution(self, results: List[Tuple[int, float]]) -> Dict[str, Any]:
        """Analyze the distribution of similarity scores"""
        if not results:
            return {'distribution_type': 'no_results'}

        scores = [score for _, score in results]

        # Calculate statistics
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        max_score = max(scores)
        min_score = min(scores)

        # Classify distribution
        if std_score < 0.1:
            dist_type = 'uniform'
        elif max_score - min_score > 0.5:
            dist_type = 'wide_spread'
        elif mean_score > 0.7:
            dist_type = 'high_similarity'
        elif mean_score < 0.3:
            dist_type = 'low_similarity'
        else:
            dist_type = 'moderate_spread'

        return {
            'distribution_type': dist_type,
            'mean_score': mean_score,
            'std_score': std_score,
            'max_score': max_score,
            'min_score': min_score,
            'score_range': max_score - min_score
        }

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics"""
        words = query.split()
        sentences = re.split(r'[.!?]+', query)

        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': statistics.mean([len(word) for word in words]) if words else 0,
            'contains_questions': '?' in query,
            'contains_numbers': any(char.isdigit() for char in query),
            'is_short': len(words) < 5,
            'is_long': len(words) > 20
        }

    def _calculate_decision_confidence(self, results: List[Tuple[int, float]]) -> float:
        """Calculate confidence in the decision"""
        if not results or len(results) < 2:
            return 0.5  # Neutral confidence

        top_score = results[0][1]
        second_score = results[1][1]

        # Confidence based on margin between top and second result
        margin = top_score - second_score

        if margin > 0.3:
            confidence = 0.9
        elif margin > 0.2:
            confidence = 0.8
        elif margin > 0.1:
            confidence = 0.7
        elif margin > 0.05:
            confidence = 0.6
        else:
            confidence = 0.5

        return confidence

    def _update_decision_patterns(self, analysis: Dict[str, Any]):
        """Update decision pattern statistics"""
        query_chars = analysis['query_characteristics']

        # Track patterns by query type
        if query_chars['contains_questions']:
            self.decision_patterns['question_queries']['count'] += 1
            self.decision_patterns['question_queries']['avg_confidence'] = (
                (self.decision_patterns['question_queries'].get('avg_confidence', 0) * (
                    self.decision_patterns['question_queries']['count'] - 1) + analysis['decision_confidence']) /
                self.decision_patterns['question_queries']['count']
            )

        if query_chars['is_short']:
            self.decision_patterns['short_queries']['count'] += 1

        # Track distribution patterns
        dist_type = analysis['result_distribution']['distribution_type']
        self.decision_patterns['distributions'][dist_type] += 1

    def get_decision_insights(self) -> Dict[str, Any]:
        """Get insights from decision analysis"""
        if not self.decision_history:
            return {'total_decisions': 0}

        total_decisions = len(self.decision_history)
        recent_decisions = list(self.decision_history)[-100:]  # Last 100 decisions

        # Confidence analysis
        confidences = [d['decision_confidence'] for d in recent_decisions]
        avg_confidence = statistics.mean(confidences)

        # Query type analysis
        query_types = defaultdict(int)
        for decision in recent_decisions:
            chars = decision['query_characteristics']
            if chars['contains_questions']:
                query_types['questions'] += 1
            if chars['is_short']:
                query_types['short'] += 1
            if chars['is_long']:
                query_types['long'] += 1

        # Performance patterns
        high_confidence_decisions = sum(1 for c in confidences if c > 0.8)
        low_confidence_decisions = sum(1 for c in confidences if c < 0.6)

        return {
            'total_decisions': total_decisions,
            'average_confidence': avg_confidence,
            'high_confidence_rate': high_confidence_decisions / len(confidences),
            'low_confidence_rate': low_confidence_decisions / len(confidences),
            'query_type_distribution': dict(query_types),
            'decision_patterns': dict(self.decision_patterns)
        }

class MetaCognitiveSOFIA:
    """
    Main meta-cognition system for SOFIA
    """

    def __init__(self, model=None):
        self.model = model
        self.confidence_estimator = ConfidenceEstimator()
        self.error_detector = ErrorDetector()
        self.decision_analyzer = DecisionAnalyzer()

        # Meta-cognitive state
        self.self_awareness_level = 0.0
        self.learning_from_errors = True

    def analyze_prediction(self, text1: str, text2: str,
                          prediction: float, ground_truth: Optional[float] = None,
                          embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Perform meta-cognitive analysis of a prediction
        """

        # Estimate confidence if embeddings available
        confidence = 0.5  # Default neutral confidence
        if embeddings:
            confidence = self.confidence_estimator(embeddings[0], embeddings[1]).item()

        # Create context
        context = {
            'text1_length': len(text1.split()),
            'text2_length': len(text2.split()),
            'text1': text1[:100],  # Truncated for storage
            'text2': text2[:100],
            'domain': self._infer_domain(text1, text2)
        }

        result = {
            'prediction': prediction,
            'confidence': confidence,
            'context': context
        }

        # Analyze error if ground truth available
        if ground_truth is not None:
            error_analysis = self.error_detector.detect_error(
                prediction, ground_truth, confidence, context
            )
            result['error_analysis'] = error_analysis

            # Update self-awareness based on error patterns
            self._update_self_awareness(error_analysis)

        return result

    def analyze_decision(self, query: str, results: List[Tuple[int, float]],
                        candidates: List[str]) -> Dict[str, Any]:
        """
        Analyze a decision-making process
        """

        context = {
            'num_candidates': len(candidates),
            'query_type': 'search' if len(results) > 1 else 'single',
            'top_candidate': candidates[results[0][0]] if results and candidates else None
        }

        analysis = self.decision_analyzer.analyze_decision(query, results, context)

        # Update self-awareness based on decision patterns
        self._update_self_awareness_from_decision(analysis)

        return analysis

    def _infer_domain(self, text1: str, text2: str) -> str:
        """Infer the domain/topic of the texts"""
        combined_text = (text1 + " " + text2).lower()

        # Simple domain detection
        if any(word in combined_text for word in ['computer', 'software', 'programming', 'code']):
            return 'technology'
        elif any(word in combined_text for word in ['health', 'medical', 'disease', 'treatment']):
            return 'health'
        elif any(word in combined_text for word in ['business', 'company', 'market', 'finance']):
            return 'business'
        elif any(word in combined_text for word in ['science', 'research', 'study', 'experiment']):
            return 'science'
        else:
            return 'general'

    def _update_self_awareness(self, error_analysis: Dict[str, Any]):
        """Update self-awareness based on error analysis"""
        if not error_analysis.get('is_error', False):
            # Correct prediction - slight increase in awareness
            self.self_awareness_level = min(1.0, self.self_awareness_level + 0.01)
        else:
            # Error - analyze and learn
            error_magnitude = error_analysis.get('error_magnitude', 0)
            confidence = error_analysis.get('confidence', 0)

            # Large errors with high confidence decrease awareness more
            awareness_penalty = error_magnitude * confidence * 0.1
            self.self_awareness_level = max(0.0, self.self_awareness_level - awareness_penalty)

            # But learning from errors can help recover
            if self.learning_from_errors:
                recovery = error_magnitude * 0.05  # Learn from mistakes
                self.self_awareness_level = min(1.0, self.self_awareness_level + recovery)

    def _update_self_awareness_from_decision(self, decision_analysis: Dict[str, Any]):
        """Update self-awareness based on decision analysis"""
        confidence = decision_analysis.get('decision_confidence', 0.5)

        # High confidence decisions increase awareness
        if confidence > 0.8:
            self.self_awareness_level = min(1.0, self.self_awareness_level + 0.005)
        elif confidence < 0.4:
            # Low confidence decisions slightly decrease awareness
            self.self_awareness_level = max(0.0, self.self_awareness_level - 0.002)

    def get_meta_cognitive_state(self) -> Dict[str, Any]:
        """Get current meta-cognitive state"""
        return {
            'self_awareness_level': self.self_awareness_level,
            'error_statistics': self.error_detector.get_error_statistics(),
            'decision_insights': self.decision_analyzer.get_decision_insights(),
            'confidence_in_abilities': self._assess_ability_confidence(),
            'learning_active': self.learning_from_errors
        }

    def _assess_ability_confidence(self) -> Dict[str, float]:
        """Assess confidence in different abilities"""
        error_stats = self.error_detector.get_error_statistics()
        decision_insights = self.decision_analyzer.get_decision_insights()

        # Base confidence on error rates and decision patterns
        error_rate = error_stats.get('error_rate', 0.5)
        avg_decision_confidence = decision_insights.get('average_confidence', 0.5)

        return {
            'similarity_prediction': 1.0 - error_rate,
            'decision_making': avg_decision_confidence,
            'error_detection': min(1.0, self.self_awareness_level + 0.3),
            'domain_adaptation': 0.7 if error_stats.get('error_patterns', {}).get('domain_general', 0) < 10 else 0.5
        }

    def reflect_on_performance(self) -> Dict[str, Any]:
        """Perform self-reflection on recent performance"""
        state = self.get_meta_cognitive_state()

        reflection = {
            'overall_assessment': self._assess_overall_performance(state),
            'strengths': self._identify_strengths(state),
            'weaknesses': self._identify_weaknesses(state),
            'improvement_suggestions': self._generate_improvement_suggestions(state),
            'confidence_level': state['self_awareness_level']
        }

        return reflection

    def _assess_overall_performance(self, state: Dict[str, Any]) -> str:
        """Assess overall performance level"""
        awareness = state['self_awareness_level']
        error_rate = state['error_statistics'].get('error_rate', 0.5)
        decision_confidence = state['decision_insights'].get('average_confidence', 0.5)

        overall_score = (awareness + (1 - error_rate) + decision_confidence) / 3

        if overall_score > 0.8:
            return "excellent"
        elif overall_score > 0.6:
            return "good"
        elif overall_score > 0.4:
            return "adequate"
        else:
            return "needs_improvement"

    def _identify_strengths(self, state: Dict[str, Any]) -> List[str]:
        """Identify current strengths"""
        strengths = []

        if state['self_awareness_level'] > 0.7:
            strengths.append("High self-awareness and error detection")

        error_rate = state['error_statistics'].get('error_rate', 0.5)
        if error_rate < 0.2:
            strengths.append("Low error rate in predictions")

        decision_conf = state['decision_insights'].get('average_confidence', 0.5)
        if decision_conf > 0.8:
            strengths.append("High confidence in decision making")

        if not strengths:
            strengths.append("Continuous learning capability")

        return strengths

    def _identify_weaknesses(self, state: Dict[str, Any]) -> List[str]:
        """Identify current weaknesses"""
        weaknesses = []

        if state['self_awareness_level'] < 0.3:
            weaknesses.append("Limited self-awareness")

        error_patterns = state['error_statistics'].get('error_patterns', {})
        if error_patterns.get('large_error', 0) > 5:
            weaknesses.append("Frequent large prediction errors")

        if error_patterns.get('low_confidence_high_error', 0) > 3:
            weaknesses.append("Overconfidence in incorrect predictions")

        decision_insights = state['decision_insights']
        if decision_insights.get('low_confidence_rate', 0) > 0.3:
            weaknesses.append("Low confidence in many decisions")

        if not weaknesses:
            weaknesses.append("Still learning and adapting")

        return weaknesses

    def _generate_improvement_suggestions(self, state: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improvement"""
        suggestions = []

        error_stats = state['error_statistics']
        if error_stats.get('error_rate', 0) > 0.3:
            suggestions.append("Focus on reducing prediction errors through additional training")

        if state['self_awareness_level'] < 0.5:
            suggestions.append("Improve self-awareness by analyzing more prediction outcomes")

        decision_insights = state['decision_insights']
        if decision_insights.get('low_confidence_rate', 0) > 0.2:
            suggestions.append("Work on increasing decision confidence through better calibration")

        error_patterns = error_stats.get('error_patterns', {})
        if error_patterns.get('domain_general', 0) > error_patterns.get('domain_technology', 0):
            suggestions.append("Specialize more in technology domain where errors are lower")

        if len(suggestions) == 0:
            suggestions.append("Continue current learning approach - performance is stable")

        return suggestions


# Example usage
if __name__ == "__main__":
    print("SOFIA Meta-Cognition System")
    print("This system provides self-awareness, error detection, and decision analysis")

    # Example usage would be integrated with SOFIA model
    """
    from sofia_model import SOFIAModel
    from sofia_meta_cognition import MetaCognitiveSOFIA

    sofia = SOFIAModel()
    meta_sofia = MetaCognitiveSOFIA(sofia)

    # Analyze a prediction
    result = meta_sofia.analyze_prediction(
        "Hello world", "Hi there",
        prediction=0.85, ground_truth=0.9
    )

    # Analyze a decision
    decision_analysis = meta_sofia.analyze_decision(
        "What is AI?", [(0, 0.9), (1, 0.7)], ["AI definition", "Weather info"]
    )

    # Get self-reflection
    reflection = meta_sofia.reflect_on_performance()
    print("Self-reflection:", reflection)
    """
