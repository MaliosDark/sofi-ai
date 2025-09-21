#!/usr/bin/env python3
"""
Test script for SOFIA Meta-Cognition System
"""

import sys
import os
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sofia_meta_cognition import (
    ConfidenceEstimator, ErrorDetector, DecisionAnalyzer, MetaCognitiveSOFIA
)

def test_confidence_estimator():
    """Test the confidence estimation system"""
    print("Testing Confidence Estimator...")

    estimator = ConfidenceEstimator()

    # Create dummy embeddings
    emb1 = torch.randn(2, 768)  # Batch of 2
    emb2 = torch.randn(2, 768)

    confidence = estimator(emb1, emb2)

    assert confidence.shape == (2,), f"Expected shape (2,), got {confidence.shape}"
    assert torch.all((confidence >= 0) & (confidence <= 1)), "Confidence should be between 0 and 1"

    print("✓ Confidence Estimator tests passed")
    return True

def test_error_detector():
    """Test the error detection system"""
    print("Testing Error Detector...")

    detector = ErrorDetector(error_threshold=0.2)

    # Test error detection
    error_info = detector.detect_error(
        prediction=0.8, ground_truth=0.9, confidence=0.7,
        context={'text1_length': 5, 'text2_length': 6}
    )

    assert 'is_error' in error_info, "Error info should contain is_error"
    assert 'error_magnitude' in error_info, "Error info should contain error_magnitude"
    assert abs(error_info['error_magnitude'] - 0.1) < 1e-10, f"Expected error magnitude ~0.1, got {error_info['error_magnitude']}"

    # Test with large error
    error_info2 = detector.detect_error(
        prediction=0.3, ground_truth=0.9, confidence=0.8,
        context={'text1_length': 10, 'text2_length': 3}
    )

    assert error_info2['is_error'] == True, "Should detect error with large magnitude"

    # Test statistics
    stats = detector.get_error_statistics()
    assert 'total_errors' in stats, "Statistics should include total_errors"
    assert stats['total_errors'] >= 1, "Should have at least 1 error recorded"

    print("✓ Error Detector tests passed")
    return True

def test_decision_analyzer():
    """Test the decision analysis system"""
    print("Testing Decision Analyzer...")

    analyzer = DecisionAnalyzer()

    # Test decision analysis
    results = [(0, 0.9), (1, 0.6), (2, 0.4)]
    candidates = ["AI definition", "Weather info", "Sports news"]

    analysis = analyzer.analyze_decision(
        "What is artificial intelligence?",
        results, {'num_candidates': 3}
    )

    assert 'decision_confidence' in analysis, "Analysis should include decision_confidence"
    assert analysis['decision_confidence'] > 0.8, f"Expected high confidence, got {analysis['decision_confidence']}"

    # Test insights
    insights = analyzer.get_decision_insights()
    assert 'total_decisions' in insights, "Insights should include total_decisions"
    assert insights['total_decisions'] >= 1, "Should have at least 1 decision recorded"

    print("✓ Decision Analyzer tests passed")
    return True

def test_meta_cognitive_integration():
    """Test the integrated meta-cognitive system"""
    print("Testing Meta-Cognitive SOFIA Integration...")

    meta_sofia = MetaCognitiveSOFIA()

    # Test prediction analysis
    analysis = meta_sofia.analyze_prediction(
        "Hello world", "Hi there",
        prediction=0.85, ground_truth=0.82
    )

    assert 'prediction' in analysis, "Analysis should include prediction"
    assert 'confidence' in analysis, "Analysis should include confidence"
    assert 'error_analysis' in analysis, "Analysis should include error_analysis"

    # Test decision analysis
    decision_analysis = meta_sofia.analyze_decision(
        "What is AI?", [(0, 0.9), (1, 0.7)], ["AI definition", "Weather"]
    )

    assert 'decision_confidence' in decision_analysis, "Decision analysis should include confidence"

    # Test self-reflection
    reflection = meta_sofia.reflect_on_performance()

    assert 'overall_assessment' in reflection, "Reflection should include overall_assessment"
    assert 'strengths' in reflection, "Reflection should include strengths"
    assert 'weaknesses' in reflection, "Reflection should include weaknesses"
    assert 'improvement_suggestions' in reflection, "Reflection should include improvement_suggestions"

    # Test meta-cognitive state
    state = meta_sofia.get_meta_cognitive_state()

    assert 'self_awareness_level' in state, "State should include self_awareness_level"
    assert 'error_statistics' in state, "State should include error_statistics"
    assert 'decision_insights' in state, "State should include decision_insights"

    print("✓ Meta-Cognitive SOFIA Integration tests passed")
    return True

def test_self_reflection():
    """Test the self-reflection capabilities"""
    print("Testing Self-Reflection Capabilities...")

    meta_sofia = MetaCognitiveSOFIA()

    # Simulate some performance data
    for i in range(10):
        # Mix of correct and incorrect predictions
        is_correct = i % 3 != 0  # 2/3 correct
        prediction = 0.8 if is_correct else 0.3
        ground_truth = 0.8 if is_correct else 0.8

        meta_sofia.analyze_prediction(
            f"Text {i}", f"Similar text {i}",
            prediction=prediction, ground_truth=ground_truth
        )

    # Test reflection
    reflection = meta_sofia.reflect_on_performance()

    assert reflection['overall_assessment'] in ['excellent', 'good', 'adequate', 'needs_improvement']
    assert isinstance(reflection['strengths'], list), "Strengths should be a list"
    assert isinstance(reflection['weaknesses'], list), "Weaknesses should be a list"
    assert isinstance(reflection['improvement_suggestions'], list), "Suggestions should be a list"
    assert 0 <= reflection['confidence_level'] <= 1, "Confidence level should be between 0 and 1"

    print("✓ Self-Reflection tests passed")
    return True

def main():
    """Run all meta-cognition tests"""
    print("🧠 Testing SOFIA Meta-Cognition System")
    print("=" * 50)

    tests = [
        test_confidence_estimator,
        test_error_detector,
        test_decision_analyzer,
        test_meta_cognitive_integration,
        test_self_reflection
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All meta-cognition tests passed!")
        print("SOFIA now has self-awareness, error detection, and decision analysis capabilities.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
