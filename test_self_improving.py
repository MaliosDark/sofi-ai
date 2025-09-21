#!/usr/bin/env python3
"""
Test script for SOFIA Self-Improving Learning System
"""

import sys
import os
import time
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sofia_self_improving import PerformanceMonitor, AdaptiveParameterTuner, SelfImprovingSOFIA

class MockModel:
    """Mock model for testing"""
    def __init__(self):
        self.parameters = {'lr': 1e-4, 'batch_size': 32}

def test_performance_monitor():
    """Test the performance monitoring system"""
    print("Testing Performance Monitor...")

    monitor = PerformanceMonitor(history_size=100)

    # Record some performance metrics
    monitor.record_performance('similarity', {'mteb_score': 62.1, 'similarity_accuracy': 0.88})
    time.sleep(0.1)  # Small delay
    monitor.record_performance('similarity', {'mteb_score': 63.8, 'similarity_accuracy': 0.91})
    time.sleep(0.1)
    monitor.record_performance('similarity', {'mteb_score': 65.1, 'similarity_accuracy': 0.93})

    # Test performance retrieval
    recent = monitor.get_recent_performance(hours=1)
    assert len(recent) >= 3, f"Expected at least 3 records, got {len(recent)}"

    # Test trend analysis
    trend = monitor.get_performance_trend('mteb_score', hours=1)
    assert trend['trend'] == 'improving', f"Expected improving trend, got {trend['trend']}"
    assert trend['change'] > 0, f"Expected positive change, got {trend['change']}"

    print("✓ Performance Monitor tests passed")
    return True

def test_adaptive_tuner():
    """Test the adaptive parameter tuning system"""
    print("Testing Adaptive Parameter Tuner...")

    mock_model = MockModel()
    tuner = AdaptiveParameterTuner(mock_model)

    # Test parameter tuning with improving performance
    params1 = tuner.tune_parameters({'mteb_score': 60.0})
    params2 = tuner.tune_parameters({'mteb_score': 65.0})  # Better performance

    # Learning rate should increase with better performance
    assert params2['learning_rate'] >= params1['learning_rate'], "Learning rate should increase with better performance"

    print("✓ Adaptive Parameter Tuner tests passed")
    return True

def test_self_improving_integration():
    """Test the integrated self-improving system"""
    print("Testing Self-Improving SOFIA Integration...")

    mock_model = MockModel()
    si_sofia = SelfImprovingSOFIA(mock_model)

    # Test recording performance
    si_sofia.record_task_performance('similarity', {'mteb_score': 65.1, 'similarity_accuracy': 0.93})

    # Test adding feedback data
    feedback_pairs = [("hello", "hi"), ("machine learning", "AI")]
    quality_scores = [0.9, 0.8]
    si_sofia.add_feedback_data(feedback_pairs, quality_scores)

    # Test system status
    status = si_sofia.get_system_status()
    assert 'is_learning' in status, "Status should include learning state"
    assert 'buffered_data' in status, "Status should include buffered data count"
    assert status['buffered_data'] == 2, f"Expected 2 buffered items, got {status['buffered_data']}"

    # Test state saving/loading
    si_sofia.save_state()
    assert os.path.exists('sofia_self_improvement_state.json'), "State file should be created"

    # Create new instance and load state
    si_sofia2 = SelfImprovingSOFIA(mock_model)
    status2 = si_sofia2.get_system_status()
    assert len(status2['recent_performance']) > 0, "State should be loaded with performance history"

    # Cleanup
    if os.path.exists('sofia_self_improvement_state.json'):
        os.remove('sofia_self_improvement_state.json')

    print("✓ Self-Improving SOFIA Integration tests passed")
    return True

def test_continuous_learning_simulation():
    """Test continuous learning simulation"""
    print("Testing Continuous Learning Simulation...")

    mock_model = MockModel()
    si_sofia = SelfImprovingSOFIA(mock_model)

    # Add some training data
    for i in range(60):  # More than min_data_points (50)
        si_sofia.add_feedback_data([("text" + str(i), "similar" + str(i))], [0.8])

    # Check that data is buffered
    status = si_sofia.get_system_status()
    assert status['buffered_data'] >= 50, f"Expected at least 50 buffered items, got {status['buffered_data']}"

    # Note: We don't start actual continuous learning in tests to avoid threading issues
    # In real usage, you would call: si_sofia.start_self_improvement()

    print("✓ Continuous Learning Simulation tests passed")
    return True

def main():
    """Run all self-improving system tests"""
    print("🧠 Testing SOFIA Self-Improving Learning System")
    print("=" * 60)

    tests = [
        test_performance_monitor,
        test_adaptive_tuner,
        test_self_improving_integration,
        test_continuous_learning_simulation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All self-improving system tests passed!")
        print("SOFIA can now monitor performance and improve automatically.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
