#!/usr/bin/env python3
"""
Test AGI Training Capabilities
Test script to verify SOFIA's AGI training integration works correctly.
"""

import sys
import os
sys.path.append('.')

def test_agi_components():
    """Test that all AGI components can be imported and initialized"""
    print("Testing AGI Components...")

    # Test emotional intelligence
    try:
        from sofia_emotional_intelligence import EmotionalAnalyzer
        analyzer = EmotionalAnalyzer()
        result = analyzer.analyze_emotion("I feel happy today")
        print(f"✅ Emotional Intelligence: {result['primary_emotion']}")
    except ImportError:
        print("⚠️  Emotional Intelligence not available")

    # Test reinforcement learning
    try:
        from sofia_reinforcement_learning import ReinforcementLearner
        rl = ReinforcementLearner()
        print("✅ Reinforcement Learning: Initialized")
    except ImportError:
        print("⚠️  Reinforcement Learning not available")

    # Test LLM integration
    try:
        from sofia_llm_integration import SOFIALanguageModel
        llm = SOFIALanguageModel()
        print("✅ LLM Integration: Initialized")
    except ImportError:
        print("⚠️  LLM Integration not available")

def test_agi_training_setup():
    """Test AGI training setup"""
    print("\nTesting AGI Training Setup...")

    try:
        from sofia.train_sofia import AGITrainer, EmotionalDataset, ConversationDataset
        from sentence_transformers import SentenceTransformer

        # Test datasets
        emotional_ds = EmotionalDataset()
        conversation_ds = ConversationDataset()

        print(f"✅ Emotional Dataset: {len(emotional_ds)} samples")
        print(f"✅ Conversation Dataset: {len(conversation_ds)} samples")

        # Test AGI trainer
        config = {"agi_training": {"enable_emotional_training": True}}
        agi_trainer = AGITrainer(config)
        print("✅ AGI Trainer: Initialized")

    except Exception as e:
        print(f"❌ AGI Training Setup Error: {e}")

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting Configuration...")

    try:
        import yaml
        with open('sofia/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        agi_config = config.get('agi_training', {})
        print(f"✅ AGI Training Config: {agi_config.keys()}")
        print(f"   - Emotional Training: {agi_config.get('enable_emotional_training', False)}")
        print(f"   - RL Training: {agi_config.get('enable_reinforcement_learning', False)}")
        print(f"   - Conversation Training: {agi_config.get('enable_conversation_training', False)}")

    except Exception as e:
        print(f"❌ Config Loading Error: {e}")

if __name__ == "__main__":
    print("🚀 SOFIA AGI Training Test Suite")
    print("=" * 50)

    test_agi_components()
    test_agi_training_setup()
    test_config_loading()

    print("\n" + "=" * 50)
    print("🎯 AGI Training Integration Test Complete!")
    print("\nTo run full AGI training:")
    print("  cd sofia && python train_sofia.py --config config.yaml --agi-training")
