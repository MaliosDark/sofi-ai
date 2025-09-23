#!/usr/bin/env python3
"""
SOFIA Emotional AGI Test Script
Tests the emotional intelligence and reinforcement learning capabilities
"""

import asyncio
import json
from datetime import datetime
from sofia_emotional_intelligence import EmotionalAnalyzer, EmotionalMemory
from sofia_reinforcement_learning import SelfImprovingSOFIA

class SOFIATester:
    """Simple tester for SOFIA's emotional and learning capabilities"""

    def __init__(self):
        self.emotional_analyzer = EmotionalAnalyzer()
        self.emotional_memory = EmotionalMemory()
        self.sofia_rl = SelfImprovingSOFIA()

    def test_emotional_analysis(self):
        """Test emotional analysis capabilities"""
        print("🧠 EMOTIONAL ANALYSIS TEST")
        print("=" * 50)

        test_phrases = [
            'I am very frustrated with this technical problem',
            'I am scared that it won\'t turn out well',
            'I feel lonely and have no one to talk to',
            'I am worried about the future',
            'I am very happy with the project\'s progress!'
        ]
        for i, text in enumerate(test_phrases, 1):
            analysis = self.emotional_analyzer.analyze_emotion(text)
            empathy = self.emotional_analyzer.get_empathy_response(analysis)

            print(f"{i}. Text: {text}")
            print(f"   Emotion: {analysis['primary_emotion']} (intensity: {analysis['intensity']:.2f})")
            print(f"   State: {analysis['emotional_state']}")
            print(f"   Empathy: {empathy}")
            print()

    def test_conversation_flow(self, user_id: str = "test_user"):
        """Test conversational flow with emotional memory"""
        print("💬 CONVERSATION TEST WITH EMOTIONAL MEMORY")
        print("=" * 50)

        # Run async conversation test
        asyncio.run(self._async_conversation_test(user_id))

    async def _async_conversation_test(self, user_id: str):
        """Async helper for conversation testing"""
        conversation = [
            "Hello SOFIA, how are you?",
            "I'm a bit worried about the project.",
            "The project is going very well! I'm happy.",
            "But I'm scared that something might go wrong.",
            "Thanks for listening, I feel better now.",
            "What do you think about my mood lately?"
        ]

        for i, message in enumerate(conversation, 1):
            print(f"\n--- Interaction {i} ---")
            print(f"User: {message}")

            # Analyze emotion
            emotion_analysis = self.emotional_analyzer.analyze_emotion(message)
            print(f"Analysis: {emotion_analysis['primary_emotion']} ({emotion_analysis['intensity']:.2f})")

            # Get SOFIA's response
            result = await self.sofia_rl.process_user_input(message, user_id)
            print(f"SOFIA: {result['response']}")

            # Update emotional memory
            self.emotional_memory.update_emotional_profile(user_id, emotion_analysis)

            # Show relationship insights
            insights = self.emotional_memory.get_relationship_insights(user_id)
            if insights:
                print(f"Relational context: {insights[0]}")

            print(f"Calculated reward: {result['reward_calculated']:.2f}")

    def test_learning_adaptation(self):
        """Test how SOFIA learns and adapts over multiple interactions"""
        print("📈 LEARNING AND ADAPTATION TEST")
        print("=" * 50)

        # Run async learning test
        asyncio.run(self._async_learning_test())

    async def _async_learning_test(self):
        """Async helper for learning adaptation testing"""
        # Simulate multiple users with different emotional patterns
        users = {
            "optimistic": [
                "Everything is going perfectly!",
                "I'm very happy with the results.",
                "What a wonder! This is incredible."
            ],
            "worried": [
                "I'm worried about the deadlines.",
                "I'm scared that it won't turn out well.",
                "This is stressing me out a lot."
            ],
            "grateful": [
                "Thank you very much for your help.",
                "I really appreciate what you do.",
                "I'm very grateful for the support."
            ]
        }

        for user_type, messages in users.items():
            print(f"\n--- User type: {user_type.upper()} ---")

            for msg in messages:
                result = await self.sofia_rl.process_user_input(msg, f"user_{user_type}")
                print(f"Input: {msg}")
                print(f"Response: {result['response'][:80]}...")
                print(f"Reward: {result['reward_calculated']:.2f}")
                print()

        # Show learning statistics
        stats = self.sofia_rl.reinforcement_learner.get_learning_stats()
        print("📊 LEARNING STATISTICS:")
        print(f"- Learned states: {stats['total_states_learned']}")
        print(f"- Total interactions: {stats['total_interactions']}")
        print(f"- Average reward: {stats['average_reward']:.2f}")
        print(f"- Improvement actions: {stats['improvement_actions']}")

    def self_test_questions(self):
        """Self-test mode with predefined questions covering various emotional scenarios"""
        print("🧪 SELF-TEST MODE - SOFIA's Emotional Intelligence Assessment")
        print("=" * 70)

        # Run async self-test
        asyncio.run(self._async_self_test())

    async def _async_self_test(self):
        """Async helper for self-testing"""

        test_questions = [
            # Positive emotions
            "I'm very happy because I finished my project successfully!",
            "Thank you so much for your help, I really appreciate it!",
            "I'm so proud of what we've accomplished together.",
            "What joy! My team won the championship.",

            # Negative emotions
            "I'm feeling really sad today, everything seems to go wrong.",
            "I'm very frustrated with this technical problem that won't get resolved.",
            "I have a lot of anxiety about the upcoming presentation.",
            "I feel lonely and have no one to talk to.",

            # Mixed emotions
            "I'm excited about the new opportunity but also nervous about the changes.",
            "I'm happy with the results but worried about the future.",
            "I feel grateful for the support but overwhelmed by the workload.",

            # Questions about emotional state
            "How are you feeling today?",
            "What do you think about my current emotional state?",
            "Can you help me understand why I'm feeling this way?",

            # Work-related scenarios
            "My boss criticized my work and I'm feeling demotivated.",
            "The project is going very well! I'm motivated.",
            "I'm stressed about meeting the deadline.",

            # Personal/relationship scenarios
            "I had a fight with my friend and I feel guilty.",
            "I'm in love and very happy with my relationship.",
            "My family doesn't understand me and I feel isolated.",

            # Edge cases and challenges
            "This is terrible, everything sucks!",
            "I don't know what to do, I'm completely lost.",
            "Why does everything go wrong for me?",
            "You stupid AI, you don't understand anything!",

            # Recovery and growth
            "I made a mistake but I'm ready to learn from it.",
            "After a bad day, how can I improve my mood?",
            "I'm working on being more positive, can you help me?"
        ]

        user_id = "self_test_user"
        results = []

        print(f"Testing with {len(test_questions)} predefined questions...\n")

        for i, question in enumerate(test_questions, 1):
            print(f"--- Test {i:2d}/{len(test_questions)} ---")
            print(f"User: {question}")

            try:
                # Process with full SOFIA
                result = await self.sofia_rl.process_user_input(question, user_id)

                print(f"SOFIA: {result['response']}")
                print(f"[Emotion: {result['emotion_analysis']['primary_emotion']} | "
                      f"Intensity: {result['emotion_analysis']['intensity']:.2f} | "
                      f"Reward: {result['reward_calculated']:.2f}]")

                # Store result for summary
                results.append({
                    'question': question,
                    'emotion': result['emotion_analysis']['primary_emotion'],
                    'intensity': result['emotion_analysis']['intensity'],
                    'reward': result['reward_calculated'],
                    'response_length': len(result['response'])
                })

            except Exception as e:
                print(f"❌ Error processing question: {e}")
                results.append({
                    'question': question,
                    'error': str(e)
                })

            print()

        # Generate summary report
        self._generate_self_test_report(results)

    def _generate_self_test_report(self, results):
        """Generate a comprehensive report of the self-test results"""
        print("📊 SELF-TEST RESULTS SUMMARY")
        print("=" * 70)

        successful_tests = [r for r in results if 'error' not in r]
        failed_tests = [r for r in results if 'error' in r]

        print(f"Total questions tested: {len(results)}")
        print(f"Successful responses: {len(successful_tests)}")
        print(f"Failed responses: {len(failed_tests)}")

        if successful_tests:
            # Emotion distribution
            emotions = {}
            for r in successful_tests:
                emotion = r['emotion']
                emotions[emotion] = emotions.get(emotion, 0) + 1

            print(f"\n🎭 Emotion Detection Distribution:")
            for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(successful_tests)) * 100
                print(f"  {emotion}: {count} ({percentage:.1f}%)")

            # Reward statistics
            rewards = [r['reward'] for r in successful_tests]
            avg_reward = sum(rewards) / len(rewards)
            max_reward = max(rewards)
            min_reward = min(rewards)

            print(f"\n💰 Reward Statistics:")
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Highest reward: {max_reward:.2f}")
            print(f"  Lowest reward: {min_reward:.2f}")

            # Response quality metrics
            response_lengths = [r['response_length'] for r in successful_tests]
            avg_length = sum(response_lengths) / len(response_lengths)

            print(f"\n📝 Response Quality:")
            print(f"  Average response length: {avg_length:.0f} characters")
            print(f"  Longest response: {max(response_lengths)} characters")
            print(f"  Shortest response: {min(response_lengths)} characters")

        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for i, failure in enumerate(failed_tests, 1):
                print(f"  {i}. {failure['question'][:50]}... -> {failure['error']}")

        # Learning statistics
        try:
            stats = self.sofia_rl.reinforcement_learner.get_learning_stats()
            print(f"\n🧠 Learning Progress:")
            print(f"  States learned: {stats['total_states_learned']}")
            print(f"  Total interactions: {stats['total_interactions']}")
            print(f"  Average reward: {stats['average_reward']:.2f}")
            print(f"  Improvement actions: {stats['improvement_actions']}")
        except:
            print("\n🧠 Learning statistics not available")

        print(f"\n✅ Self-test completed! SOFIA demonstrated emotional intelligence across {len(successful_tests)} scenarios.")

def main():
    """Main test function"""
    print("🤖 SOFIA EMOTIONAL AGI TEST SUITE")
    print("=" * 60)
    print("Testing emotional intelligence and reinforcement learning capabilities")
    print()

    tester = SOFIATester()

    # Run tests
    try:
        print("1. Running emotional analysis tests...")
        tester.test_emotional_analysis()

        print("\n2. Running conversation test with memory...")
        tester.test_conversation_flow()

        print("\n3. Running learning adaptation test...")
        tester.test_learning_adaptation()

        print("\n4. Running self-test with predefined questions...")
        tester.self_test_questions()

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
