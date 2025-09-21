#!/usr/bin/env python3
"""
SOFIA Advanced Reasoning Engine
Implements task decomposition, strategy selection, and logical reasoning
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
from collections import defaultdict, deque
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Task:
    """
    Represents a task that can be decomposed and reasoned about
    """

    def __init__(self, description: str, complexity: int = 1, dependencies: List[str] = None):
        self.description = description
        self.complexity = complexity  # 1-10 scale
        self.dependencies = dependencies or []
        self.subtasks = []
        self.completed = False
        self.created_at = datetime.now()
        self.estimated_time = self._estimate_time()

    def _estimate_time(self) -> float:
        """Estimate time required based on complexity and description"""
        base_time = self.complexity * 5  # 5 minutes per complexity unit

        # Adjust based on keywords
        desc_lower = self.description.lower()
        if any(word in desc_lower for word in ['research', 'analyze', 'investigate']):
            base_time *= 1.5
        if any(word in desc_lower for word in ['create', 'build', 'implement']):
            base_time *= 2.0
        if any(word in desc_lower for word in ['simple', 'quick', 'basic']):
            base_time *= 0.5

        return base_time

    def add_subtask(self, subtask: 'Task'):
        """Add a subtask to this task"""
        self.subtasks.append(subtask)

    def mark_completed(self):
        """Mark this task as completed"""
        self.completed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            'description': self.description,
            'complexity': self.complexity,
            'dependencies': self.dependencies,
            'subtasks': [st.to_dict() for st in self.subtasks],
            'completed': self.completed,
            'estimated_time': self.estimated_time,
            'created_at': self.created_at.isoformat()
        }

class ReasoningStrategy:
    """
    Represents a reasoning strategy for solving problems
    """

    def __init__(self, name: str, description: str, applicable_domains: List[str],
                 success_rate: float = 0.7, avg_time: float = 10.0):
        self.name = name
        self.description = description
        self.applicable_domains = applicable_domains
        self.success_rate = success_rate
        self.avg_time = avg_time
        self.usage_count = 0
        self.success_count = 0

    def is_applicable(self, problem_domain: str, problem_complexity: int) -> bool:
        """Check if this strategy is applicable to the given problem"""
        domain_match = problem_domain in self.applicable_domains or 'general' in self.applicable_domains

        # Some strategies work better for different complexity levels
        if 'simple_problems' in self.applicable_domains and problem_complexity <= 3:
            return True
        if 'complex_problems' in self.applicable_domains and problem_complexity >= 7:
            return True

        return domain_match

    def record_usage(self, success: bool):
        """Record usage of this strategy"""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.success_rate = self.success_count / self.usage_count

    def get_effectiveness_score(self) -> float:
        """Get effectiveness score for strategy selection"""
        # Combine success rate with usage experience
        experience_factor = min(1.0, self.usage_count / 10)  # Diminishing returns after 10 uses
        return self.success_rate * (0.5 + 0.5 * experience_factor)

class TaskDecomposer:
    """
    Decomposes complex tasks into manageable subtasks
    """

    def __init__(self):
        self.decomposition_patterns = self._load_patterns()

    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load task decomposition patterns"""
        return {
            'research': [
                'Define research question',
                'Identify information sources',
                'Gather relevant data',
                'Analyze findings',
                'Synthesize conclusions'
            ],
            'implementation': [
                'Analyze requirements',
                'Design solution architecture',
                'Implement core functionality',
                'Add error handling',
                'Test implementation',
                'Document solution'
            ],
            'problem_solving': [
                'Understand the problem',
                'Break down into components',
                'Identify potential solutions',
                'Evaluate solution options',
                'Implement chosen solution',
                'Verify solution works'
            ],
            'learning': [
                'Assess current knowledge',
                'Identify learning objectives',
                'Find learning resources',
                'Study materials',
                'Practice concepts',
                'Assess understanding'
            ]
        }

    def decompose_task(self, task_description: str, complexity: int = 5) -> Task:
        """Decompose a task into subtasks"""
        # Identify task type
        task_type = self._classify_task_type(task_description)

        # Create main task
        main_task = Task(task_description, complexity)

        # Get decomposition pattern
        if task_type in self.decomposition_patterns:
            pattern = self.decomposition_patterns[task_type]

            # Adjust pattern based on complexity
            if complexity <= 3:
                # Simplify for simple tasks
                pattern = pattern[:3]
            elif complexity >= 8:
                # Expand for complex tasks
                pattern.extend([
                    'Review and refine',
                    'Optimize performance',
                    'Prepare for deployment'
                ])

            # Create subtasks
            for i, subtask_desc in enumerate(pattern):
                subtask_complexity = max(1, complexity // len(pattern))
                subtask = Task(subtask_desc, subtask_complexity)
                main_task.add_subtask(subtask)

        else:
            # Generic decomposition for unknown task types
            subtasks = self._generic_decomposition(task_description, complexity)
            for subtask_desc in subtasks:
                subtask = Task(subtask_desc, max(1, complexity // len(subtasks)))
                main_task.add_subtask(subtask)

        return main_task

    def _classify_task_type(self, description: str) -> str:
        """Classify the type of task"""
        desc_lower = description.lower()

        if any(word in desc_lower for word in ['research', 'investigate', 'analyze', 'study']):
            return 'research'
        elif any(word in desc_lower for word in ['implement', 'build', 'create', 'develop']):
            return 'implementation'
        elif any(word in desc_lower for word in ['solve', 'fix', 'resolve', 'answer']):
            return 'problem_solving'
        elif any(word in desc_lower for word in ['learn', 'understand', 'master', 'study']):
            return 'learning'
        else:
            return 'general'

    def _generic_decomposition(self, description: str, complexity: int) -> List[str]:
        """Generic task decomposition when no specific pattern matches"""
        base_steps = [
            'Plan the approach',
            'Gather necessary resources',
            'Execute the main work',
            'Review and verify results'
        ]

        if complexity > 5:
            base_steps.insert(1, 'Break down into smaller steps')
            base_steps.insert(-1, 'Test and validate')

        return base_steps

class StrategySelector:
    """
    Selects the best reasoning strategy for a given problem
    """

    def __init__(self):
        self.strategies = self._initialize_strategies()

    def _initialize_strategies(self) -> List[ReasoningStrategy]:
        """Initialize available reasoning strategies"""
        return [
            ReasoningStrategy(
                "analytical_reasoning",
                "Break down problem into components and analyze systematically",
                ["mathematics", "science", "engineering", "general"],
                success_rate=0.85,
                avg_time=12.0
            ),
            ReasoningStrategy(
                "creative_problem_solving",
                "Use creative thinking and brainstorming for novel solutions",
                ["design", "innovation", "art", "general"],
                success_rate=0.75,
                avg_time=15.0
            ),
            ReasoningStrategy(
                "deductive_reasoning",
                "Apply logical deduction from general principles to specific cases",
                ["logic", "philosophy", "mathematics", "law"],
                success_rate=0.80,
                avg_time=10.0
            ),
            ReasoningStrategy(
                "inductive_reasoning",
                "Draw general conclusions from specific observations",
                ["science", "research", "data_analysis"],
                success_rate=0.70,
                avg_time=18.0
            ),
            ReasoningStrategy(
                "case_based_reasoning",
                "Solve problems by adapting solutions from similar past cases",
                ["medicine", "law", "customer_service", "general"],
                success_rate=0.78,
                avg_time=8.0
            ),
            ReasoningStrategy(
                "algorithmic_approach",
                "Apply step-by-step algorithmic procedures",
                ["programming", "mathematics", "engineering"],
                success_rate=0.90,
                avg_time=6.0
            ),
            ReasoningStrategy(
                "intuitive_reasoning",
                "Rely on intuition and experience for quick solutions",
                ["simple_problems", "general"],
                success_rate=0.65,
                avg_time=3.0
            )
        ]

    def select_strategy(self, problem_description: str, problem_complexity: int = 5,
                       time_constraint: Optional[float] = None) -> ReasoningStrategy:
        """
        Select the best reasoning strategy for the given problem

        Args:
            problem_description: Description of the problem
            problem_complexity: Complexity level (1-10)
            time_constraint: Maximum time allowed (minutes)

        Returns:
            Selected reasoning strategy
        """

        # Identify problem domain
        problem_domain = self._identify_domain(problem_description)

        # Filter applicable strategies
        applicable_strategies = [
            strategy for strategy in self.strategies
            if strategy.is_applicable(problem_domain, problem_complexity)
        ]

        if not applicable_strategies:
            # Fallback to general strategies
            applicable_strategies = [
                strategy for strategy in self.strategies
                if 'general' in strategy.applicable_domains
            ]

        # Apply time constraint if specified
        if time_constraint:
            applicable_strategies = [
                strategy for strategy in applicable_strategies
                if strategy.avg_time <= time_constraint
            ]

        if not applicable_strategies:
            # Ultimate fallback
            return self.strategies[0]

        # Score strategies based on effectiveness and time
        scored_strategies = []
        for strategy in applicable_strategies:
            effectiveness = strategy.get_effectiveness_score()

            # Penalize slow strategies for complex problems under time pressure
            time_penalty = 0
            if time_constraint and strategy.avg_time > time_constraint * 0.8:
                time_penalty = 0.2

            final_score = effectiveness * (1 - time_penalty)
            scored_strategies.append((final_score, strategy))

        # Select highest scoring strategy
        scored_strategies.sort(reverse=True)
        selected_strategy = scored_strategies[0][1]

        logger.info(f"Selected strategy: {selected_strategy.name} (score: {scored_strategies[0][0]:.3f})")
        return selected_strategy

    def _identify_domain(self, description: str) -> str:
        """Identify the problem domain from description"""
        desc_lower = description.lower()

        domain_keywords = {
            'mathematics': ['math', 'calculate', 'equation', 'algebra', 'geometry'],
            'programming': ['code', 'program', 'software', 'algorithm', 'debug'],
            'science': ['experiment', 'hypothesis', 'theory', 'research', 'data'],
            'engineering': ['design', 'build', 'construct', 'system', 'architecture'],
            'business': ['profit', 'market', 'customer', 'strategy', 'finance'],
            'medicine': ['patient', 'diagnosis', 'treatment', 'health', 'medical'],
            'law': ['legal', 'contract', 'regulation', 'court', 'justice'],
            'education': ['learn', 'teach', 'student', 'course', 'knowledge']
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                return domain

        return 'general'

class LogicalReasoner:
    """
    Performs logical reasoning and inference
    """

    def __init__(self):
        self.knowledge_base = defaultdict(list)
        self.inference_rules = self._load_inference_rules()

    def _load_inference_rules(self) -> List[Dict[str, Any]]:
        """Load basic inference rules"""
        return [
            {
                'name': 'modus_ponens',
                'description': 'If P implies Q and P is true, then Q is true',
                'pattern': lambda premises: self._check_modus_ponens(premises)
            },
            {
                'name': 'transitivity',
                'description': 'If A relates to B and B relates to C, then A relates to C',
                'pattern': lambda premises: self._check_transitivity(premises)
            },
            {
                'name': 'contradiction_detection',
                'description': 'Detect logical contradictions',
                'pattern': lambda premises: self._check_contradiction(premises)
            }
        ]

    def add_knowledge(self, fact: str, category: str = 'general'):
        """Add a fact to the knowledge base"""
        self.knowledge_base[category].append({
            'fact': fact,
            'added_at': datetime.now(),
            'confidence': 1.0
        })

    def draw_inference(self, premises: List[str]) -> Dict[str, Any]:
        """Draw logical inferences from premises"""
        inferences = []

        # Apply inference rules
        for rule in self.inference_rules:
            result = rule['pattern'](premises)
            if result:
                inferences.append({
                    'rule': rule['name'],
                    'conclusion': result,
                    'confidence': 0.8  # Base confidence
                })

        # Look for patterns in knowledge base
        kb_inferences = self._knowledge_base_inference(premises)
        inferences.extend(kb_inferences)

        return {
            'inferences': inferences,
            'premises_used': len(premises),
            'total_inferences': len(inferences)
        }

    def _check_modus_ponens(self, premises: List[str]) -> Optional[str]:
        """Check for modus ponens pattern: If P then Q, P -> Q"""
        # Simplified implementation
        for premise in premises:
            if 'if' in premise.lower() and 'then' in premise.lower():
                # This is a conditional premise
                parts = re.split(r'\s+(?:if|then)\s+', premise, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    condition = parts[0].strip()
                    conclusion = parts[1].strip()

                    # Check if condition is in other premises
                    for other_premise in premises:
                        if other_premise != premise and condition.lower() in other_premise.lower():
                            return conclusion

        return None

    def _check_transitivity(self, premises: List[str]) -> Optional[str]:
        """Check for transitivity pattern"""
        # Simplified implementation - look for chains
        relationships = []
        for premise in premises:
            # Look for patterns like "A is related to B"
            match = re.search(r'(.+?)\s+(?:is|are|relates? to)\s+(.+)', premise, re.IGNORECASE)
            if match:
                relationships.append((match.group(1).strip(), match.group(2).strip()))

        # Check for transitive chains
        for i, (a1, b1) in enumerate(relationships):
            for j, (a2, b2) in enumerate(relationships):
                if i != j and b1.lower() == a2.lower():
                    return f"{a1} relates to {b2}"

        return None

    def _check_contradiction(self, premises: List[str]) -> Optional[str]:
        """Check for contradictions"""
        statements = []
        negations = []

        for premise in premises:
            premise_lower = premise.lower()
            statements.append(premise_lower)

            # Simple negation detection
            if premise_lower.startswith(('not ', 'no ', 'never ')):
                negations.append(premise_lower[4:])
            elif 'not' in premise_lower:
                # Split on 'not'
                parts = premise_lower.split('not', 1)
                if len(parts) == 2:
                    negations.append(parts[0].strip() + parts[1].strip())

        # Check for contradictions
        for stmt in statements:
            for neg in negations:
                if self._are_contradictory(stmt, neg):
                    return f"Contradiction detected: '{stmt}' vs 'not {neg}'"

        return None

    def _are_contradictory(self, stmt1: str, stmt2: str) -> bool:
        """Check if two statements are contradictory"""
        # Very simplified contradiction detection
        return stmt1.strip() == stmt2.strip()

    def _knowledge_base_inference(self, premises: List[str]) -> List[Dict[str, Any]]:
        """Draw inferences using knowledge base"""
        inferences = []

        for category, facts in self.knowledge_base.items():
            for fact in facts:
                fact_text = fact['fact']

                # Check if any premise relates to this fact
                for premise in premises:
                    if self._texts_related(premise, fact_text):
                        inferences.append({
                            'rule': 'knowledge_base_similarity',
                            'conclusion': f"Related knowledge: {fact_text}",
                            'confidence': fact['confidence'] * 0.7,
                            'category': category
                        })

        return inferences

    def _texts_related(self, text1: str, text2: str) -> bool:
        """Check if two texts are related (simplified)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Check for word overlap
        overlap = len(words1.intersection(words2))
        return overlap >= 2  # At least 2 words in common

class AdvancedReasoningEngine:
    """
    Main advanced reasoning engine combining all reasoning capabilities
    """

    def __init__(self):
        self.task_decomposer = TaskDecomposer()
        self.strategy_selector = StrategySelector()
        self.logical_reasoner = LogicalReasoner()

        # Reasoning history
        self.reasoning_history = deque(maxlen=1000)

    def reason_about_task(self, task_description: str,
                         complexity: int = 5,
                         time_constraint: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform comprehensive reasoning about a task

        Args:
            task_description: Description of the task
            complexity: Task complexity (1-10)
            time_constraint: Time limit in minutes

        Returns:
            Reasoning results
        """

        reasoning_start = datetime.now()

        # 1. Decompose the task
        decomposed_task = self.task_decomposer.decompose_task(task_description, complexity)

        # 2. Select reasoning strategy
        strategy = self.strategy_selector.select_strategy(
            task_description, complexity, time_constraint
        )

        # 3. Perform logical reasoning
        task_premises = [task_description]
        if decomposed_task.subtasks:
            task_premises.extend([st.description for st in decomposed_task.subtasks])

        logical_inferences = self.logical_reasoner.draw_inference(task_premises)

        # 4. Generate execution plan
        execution_plan = self._generate_execution_plan(decomposed_task, strategy)

        # 5. Assess feasibility
        feasibility = self._assess_feasibility(decomposed_task, time_constraint)

        reasoning_result = {
            'task_analysis': {
                'original_task': task_description,
                'complexity': complexity,
                'estimated_total_time': decomposed_task.estimated_time,
                'subtasks_count': len(decomposed_task.subtasks)
            },
            'decomposition': decomposed_task.to_dict(),
            'selected_strategy': {
                'name': strategy.name,
                'description': strategy.description,
                'expected_success_rate': strategy.success_rate,
                'estimated_time': strategy.avg_time
            },
            'logical_inferences': logical_inferences,
            'execution_plan': execution_plan,
            'feasibility_assessment': feasibility,
            'reasoning_time': (datetime.now() - reasoning_start).total_seconds()
        }

        # Record in history
        self.reasoning_history.append({
            'timestamp': datetime.now().isoformat(),
            'task': task_description,
            'result': reasoning_result
        })

        return reasoning_result

    def _generate_execution_plan(self, task: Task, strategy: ReasoningStrategy) -> List[Dict[str, Any]]:
        """Generate a step-by-step execution plan"""
        plan = []

        # Add strategy-specific preparation steps
        if strategy.name == 'analytical_reasoning':
            plan.append({
                'step': 'analysis',
                'description': 'Analyze task components systematically',
                'estimated_time': 5.0
            })
        elif strategy.name == 'creative_problem_solving':
            plan.append({
                'step': 'brainstorming',
                'description': 'Generate multiple solution approaches',
                'estimated_time': 10.0
            })

        # Add task subtasks
        for i, subtask in enumerate(task.subtasks):
            plan.append({
                'step': f'subtask_{i+1}',
                'description': subtask.description,
                'estimated_time': subtask.estimated_time,
                'dependencies': subtask.dependencies
            })

        # Add verification step
        plan.append({
            'step': 'verification',
            'description': 'Verify solution meets requirements',
            'estimated_time': 3.0
        })

        return plan

    def _assess_feasibility(self, task: Task, time_constraint: Optional[float]) -> Dict[str, Any]:
        """Assess the feasibility of completing the task"""
        total_estimated_time = task.estimated_time

        if time_constraint and total_estimated_time > time_constraint:
            return {
                'feasible': False,
                'reason': f'Estimated time ({total_estimated_time:.1f} min) exceeds constraint ({time_constraint} min)',
                'recommendation': 'Break task into smaller parts or extend time limit'
            }

        # Assess based on complexity and subtasks
        complexity_score = task.complexity / 10.0
        subtask_score = len(task.subtasks) / 10.0  # Normalize

        feasibility_score = 1.0 - (complexity_score * 0.4 + subtask_score * 0.3)

        if feasibility_score > 0.7:
            assessment = 'highly_feasible'
        elif feasibility_score > 0.4:
            assessment = 'moderately_feasible'
        else:
            assessment = 'challenging'

        return {
            'feasible': feasibility_score > 0.3,
            'assessment': assessment,
            'feasibility_score': feasibility_score,
            'estimated_time': total_estimated_time
        }

    def learn_from_experience(self, task_description: str, success: bool, actual_time: float):
        """Learn from task execution experience"""
        # Find the reasoning result for this task
        for record in reversed(self.reasoning_history):
            if record['task'] == task_description:
                reasoning_result = record['result']
                strategy_name = reasoning_result['selected_strategy']['name']

                # Find the strategy and update its performance
                for strategy in self.strategy_selector.strategies:
                    if strategy.name == strategy_name:
                        strategy.record_usage(success)

                        # Update time estimate
                        old_time = strategy.avg_time
                        strategy.avg_time = (old_time + actual_time) / 2  # Simple averaging

                        logger.info(f"Updated strategy {strategy_name}: success_rate={strategy.success_rate:.3f}, avg_time={strategy.avg_time:.1f}")
                        break

                break

    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning performance"""
        if not self.reasoning_history:
            return {'total_reasoning_sessions': 0}

        total_sessions = len(self.reasoning_history)
        recent_sessions = list(self.reasoning_history)[-50:]  # Last 50 sessions

        # Calculate average reasoning time
        reasoning_times = [r['result']['reasoning_time'] for r in recent_sessions]
        avg_reasoning_time = sum(reasoning_times) / len(reasoning_times)

        # Strategy usage statistics
        strategy_usage = defaultdict(int)
        for record in recent_sessions:
            strategy_name = record['result']['selected_strategy']['name']
            strategy_usage[strategy_name] += 1

        return {
            'total_reasoning_sessions': total_sessions,
            'average_reasoning_time': avg_reasoning_time,
            'strategy_usage': dict(strategy_usage),
            'most_used_strategy': max(strategy_usage.items(), key=lambda x: x[1])[0] if strategy_usage else None
        }


# Example usage
if __name__ == "__main__":
    print("SOFIA Advanced Reasoning Engine")
    print("This system provides task decomposition, strategy selection, and logical reasoning")

    # Example usage would be integrated with SOFIA
    """
    from sofia_reasoning import AdvancedReasoningEngine

    reasoner = AdvancedReasoningEngine()

    # Reason about a complex task
    result = reasoner.reason_about_task(
        "Implement a machine learning model for image classification",
        complexity=8,
        time_constraint=120  # 2 hours
    )

    print("Reasoning result:", result)

    # Learn from execution
    reasoner.learn_from_experience(
        "Implement a machine learning model for image classification",
        success=True,
        actual_time=95.0
    )
    """
