#!/usr/bin/env python3
"""
OOLONG Benchmark Runner for RLM

Reproduces the OOLONG trec_coarse experiments from the RLM paper.
This benchmark tests O(N) complexity tasks requiring semantic classification
of all entries in a dataset.

Usage:
    python run_oolong.py --provider claude-code --max-tasks 10
    python run_oolong.py --provider anthropic --max-tasks 50 --verbose
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)

from rlm import run_rlm, RecursiveLanguageModel, RLMConfig


def load_oolong_trec_coarse() -> List[Dict[str, Any]]:
    """Load OOLONG trec_coarse benchmark from HuggingFace."""
    print("Loading OOLONG trec_coarse dataset...")

    try:
        # Try loading from HuggingFace
        dataset = load_dataset("oolongbench/oolong-synth", "trec_coarse", split="test")
        tasks = list(dataset)
        print(f"Loaded {len(tasks)} tasks from HuggingFace")
        return tasks
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Falling back to sample data...")
        return get_sample_oolong_tasks()


def get_sample_oolong_tasks() -> List[Dict[str, Any]]:
    """Return sample OOLONG-style tasks for testing."""
    # Sample task structure matching OOLONG format
    sample_context = """The following lines contain 100 general-knowledge questions, one per line. Each line has a User ID, which is not necessarily unique.

Date: Sep 06, 2023 || User: 14512 || Instance: What is a tonne ?
Date: Jun 21, 2023 || User: 16285 || Instance: Where is the Orange Bowl ?
Date: May 16, 2022 || User: 52632 || Instance: What is the current unemployment rate in the U.S. ?
Date: Jan 14, 2024 || User: 88142 || Instance: What King is satirized in the line : `` The King was in the countinghouse , counting all his money '' ?
Date: Sep 11, 2024 || User: 44465 || Instance: On average , how long time does it take to type a screenplay ?
Date: Jan 09, 2024 || User: 17538 || Instance: What articles of clothing are tokens in Monopoly ?
Date: Feb 26, 2024 || User: 88488 || Instance: What U.S. President was the first to breed mules ?
Date: Apr 08, 2023 || User: 33449 || Instance: How do you recognize anorexia ?
Date: May 16, 2024 || User: 95871 || Instance: Name the child left on a doorstep at the beginning of Gasoline Alley .
Date: Mar 22, 2023 || User: 21456 || Instance: What does DNA stand for ?
"""

    return [
        {
            "context": sample_context,
            "query": "In the above data, is label 'description and abstract concept' more common, less common, or the same frequency as label 'numeric value'? Each of the questions can be labelled as one of the labels: description and abstract concept, entity, human being, numeric value, location, abbreviation. Give your final answer in the form 'Answer: description and abstract concept is [X] numeric value', where [X] is 'more common than', 'less common than', or 'same frequency as'.",
            "answer": "less common than",
            "task_id": "sample_1"
        }
    ]


def score_oolong_answer(predicted: str, expected: str) -> float:
    """
    Score OOLONG answer using the paper's scoring function.

    For numeric answers: score = 0.75^|y - ŷ|
    For other answers: exact match (1.0 or 0.0)
    """
    predicted = predicted.strip().lower()
    expected = expected.strip().lower()

    # Try to extract numeric values
    try:
        pred_num = float(predicted)
        exp_num = float(expected)
        return 0.75 ** abs(exp_num - pred_num)
    except ValueError:
        pass

    # Exact match for non-numeric
    if expected in predicted:
        return 1.0
    return 0.0


def run_oolong_benchmark(
    provider: str = "claude-code",
    model: Optional[str] = None,
    max_tasks: int = 50,
    max_iterations: int = 20,
    verbose: bool = True,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run the OOLONG benchmark.

    Args:
        provider: LLM provider ("anthropic", "openai", "claude-code")
        model: Model name (optional)
        max_tasks: Maximum number of tasks to run
        max_iterations: Max REPL iterations per task
        verbose: Print progress
        save_results: Save results to JSON file

    Returns:
        Dictionary with benchmark results
    """
    tasks = load_oolong_trec_coarse()[:max_tasks]

    results = {
        "benchmark": "OOLONG-trec_coarse",
        "provider": provider,
        "model": model,
        "max_iterations": max_iterations,
        "tasks": [],
        "total_score": 0.0,
        "total_cost": 0.0,
        "total_time": 0.0,
        "num_tasks": len(tasks)
    }

    config = RLMConfig(
        api_provider=provider,
        root_model=model,
        sub_model=model,
        max_iterations=max_iterations,
        verbose=verbose
    )

    rlm = RecursiveLanguageModel(config)

    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Task {i+1}/{len(tasks)}")
        print(f"{'='*60}")

        context = task.get("context", task.get("input", ""))
        query = task.get("query", task.get("question", ""))
        expected = task.get("answer", task.get("expected", ""))

        start_time = time.time()

        try:
            answer = rlm.run(query, context)
            trajectory = rlm.get_trajectory()

            task_time = time.time() - start_time
            task_cost = trajectory.total_cost if trajectory else 0.0
            score = score_oolong_answer(answer, expected)

            task_result = {
                "task_id": task.get("task_id", i),
                "answer": answer,
                "expected": expected,
                "score": score,
                "cost": task_cost,
                "time": task_time,
                "iterations": len(trajectory.iterations) if trajectory else 0,
                "sub_calls": len(trajectory.sub_calls) if trajectory else 0
            }

            results["tasks"].append(task_result)
            results["total_score"] += score
            results["total_cost"] += task_cost
            results["total_time"] += task_time

            print(f"\nAnswer: {answer[:200]}...")
            print(f"Expected: {expected}")
            print(f"Score: {score:.2f}")
            print(f"Cost: ${task_cost:.4f}, Time: {task_time:.2f}s")

        except Exception as e:
            print(f"Error on task {i}: {e}")
            results["tasks"].append({
                "task_id": task.get("task_id", i),
                "error": str(e),
                "score": 0.0
            })

    # Calculate final metrics
    results["average_score"] = results["total_score"] / len(tasks) * 100
    results["average_cost"] = results["total_cost"] / len(tasks)
    results["average_time"] = results["total_time"] / len(tasks)

    print(f"\n{'='*60}")
    print("OOLONG Benchmark Results")
    print(f"{'='*60}")
    print(f"Tasks completed: {len(tasks)}")
    print(f"Average Score: {results['average_score']:.2f}%")
    print(f"Total Cost: ${results['total_cost']:.4f}")
    print(f"Average Cost per Task: ${results['average_cost']:.4f}")
    print(f"Total Time: {results['total_time']:.2f}s")

    if save_results:
        output_path = Path(__file__).parent / "results" / f"oolong_results_{int(time.time())}.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run OOLONG benchmark with RLM")
    parser.add_argument("--provider", type=str, default="claude-code",
                       choices=["anthropic", "openai", "claude-code"],
                       help="LLM provider")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (optional)")
    parser.add_argument("--max-tasks", type=int, default=50,
                       help="Maximum number of tasks to run")
    parser.add_argument("--max-iterations", type=int, default=20,
                       help="Max REPL iterations per task")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print verbose output")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to file")

    args = parser.parse_args()

    run_oolong_benchmark(
        provider=args.provider,
        model=args.model,
        max_tasks=args.max_tasks,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
