#!/usr/bin/env python3
"""
Quick RLM Test Script

A simple test to verify your RLM setup works before running full benchmarks.
Uses synthetic data that doesn't require downloading external datasets.

Usage:
    python quick_test.py                    # Uses claude-code provider
    python quick_test.py --provider anthropic
    python quick_test.py --all              # Run all test types
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm import run_rlm, RecursiveLanguageModel, RLMConfig


# =============================================================================
# Test Data (Synthetic - No External Downloads Required)
# =============================================================================

# O(1) Test: Single needle in haystack (S-NIAH style)
SNIAH_TEST = {
    "name": "S-NIAH (O(1) Complexity)",
    "context": """
This is a collection of random text passages about various topics.

The history of computing began with mechanical calculators in the 17th century.
Charles Babbage designed the Analytical Engine in the 1830s.
Ada Lovelace wrote the first computer program for Babbage's machine.

Random filler text about weather patterns and climate science.
The atmosphere contains nitrogen, oxygen, and trace gases.
Weather forecasting uses complex mathematical models.

The special magic number for Project Alpha is: 7429163

More random text about biology and ecosystems.
Photosynthesis converts sunlight into chemical energy.
Plants release oxygen as a byproduct of this process.

Additional filler content about geography and geology.
The Earth's crust is divided into tectonic plates.
Continental drift was first proposed by Alfred Wegener.
""",
    "query": "What is the special magic number for Project Alpha?",
    "expected": "7429163"
}


# O(N) Test: Aggregation over all entries (OOLONG style)
OOLONG_TEST = {
    "name": "OOLONG-style (O(N) Complexity)",
    "context": """The following entries contain questions with user IDs. Each question's answer type can be categorized.

User: 1001 || Question: What year did World War II end?
User: 1002 || Question: Who painted the Mona Lisa?
User: 1003 || Question: What is the capital of France?
User: 1001 || Question: How many legs does a spider have?
User: 1004 || Question: What does CPU stand for?
User: 1002 || Question: Where is Mount Everest located?
User: 1005 || Question: Who wrote Romeo and Juliet?
User: 1003 || Question: What is the speed of light in m/s?
User: 1001 || Question: What does HTML stand for?
User: 1006 || Question: Who was the first president of the United States?
User: 1002 || Question: What is the largest planet in our solar system?
User: 1004 || Question: Where is the Great Wall located?
User: 1005 || Question: How many bones are in the human body?
User: 1003 || Question: What does NASA stand for?
User: 1006 || Question: Who discovered penicillin?
""",
    "query": """Count the total number of questions in the data. Also count how many questions are about:
1. Abbreviations/acronyms (what does X stand for?)
2. Locations (where is X?)
3. People (who did/was X?)
4. Numbers (how many X?)

Report in format: Total: N, Abbreviations: N, Locations: N, People: N, Numbers: N""",
    "expected_contains": ["15", "4", "3", "4", "3"]  # Approximate expected values
}


# O(N²) Test: Pairwise relationships (OOLONG-Pairs style)
PAIRS_TEST = {
    "name": "OOLONG-Pairs-style (O(N²) Complexity)",
    "context": """The following entries show user preferences:

User: A || Likes: Python, JavaScript
User: B || Likes: Python, Java, C++
User: C || Likes: JavaScript, TypeScript
User: D || Likes: Python, Rust
User: E || Likes: Java, Kotlin
User: F || Likes: Python, JavaScript, TypeScript
""",
    "query": """Find all pairs of users who share at least one programming language preference.
List pairs as (User1, User2) where User1 comes alphabetically before User2.
Example format: (A, B), (A, C), etc.""",
    "expected_contains": ["(A, B)", "(A, C)", "(A, D)", "(A, F)", "(B, D)", "(B, E)", "(C, F)"]
}


def run_test(test_data: dict, provider: str, model: str = None, verbose: bool = True) -> dict:
    """Run a single test and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {test_data['name']}")
    print(f"{'='*60}")

    if verbose:
        print(f"Query: {test_data['query'][:100]}...")
        print(f"Context length: {len(test_data['context'])} chars")

    config = RLMConfig(
        api_provider=provider,
        root_model=model,
        sub_model=model,
        max_iterations=15,
        verbose=verbose
    )

    rlm = RecursiveLanguageModel(config)
    start_time = time.time()

    try:
        answer = rlm.run(test_data["query"], test_data["context"])
        trajectory = rlm.get_trajectory()
        elapsed = time.time() - start_time

        # Check if answer contains expected content
        if "expected" in test_data:
            passed = test_data["expected"].lower() in answer.lower()
        elif "expected_contains" in test_data:
            passed = all(exp in answer for exp in test_data["expected_contains"])
        else:
            passed = True  # No validation

        result = {
            "name": test_data["name"],
            "passed": passed,
            "answer": answer,
            "time": elapsed,
            "cost": trajectory.total_cost if trajectory else 0,
            "iterations": len(trajectory.iterations) if trajectory else 0,
            "sub_calls": len(trajectory.sub_calls) if trajectory else 0
        }

        print(f"\nAnswer: {answer[:300]}...")
        print(f"Time: {elapsed:.2f}s")
        print(f"Cost: ${result['cost']:.4f}")
        print(f"Iterations: {result['iterations']}, Sub-calls: {result['sub_calls']}")
        print(f"Status: {'✓ PASSED' if passed else '✗ FAILED'}")

        return result

    except Exception as e:
        print(f"Error: {e}")
        return {
            "name": test_data["name"],
            "passed": False,
            "error": str(e),
            "time": time.time() - start_time
        }


def main():
    parser = argparse.ArgumentParser(description="Quick RLM functionality test")
    parser.add_argument("--provider", type=str, default="claude-code",
                       choices=["anthropic", "openai", "claude-code"],
                       help="LLM provider")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (optional)")
    parser.add_argument("--test", type=str, default="sniah",
                       choices=["sniah", "oolong", "pairs", "all"],
                       help="Test type to run")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    args = parser.parse_args()

    print("="*60)
    print("RLM Quick Test")
    print("="*60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")

    tests_to_run = []

    if args.all or args.test == "all":
        tests_to_run = [SNIAH_TEST, OOLONG_TEST, PAIRS_TEST]
    elif args.test == "sniah":
        tests_to_run = [SNIAH_TEST]
    elif args.test == "oolong":
        tests_to_run = [OOLONG_TEST]
    elif args.test == "pairs":
        tests_to_run = [PAIRS_TEST]

    results = []
    for test in tests_to_run:
        result = run_test(test, args.provider, args.model, verbose=not args.quiet)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    total_time = sum(r.get("time", 0) for r in results)
    total_cost = sum(r.get("cost", 0) for r in results)

    for r in results:
        status = "✓" if r.get("passed") else "✗"
        print(f"{status} {r['name']}: {r.get('time', 0):.2f}s, ${r.get('cost', 0):.4f}")

    print(f"\nTotal: {passed}/{total} passed")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Cost: ${total_cost:.4f}")

    if passed == total:
        print("\n✓ All tests passed! RLM is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
