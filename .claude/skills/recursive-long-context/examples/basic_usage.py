#!/usr/bin/env python3
"""
Basic usage example for the Recursive Language Model (RLM).

This example demonstrates:
1. Simple query on a long document
2. Using the convenience function
3. Accessing the trajectory for debugging
"""

import sys
sys.path.insert(0, '..')

from rlm import run_rlm, RecursiveLanguageModel, RLMConfig


def example_simple():
    """Simple example using the convenience function."""
    print("=" * 60)
    print("Example 1: Simple Usage")
    print("=" * 60)

    # Simulate a long document (in real use, this would be much longer)
    long_document = """
    Chapter 1: Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.
    The field has grown significantly since the 1950s when Arthur Samuel
    first coined the term while working at IBM.

    Key concepts in machine learning include:
    - Supervised learning: Learning from labeled examples
    - Unsupervised learning: Finding patterns in unlabeled data
    - Reinforcement learning: Learning through trial and error

    Chapter 2: Neural Networks

    Neural networks are computing systems inspired by biological neural
    networks. The basic unit is the artificial neuron, which receives
    inputs, applies weights, and produces an output through an activation
    function.

    Deep learning refers to neural networks with many layers, enabling
    the learning of hierarchical representations. The backpropagation
    algorithm, discovered in the 1980s, made training deep networks
    practical.

    Chapter 3: Applications

    Machine learning is now used in:
    - Image recognition (computer vision)
    - Natural language processing
    - Recommendation systems
    - Autonomous vehicles
    - Medical diagnosis
    - Financial trading

    The field continues to evolve rapidly with new architectures like
    transformers revolutionizing NLP tasks.
    """ * 100  # Repeat to simulate longer document

    query = "What are the three main types of machine learning mentioned, and when was the term 'machine learning' first coined?"

    print(f"Document length: {len(long_document):,} characters")
    print(f"Query: {query}")
    print()

    # Run RLM
    answer = run_rlm(
        query=query,
        context=long_document,
        api_provider="anthropic",
        max_iterations=10,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)


def example_with_config():
    """Example using full configuration and trajectory."""
    print("\n" + "=" * 60)
    print("Example 2: Advanced Usage with Config")
    print("=" * 60)

    # Create a dataset of entries
    entries = []
    for i in range(500):
        category = ["product", "service", "support", "billing"][i % 4]
        entries.append(f"Entry {i}: Category={category}, Status=active, Value=${i * 10}")

    context = "\n".join(entries)

    query = "Count how many entries are in each category and calculate the total value for 'product' category entries."

    print(f"Context: {len(entries)} entries, {len(context):,} characters")
    print(f"Query: {query}")
    print()

    # Create custom config
    config = RLMConfig(
        api_provider="anthropic",
        root_model="claude-sonnet-4-20250514",
        sub_model="claude-sonnet-4-20250514",  # Use smaller model for sub-calls
        max_iterations=15,
        verbose=True
    )

    # Run RLM
    rlm = RecursiveLanguageModel(config)
    answer = rlm.run(query, context)

    # Get trajectory for analysis
    trajectory = rlm.get_trajectory()

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY:")
    print("=" * 60)
    print(f"Total iterations: {len(trajectory.iterations)}")
    print(f"Total sub-LLM calls: {len(trajectory.sub_calls)}")
    print(f"Total cost: ${trajectory.total_cost:.4f}")
    print(f"Total time: {trajectory.total_time:.2f}s")


if __name__ == "__main__":
    print("RLM Basic Usage Examples")
    print("Note: These examples require an API key.")
    print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
    print()

    # Check for API key
    import os
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: No API key found. Examples will fail.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run examples.")
        sys.exit(1)

    # Run examples
    example_simple()
    example_with_config()
