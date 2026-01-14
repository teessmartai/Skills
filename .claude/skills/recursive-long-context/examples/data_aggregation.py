#!/usr/bin/env python3
"""
Data Aggregation example for the Recursive Language Model (RLM).

This example demonstrates using RLM for aggregating information
across many data entries, similar to the OOLONG benchmark from the paper.
The task requires semantic classification of each entry, which cannot
be done with simple pattern matching.
"""

import sys
import random
sys.path.insert(0, '..')

from rlm import RecursiveLanguageModel, RLMConfig


def create_sample_dataset(num_entries: int = 200):
    """
    Create a sample dataset of questions that need semantic classification.

    Each question's answer falls into one of these categories:
    - numeric value: Questions about counts, dates, measurements
    - entity: Questions about objects, organizations, products
    - location: Questions about places
    - description: Questions seeking explanations or definitions
    - abbreviation: Questions about acronyms or short forms
    - human being: Questions about people
    """

    templates = {
        'numeric value': [
            "How many planets are in our solar system?",
            "What year did World War II end?",
            "How tall is Mount Everest in meters?",
            "What is the population of Tokyo?",
            "How many bones are in the human body?",
            "What is the speed of light in km/s?",
            "In what year was the iPhone first released?",
            "How many states are in the United States?",
        ],
        'entity': [
            "What is the largest ocean on Earth?",
            "What company makes the iPhone?",
            "What is the chemical symbol for gold?",
            "What programming language was created by Guido van Rossum?",
            "What is the capital building of the United States called?",
            "What sport uses a puck?",
            "What instrument has 88 keys?",
            "What gas do plants absorb from the atmosphere?",
        ],
        'location': [
            "Where is the Eiffel Tower located?",
            "In which country is the Amazon rainforest?",
            "Where was pizza invented?",
            "Which city hosts the Olympic headquarters?",
            "Where is Silicon Valley?",
            "In what country is Mount Fuji?",
            "Where is the Great Barrier Reef?",
            "Which continent has the Sahara Desert?",
        ],
        'description': [
            "What is photosynthesis?",
            "How does gravity work?",
            "What causes rainbows?",
            "Why is the sky blue?",
            "What is machine learning?",
            "How do vaccines work?",
            "What is the purpose of DNA?",
            "Why do leaves change color in fall?",
        ],
        'abbreviation': [
            "What does NASA stand for?",
            "What is the full form of CPU?",
            "What does HTTP mean?",
            "What does DNA stand for?",
            "What is the expansion of WiFi?",
            "What does CEO mean?",
            "What is the full form of USB?",
            "What does AI stand for?",
        ],
        'human being': [
            "Who painted the Mona Lisa?",
            "Who invented the telephone?",
            "Who wrote Romeo and Juliet?",
            "Who was the first person on the moon?",
            "Who founded Microsoft?",
            "Who discovered penicillin?",
            "Who is the author of Harry Potter?",
            "Who invented the light bulb?",
        ]
    }

    entries = []
    for i in range(num_entries):
        # Pick a random category and question
        category = random.choice(list(templates.keys()))
        question = random.choice(templates[category])

        # Create entry with user ID and timestamp
        user_id = random.randint(10000, 99999)
        date = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

        entry = f"Date: {date} || User: {user_id} || Instance: {question}"
        entries.append(entry)

    return entries


def main():
    print("=" * 60)
    print("RLM Data Aggregation Example")
    print("=" * 60)
    print()

    # Create dataset
    entries = create_sample_dataset(num_entries=150)
    context = "\n".join(entries)

    print(f"Dataset: {len(entries)} entries")
    print(f"Context size: {len(context):,} characters")
    print()
    print("Sample entries:")
    for entry in entries[:5]:
        print(f"  {entry}")
    print()

    # Aggregation query - requires semantic understanding of each entry
    query = """
    The context contains questions, one per line. Each question's answer
    falls into one of 6 categories:
    - numeric value: Questions about numbers, counts, dates, measurements
    - entity: Questions about things, objects, products, organizations
    - location: Questions about places, countries, cities
    - description: Questions seeking explanations or definitions
    - abbreviation: Questions about what acronyms stand for
    - human being: Questions about specific people

    Task: Count how many questions fall into each category.
    Return the counts for all 6 categories.

    Note: You must analyze the MEANING of each question to classify it,
    not just look for keywords.
    """

    print("Query:", query.strip()[:200], "...")
    print()

    # Configure RLM
    config = RLMConfig(
        api_provider="anthropic",
        root_model="claude-sonnet-4-20250514",
        sub_model="claude-sonnet-4-20250514",
        max_iterations=20,
        verbose=True
    )

    # Run RLM
    rlm = RecursiveLanguageModel(config)
    answer = rlm.run(query, context)

    # Display results
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(answer)

    # Show trajectory stats
    trajectory = rlm.get_trajectory()
    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS:")
    print("=" * 60)
    print(f"Iterations: {len(trajectory.iterations)}")
    print(f"Sub-LLM calls: {len(trajectory.sub_calls)}")
    print(f"Total cost: ${trajectory.total_cost:.4f}")
    print(f"Processing time: {trajectory.total_time:.2f}s")

    # Calculate actual distribution for verification
    print("\n" + "=" * 60)
    print("NOTE: The dataset was randomly generated.")
    print("Each run will have different distributions.")
    print("=" * 60)


if __name__ == "__main__":
    import os

    # Set random seed for reproducibility in demo
    random.seed(42)

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: No API key found.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    main()
