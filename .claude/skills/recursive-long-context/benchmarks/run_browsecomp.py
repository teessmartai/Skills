#!/usr/bin/env python3
"""
BrowseComp+ Benchmark Runner for RLM

Reproduces the BrowseComp+ experiments from the RLM paper.
This is the most challenging benchmark - multi-hop reasoning over
1000 documents (6M-11M tokens total).

WARNING: This benchmark is expensive to run! Each query costs ~$1 with RLM.

Usage:
    python run_browsecomp.py --provider claude-code --num-docs 100 --max-tasks 5
    python run_browsecomp.py --provider anthropic --num-docs 1000 --max-tasks 20
"""

import argparse
import json
import os
import sys
import time
import random
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


def load_browsecomp_plus(num_docs: int = 1000, max_tasks: int = 150) -> List[Dict[str, Any]]:
    """
    Load BrowseComp+ benchmark from HuggingFace.

    Args:
        num_docs: Number of documents to include per task (paper uses 1000)
        max_tasks: Maximum number of tasks to load

    Returns:
        List of task dictionaries with context and queries
    """
    print(f"Loading BrowseComp+ dataset (num_docs={num_docs})...")

    try:
        # Load queries
        queries_dataset = load_dataset("Tevatron/browsecomp-plus", "queries", split="test")

        # Load corpus
        corpus_dataset = load_dataset("Tevatron/browsecomp-plus", "corpus", split="train")

        # Load qrels (query-document relevance)
        qrels_dataset = load_dataset("Tevatron/browsecomp-plus", "qrels", split="test")

        # Build corpus index
        corpus_dict = {doc["_id"]: doc["text"] for doc in corpus_dataset}

        # Build qrels mapping
        qrels_dict = {}
        for qrel in qrels_dataset:
            qid = qrel["query-id"]
            did = qrel["corpus-id"]
            if qid not in qrels_dict:
                qrels_dict[qid] = []
            qrels_dict[qid].append(did)

        tasks = []
        all_doc_ids = list(corpus_dict.keys())

        for query in list(queries_dataset)[:max_tasks]:
            qid = query["_id"]
            query_text = query["text"]
            answer = query.get("answer", query.get("answers", [""])[0] if "answers" in query else "")

            # Get relevant documents
            relevant_docs = qrels_dict.get(qid, [])

            # Sample additional documents to reach num_docs
            other_docs = [d for d in all_doc_ids if d not in relevant_docs]
            sample_size = min(num_docs - len(relevant_docs), len(other_docs))
            sampled_docs = random.sample(other_docs, sample_size) if sample_size > 0 else []

            # Combine and shuffle
            doc_ids = relevant_docs + sampled_docs
            random.shuffle(doc_ids)

            # Build context from documents
            context_parts = []
            for i, doc_id in enumerate(doc_ids[:num_docs]):
                if doc_id in corpus_dict:
                    context_parts.append(f"[Document {i+1}]\n{corpus_dict[doc_id]}")

            context = "\n\n".join(context_parts)

            tasks.append({
                "task_id": qid,
                "query": query_text,
                "context": context,
                "answer": answer,
                "num_docs": len(doc_ids),
                "num_relevant": len(relevant_docs)
            })

        print(f"Loaded {len(tasks)} tasks")
        return tasks

    except Exception as e:
        print(f"Error loading BrowseComp+: {e}")
        print("Using sample data for testing...")
        return get_sample_browsecomp_tasks()


def get_sample_browsecomp_tasks() -> List[Dict[str, Any]]:
    """Return sample BrowseComp+ style tasks for testing."""
    sample_context = """
[Document 1]
The Dinengdeng Festival is celebrated annually in Agoo, La Union, Philippines.
First held in 2005, the festival celebrates the town's famous vegetable stew dish.
The festival features cooking competitions, cultural performances, and a beauty pageant.

[Document 2]
Random article about climate change and environmental policies.
Global temperatures have risen significantly over the past century.
International agreements aim to reduce carbon emissions.

[Document 3]
The 13th Dinengdeng Festival was held in 2017 in Agoo, La Union.
The Festival of Festivals competition crowned three winners, all from La Union province.
Miss Agoo 2017 was won by Maria Camille Dalmacio.

[Document 4]
Article about Philippine cuisine and traditional dishes.
Dinengdeng is a vegetable stew using fish and bagoong (fermented fish paste).
It is a staple dish in the Ilocos region.

[Document 5]
More random content about Asian festivals and celebrations.
Various countries in Southeast Asia have unique cultural festivals.
Food festivals are popular tourist attractions.
"""

    return [{
        "task_id": "sample_1",
        "query": "What is the name of the person who won the beauty pageant at the 13th Dinengdeng Festival?",
        "context": sample_context,
        "answer": "Maria Camille Dalmacio",
        "num_docs": 5,
        "num_relevant": 2
    }]


def run_browsecomp_benchmark(
    provider: str = "claude-code",
    model: Optional[str] = None,
    num_docs: int = 1000,
    max_tasks: int = 20,
    max_iterations: int = 20,
    verbose: bool = True,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run the BrowseComp+ benchmark.

    Args:
        provider: LLM provider
        model: Model name
        num_docs: Documents per task (paper uses 1000)
        max_tasks: Maximum tasks to run (paper uses 150)
        max_iterations: Max REPL iterations
        verbose: Print progress
        save_results: Save results to file

    Returns:
        Benchmark results dictionary
    """
    tasks = load_browsecomp_plus(num_docs=num_docs, max_tasks=max_tasks)

    results = {
        "benchmark": "BrowseComp-Plus",
        "provider": provider,
        "model": model,
        "num_docs": num_docs,
        "max_iterations": max_iterations,
        "tasks": [],
        "correct": 0,
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
        print(f"Task {i+1}/{len(tasks)} (ID: {task['task_id']})")
        print(f"Documents: {task['num_docs']}, Relevant: {task['num_relevant']}")
        print(f"{'='*60}")

        context = task["context"]
        query = task["query"]
        expected = task["answer"]

        print(f"Query: {query[:200]}...")
        print(f"Context length: {len(context):,} chars (~{len(context)//4:,} tokens)")

        start_time = time.time()

        try:
            answer = rlm.run(query, context)
            trajectory = rlm.get_trajectory()

            task_time = time.time() - start_time
            task_cost = trajectory.total_cost if trajectory else 0.0

            # Check correctness (case-insensitive partial match)
            correct = expected.lower() in answer.lower()

            task_result = {
                "task_id": task["task_id"],
                "answer": answer,
                "expected": expected,
                "correct": correct,
                "cost": task_cost,
                "time": task_time,
                "iterations": len(trajectory.iterations) if trajectory else 0,
                "sub_calls": len(trajectory.sub_calls) if trajectory else 0
            }

            results["tasks"].append(task_result)
            if correct:
                results["correct"] += 1
            results["total_cost"] += task_cost
            results["total_time"] += task_time

            print(f"\nAnswer: {answer[:300]}...")
            print(f"Expected: {expected}")
            print(f"Correct: {'✓ YES' if correct else '✗ NO'}")
            print(f"Cost: ${task_cost:.4f}, Time: {task_time:.2f}s")

        except Exception as e:
            print(f"Error: {e}")
            results["tasks"].append({
                "task_id": task["task_id"],
                "error": str(e),
                "correct": False
            })

    # Calculate final metrics
    results["accuracy"] = results["correct"] / len(tasks) * 100
    results["average_cost"] = results["total_cost"] / len(tasks)
    results["average_time"] = results["total_time"] / len(tasks)

    print(f"\n{'='*60}")
    print("BrowseComp+ Benchmark Results")
    print(f"{'='*60}")
    print(f"Tasks: {len(tasks)}")
    print(f"Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{len(tasks)})")
    print(f"Total Cost: ${results['total_cost']:.2f}")
    print(f"Average Cost: ${results['average_cost']:.4f}/task")
    print(f"Total Time: {results['total_time']:.2f}s")

    if save_results:
        output_path = Path(__file__).parent / "results" / f"browsecomp_results_{int(time.time())}.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run BrowseComp+ benchmark with RLM")
    parser.add_argument("--provider", type=str, default="claude-code",
                       choices=["anthropic", "openai", "claude-code"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-docs", type=int, default=1000,
                       help="Documents per task (paper uses 1000)")
    parser.add_argument("--max-tasks", type=int, default=20,
                       help="Max tasks to run (paper uses 150)")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    print("⚠️  WARNING: BrowseComp+ is expensive!")
    print(f"   Estimated cost: ~${args.max_tasks * 1.0:.2f} for {args.max_tasks} tasks")
    print()

    run_browsecomp_benchmark(
        provider=args.provider,
        model=args.model,
        num_docs=args.num_docs,
        max_tasks=args.max_tasks,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
