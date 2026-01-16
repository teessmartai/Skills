#!/usr/bin/env python3
"""
Command-line interface for the Recursive Language Model (RLM).

Usage:
    python cli.py --query "Your question" --context-file document.txt
    python cli.py --query "Your question" --context "Direct text input"
    python cli.py --query "Your question" --context-dir ./documents/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from rlm import RecursiveLanguageModel, RLMConfig, RLMTrajectory


def load_context_from_file(file_path: str) -> str:
    """Load context from a single file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Handle different file types
    suffix = path.suffix.lower()

    if suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, str):
                return data
            else:
                return json.dumps(data, indent=2)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


def load_context_from_directory(dir_path: str, extensions: Optional[List[str]] = None) -> str:
    """Load and concatenate context from all files in a directory."""
    path = Path(dir_path)
    if not path.exists() or not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    if extensions is None:
        extensions = ['.txt', '.md', '.py', '.json', '.csv', '.html', '.xml']

    contents = []
    for file_path in sorted(path.rglob('*')):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Add file marker
                    relative_path = file_path.relative_to(path)
                    contents.append(f"=== FILE: {relative_path} ===\n{content}\n")
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)

    if not contents:
        raise ValueError(f"No readable files found in {dir_path}")

    return "\n".join(contents)


def save_trajectory(trajectory: RLMTrajectory, output_path: str):
    """Save trajectory to a JSON file."""
    data = {
        "query": trajectory.query,
        "context_length": trajectory.context_length,
        "final_answer": trajectory.final_answer,
        "total_cost": trajectory.total_cost,
        "total_time": trajectory.total_time,
        "num_iterations": len(trajectory.iterations),
        "num_sub_calls": len(trajectory.sub_calls),
        "iterations": trajectory.iterations,
        "sub_calls": [
            {
                "query": sc.query,
                "response": sc.response,
                "cost": sc.cost,
                "tokens": sc.tokens
            }
            for sc in trajectory.sub_calls
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Trajectory saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursive Language Model (RLM) CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file (uses Claude Code by default)
  python cli.py -q "Summarize the main findings" -f report.txt

  # Process a directory of documents
  python cli.py -q "Find all mentions of pricing" -d ./documents/

  # Use direct context input
  python cli.py -q "Count the items" -c "item1\\nitem2\\nitem3"

  # Use Anthropic API directly (requires API key)
  python cli.py -q "Analyze the data" -f data.txt --provider anthropic

  # Use OpenAI API directly (requires API key)
  python cli.py -q "Analyze the code" -f code.py --provider openai --model gpt-4o

  # Save trajectory for debugging
  python cli.py -q "Complex analysis" -f data.txt --save-trajectory results.json
        """
    )

    # Query argument (required)
    parser.add_argument(
        '-q', '--query',
        required=True,
        help='The question or task to answer'
    )

    # Context source (one required)
    context_group = parser.add_mutually_exclusive_group(required=True)
    context_group.add_argument(
        '-f', '--context-file',
        help='Path to a file containing the context'
    )
    context_group.add_argument(
        '-d', '--context-dir',
        help='Path to a directory containing context files'
    )
    context_group.add_argument(
        '-c', '--context',
        help='Direct context string input'
    )

    # API configuration
    parser.add_argument(
        '--provider',
        choices=['claude-code', 'claude-code-cli', 'anthropic', 'openai'],
        default='claude-code',
        help='LLM provider: claude-code (default, uses Claude Code subscription), '
             'claude-code-cli (CLI fallback), anthropic (API), openai (API)'
    )
    parser.add_argument(
        '--api-key',
        help='API key (only needed for anthropic/openai providers)'
    )
    parser.add_argument(
        '--model',
        help='Model to use (default depends on provider)'
    )
    parser.add_argument(
        '--sub-model',
        help='Model for sub-LLM calls (default: same as main model)'
    )

    # Processing options
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=20,
        help='Maximum REPL iterations (default: 20)'
    )
    parser.add_argument(
        '--max-output-chars',
        type=int,
        default=30000,
        help='Maximum output chars per execution (default: 30000)'
    )

    # Output options
    parser.add_argument(
        '--quiet', '-Q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--save-trajectory',
        metavar='FILE',
        help='Save full trajectory to JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        metavar='FILE',
        help='Save final answer to file'
    )

    args = parser.parse_args()

    # Load context
    try:
        if args.context_file:
            print(f"Loading context from file: {args.context_file}")
            context = load_context_from_file(args.context_file)
        elif args.context_dir:
            print(f"Loading context from directory: {args.context_dir}")
            context = load_context_from_directory(args.context_dir)
        else:
            context = args.context

        print(f"Context loaded: {len(context):,} characters")

    except Exception as e:
        print(f"Error loading context: {e}", file=sys.stderr)
        sys.exit(1)

    # Create config
    config = RLMConfig(
        api_provider=args.provider,
        api_key=args.api_key,
        root_model=args.model,
        sub_model=args.sub_model,
        max_iterations=args.max_iterations,
        max_output_chars=args.max_output_chars,
        verbose=not args.quiet
    )

    # Run RLM
    try:
        rlm = RecursiveLanguageModel(config)
        answer = rlm.run(args.query, context)

        # Output results
        print("\n" + "=" * 60)
        print("FINAL ANSWER:")
        print("=" * 60)
        print(answer)

        # Save trajectory if requested
        if args.save_trajectory:
            trajectory = rlm.get_trajectory()
            if trajectory:
                save_trajectory(trajectory, args.save_trajectory)

        # Save answer to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(answer)
            print(f"\nAnswer saved to: {args.output}")

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error running RLM: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
