"""
Recursive Language Model (RLM) Package

A general inference strategy that treats long prompts as part of an
external environment and allows the LLM to programmatically examine,
decompose, and recursively call itself over snippets of the prompt.

Based on "Recursive Language Models" paper from MIT CSAIL.

Usage:
    from recursive_long_context import run_rlm, RecursiveLanguageModel, RLMConfig

    # Simple usage with API
    answer = run_rlm(
        query="Your question here",
        context=your_long_context,
        api_provider="anthropic"
    )

    # Using Claude Code CLI (no API key needed)
    answer = run_rlm(
        query="Your question here",
        context=your_long_context,
        api_provider="claude-code"
    )

    # Advanced usage
    config = RLMConfig(
        api_provider="anthropic",
        root_model="claude-sonnet-4-20250514",
        max_iterations=20,
        verbose=True
    )
    rlm = RecursiveLanguageModel(config)
    answer = rlm.run(query, context)
    trajectory = rlm.get_trajectory()
"""

from .rlm import (
    RecursiveLanguageModel,
    RLMConfig,
    RLMTrajectory,
    REPLEnvironment,
    ExecutionResult,
    SubCallResult,
    LLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    ClaudeCodeProvider,
    run_rlm,
)

__version__ = "0.1.0"
__author__ = "Based on MIT CSAIL RLM Paper"

__all__ = [
    "RecursiveLanguageModel",
    "RLMConfig",
    "RLMTrajectory",
    "REPLEnvironment",
    "ExecutionResult",
    "SubCallResult",
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "ClaudeCodeProvider",
    "run_rlm",
]
