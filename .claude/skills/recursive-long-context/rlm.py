"""
Recursive Language Model (RLM) Implementation

Based on "Recursive Language Models" paper from MIT CSAIL (arXiv:2512.24601)
by Alex L. Zhang, Tim Kraska, and Omar Khattab.

This module provides the core RLM functionality:
- Loads context as a variable in a Python REPL environment
- Provides llm_query() for recursive LLM sub-calls
- Executes code blocks from LLM responses iteratively
- Handles FINAL() and FINAL_VAR() output mechanisms
"""

import os
import re
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class RLMConfig:
    """Configuration for the RLM."""
    max_iterations: int = 20
    max_recursion_depth: int = 1
    sub_model: Optional[str] = None  # Model for sub-calls (defaults to same as root)
    root_model: Optional[str] = None
    max_output_chars: int = 30000  # Truncate REPL output to avoid context overflow
    timeout_per_execution: int = 300  # 5 minutes per code execution
    verbose: bool = True
    api_provider: str = "claude-code"  # "claude-code", "anthropic", "openai", or "custom"
    api_key: Optional[str] = None
    cwd: Optional[str] = None  # Working directory for Claude Code


@dataclass
class ExecutionResult:
    """Result from a single code execution."""
    code: str
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class SubCallResult:
    """Result from a recursive LLM sub-call."""
    query: str
    response: str
    cost: float = 0.0
    tokens: int = 0


@dataclass
class RLMTrajectory:
    """Complete trajectory of an RLM run."""
    query: str
    context_length: int
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    sub_calls: List[SubCallResult] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_cost: float = 0.0
    total_time: float = 0.0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, float, int]:
        """
        Query the LLM.

        Returns:
            Tuple of (response_text, cost, token_count)
        """
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        return self._client

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, float, int]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens

        # Approximate cost calculation (adjust based on actual pricing)
        cost = (input_tokens * 0.003 + output_tokens * 0.015) / 1000

        return response_text, cost, input_tokens + output_tokens


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        return self._client

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, float, int]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=8192
        )

        response_text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Approximate cost (adjust based on model)
        cost = (input_tokens * 0.005 + output_tokens * 0.015) / 1000

        return response_text, cost, input_tokens + output_tokens


class ClaudeCodeProvider(LLMProvider):
    """
    Claude Code provider using the Claude Agent SDK.

    Uses your existing Claude Code installation and authentication.
    No separate API key required - uses your Claude Code subscription.
    """

    def __init__(self, model: Optional[str] = None, cwd: Optional[str] = None):
        self.model = model  # Currently ignored, uses Claude Code's default
        self.cwd = cwd
        self._sdk_available = None

    def _check_sdk(self):
        """Check if Claude Agent SDK is available."""
        if self._sdk_available is None:
            try:
                import claude_agent_sdk
                self._sdk_available = True
            except ImportError:
                self._sdk_available = False
        return self._sdk_available

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, float, int]:
        """Query Claude Code using the Claude Agent SDK."""
        if not self._check_sdk():
            raise ImportError(
                "Claude Agent SDK not found. Install with: pip install claude-agent-sdk\n"
                "Alternatively, use api_provider='anthropic' with an API key."
            )

        import anyio
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions, AssistantMessage, TextBlock

        async def _run_query():
            options = ClaudeAgentOptions(
                system_prompt=system_prompt or "You are a helpful assistant.",
                max_turns=1,
                allowed_tools=[],  # No tools for sub-queries
            )
            if self.cwd:
                options.cwd = self.cwd

            response_text = ""
            async for message in sdk_query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text

            return response_text

        response_text = anyio.run(_run_query)

        # Claude Code doesn't expose token counts or costs directly
        # Estimate based on character count (rough approximation)
        estimated_tokens = (len(prompt) + len(response_text)) // 4
        cost = 0.0  # Cost is handled by Claude Code subscription

        return response_text, cost, estimated_tokens


class ClaudeCodeCLIProvider(LLMProvider):
    """
    Claude Code provider using CLI subprocess (fallback).

    Uses `claude -p` for non-interactive queries.
    Simpler but less feature-rich than SDK approach.
    """

    def __init__(self, model: Optional[str] = None, cwd: Optional[str] = None):
        self.model = model
        self.cwd = cwd

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, float, int]:
        """Query Claude Code using CLI subprocess."""
        import subprocess
        import shutil

        # Check if claude CLI is available
        claude_path = shutil.which("claude")
        if not claude_path:
            raise FileNotFoundError(
                "Claude Code CLI not found. Please install Claude Code or use a different provider.\n"
                "Install: https://docs.anthropic.com/en/docs/claude-code\n"
                "Or use api_provider='anthropic' with an API key."
            )

        # Build command
        cmd = [claude_path, "-p", prompt]

        # Add system prompt if provided
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.cwd
            )

            if result.returncode != 0:
                raise RuntimeError(f"Claude Code CLI failed: {result.stderr}")

            response_text = result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise TimeoutError("Claude Code CLI timed out after 300 seconds")

        # Estimate tokens (CLI doesn't provide this)
        estimated_tokens = (len(prompt) + len(response_text)) // 4
        cost = 0.0  # Cost handled by subscription

        return response_text, cost, estimated_tokens


class REPLEnvironment:
    """
    Python REPL environment for RLM execution.

    Provides:
    - A 'context' variable containing the input
    - An 'llm_query' function for recursive sub-calls
    - Execution of arbitrary Python code
    - Output capture and truncation
    """

    def __init__(
        self,
        context: Union[str, List[str]],
        llm_query_fn: Callable[[str], str],
        max_output_chars: int = 30000
    ):
        self.context = context
        self.llm_query_fn = llm_query_fn
        self.max_output_chars = max_output_chars

        # Initialize the execution namespace
        self.namespace: Dict[str, Any] = {
            'context': context,
            'llm_query': llm_query_fn,
            '__builtins__': __builtins__,
        }

        # Import common modules into namespace
        self._setup_imports()

    def _setup_imports(self):
        """Pre-import common modules into the namespace."""
        import_statements = [
            "import re",
            "import json",
            "import math",
            "from collections import Counter, defaultdict",
        ]

        for stmt in import_statements:
            try:
                exec(stmt, self.namespace)
            except Exception:
                pass  # Ignore import errors

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the REPL environment."""
        import io
        from contextlib import redirect_stdout, redirect_stderr

        start_time = time.time()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        error = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self.namespace)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        execution_time = time.time() - start_time

        output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        if stderr_output:
            output += f"\n[stderr]: {stderr_output}"

        # Truncate output if too long
        if len(output) > self.max_output_chars:
            output = output[:self.max_output_chars] + f"\n... [Output truncated at {self.max_output_chars} chars]"

        return ExecutionResult(
            code=code,
            output=output,
            error=error,
            execution_time=execution_time
        )

    def get_variable(self, name: str) -> Any:
        """Get a variable from the namespace."""
        return self.namespace.get(name)


class RecursiveLanguageModel:
    """
    Recursive Language Model (RLM) implementation.

    Treats long prompts as part of an external environment and allows
    the LLM to programmatically examine, decompose, and recursively
    call itself over snippets of the prompt.
    """

    SYSTEM_PROMPT = '''You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs.

Your context is a {context_type} with {context_total_length} total characters.

The REPL environment is initialized with:
1. A 'context' variable containing the input data. Check its structure before processing.
2. An 'llm_query(prompt)' function to query a sub-LLM (handles ~500K chars). Use this for semantic analysis.
3. Use 'print()' to view outputs and continue reasoning.

IMPORTANT: Batch LLM queries efficiently. Instead of calling llm_query() per item, batch items together (e.g., 50-100 items per call) to reduce costs.

When you want to execute Python code, wrap it in triple backticks with 'repl':

```repl
# Example: probe the context
print(f"Context length: {len(context)}")
print(f"First 500 chars: {context[:500]}")
```

Strategy for long contexts:
1. First, probe the context structure (print samples, check format)
2. Design a chunking/filtering strategy based on the task
3. Use code for filtering/aggregation, llm_query() for semantic understanding
4. Build up your answer iteratively, storing results in variables

When finished, provide your final answer using one of:
- FINAL(your answer here) - for direct text answers
- FINAL_VAR(variable_name) - to return a variable's value

Think step by step. Execute code in your response. Remember to answer the original query.'''

    def __init__(self, config: Optional[RLMConfig] = None):
        self.config = config or RLMConfig()

        # Initialize LLM providers
        self.root_provider = self._create_provider(self.config.root_model)
        self.sub_provider = self._create_provider(
            self.config.sub_model or self.config.root_model
        )

        self.trajectory: Optional[RLMTrajectory] = None
        self.current_depth = 0

    def _create_provider(self, model: Optional[str] = None) -> LLMProvider:
        """Create an LLM provider based on config."""
        if self.config.api_provider == "claude-code":
            # Try SDK first, fall back to CLI
            try:
                provider = ClaudeCodeProvider(model=model, cwd=self.config.cwd)
                provider._check_sdk()  # Verify SDK is available
                if provider._sdk_available:
                    return provider
            except Exception:
                pass
            # Fall back to CLI
            return ClaudeCodeCLIProvider(model=model, cwd=self.config.cwd)
        elif self.config.api_provider == "claude-code-cli":
            return ClaudeCodeCLIProvider(model=model, cwd=self.config.cwd)
        elif self.config.api_provider == "anthropic":
            return AnthropicProvider(
                api_key=self.config.api_key,
                model=model or "claude-sonnet-4-20250514"
            )
        elif self.config.api_provider == "openai":
            return OpenAIProvider(
                api_key=self.config.api_key,
                model=model or "gpt-4o"
            )
        else:
            raise ValueError(f"Unknown API provider: {self.config.api_provider}. "
                           f"Valid options: claude-code, claude-code-cli, anthropic, openai")

    def _llm_query(self, prompt: str) -> str:
        """
        Function provided to the REPL for recursive LLM sub-calls.
        """
        if self.current_depth >= self.config.max_recursion_depth:
            # At max depth, use direct LLM call without recursion
            response, cost, tokens = self.sub_provider.query(prompt)
        else:
            # Could implement deeper recursion here
            response, cost, tokens = self.sub_provider.query(prompt)

        # Track the sub-call
        sub_call = SubCallResult(
            query=prompt[:500] + "..." if len(prompt) > 500 else prompt,
            response=response[:500] + "..." if len(response) > 500 else response,
            cost=cost,
            tokens=tokens
        )

        if self.trajectory:
            self.trajectory.sub_calls.append(sub_call)
            self.trajectory.total_cost += cost

        if self.config.verbose:
            print(f"  [Sub-LLM call] Cost: ${cost:.4f}, Tokens: {tokens}")

        return response

    def _extract_code_blocks(self, response: str) -> List[str]:
        """Extract ```repl code blocks from LLM response."""
        pattern = r'```repl\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        return matches

    def _extract_final_answer(self, response: str, repl: REPLEnvironment) -> Optional[str]:
        """Extract FINAL() or FINAL_VAR() answer from response."""
        # Check for FINAL(...)
        final_match = re.search(r'FINAL\((.*?)\)(?:\s*$|\n)', response, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()

        # Check for FINAL_VAR(...)
        final_var_match = re.search(r'FINAL_VAR\((\w+)\)', response)
        if final_var_match:
            var_name = final_var_match.group(1)
            value = repl.get_variable(var_name)
            if value is not None:
                return str(value)
            else:
                return f"[Error: Variable '{var_name}' not found]"

        return None

    def _format_system_prompt(self, context: Union[str, List[str]]) -> str:
        """Format the system prompt with context information."""
        if isinstance(context, list):
            context_type = f"list of {len(context)} items"
            context_total_length = sum(len(str(c)) for c in context)
        else:
            context_type = "string"
            context_total_length = len(context)

        return self.SYSTEM_PROMPT.format(
            context_type=context_type,
            context_total_length=f"{context_total_length:,}"
        )

    def run(self, query: str, context: Union[str, List[str]]) -> str:
        """
        Run the RLM on a query with given context.

        Args:
            query: The question/task to answer
            context: The long context (string or list of strings)

        Returns:
            The final answer string
        """
        start_time = time.time()

        # Initialize trajectory tracking
        context_length = len(context) if isinstance(context, str) else sum(len(c) for c in context)
        self.trajectory = RLMTrajectory(
            query=query,
            context_length=context_length
        )

        # Create REPL environment
        repl = REPLEnvironment(
            context=context,
            llm_query_fn=self._llm_query,
            max_output_chars=self.config.max_output_chars
        )

        # Format system prompt
        system_prompt = self._format_system_prompt(context)

        # Build conversation history
        conversation = [f"Query: {query}"]

        if self.config.verbose:
            print(f"Starting RLM processing...")
            print(f"  Context length: {context_length:,} characters")
            print(f"  Max iterations: {self.config.max_iterations}")
            print()

        final_answer = None

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"=== Iteration {iteration + 1} ===")

            # Query the root LLM
            full_prompt = "\n\n".join(conversation)
            response, cost, tokens = self.root_provider.query(full_prompt, system_prompt)

            self.trajectory.total_cost += cost

            if self.config.verbose:
                print(f"  Root LLM cost: ${cost:.4f}, Tokens: {tokens}")

            # Record iteration
            iteration_record = {
                "iteration": iteration + 1,
                "response_preview": response[:500] + "..." if len(response) > 500 else response,
                "cost": cost,
                "tokens": tokens,
                "code_executions": []
            }

            # Extract and execute code blocks
            code_blocks = self._extract_code_blocks(response)

            execution_output = ""
            for i, code in enumerate(code_blocks):
                if self.config.verbose:
                    print(f"  Executing code block {i + 1}...")

                result = repl.execute(code)

                iteration_record["code_executions"].append({
                    "code": code[:200] + "..." if len(code) > 200 else code,
                    "output_preview": result.output[:200] if result.output else "",
                    "error": result.error,
                    "time": result.execution_time
                })

                if result.error:
                    execution_output += f"\n[Execution {i + 1} Error]:\n{result.error}\n"
                    if self.config.verbose:
                        print(f"    Error: {result.error[:100]}...")
                elif result.output:
                    execution_output += f"\n[Execution {i + 1} Output]:\n{result.output}\n"
                    if self.config.verbose:
                        preview = result.output[:200].replace('\n', ' ')
                        print(f"    Output: {preview}...")

            self.trajectory.iterations.append(iteration_record)

            # Check for final answer
            final_answer = self._extract_final_answer(response, repl)
            if final_answer:
                if self.config.verbose:
                    print(f"\n  Final answer found!")
                break

            # Add response and execution results to conversation
            conversation.append(f"Assistant: {response}")
            if execution_output:
                conversation.append(f"[REPL Output]:{execution_output}")

            # Prompt to continue
            conversation.append("Continue processing or provide your final answer using FINAL() or FINAL_VAR().")

        if final_answer is None:
            final_answer = "[Max iterations reached without final answer]"

        self.trajectory.final_answer = final_answer
        self.trajectory.total_time = time.time() - start_time

        if self.config.verbose:
            print(f"\n=== RLM Complete ===")
            print(f"  Total time: {self.trajectory.total_time:.2f}s")
            print(f"  Total cost: ${self.trajectory.total_cost:.4f}")
            print(f"  Sub-LLM calls: {len(self.trajectory.sub_calls)}")
            print(f"  Iterations: {len(self.trajectory.iterations)}")

        return final_answer

    def get_trajectory(self) -> Optional[RLMTrajectory]:
        """Get the trajectory from the last run."""
        return self.trajectory


def run_rlm(
    query: str,
    context: Union[str, List[str]],
    api_provider: str = "claude-code",
    api_key: Optional[str] = None,
    root_model: Optional[str] = None,
    sub_model: Optional[str] = None,
    max_iterations: int = 20,
    verbose: bool = True,
    cwd: Optional[str] = None
) -> str:
    """
    Convenience function to run RLM.

    Args:
        query: The question/task to answer
        context: The long context (string or list of strings)
        api_provider: "claude-code" (default), "claude-code-cli", "anthropic", or "openai"
        api_key: API key (only needed for anthropic/openai providers)
        root_model: Model for root LLM (ignored for claude-code)
        sub_model: Model for sub-calls (defaults to root_model)
        max_iterations: Maximum REPL iterations
        verbose: Print progress
        cwd: Working directory for Claude Code

    Returns:
        The final answer string
    """
    config = RLMConfig(
        api_provider=api_provider,
        api_key=api_key,
        root_model=root_model,
        sub_model=sub_model,
        max_iterations=max_iterations,
        verbose=verbose,
        cwd=cwd
    )

    rlm = RecursiveLanguageModel(config)
    return rlm.run(query, context)


if __name__ == "__main__":
    # Example usage
    print("Recursive Language Model (RLM) Implementation")
    print("=" * 50)
    print()
    print("Providers:")
    print("  - claude-code (default): Uses Claude Code via SDK or CLI")
    print("  - claude-code-cli: Uses Claude Code CLI directly")
    print("  - anthropic: Uses Anthropic API directly (requires API key)")
    print("  - openai: Uses OpenAI API directly (requires API key)")
    print()
    print("Usage:")
    print("  from rlm import run_rlm, RecursiveLanguageModel, RLMConfig")
    print()
    print("  # Simple usage with Claude Code (default, no API key needed)")
    print('  answer = run_rlm(')
    print('      query="What is the main topic of this document?",')
    print('      context=long_document_text')
    print('  )')
    print()
    print("  # With Anthropic API directly")
    print('  answer = run_rlm(')
    print('      query="What is the main topic of this document?",')
    print('      context=long_document_text,')
    print('      api_provider="anthropic"  # Requires ANTHROPIC_API_KEY')
    print('  )')
    print()
    print("  # Advanced usage with config")
    print('  config = RLMConfig(')
    print('      api_provider="claude-code",  # Uses your Claude Code subscription')
    print('      max_iterations=15,')
    print('      verbose=True')
    print('  )')
    print('  rlm = RecursiveLanguageModel(config)')
    print('  answer = rlm.run(query, context)')
    print('  trajectory = rlm.get_trajectory()')
    print()
    print("See examples/ directory for complete examples.")
