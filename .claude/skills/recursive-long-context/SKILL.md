---
name: recursive-long-context
description: Process arbitrarily long inputs (documents, codebases, datasets) that exceed context windows using the Recursive Language Model (RLM) approach. Use when dealing with very long documents, large datasets, multi-hop reasoning across many sources, or any task requiring analysis of content too large for direct processing.
---

# Recursive Long Context Processing Skill

Process arbitrarily long inputs by treating them as an external environment that can be programmatically examined, decomposed, and recursively processed. Based on the Recursive Language Models (RLM) research from MIT CSAIL.

## Quick Start

### Installation

```bash
cd .claude/skills/recursive-long-context
pip install -r requirements.txt
```

### Python Usage

```python
from rlm import run_rlm, RecursiveLanguageModel, RLMConfig

# Simple usage with Claude Code (default - uses your subscription)
answer = run_rlm(
    query="What are the main findings in this document?",
    context=your_long_document,
    verbose=True
)

# Advanced usage with configuration
config = RLMConfig(
    api_provider="claude-code",  # Default - no API key needed
    max_iterations=20,
    verbose=True
)

rlm = RecursiveLanguageModel(config)
answer = rlm.run(query, context)
```

### Alternative: Direct API Usage

If you prefer direct API access instead of Claude Code:

```bash
export ANTHROPIC_API_KEY="your-api-key"  # For Anthropic
# OR
export OPENAI_API_KEY="your-api-key"     # For OpenAI
```

```python
# With Anthropic API directly
answer = run_rlm(
    query="Analyze this document",
    context=document,
    api_provider="anthropic"  # Requires ANTHROPIC_API_KEY
)
```

### Command Line Usage

```bash
# Using Claude Code (default)
python cli.py -q "Summarize the main findings" -f document.txt
python cli.py -q "Find all mentions of pricing" -d ./documents/

# Using Anthropic API directly
python cli.py -q "Count entries by category" -f data.txt --provider anthropic
```

### Available Providers

| Provider | Description | API Key Required |
|----------|-------------|------------------|
| `claude-code` | Claude Code SDK (default) | No (uses subscription) |
| `claude-code-cli` | Claude Code CLI fallback | No (uses subscription) |
| `anthropic` | Anthropic API directly | Yes |
| `openai` | OpenAI API directly | Yes |

## Core Concept

Traditional LLMs struggle with long contexts due to "context rot." RLMs solve this by:

1. **Treating the prompt as an environment variable** - Long input loaded as string variable, not fed directly to model
2. **Programmatic interaction** - Write code to peek, filter, and decompose the context
3. **Recursive sub-calls** - Invoke LLM calls on smaller chunks for semantic analysis
4. **Result aggregation** - Combine findings into comprehensive answer

## Task Complexity Guide

| Complexity | Description | Strategy |
|------------|-------------|----------|
| **O(1)** | Find single piece of info | Keyword filtering + targeted analysis |
| **O(N)** | Examine every entry | Chunked processing with aggregation |
| **O(N²)** | Compare pairs of entries | Pairwise classification |

Higher complexity tasks benefit most from RLM processing.

## Information Gathering

Before processing, gather:

**Context Info:**
- Type of content (documents, code, data, mixed)
- Approximate size (characters, lines, documents)
- Structure (delimiters, headers, format)

**Task Info:**
- What question needs answered?
- Dense or sparse access required?
- Expected output format

**Preferences:**
- Speed vs thoroughness priority
- Budget constraints on API calls

## Processing Strategies

See [code-patterns.md](references/code-patterns.md) for detailed implementation code.

### Strategy 1: Keyword Filtering + Targeted Analysis
**Best for:** O(1) complexity, finding specific information

1. Use code to filter relevant chunks by keyword
2. Search for multiple related keywords
3. Analyze promising chunks with LLM sub-calls

### Strategy 2: Chunked Processing with Aggregation
**Best for:** O(N) complexity, examining all content

1. Determine chunking strategy based on content structure
2. Process in batches with LLM sub-calls
3. Aggregate all findings into final answer

### Strategy 3: Hierarchical Decomposition
**Best for:** Complex multi-hop reasoning, document collections

1. First pass: High-level structure analysis
2. Second pass: Targeted deep analysis by section
3. Final synthesis across all summaries

### Strategy 4: Pairwise Analysis
**Best for:** O(N²) complexity, finding relationships

1. Classify all entries into categories
2. Build classification index
3. Generate pairs meeting criteria

## LLM Sub-Query Best Practices

1. **Specific prompts**: Request exact output format, one category at a time
2. **Batch queries**: Process 50-100 items per LLM call to reduce costs
3. **Context sizing**: Keep chunks under 400K characters for sub-calls
4. **Filter first**: Use code to eliminate irrelevant content before LLM calls

## Output Format

Provide:
1. **Direct Answer**: Clear response to original question
2. **Methodology Summary**: Strategy used, batches processed, LLM sub-calls made
3. **Confidence Assessment**: High/Medium/Low with explanation

## Cost Optimization

| Approach | Quality | Cost | Speed |
|----------|---------|------|-------|
| Every line individually | Highest | Highest | Slowest |
| Batches of 50-100 | High | Medium | Medium |
| Batches of 500+ | Medium | Low | Fast |
| Keyword filter only | Varies | Lowest | Fastest |

**Recommendation**: Start with batches of 50-100 items for semantic tasks.

## References

- **Paper**: "Recursive Language Models" (arXiv:2512.24601)
- **Authors**: Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL)
- **Key Insight**: Long prompts should be treated as part of the environment that the LLM can symbolically interact with
