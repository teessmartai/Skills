# RLM Benchmark Testing Guide

This guide explains how to reproduce the experiments from the "Recursive Language Models" paper (arXiv:2512.24601).

## Benchmark Sources

| Benchmark | Source | Download |
|-----------|--------|----------|
| **S-NIAH** | NVIDIA RULER | [GitHub](https://github.com/NVIDIA/RULER) |
| **OOLONG** | CMU/LTI | [GitHub](https://github.com/abertsch72/oolong), [HuggingFace](https://huggingface.co/oolongbench) |
| **OOLONG-Pairs** | RLM Paper | See Appendix E.1 in paper |
| **BrowseComp+** | Tevatron | [GitHub](https://github.com/texttron/BrowseComp-Plus), [HuggingFace](https://huggingface.co/datasets/Tevatron/browsecomp-plus) |
| **LongBench-v2 CodeQA** | THUDM | [GitHub](https://github.com/THUDM/LongBench), [HuggingFace](https://huggingface.co/datasets/THUDM/LongBench-v2) |

## Quick Setup

```bash
# Install dependencies
cd .claude/skills/recursive-long-context
pip install -r requirements.txt
pip install datasets  # For HuggingFace datasets

# Set API key (choose one)
export ANTHROPIC_API_KEY="your-key"
# OR use Claude Code provider (no key needed)
```

## 1. OOLONG Benchmark (Recommended Starting Point)

OOLONG is the most accessible benchmark - 50 tasks, ~131K tokens, tests O(N) complexity.

### Download

```python
from datasets import load_dataset

# Load OOLONG trec_coarse split
dataset = load_dataset("oolongbench/oolong-synth", "trec_coarse")
```

### Run Test

```bash
python benchmarks/run_oolong.py --provider claude-code --max-tasks 10
```

## 2. S-NIAH Benchmark

Tests O(1) complexity - finding a single needle in varying haystack sizes.

### Setup

```bash
git clone https://github.com/NVIDIA/RULER.git
cd RULER
# Follow RULER setup instructions
```

### Generate Tasks

```python
# RULER provides scripts to generate S-NIAH tasks at various lengths
# See RULER/scripts/data/synthetic/niah.py
```

## 3. BrowseComp+ Benchmark

Most challenging - 6M-11M tokens, multi-hop reasoning over 1000 documents.

### Download

```python
from datasets import load_dataset

# Load BrowseComp+ with 100K document corpus
dataset = load_dataset("Tevatron/browsecomp-plus")
```

### Run Test

```bash
python benchmarks/run_browsecomp.py --num-docs 1000 --max-tasks 20
```

## 4. LongBench-v2 CodeQA

Code repository understanding tasks.

### Download

```python
from datasets import load_dataset

dataset = load_dataset("THUDM/LongBench-v2", split="train")
# Filter for code_repo_understanding domain
code_qa = [x for x in dataset if x["domain"] == "code_repo_understanding"]
```

## Evaluation Metrics

| Benchmark | Metric | Paper Results (GPT-5 RLM) |
|-----------|--------|---------------------------|
| S-NIAH | Accuracy % | ~95-100% |
| OOLONG | Score (0.75^|y-ŷ|) | 56.50% |
| OOLONG-Pairs | F1 Score | 58.00% |
| BrowseComp+ (1K) | Accuracy % | 91.33% |
| CodeQA | Accuracy % | 62.00% |

## Baseline Comparisons (RLM vs Standard LLM)

The benchmark scripts currently only run with RLM. To compare against a baseline (standard LLM without RLM), you can run direct queries through the Anthropic/OpenAI API on the same tasks.

**Paper-reported baseline comparisons:**

| Benchmark | Baseline (Standard) | With RLM | Improvement |
|-----------|---------------------|----------|-------------|
| S-NIAH (4M tokens) | ~0% | ~95-100% | Massive |
| OOLONG | 44.93% | 56.50% | +11.57% |
| OOLONG-Pairs | 50.10% | 58.00% | +7.90% |
| BrowseComp+ (1K docs) | 5.77%* | 91.33% | +85.56% |
| CodeQA | 53.00% | 62.00% | +9.00% |

*\*BrowseComp+ baseline uses RAG retrieval since full context exceeds model limits*

**Key insight:** The larger the context, the more dramatic the improvement. Standard LLMs suffer from "context rot" on very long inputs, while RLM maintains accuracy by processing content programmatically.

**Running your own baselines:**
```python
# Baseline (direct query without RLM)
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{"role": "user", "content": f"{context}\n\nQuestion: {query}"}]
)
baseline_answer = response.content[0].text

# RLM (recursive processing)
from rlm import run_rlm
rlm_answer = run_rlm(query, context, api_provider="anthropic")
```

## Cost Estimates

Based on paper's experiments:

| Benchmark | Avg Cost per Query (RLM) |
|-----------|-------------------------|
| OOLONG | $0.43 |
| OOLONG-Pairs | $0.33 |
| BrowseComp+ (1K) | $0.99 |
| CodeQA | $0.11 |

**Note**: Using `claude-code` provider has $0 API cost (uses subscription).

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601)
- [OOLONG Paper](https://arxiv.org/abs/2511.02817)
- [BrowseComp+ Paper](https://arxiv.org/abs/2508.06600)
- [RULER Paper](https://arxiv.org/abs/2404.06654)
- [LongBench-v2 Paper](https://arxiv.org/abs/2412.15204)
