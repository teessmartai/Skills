---
name: recursive-long-context
description: Process arbitrarily long inputs (documents, codebases, datasets) that exceed context windows using the Recursive Language Model (RLM) approach. Use when dealing with very long documents, large datasets, multi-hop reasoning across many sources, or any task requiring analysis of content too large for direct processing.
---

# Recursive Long Context Processing Skill

Help users process arbitrarily long inputs by treating them as part of an external environment that can be programmatically examined, decomposed, and recursively processed. Based on the Recursive Language Models (RLM) research from MIT CSAIL.

## Quick Start

### Installation

```bash
cd Skills/recursive-long-context
pip install -r requirements.txt
```

### Set API Key

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key"

# OR for OpenAI
export OPENAI_API_KEY="your-api-key"
```

### Python Usage

```python
from rlm import run_rlm, RecursiveLanguageModel, RLMConfig

# Simple usage
answer = run_rlm(
    query="What are the main findings in this document?",
    context=your_long_document,
    api_provider="anthropic",
    verbose=True
)

# Advanced usage with configuration
config = RLMConfig(
    api_provider="anthropic",
    root_model="claude-sonnet-4-20250514",
    sub_model="claude-sonnet-4-20250514",  # Model for recursive sub-calls
    max_iterations=20,
    verbose=True
)

rlm = RecursiveLanguageModel(config)
answer = rlm.run(query, context)

# Access processing trajectory for debugging
trajectory = rlm.get_trajectory()
print(f"Cost: ${trajectory.total_cost:.4f}")
print(f"Sub-LLM calls: {len(trajectory.sub_calls)}")
```

### Command Line Usage

```bash
# Process a single file
python cli.py -q "Summarize the main findings" -f document.txt

# Process a directory of documents
python cli.py -q "Find all mentions of pricing" -d ./documents/

# With options
python cli.py -q "Count entries by category" -f data.txt \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --max-iterations 15 \
    --save-trajectory results.json
```

### Example Scripts

```bash
# Basic usage examples
python examples/basic_usage.py

# Document QA (multi-hop reasoning)
python examples/document_qa.py

# Data aggregation (semantic classification)
python examples/data_aggregation.py
```

## Core Concept

Traditional LLMs struggle with long contexts due to "context rot" - performance degradation as context length increases. RLMs solve this by:

1. **Treating the prompt as an environment variable** - The long input is loaded as a string variable, not fed directly to the model
2. **Programmatic interaction** - Write code to peek into, filter, and decompose the context
3. **Recursive sub-calls** - Invoke LLM calls on smaller chunks to perform semantic analysis
4. **Result aggregation** - Combine findings to form a comprehensive answer

## When to Use This Skill

Use this approach when:
- Input exceeds the model's effective context window
- Task requires dense access to many parts of a long document
- Multi-hop reasoning across multiple documents is needed
- Information aggregation from large datasets is required
- Answer depends on examining nearly every part of the input
- Standard summarization would lose critical details

### Task Complexity Categories

| Complexity | Description | Example Tasks |
|------------|-------------|---------------|
| **O(1)** | Answer requires finding a single piece of information | Needle-in-haystack, single fact retrieval |
| **O(N)** | Answer requires examining every entry once | Counting, categorizing all items, aggregation |
| **O(N²)** | Answer requires comparing pairs of entries | Finding all pairs matching criteria, relationship mapping |

Higher complexity tasks benefit most from RLM processing.

## Information Gathering Phase

Before processing, gather essential information about the task:

### Required Information Checklist

#### Context Information
- [ ] Type of content (documents, code, data entries, mixed)
- [ ] Approximate size (characters, lines, number of documents)
- [ ] Structure of the content (delimiters, headers, format)
- [ ] Is content pre-chunked or needs chunking?

#### Task Information
- [ ] What question needs to be answered?
- [ ] Does the answer require:
  - Finding specific information (sparse access)?
  - Examining most/all of the content (dense access)?
  - Semantic understanding vs. pattern matching?
  - Aggregating results across chunks?
- [ ] Expected output format

#### Processing Preferences
- [ ] Priority: Speed vs. thoroughness?
- [ ] Budget constraints on API calls?
- [ ] Acceptable confidence level?

### Sample Questions Script

```
I'll help you process this long content. Let me understand the task:

**About the Content:**
1. What type of content is this? (documents, code, data, mixed)
2. Approximately how large is it? (characters, pages, number of files)
3. How is it structured? (one continuous text, separate documents, data entries)

**About Your Task:**
4. What question do you need answered?
5. Does your answer require looking at most of the content, or finding specific pieces?
6. Do you need semantic understanding (meaning) or pattern matching (keywords)?

**Processing:**
7. What's more important - speed or thoroughness?
8. Any constraints on processing time or API calls?
```

## Processing Strategy Selection

Based on task complexity, select the appropriate strategy:

### Strategy 1: Keyword Filtering + Targeted Analysis
**Best for:** O(1) complexity, finding specific information

```python
# 1. Use code to filter relevant chunks
def find_snippets(keyword, window=500, max_hits=10):
    hits = []
    for i, chunk in enumerate(context):
        idx = chunk.lower().find(keyword.lower())
        if idx != -1:
            s = max(0, idx - window)
            e = min(len(chunk), idx + len(keyword) + window)
            hits.append((i, chunk[s:e]))
            if len(hits) >= max_hits:
                return hits
    return hits

# 2. Search for relevant keywords
keywords = ["target_term", "related_term", "alternative_term"]
results = {}
for kw in keywords:
    results[kw] = find_snippets(kw, window=400, max_hits=5)

# 3. Analyze promising chunks with LLM sub-calls
for kw, hits in results.items():
    for i, snippet in hits:
        answer = llm_query(f"Extract information about [topic] from this text:\n{snippet}")
        print(f"Chunk {i}: {answer}")
```

### Strategy 2: Chunked Processing with Aggregation
**Best for:** O(N) complexity, examining all content

```python
# 1. Determine chunking strategy
lines = context.strip().split('\n')
print(f"Total lines: {len(lines)}")

# 2. Process in batches with LLM sub-calls
batch_size = 100  # Adjust based on content density
findings = []

for i in range(0, len(lines), batch_size):
    batch = lines[i:i+batch_size]
    batch_str = "\n".join(batch)

    result = llm_query(f"""
    Analyze the following entries and extract [required information].

    Entries:
    {batch_str}

    Return findings in format: [specified format]
    """)

    findings.append(result)
    print(f"Processed batch {i//batch_size + 1}/{(len(lines)-1)//batch_size + 1}")

# 3. Aggregate all findings
final_answer = llm_query(f"""
Combine these findings to answer the original question: [question]

Findings from all batches:
{chr(10).join(findings)}
""")
```

### Strategy 3: Hierarchical Decomposition
**Best for:** Complex multi-hop reasoning, document collections

```python
# 1. First pass: High-level structure analysis
structure = llm_query(f"""
Analyze the first 5000 characters to understand the document structure:
{context[:5000]}

Identify: document boundaries, section headers, key themes.
""")
print(f"Structure: {structure}")

# 2. Second pass: Targeted deep analysis
# Split by identified structure (e.g., document boundaries, headers)
import re
sections = re.split(r'### (.+)', context)

summaries = []
for i in range(1, len(sections), 2):
    header = sections[i]
    content = sections[i+1]

    summary = llm_query(f"""
    Summarize this section titled '{header}' focusing on [specific aspects]:
    {content[:50000]}
    """)
    summaries.append(f"{header}: {summary}")

# 3. Final synthesis
answer = llm_query(f"""
Based on these section summaries, answer: [original question]

Summaries:
{chr(10).join(summaries)}
""")
```

### Strategy 4: Pairwise Analysis
**Best for:** O(N²) complexity, finding relationships between entries

```python
# 1. First, classify all entries
def classify_entry(entry):
    return llm_query(f"""
    Classify this entry into one of these categories: [categories]
    Entry: {entry}
    Respond with ONLY the category name.
    """)

# 2. Build classification index
entries = context.strip().split('\n')
classifications = {}

for i, entry in enumerate(entries):
    if entry.strip():
        cat = classify_entry(entry)
        if cat not in classifications:
            classifications[cat] = []
        classifications[cat].append((i, entry))

    if (i + 1) % 50 == 0:
        print(f"Classified {i+1}/{len(entries)} entries")

# 3. Find pairs meeting criteria
matching_pairs = []
target_categories = ["category_a", "category_b"]

# Get entries matching criteria
candidates = []
for cat in target_categories:
    candidates.extend(classifications.get(cat, []))

# Generate pairs
for i in range(len(candidates)):
    for j in range(i+1, len(candidates)):
        id1, entry1 = candidates[i]
        id2, entry2 = candidates[j]
        matching_pairs.append((min(id1, id2), max(id1, id2)))

print(f"Found {len(matching_pairs)} matching pairs")
```

## Code Patterns Reference

### Probing the Context

```python
# Check size and structure
print(f"Context length: {len(context)} characters")
print(f"First 500 characters:\n{context[:500]}")

# Count lines/entries
lines = context.strip().split('\n')
print(f"Total lines: {len(lines)}")
print(f"First 10 lines:")
for i, line in enumerate(lines[:10]):
    print(f"{i+1}: {line}")
```

### Keyword Search with Context Window

```python
def find_snippets(keyword, window=200, max_hits=10):
    """Find snippets containing keyword with surrounding context."""
    hits = []
    start = 0
    text_lower = context.lower()
    keyword_lower = keyword.lower()

    while True:
        idx = text_lower.find(keyword_lower, start)
        if idx == -1:
            break
        s = max(0, idx - window)
        e = min(len(context), idx + len(keyword) + window)
        hits.append((idx, context[s:e]))
        if len(hits) >= max_hits:
            return hits
        start = idx + 1
    return hits
```

### Batch Processing with Progress

```python
def process_in_batches(items, batch_size, query_template):
    """Process items in batches with LLM sub-calls."""
    results = []
    total_batches = (len(items) - 1) // batch_size + 1

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_str = "\n".join(batch)

        result = llm_query(query_template.format(batch=batch_str))
        results.append(result)

        print(f"Processed batch {i//batch_size + 1}/{total_batches}")

    return results
```

### Regex-Based Filtering

```python
import re

# Find all entries matching a pattern
pattern = r'Date: (\d{4}-\d{2}-\d{2})'
matches = re.findall(pattern, context)
print(f"Found {len(matches)} date entries")

# Split by markers
sections = re.split(r'\n---\n', context)
print(f"Found {len(sections)} sections")

# Extract structured data
entries = re.findall(r'User: (\d+) \| Instance: (.+)', context)
for user_id, instance in entries[:5]:
    print(f"User {user_id}: {instance}")
```

### Building Aggregated Output

```python
# Store results in a variable for final output
final_results = []

for i, chunk in enumerate(chunks):
    result = llm_query(f"Process this chunk: {chunk}")
    final_results.append(result)

# Format final output
formatted_output = "\n".join([f"- {r}" for r in final_results])
print(f"Final results:\n{formatted_output}")

# Return via variable (for very long outputs)
FINAL_VAR(formatted_output)
```

## LLM Sub-Query Best Practices

### Query Design

```python
# GOOD: Specific, structured query
result = llm_query(f"""
Classify the following question into exactly ONE category:
- numeric value
- entity
- location
- description and abstract concept
- abbreviation
- human being

Question: {question}

Respond with ONLY the category name, nothing else.
""")

# BAD: Vague query
result = llm_query(f"What is this about? {question}")
```

### Batching for Efficiency

```python
# GOOD: Batch related queries
batch_size = 50  # Process 50 items per LLM call
for i in range(0, len(items), batch_size):
    batch = items[i:i+batch_size]
    results = llm_query(f"Process all these items:\n{chr(10).join(batch)}")

# BAD: One LLM call per item (expensive and slow)
for item in items:  # If items has 1000 entries, this makes 1000 API calls!
    result = llm_query(f"Process: {item}")
```

### Context Sizing

```python
# Check if chunk fits in sub-LLM context (~500K characters safe)
MAX_CHUNK_SIZE = 400000  # Leave room for prompt and response

if len(chunk) > MAX_CHUNK_SIZE:
    # Split further
    sub_chunks = [chunk[i:i+MAX_CHUNK_SIZE] for i in range(0, len(chunk), MAX_CHUNK_SIZE)]
    for sub in sub_chunks:
        result = llm_query(f"Process: {sub}")
else:
    result = llm_query(f"Process: {chunk}")
```

## Output Format

### Progress Updates

Provide regular progress updates during processing:

```
## Processing Status

**Phase 1: Context Analysis**
- Total size: 2.5M characters
- Structure: 1,000 data entries, newline-separated
- Chunking strategy: 100 entries per batch (10 batches total)

**Phase 2: Batch Processing**
- Batch 1/10: Complete - Found 15 matching entries
- Batch 2/10: Complete - Found 23 matching entries
- Batch 3/10: Processing...

**Phase 3: Aggregation**
- Combining results from all batches...
```

### Final Answer Format

```
## Results

### Summary
[Brief answer to the original question]

### Detailed Findings
[Structured breakdown of results]

### Methodology
- Strategy used: [Chunked Processing with Aggregation]
- Batches processed: [10]
- Total LLM sub-calls: [12]
- Key filtering criteria: [description]

### Confidence
[High/Medium/Low] - [Explanation of confidence level]
```

### For Long Output Tasks

When the output itself is very long (e.g., listing all matching pairs):

```python
# Build output incrementally
output_lines = []

for pair in matching_pairs:
    output_lines.append(f"({pair[0]}, {pair[1]})")

# Store in variable
final_result = "\n".join(output_lines)

# Report statistics
print(f"Total results: {len(output_lines)}")
print(f"First 10 results:")
for line in output_lines[:10]:
    print(line)

# Return full result via variable
FINAL_VAR(final_result)
```

## Error Handling

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Sub-LLM timeout | Chunk too large | Reduce batch size |
| Inconsistent classifications | Ambiguous prompts | Make classification criteria more specific |
| Missing results | Filtering too aggressive | Expand search terms, reduce strictness |
| High costs | Too many sub-calls | Increase batch size, filter before sub-calling |
| Context rot in sub-calls | Chunks still too large | Further decompose, use hierarchical approach |

### Verification Strategies

```python
# Cross-check results with different approaches
approach1_result = keyword_based_search(context, query)
approach2_result = semantic_analysis(context, query)

if approach1_result != approach2_result:
    # Investigate discrepancy
    verification = llm_query(f"""
    Two approaches gave different results:
    Approach 1: {approach1_result}
    Approach 2: {approach2_result}

    Verify which is correct by checking: [specific criteria]
    """)
```

## Example Workflows

### Example 1: Finding Information in Large Document Corpus

**Task:** Find the answer to a multi-hop question across 1000 documents

```python
# 1. Probe the structure
print(f"Total characters: {len(context)}")
# Output: Total characters: 8,300,000

# 2. Extract keywords from question
question = "What festival in La Union celebrates a fish stew dish?"
keywords = ["festival", "La Union", "fish", "stew", "celebration"]

# 3. Filter relevant documents
for kw in keywords:
    hits = find_snippets(kw, window=500, max_hits=5)
    print(f"'{kw}': {len(hits)} hits")
    for idx, snippet in hits:
        print(f"  Position {idx}: ...{snippet[:100]}...")

# 4. Deep analyze promising chunks
promising_chunk = context[chunk_start:chunk_end]
answer = llm_query(f"""
Based on this document, answer: {question}

Document:
{promising_chunk}
""")
```

### Example 2: Aggregating Classifications

**Task:** Count entries by semantic category in a 100K line dataset

```python
# 1. Sample to understand format
lines = context.strip().split('\n')
print(f"Total lines: {len(lines)}")
for line in lines[:5]:
    print(line)

# 2. Define classification task
def classify_batch(batch_lines):
    return llm_query(f"""
    Classify each of these questions into one of 6 categories:
    - numeric value, entity, location, description, abbreviation, human being

    Questions:
    {chr(10).join(batch_lines)}

    Return format: "line_number: category" for each line.
    """)

# 3. Process in batches
all_classifications = []
batch_size = 100
for i in range(0, len(lines), batch_size):
    batch = lines[i:i+batch_size]
    result = classify_batch(batch)
    all_classifications.append(result)
    print(f"Batch {i//batch_size + 1}: done")

# 4. Aggregate counts
category_counts = {}
for classification_result in all_classifications:
    # Parse and count...
    pass

print(f"Final counts: {category_counts}")
```

### Example 3: Finding All Matching Pairs

**Task:** List all user ID pairs where both users have specific characteristics

```python
# 1. First pass: identify qualifying users
qualifying_users = set()

for i in range(0, len(entries), 100):
    batch = entries[i:i+100]
    result = llm_query(f"""
    From these entries, list user IDs that have [criteria].
    Entries: {chr(10).join(batch)}
    Return only the user IDs, one per line.
    """)

    for line in result.strip().split('\n'):
        if line.strip().isdigit():
            qualifying_users.add(int(line.strip()))

print(f"Found {len(qualifying_users)} qualifying users")

# 2. Generate all pairs
users_sorted = sorted(qualifying_users)
pairs = []
for i in range(len(users_sorted)):
    for j in range(i+1, len(users_sorted)):
        pairs.append((users_sorted[i], users_sorted[j]))

# 3. Format output
output = "\n".join([f"({p[0]}, {p[1]})" for p in pairs])
print(f"Total pairs: {len(pairs)}")
FINAL_VAR(output)
```

## Cost and Performance Optimization

### Reducing API Costs

1. **Filter before sub-calling**: Use code to eliminate irrelevant content
2. **Maximize batch sizes**: Process as many items as possible per LLM call
3. **Use smaller models for simple tasks**: Classification can use smaller/cheaper models
4. **Cache intermediate results**: Store in variables to avoid reprocessing

### Improving Speed

1. **Smart chunking**: Chunk by logical boundaries (documents, sections) not arbitrary size
2. **Early termination**: Stop when answer is found with sufficient confidence
3. **Parallel-friendly decomposition**: Design chunks that can be processed independently

### Quality vs. Cost Tradeoffs

| Approach | Quality | Cost | Speed |
|----------|---------|------|-------|
| Every line individually | Highest | Highest | Slowest |
| Batches of 50-100 | High | Medium | Medium |
| Batches of 500+ | Medium | Low | Fast |
| Keyword filter only | Varies | Lowest | Fastest |

**Recommendation**: Start with batches of 50-100 items for semantic tasks, adjust based on results.

## Wrap Up

After processing, provide:

1. **Direct Answer**: Clear response to the original question
2. **Methodology Summary**: How the content was processed
3. **Confidence Assessment**: How reliable the answer is
4. **Verification Option**: Offer to double-check specific findings
5. **Alternative Approaches**: If results are uncertain, suggest other strategies

```
## Processing Complete

### Answer
[Direct answer to the question]

### How This Was Processed
- Total content: [size]
- Strategy: [approach used]
- Batches/chunks: [number]
- LLM sub-calls: [number]

### Confidence: [High/Medium/Low]
[Explanation]

### Would You Like Me To:
- [ ] Verify specific findings?
- [ ] Try a different approach?
- [ ] Provide more detail on any section?
```

## References

This skill is based on the Recursive Language Models (RLM) research:

- **Paper**: "Recursive Language Models" (arXiv:2512.24601)
- **Authors**: Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL)
- **Key Insight**: Long prompts should be treated as part of the environment that the LLM can symbolically interact with, not fed directly into the neural network
