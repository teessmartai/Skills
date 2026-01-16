# RLM Code Patterns Reference

Detailed code patterns for implementing Recursive Language Model processing strategies.

## Strategy 1: Keyword Filtering + Targeted Analysis
**Best for:** O(1) complexity, finding specific information

```python
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

# Search for relevant keywords
keywords = ["target_term", "related_term", "alternative_term"]
results = {}
for kw in keywords:
    results[kw] = find_snippets(kw, window=400, max_hits=5)

# Analyze promising chunks with LLM sub-calls
for kw, hits in results.items():
    for i, snippet in hits:
        answer = llm_query(f"Extract information about [topic] from this text:\n{snippet}")
        print(f"Chunk {i}: {answer}")
```

## Strategy 2: Chunked Processing with Aggregation
**Best for:** O(N) complexity, examining all content

```python
lines = context.strip().split('\n')
print(f"Total lines: {len(lines)}")

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

# Aggregate all findings
final_answer = llm_query(f"""
Combine these findings to answer the original question: [question]

Findings from all batches:
{chr(10).join(findings)}
""")
```

## Strategy 3: Hierarchical Decomposition
**Best for:** Complex multi-hop reasoning, document collections

```python
# First pass: High-level structure analysis
structure = llm_query(f"""
Analyze the first 5000 characters to understand the document structure:
{context[:5000]}

Identify: document boundaries, section headers, key themes.
""")
print(f"Structure: {structure}")

# Second pass: Targeted deep analysis
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

# Final synthesis
answer = llm_query(f"""
Based on these section summaries, answer: [original question]

Summaries:
{chr(10).join(summaries)}
""")
```

## Strategy 4: Pairwise Analysis
**Best for:** O(N²) complexity, finding relationships between entries

```python
def classify_entry(entry):
    return llm_query(f"""
    Classify this entry into one of these categories: [categories]
    Entry: {entry}
    Respond with ONLY the category name.
    """)

# Build classification index
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

# Find pairs meeting criteria
matching_pairs = []
target_categories = ["category_a", "category_b"]

candidates = []
for cat in target_categories:
    candidates.extend(classifications.get(cat, []))

for i in range(len(candidates)):
    for j in range(i+1, len(candidates)):
        id1, entry1 = candidates[i]
        id2, entry2 = candidates[j]
        matching_pairs.append((min(id1, id2), max(id1, id2)))

print(f"Found {len(matching_pairs)} matching pairs")
```

## Helper Functions

### Probing the Context
```python
print(f"Context length: {len(context)} characters")
print(f"First 500 characters:\n{context[:500]}")

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

## LLM Sub-Query Best Practices

### Good Query Design
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
```

### Batching for Efficiency
```python
# GOOD: Batch related queries
batch_size = 50  # Process 50 items per LLM call
for i in range(0, len(items), batch_size):
    batch = items[i:i+batch_size]
    results = llm_query(f"Process all these items:\n{chr(10).join(batch)}")

# BAD: One LLM call per item (expensive and slow)
# for item in items:  # If items has 1000 entries, this makes 1000 API calls!
#     result = llm_query(f"Process: {item}")
```

### Context Sizing
```python
MAX_CHUNK_SIZE = 400000  # Leave room for prompt and response

if len(chunk) > MAX_CHUNK_SIZE:
    sub_chunks = [chunk[i:i+MAX_CHUNK_SIZE] for i in range(0, len(chunk), MAX_CHUNK_SIZE)]
    for sub in sub_chunks:
        result = llm_query(f"Process: {sub}")
else:
    result = llm_query(f"Process: {chunk}")
```

## Error Handling

| Issue | Cause | Solution |
|-------|-------|----------|
| Sub-LLM timeout | Chunk too large | Reduce batch size |
| Inconsistent classifications | Ambiguous prompts | Make classification criteria more specific |
| Missing results | Filtering too aggressive | Expand search terms, reduce strictness |
| High costs | Too many sub-calls | Increase batch size, filter before sub-calling |
| Context rot in sub-calls | Chunks still too large | Further decompose, use hierarchical approach |

## Verification Strategy
```python
# Cross-check results with different approaches
approach1_result = keyword_based_search(context, query)
approach2_result = semantic_analysis(context, query)

if approach1_result != approach2_result:
    verification = llm_query(f"""
    Two approaches gave different results:
    Approach 1: {approach1_result}
    Approach 2: {approach2_result}

    Verify which is correct by checking: [specific criteria]
    """)
```
