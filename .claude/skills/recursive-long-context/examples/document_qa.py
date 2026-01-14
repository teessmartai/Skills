#!/usr/bin/env python3
"""
Document QA example for the Recursive Language Model (RLM).

This example demonstrates using RLM for multi-hop question answering
across a large document corpus, similar to the BrowseComp+ benchmark
from the paper.
"""

import sys
sys.path.insert(0, '..')

from rlm import RecursiveLanguageModel, RLMConfig


def create_sample_corpus():
    """Create a sample document corpus for demonstration."""

    documents = []

    # Document 1: Company Overview
    documents.append("""
=== DOCUMENT: TechCorp Annual Report 2024 ===

TechCorp Inc. was founded in 2010 by Sarah Chen and Michael Rodriguez
in Palo Alto, California. The company started as a small startup focused
on cloud computing solutions.

Key Milestones:
- 2010: Company founded with $500K seed funding
- 2012: Launched CloudSync product, reached 10K users
- 2015: Series A funding of $15M led by Venture Capital Partners
- 2018: Acquired DataFlow Inc. for $50M
- 2020: IPO on NASDAQ at $45 per share
- 2023: Revenue reached $500M, 5000 employees globally

Current Leadership:
- CEO: Sarah Chen (co-founder)
- CTO: James Wu (joined 2015)
- CFO: Lisa Park (joined 2019)
- VP Engineering: Michael Rodriguez (co-founder)

The company's flagship product, CloudSync, now serves over 2 million
enterprise customers in 45 countries.
    """)

    # Document 2: Product Information
    documents.append("""
=== DOCUMENT: TechCorp Product Catalog ===

CloudSync Enterprise Suite:
- CloudSync Core: Basic file synchronization ($10/user/month)
- CloudSync Pro: Advanced features with API access ($25/user/month)
- CloudSync Enterprise: Full suite with dedicated support ($50/user/month)

DataFlow Integration (acquired 2018):
- Real-time data pipeline management
- Supports 50+ data source connectors
- Processing capacity: 1TB/hour per instance

New in 2024:
- CloudSync AI: Machine learning-powered analytics
- Launched Q2 2024, already 50K users
- Pricing: Starting at $100/user/month
    """)

    # Document 3: Financial Details
    documents.append("""
=== DOCUMENT: TechCorp Q4 2024 Financial Summary ===

Revenue Breakdown:
- CloudSync subscriptions: $350M (70%)
- DataFlow services: $100M (20%)
- Professional services: $50M (10%)

Geographic Distribution:
- North America: 55%
- Europe: 25%
- Asia Pacific: 15%
- Rest of World: 5%

Notable Customers:
- Global Bank Corp (signed 2023, $5M annual contract)
- National Healthcare System (signed 2024, $3M annual contract)
- RetailMax Inc. (signed 2022, $2M annual contract)

Stock Performance:
- IPO price (2020): $45
- Current price: $127
- Market cap: $12.5B
    """)

    # Add more documents to increase corpus size
    for i in range(20):
        documents.append(f"""
=== DOCUMENT: Industry News Article {i+1} ===

Tech Industry Update - Week {i+1} of 2024

Various technology companies announced new products and partnerships
this week. Cloud computing continues to grow at 25% annually.

Notable mentions:
- Company A launched new AI features
- Company B acquired startup for $100M
- Company C reached 1M customers milestone

Market Analysis:
The enterprise software market is expected to reach $800B by 2025.
Key growth drivers include remote work adoption and digital transformation.

TechCorp was mentioned in relation to their CloudSync AI launch,
which analysts predict will add $50M to annual revenue.
        """)

    return "\n\n".join(documents)


def main():
    print("=" * 60)
    print("RLM Document QA Example")
    print("=" * 60)
    print()

    # Create sample corpus
    corpus = create_sample_corpus()
    print(f"Corpus size: {len(corpus):,} characters")
    print(f"Number of documents: {corpus.count('=== DOCUMENT:')}")
    print()

    # Multi-hop question that requires finding and connecting information
    query = """
    Answer this multi-hop question:
    1. Who founded TechCorp and in what year?
    2. What company did TechCorp acquire, and for how much?
    3. What is the current stock price compared to the IPO price?
    4. What is TechCorp's total revenue and what percentage comes from CloudSync?

    Provide a comprehensive answer connecting all these facts.
    """

    print("Query:", query.strip())
    print()

    # Configure RLM
    config = RLMConfig(
        api_provider="anthropic",
        root_model="claude-sonnet-4-20250514",
        sub_model="claude-sonnet-4-20250514",
        max_iterations=15,
        verbose=True
    )

    # Run RLM
    rlm = RecursiveLanguageModel(config)
    answer = rlm.run(query, corpus)

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


if __name__ == "__main__":
    import os

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: No API key found.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    main()
