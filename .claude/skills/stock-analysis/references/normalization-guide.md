# Earnings Normalization Guide

How to identify, quantify, and adjust for one-time items to get a clearer picture of sustainable earnings.

## Why Normalize Earnings?

Reported earnings (GAAP) include many items that don't reflect ongoing operations:
- One-time gains inflate earnings temporarily
- Restructuring charges depress earnings temporarily
- "Non-recurring" items often recur regularly

**Normalized earnings** attempt to show what a company would earn in a typical year, excluding unusual items.

## Where to Find One-Time Items

### SEC Filings

| Filing | What to Search For |
|--------|-------------------|
| **10-K** (Annual Report) | Full year unusual items, typically in Notes to Financial Statements |
| **10-Q** (Quarterly Report) | Quarterly unusual items |
| **8-K** | Material events as they happen |
| **Earnings Release** | Company's own reconciliation of GAAP to Non-GAAP |

### Specific Sections to Review

1. **Notes to Financial Statements**
   - Look for notes titled: "Restructuring", "Impairments", "Discontinued Operations", "Legal Matters"

2. **Management Discussion & Analysis (MD&A)**
   - Explains unusual items and their impact
   - Often provides adjusted earnings metrics

3. **Non-GAAP Reconciliation**
   - Companies often provide their own adjusted numbers
   - Lists items they consider non-recurring
   - Be skeptical - companies may exclude recurring expenses

### Search Queries

```
[ticker] 10-K restructuring charges
[ticker] non-recurring items annual report
[ticker] GAAP vs non-GAAP earnings reconciliation
[ticker] unusual expenses [year]
[ticker] one-time charges history
[ticker] adjusted earnings reconciliation
```

## Categories of One-Time Items

### Expense Items (Add Back for Normalized Earnings)

#### Restructuring Charges
- **What**: Costs for layoffs, facility closures, organizational changes
- **Where Found**: Income statement as separate line item or in operating expenses
- **Red Flag**: If restructuring appears every year, may be ongoing cost of doing business

**Example Adjustments**:
| Item | Typical Range | Notes |
|------|--------------|-------|
| Severance payments | $10M-$500M | Size depends on workforce affected |
| Facility closure costs | $5M-$200M | Lease terminations, moving costs |
| Asset write-offs | $10M-$1B+ | Equipment, inventory obsolescence |

#### Asset Impairments
- **What**: Write-downs of goodwill, intangibles, or fixed assets
- **Where Found**: Income statement, often in operating expenses
- **Signal**: May indicate overpaid acquisitions or deteriorating business

**Types**:
| Type | Trigger | Impact |
|------|---------|--------|
| Goodwill impairment | Acquisition underperformed | Large, non-cash charge |
| Intangible impairment | Technology/patents obsolete | Varies |
| PP&E impairment | Asset obsolete or damaged | Varies |

#### Litigation & Settlements
- **What**: Legal costs, settlements, fines
- **Where Found**: Income statement or notes
- **Consideration**: Industry-dependent; some industries have regular legal costs

#### Acquisition-Related Costs
- **What**: M&A transaction costs, integration expenses
- **Where Found**: Operating expenses, often separately disclosed
- **Note**: Integration costs can last 2-3 years post-acquisition

#### Natural Disasters / Force Majeure
- **What**: Costs from disasters, pandemics, supply chain disruptions
- **Where Found**: Operating expenses or separately disclosed
- **Note**: Insurance recoveries may come in subsequent periods

### Revenue Items (Subtract for Normalized Earnings)

#### Large One-Time Contracts
- **What**: Unusually large deals that won't repeat
- **How to Identify**: Significant revenue jump without corresponding customer growth
- **Signal**: Customer concentration, reliance on single deals

#### Settlement Receipts
- **What**: Proceeds from won lawsuits, patent infringement
- **Where Found**: Other income or noted in operating income
- **Example**: Tech companies winning patent cases

#### Asset Sale Gains
- **What**: Profit from selling real estate, business units, investments
- **Where Found**: Other income, gains on sale of assets
- **Common**: Real estate holdings, divestitures

#### Insurance Proceeds
- **What**: Payments for covered losses
- **Where Found**: Other income
- **Note**: May offset expenses in same or different period

#### Tax Benefits
- **What**: One-time tax credits, refunds, changes in deferred taxes
- **Where Found**: Tax expense line, notes on taxes
- **Example**: Tax cuts leading to revaluation of deferred taxes

## Normalization Process

### Step 1: Gather Raw Data

Create a table for each year:

| Year | Reported EPS | Shares Outstanding | Reported Net Income |
|------|-------------|-------------------|---------------------|
| 2024 | $X.XX | XXM | $XXM |
| 2023 | $X.XX | XXM | $XXM |
| ... | | | |

### Step 2: Identify One-Time Items

For each year, list unusual items:

| Year | Item | Type | Pre-Tax Amount | Tax Rate | After-Tax Impact |
|------|------|------|----------------|----------|------------------|
| 2024 | Restructuring | Expense | $150M | 25% | $112.5M |
| 2024 | Settlement gain | Revenue | $50M | 25% | -$37.5M |
| 2023 | Goodwill impairment | Expense | $500M | 0%* | $500M |

*Note: Goodwill impairments are often not tax-deductible

### Step 3: Calculate Adjusted Net Income

```
Adjusted Net Income = Reported Net Income
                    + After-Tax Expenses Added Back
                    - After-Tax One-Time Revenues
```

### Step 4: Calculate Normalized EPS

```
Normalized EPS = Adjusted Net Income / Diluted Shares Outstanding
```

### Step 5: Recalculate Valuation Ratios

| Metric | Using GAAP | Using Normalized |
|--------|------------|------------------|
| P/E Ratio | Current Price / GAAP EPS | Current Price / Normalized EPS |
| EV/EBITDA | EV / GAAP EBITDA | EV / Adjusted EBITDA |

## Tracking "Recurring" One-Time Items

### The Problem

Many companies have "one-time" items every single year:
- "Restructuring" charges for 5 years straight
- "Non-recurring" legal costs annually
- "Unusual" acquisition costs while acquiring every year

### How to Track

Create a frequency table:

| Item Type | 2020 | 2021 | 2022 | 2023 | 2024 | Frequency |
|-----------|------|------|------|------|------|-----------|
| Restructuring | $50M | - | $80M | $30M | $75M | 4 of 5 years |
| Legal costs | $10M | $15M | $20M | $25M | $30M | 5 of 5 years |
| Acquisition costs | - | $100M | - | $150M | $200M | 3 of 5 years |

### Decision Framework

| Frequency | Treatment |
|-----------|-----------|
| 1 time in 5 years | Truly one-time, adjust out |
| 2 times in 5 years | Likely one-time, adjust out with note |
| 3 times in 5 years | Partially recurring, adjust out half |
| 4-5 times in 5 years | Recurring cost of business, don't adjust |

### Average Approach for Recurring Items

If "one-time" items recur, calculate 5-year average:

```
Average Annual Restructuring = Sum of 5 Years / 5
Adjusted Earnings = GAAP + Actual Charge - Average Charge
```

This normalizes without completely ignoring real costs.

## Red Flags in Non-GAAP Adjustments

### When Company Adjustments Are Suspect

1. **Stock-Based Compensation Excluded**
   - SBC is a real cost to shareholders (dilution)
   - Many tech companies exclude it
   - Consider adding back a portion

2. **"Adjusted EBITDA" Excludes Too Much**
   - Legitimate: Interest, taxes, depreciation, amortization
   - Questionable: SBC, restructuring (if recurring), acquisition costs (if frequent acquirer)

3. **Growing Gap Between GAAP and Non-GAAP**
   - If non-GAAP consistently exceeds GAAP by increasing margins
   - May be aggressively excluding real costs

4. **Never Positive on GAAP Basis**
   - Company has never made GAAP profit
   - But claims "adjusted profitability"
   - Adjustments may be masking unsustainable business

### Verification Approach

Compare company's non-GAAP to your own normalization:

| Metric | Company Adjusted | Your Normalized | Difference |
|--------|-----------------|-----------------|------------|
| EPS | $5.00 | $4.20 | Company adds back $0.80 more |
| EBITDA | $1B | $800M | Company excludes $200M more |

Large differences warrant investigation of what company is excluding.

## Normalization Examples

### Example 1: Tech Company with Stock-Based Compensation

```
Reported EPS: $3.00
Stock-Based Comp: $1.50 per share

Company "Adjusted" EPS: $4.50 (adds back all SBC)
Conservative Normalized: $3.75 (adds back 50% of SBC as one-time grants)
```

### Example 2: Industrial with Restructuring History

```
Year | Reported EPS | Restructuring | Company Adjusted |
2020 | $2.00 | $0.50 | $2.50 |
2021 | $2.20 | $0.00 | $2.20 |
2022 | $1.80 | $0.80 | $2.60 |
2023 | $2.50 | $0.30 | $2.80 |
2024 | $2.00 | $0.60 | $2.60 |

5-Year Average Restructuring: $0.44
True Normalized 2024 EPS: $2.00 + $0.60 - $0.44 = $2.16
(Not $2.60 as company claims)
```

### Example 3: Retail with Asset Sales

```
2024 Reported EPS: $5.00
Gain from HQ sale: $2.00 per share
Normalized EPS: $3.00

Using $5.00 P/E ratio overstates earnings quality
$3.00 is more representative of ongoing operations
```

## Data Presentation

### Summary Table Format

| Year | GAAP EPS | Adjustments | Normalized EPS | Key Items |
|------|----------|-------------|----------------|-----------|
| 2024 | $3.00 | +$0.50 | $3.50 | Restructuring ($75M) |
| 2023 | $4.00 | -$1.00 | $3.00 | Settlement gain ($150M) |
| 2022 | $2.50 | +$1.50 | $4.00 | Goodwill impairment ($200M) |

### Earnings Quality Score

Rate earnings quality based on adjustments:

| Score | Criteria |
|-------|----------|
| High | <10% adjustment, items truly one-time |
| Medium | 10-25% adjustment, some recurring items |
| Low | >25% adjustment, frequent "one-time" items |
