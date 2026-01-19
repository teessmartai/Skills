# Financial Ratios Guide

Detailed guide for calculating, interpreting, and comparing financial ratios in stock analysis.

## Valuation Ratios

### Price-to-Earnings (P/E) Ratio

**Formula**: Stock Price / Earnings Per Share (EPS)

**Variants**:
- **Trailing P/E (TTM)**: Uses last 12 months of actual earnings
- **Forward P/E**: Uses estimated earnings for next 12 months
- **Shiller P/E (CAPE)**: Uses 10-year inflation-adjusted average earnings

**Interpretation**:
| P/E Range | General Interpretation |
|-----------|----------------------|
| <10 | Potentially undervalued or facing serious challenges |
| 10-15 | Value territory for mature companies |
| 15-25 | Fair value for quality growth companies |
| 25-40 | Growth premium priced in |
| >40 | High growth expectations or speculative |

**Industry Baselines**:
| Sector | Typical P/E Range |
|--------|-------------------|
| Utilities | 12-18 |
| Financials | 10-15 |
| Consumer Staples | 18-25 |
| Industrials | 15-22 |
| Healthcare | 18-30 |
| Technology | 25-45 |
| High-Growth Tech | 40-100+ |

**Limitations**:
- Meaningless for companies with negative earnings
- Easily distorted by one-time items
- Doesn't account for growth rates
- Capital structure not considered

### Enterprise Value / EBITDA (EV/EBITDA)

**Formula**:
```
Enterprise Value = Market Cap + Total Debt - Cash & Equivalents
EBITDA = Earnings Before Interest, Taxes, Depreciation & Amortization
EV/EBITDA = Enterprise Value / EBITDA
```

**Why Use EV/EBITDA Over P/E**:
1. **Capital Structure Neutral**: Allows comparison of companies with different debt levels
2. **Depreciation Neutral**: Useful for capital-intensive businesses
3. **Tax Neutral**: Compares companies across tax jurisdictions

**Interpretation**:
| EV/EBITDA | General Interpretation |
|-----------|----------------------|
| <6 | Potentially undervalued or distressed |
| 6-10 | Fair value for mature businesses |
| 10-15 | Quality premium or growth expectations |
| 15-20 | High growth company |
| >20 | Very high growth expectations |

**Industry Benchmarks**:
| Industry | Typical EV/EBITDA |
|----------|------------------|
| Oil & Gas | 4-8 |
| Retail | 6-10 |
| Manufacturing | 7-11 |
| Healthcare | 10-15 |
| Software/SaaS | 15-30 |
| High-Growth Tech | 30-50+ |

### Price-to-Sales (P/S) Ratio

**Formula**: Market Cap / Annual Revenue

**When to Use**:
- Unprofitable growth companies
- Companies with temporarily depressed earnings
- Comparing companies with different margin profiles

**Interpretation**:
| P/S Range | General Interpretation |
|-----------|----------------------|
| <1 | Deep value or serious issues |
| 1-3 | Typical for mature businesses |
| 3-10 | Growth premium |
| >10 | High growth, high margin expected |

### Price-to-Book (P/B) Ratio

**Formula**: Stock Price / Book Value Per Share

**Book Value** = Total Assets - Total Liabilities

**Best Used For**:
- Banks and financial institutions
- Asset-heavy businesses (REITs, utilities)
- Companies potentially being acquired for assets

**Interpretation**:
| P/B Range | Interpretation |
|-----------|---------------|
| <1 | Trading below liquidation value (value trap or opportunity) |
| 1-2 | Fair value for asset-heavy businesses |
| 2-5 | Premium for quality or growth |
| >5 | Intangible assets or brand value driving premium |

### PEG Ratio

**Formula**: P/E Ratio / Annual EPS Growth Rate

**Example**: P/E of 30 with 30% growth = PEG of 1.0

**Interpretation**:
| PEG | Interpretation |
|-----|---------------|
| <1 | Undervalued relative to growth |
| 1.0 | Fairly valued |
| >1 | Overvalued relative to growth |
| >2 | Very expensive relative to growth |

**Limitations**:
- Assumes growth is sustainable
- Doesn't work with negative growth
- Quality of earnings not considered

## Profitability Ratios

### Gross Margin

**Formula**: (Revenue - Cost of Goods Sold) / Revenue

**Indicates**: Pricing power, production efficiency, competitive position

| Gross Margin | Business Type |
|--------------|---------------|
| 70-90% | Software, luxury goods |
| 50-70% | Pharmaceuticals, branded products |
| 30-50% | Consumer goods, retail |
| 10-30% | Groceries, commodities |
| <10% | Distribution, low-margin retail |

### Operating Margin

**Formula**: Operating Income / Revenue

**Indicates**: Efficiency of core operations, cost control

**What to Look For**:
- Expanding margins = operational leverage or pricing power
- Contracting margins = competition, rising costs, or investment phase
- Compare to industry peers, not across industries

### Net Profit Margin

**Formula**: Net Income / Revenue

**Indicates**: Bottom-line profitability after all expenses

**Typical Ranges**:
| Sector | Typical Net Margin |
|--------|-------------------|
| Software | 20-35% |
| Pharmaceuticals | 15-25% |
| Banking | 20-30% |
| Consumer Staples | 5-15% |
| Retail | 2-7% |
| Airlines | 2-10% |
| Grocery | 1-3% |

### Return on Equity (ROE)

**Formula**: Net Income / Shareholders' Equity

**DuPont Decomposition**:
```
ROE = Net Margin × Asset Turnover × Equity Multiplier
    = (Net Income/Revenue) × (Revenue/Assets) × (Assets/Equity)
```

**Interpretation**:
| ROE | Quality |
|-----|---------|
| >20% | Excellent |
| 15-20% | Good |
| 10-15% | Average |
| <10% | Below average |

**Warning**: High ROE can come from excessive leverage (high debt)

### Return on Invested Capital (ROIC)

**Formula**: NOPAT / Invested Capital

```
NOPAT = Operating Income × (1 - Tax Rate)
Invested Capital = Total Debt + Shareholders' Equity - Cash
```

**Why ROIC Matters**:
- Shows how efficiently capital is deployed
- ROIC > WACC = value creation
- ROIC < WACC = value destruction

**Benchmarks**:
| ROIC | Assessment |
|------|------------|
| >25% | Exceptional capital allocation |
| 15-25% | Strong |
| 10-15% | Adequate |
| <10% | Weak capital allocation |

## Financial Health Ratios

### Debt-to-Equity Ratio

**Formula**: Total Debt / Shareholders' Equity

**Interpretation by Industry**:
| Industry | Acceptable D/E |
|----------|----------------|
| Utilities | 1.0-2.0 |
| Banks | 8.0-15.0 (regulated) |
| Real Estate | 1.0-2.5 |
| Technology | 0-0.5 |
| Industrial | 0.5-1.5 |

### Interest Coverage Ratio

**Formula**: EBIT / Interest Expense

**Risk Assessment**:
| Coverage | Risk Level |
|----------|------------|
| >10 | Very safe |
| 5-10 | Comfortable |
| 2-5 | Adequate |
| 1-2 | Concerning |
| <1 | High risk of default |

### Current Ratio

**Formula**: Current Assets / Current Liabilities

**Interpretation**:
| Ratio | Liquidity Status |
|-------|-----------------|
| >2.0 | Strong liquidity |
| 1.5-2.0 | Healthy |
| 1.0-1.5 | Adequate |
| <1.0 | Potential liquidity issues |

### Free Cash Flow Yield

**Formula**: Free Cash Flow / Market Cap

```
Free Cash Flow = Operating Cash Flow - Capital Expenditures
```

**Interpretation**:
| FCF Yield | Assessment |
|-----------|------------|
| >8% | Potentially undervalued or returning capital |
| 5-8% | Healthy cash generation |
| 2-5% | Growth company reinvesting |
| <2% | Low cash generation or high capex needs |

## Historical Analysis Framework

### 5-Year Trend Analysis

For each key ratio, track:

1. **Current Value**: Today's ratio
2. **5-Year Average**: Mean over past 5 years
3. **5-Year High**: Maximum value
4. **5-Year Low**: Minimum value
5. **Standard Deviation**: Volatility of the ratio
6. **Trend Direction**: Improving, stable, or deteriorating

### Z-Score Analysis

Calculate how many standard deviations current value is from mean:

```
Z-Score = (Current Value - 5Y Average) / Standard Deviation
```

| Z-Score | Interpretation |
|---------|---------------|
| < -2 | Historically very cheap |
| -1 to -2 | Cheap |
| -1 to 1 | Normal range |
| 1 to 2 | Expensive |
| > 2 | Historically very expensive |

### Ratio Correlation Check

When multiple ratios disagree:
- P/E says cheap but EV/EBITDA says expensive → Check debt levels
- P/E says expensive but P/S says cheap → Check margins
- All ratios say expensive → Likely truly expensive
- All ratios say cheap → Value opportunity or value trap (investigate why)

## Data Sources

### Primary Sources
- **SEC EDGAR**: Official 10-K and 10-Q filings
- **Company Investor Relations**: Earnings reports, presentations

### Aggregated Data Sources
- **Yahoo Finance**: Free, comprehensive data
- **Finviz**: Screening and ratio comparisons
- **Morningstar**: Historical ratios, fair value estimates
- **Simply Wall St**: Visual ratio analysis
- **Koyfin**: Professional-grade data, some free
- **TIKR**: Detailed financials with history

### Search Queries for Historical Data
```
[ticker] historical P/E ratio chart
[ticker] 10 year EV/EBITDA history
[ticker] ROE historical trend
[ticker] financial ratios Yahoo Finance
[ticker] valuation history Morningstar
```
