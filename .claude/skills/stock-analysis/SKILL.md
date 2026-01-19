---
name: stock-analysis
description: Analyze stocks for investment decisions by examining historical financial ratios (P/E, EV/EBITDA, etc.), forecasted metrics, normalized earnings, unusual expense patterns, social media sentiment, and recent news. Use when investors want to evaluate whether a stock is a good investment based on fundamental analysis and market sentiment.
---

# Stock Analysis Skill

Help investors analyze stocks by gathering comprehensive financial data, normalizing metrics for one-time items, comparing historical vs current vs forecasted valuations, and synthesizing market sentiment and news into actionable investment insights.

## Information Gathering Phase

Before analyzing, you MUST gather the following information. Ask questions conversationally.

### Required Information

#### Stock Information
- [ ] Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
- [ ] Exchange if ambiguous (NYSE, NASDAQ, etc.)

#### Analysis Preferences
- [ ] Investment time horizon (short-term: <1 year, medium: 1-3 years, long-term: 3+ years)
- [ ] Risk tolerance (conservative, moderate, aggressive)
- [ ] Specific concerns or areas to focus on (if any)
- [ ] Comparison stocks or competitors to benchmark against (optional)

## Analysis Framework

Perform analysis in the following order, using deep research (web search) for each section.

### Phase 1: Company Overview

Search for and summarize:
- Company name and business description
- Industry and sector classification
- Market capitalization
- Key products/services and revenue segments
- Geographic revenue breakdown
- Recent major events (M&A, leadership changes, strategic shifts)

### Phase 2: Historical Financial Ratios (5-10 Years)

See [financial-ratios-guide.md](references/financial-ratios-guide.md) for detailed ratio calculations.

#### Valuation Metrics to Track

| Metric | Description | What to Look For |
|--------|-------------|------------------|
| **P/E Ratio** | Price / Earnings Per Share | Compare to historical average and industry |
| **Forward P/E** | Price / Expected EPS | Growth expectations built into price |
| **EV/EBITDA** | Enterprise Value / EBITDA | Capital structure-neutral valuation |
| **EV/Revenue** | Enterprise Value / Revenue | For high-growth or unprofitable companies |
| **P/B Ratio** | Price / Book Value | Asset-heavy industries |
| **P/S Ratio** | Price / Sales | Revenue multiple for growth stocks |
| **PEG Ratio** | P/E / Earnings Growth Rate | Growth-adjusted valuation |

#### Profitability Metrics

| Metric | Description | Benchmark |
|--------|-------------|-----------|
| **Gross Margin** | Gross Profit / Revenue | Pricing power, COGS efficiency |
| **Operating Margin** | Operating Income / Revenue | Operational efficiency |
| **Net Margin** | Net Income / Revenue | Bottom-line profitability |
| **ROE** | Net Income / Shareholders' Equity | Return on equity capital |
| **ROA** | Net Income / Total Assets | Asset efficiency |
| **ROIC** | NOPAT / Invested Capital | Capital allocation effectiveness |

#### Financial Health Metrics

| Metric | Description | Warning Signs |
|--------|-------------|---------------|
| **Debt/Equity** | Total Debt / Equity | >2.0 in most industries |
| **Interest Coverage** | EBIT / Interest Expense | <1.5 is concerning |
| **Current Ratio** | Current Assets / Current Liabilities | <1.0 is risky |
| **Quick Ratio** | (Cash + Receivables) / Current Liabilities | <0.5 is tight |
| **FCF Yield** | Free Cash Flow / Market Cap | Cash generation ability |

### Search Strategy for Historical Data

Use these search queries:
- `[ticker] historical P/E ratio 10 year chart`
- `[ticker] EV/EBITDA history`
- `[ticker] financial ratios historical data`
- `[ticker] profitability margins history`
- `[company name] annual report financial highlights`

### Phase 3: Earnings Normalization

See [normalization-guide.md](references/normalization-guide.md) for detailed methodology.

#### One-Time Items to Identify

Search annual reports (10-K) and quarterly reports (10-Q) for:

**Revenue Adjustments:**
- Large one-time contract wins or losses
- Licensing windfalls
- Settlement receipts
- Discontinued operations revenue
- Currency gains from unusual transactions

**Expense Adjustments:**
- Restructuring charges
- Goodwill/asset impairments
- Litigation settlements
- Natural disaster impacts
- Acquisition-related costs
- Executive severance packages
- Pension adjustments

#### Normalization Process

1. **Identify Items**: Search `[ticker] 10-K unusual items` and `[ticker] non-recurring charges`
2. **Quantify Impact**: Note dollar amount and which year(s)
3. **Adjust Earnings**: Add back expenses or subtract one-time revenues
4. **Recalculate Ratios**: Use normalized earnings for P/E calculations
5. **Track Frequency**: Note how often "one-time" items actually recur

#### Output: Normalization Table

| Year | Reported EPS | Adjustments | Normalized EPS | Notes |
|------|--------------|-------------|----------------|-------|
| 2024 | $X.XX | +$X.XX restructuring | $X.XX | Plant closure |
| 2023 | $X.XX | -$X.XX settlement gain | $X.XX | Patent lawsuit won |

### Phase 4: Forecasted Metrics (4-8 Quarters)

#### Data to Gather

Search for analyst estimates:
- `[ticker] analyst estimates EPS`
- `[ticker] revenue forecast next year`
- `[ticker] earnings estimates consensus`
- `[ticker] analyst price targets`

#### Forecast Information to Collect

| Metric | Current Q | Next Q | Q+2 | Q+3 | Q+4 | Next FY | FY+2 |
|--------|-----------|--------|-----|-----|-----|---------|------|
| Revenue Estimate | | | | | | | |
| EPS Estimate | | | | | | | |
| EBITDA Estimate | | | | | | | |

#### Forecast Methodology Explanation

Explain to the user HOW forecasts are determined:

1. **Sell-Side Analyst Models**: Investment bank analysts build DCF and comparable company models
2. **Company Guidance**: Management provides revenue/earnings guidance during earnings calls
3. **Consensus Aggregation**: Services like FactSet, Bloomberg aggregate multiple analyst estimates
4. **Model Inputs**: Revenue growth assumptions, margin expansion/contraction, share count changes
5. **Revision Trends**: Note if estimates are being revised up or down over past 30/60/90 days

Search: `[ticker] analyst estimate methodology` and `[ticker] earnings guidance`

### Phase 5: Valuation Comparison

Create comparison showing current vs historical vs forward:

| Metric | 5Y Low | 5Y Avg | 5Y High | Current | Forward (NTM) | Assessment |
|--------|--------|--------|---------|---------|---------------|------------|
| P/E | | | | | | Cheap/Fair/Expensive |
| EV/EBITDA | | | | | | |
| P/S | | | | | | |
| PEG | | | | | | |

### Phase 6: Social Media Sentiment

See [sentiment-and-news-guide.md](references/sentiment-and-news-guide.md) for platform-specific guidance.

#### Platforms to Search

| Platform | Search Query | What to Look For |
|----------|--------------|------------------|
| **X (Twitter)** | `$[ticker] sentiment` | Retail investor buzz, breaking news reaction |
| **Reddit** | `[ticker] r/wallstreetbets OR r/stocks OR r/investing` | Retail sentiment, DD posts |
| **StockTwits** | `[ticker] stocktwits sentiment` | Trader sentiment, message volume |
| **Seeking Alpha** | `[ticker] seeking alpha analysis` | In-depth analysis, bull/bear debates |

#### Sentiment Indicators

- **Volume**: Is discussion increasing or decreasing?
- **Tone**: Bullish, bearish, or neutral overall?
- **Key Themes**: What are people focused on?
- **Influencer Views**: Notable investors' positions/opinions?
- **Short Interest**: Are shorts covering or building?

Search: `[ticker] short interest` and `[ticker] institutional ownership changes`

### Phase 7: News Analysis

#### Recent News Search (Past 3 Months)

Search for:
- `[ticker] news past month`
- `[company name] latest news [current year]`
- `[company name] earnings report analysis`
- `[company name] press releases`

#### Industry News Search

Search for:
- `[industry] industry outlook [current year]`
- `[industry] market trends`
- `[sector] headwinds tailwinds`
- `[company name] competitors news`

#### News Categories to Track

| Category | Positive Signals | Negative Signals |
|----------|------------------|------------------|
| **Earnings** | Beat estimates, raised guidance | Missed, lowered guidance |
| **Products** | New launches, strong demand | Delays, recalls, weak reception |
| **Competition** | Market share gains | New competitors, price wars |
| **Regulation** | Favorable rulings | Investigations, fines |
| **Macro** | Tailwinds from economy | Interest rate/recession risk |
| **Management** | Strong hires, confidence | Departures, insider selling |

### Phase 8: Industry Analysis

#### Industry Position Assessment

| Factor | Assessment | Evidence |
|--------|------------|----------|
| Market Position | Leader/Challenger/Niche | Market share data |
| Competitive Moat | Wide/Narrow/None | Barriers to entry |
| Industry Growth | Growing/Stable/Declining | TAM projections |
| Cyclicality | High/Medium/Low | Historical revenue patterns |
| Disruption Risk | High/Medium/Low | Technology threats |

Search: `[industry] market size growth forecast` and `[company] competitive position analysis`

## Output Format

### Executive Summary

```
## [Company Name] ([TICKER]) Investment Analysis

**Analysis Date**: [Date]
**Current Price**: $XX.XX
**Market Cap**: $XX.XB
**52-Week Range**: $XX.XX - $XX.XX

### Investment Thesis Summary
[2-3 sentence summary of key findings and recommendation]

### Key Metrics Snapshot
| Metric | Current | 5Y Avg | Forward | vs. History |
|--------|---------|--------|---------|-------------|
| P/E | XX.X | XX.X | XX.X | Premium/Discount |
| EV/EBITDA | XX.X | XX.X | XX.X | |
| Normalized P/E | XX.X | XX.X | - | |
```

### Detailed Sections

Provide each section with findings:

1. **Historical Valuation Analysis**: Ratios over 5-10 years with charts/tables
2. **Normalized Earnings**: One-time items identified, adjusted EPS, frequency of "unusual" items
3. **Forward Estimates**: Consensus estimates, revision trends, forecast methodology
4. **Valuation Assessment**: Current valuation vs. historical range and peers
5. **Sentiment Analysis**: Social media tone, retail vs institutional sentiment
6. **News & Industry Outlook**: Key developments, risks, opportunities
7. **Bull Case vs Bear Case**: Arguments for each side
8. **Risk Factors**: Top 3-5 risks to monitor

### Final Assessment

```
### Investment Conclusion

**Valuation**: [Undervalued / Fairly Valued / Overvalued]
**Sentiment**: [Bullish / Neutral / Bearish]
**News Trend**: [Positive / Mixed / Negative]

**Recommendation Considerations**:
- For [time horizon] investors with [risk tolerance] risk tolerance
- Key catalysts to watch: [list]
- Key risks to monitor: [list]

**Important Disclaimer**: This analysis is for informational purposes only and
does not constitute financial advice. Always conduct your own research and
consult with a qualified financial advisor before making investment decisions.
```

## Analysis Quality Checklist

Before concluding, verify:
- [ ] All historical ratios gathered (5-10 years where available)
- [ ] At least 3 one-time items identified and quantified (or confirmed none exist)
- [ ] Forward estimates for next 4+ quarters gathered
- [ ] Forecast methodology explained to user
- [ ] Social media sentiment from 2+ platforms summarized
- [ ] News from past 3 months reviewed
- [ ] Industry outlook assessed
- [ ] Both bull and bear cases presented
- [ ] Clear valuation assessment provided
- [ ] Appropriate disclaimers included

## Wrap Up

After presenting the analysis, offer:
1. **Deep Dive**: More detail on any section (ratios, specific one-time items, news)
2. **Peer Comparison**: Compare to specific competitors
3. **Scenario Analysis**: What-if analysis on key assumptions
4. **Monitoring Plan**: Key metrics and events to track going forward
