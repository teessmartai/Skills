# Sentiment and News Analysis Guide

How to research and synthesize social media sentiment, news, and industry outlook for stock analysis.

## Social Media Sentiment Analysis

### Platform-Specific Guidance

#### X (Twitter)

**Search Queries**:
```
$TICKER (cashtag search for stock discussion)
$TICKER sentiment
$TICKER analysis
$TICKER earnings
"[company name]" stock
```

**What to Look For**:
| Signal | Bullish | Bearish |
|--------|---------|---------|
| Volume | Increasing discussion | Declining interest |
| Tone | Enthusiasm, price targets up | Concern, warnings |
| News reaction | Positive spin on news | Negative interpretation |
| Influencers | Major accounts bullish | Big accounts warning |

**Key Accounts to Monitor**:
- Company's official account
- CEO/CFO accounts
- Industry analysts
- Financial journalists
- Popular retail investor accounts

#### Reddit

**Key Subreddits**:
| Subreddit | Focus | Reliability |
|-----------|-------|-------------|
| r/wallstreetbets | High-risk trades, meme stocks | Low (hype-driven) |
| r/stocks | General stock discussion | Medium |
| r/investing | Long-term investing | Medium-High |
| r/SecurityAnalysis | Deep fundamental analysis | High |
| r/options | Options strategies | Varies |
| r/ValueInvesting | Value-oriented analysis | High |

**Search Queries**:
```
site:reddit.com [ticker] DD
site:reddit.com [ticker] analysis
site:reddit.com "[company name]" stock
r/stocks [ticker]
```

**What to Look For**:
- **Due Diligence (DD) posts**: Detailed research with sources
- **Discussion threads**: General sentiment in comments
- **Contrary opinions**: Both bull and bear cases being made
- **Award count**: High awards = community engagement
- **Comment quality**: Research-backed vs speculation

**Red Flags**:
- All-caps enthusiasm ("TO THE MOON")
- No supporting data or sources
- Coordinated posting patterns
- New accounts pushing specific stocks

#### StockTwits

**Search**: `stocktwits.com/symbol/[TICKER]`

**Metrics to Track**:
| Metric | What It Shows |
|--------|---------------|
| Message volume | Interest level |
| Bull/Bear ratio | Sentiment balance |
| Sentiment trend | Direction of opinion |
| Watchlist additions | Growing interest |

**Interpretation**:
| Bull/Bear Ratio | Sentiment |
|-----------------|-----------|
| >70% bullish | Potentially overenthusiastic |
| 55-70% bullish | Generally positive |
| 45-55% | Neutral/mixed |
| <45% bullish | Generally negative |

#### Seeking Alpha

**Search**: `seekingalpha.com/symbol/[TICKER]`

**Content Types**:
| Type | Quality | Bias |
|------|---------|------|
| PRO articles | Higher (vetted) | Varies |
| Free articles | Medium | Often bullish |
| Short reports | Medium-High | Bearish |
| Comments | Low-Medium | Mixed |

**What to Look For**:
- Bull vs Bear article ratio
- Quality of bearish arguments
- Author track record on stock
- Response to recent earnings

#### YouTube

**Search**: `[ticker] stock analysis [current year]`

**Channels to Consider**:
- Financial news channels (CNBC, Bloomberg clips)
- Independent analysts with track records
- Educational finance channels

**Warning Signs**:
- Clickbait titles ("This Stock Will 10x!")
- Channels that are always bullish
- No disclosure of positions

### Sentiment Aggregation Framework

Create a summary table:

| Platform | Sentiment | Confidence | Key Themes | Volume Trend |
|----------|-----------|------------|------------|--------------|
| X/Twitter | Bullish | Medium | AI growth, earnings beat | Increasing |
| Reddit | Mixed | High | Valuation concerns | Stable |
| StockTwits | Bullish | Low | Price momentum | Increasing |
| Seeking Alpha | Bearish | Medium | Competition risk | Stable |

**Overall Sentiment Score**:
- Weight by confidence level
- Note divergences (retail vs professional)
- Track changes over time

## Short Interest Analysis

### Key Metrics

| Metric | Definition | What to Watch |
|--------|------------|---------------|
| Short Interest | Total shares sold short | Absolute level |
| Short % of Float | Short shares / floating shares | >10% is elevated |
| Days to Cover | Short interest / avg daily volume | >5 days = hard to unwind |
| Short Interest Ratio | Current vs previous reporting | Rising or falling |

### Search Queries
```
[ticker] short interest
[ticker] short squeeze potential
[ticker] days to cover
[ticker] short interest history chart
```

### Interpretation

| Short % of Float | Interpretation |
|------------------|----------------|
| <3% | Minimal shorting |
| 3-10% | Normal range |
| 10-20% | Elevated, watch for catalysts |
| >20% | High short interest, squeeze potential |

### Data Sources
- FINRA (bi-monthly reports)
- Nasdaq short interest data
- Yahoo Finance statistics page
- Finviz screener

## Institutional Activity

### What to Track

| Activity | Bullish Signal | Bearish Signal |
|----------|----------------|----------------|
| 13F filings | Major funds adding | Major funds reducing |
| Insider transactions | Executives buying | Executives selling |
| Analyst ratings | Upgrades | Downgrades |
| Price targets | Targets raised | Targets lowered |

### Search Queries
```
[ticker] institutional ownership
[ticker] 13F filings
[ticker] insider transactions
[ticker] analyst ratings
[ticker] price target changes
```

### Key Sources
- SEC EDGAR (13F filings)
- WhaleWisdom (institutional tracking)
- OpenInsider (insider transactions)
- TipRanks (analyst ratings aggregation)

## News Analysis

### News Categories

#### Earnings News

| Signal | Positive | Negative |
|--------|----------|----------|
| Results vs estimates | Beat | Miss |
| Guidance | Raised | Lowered |
| Call tone | Confident | Cautious |
| Surprise | Positive | Negative |

**Search**: `[ticker] earnings [quarter] [year] results`

#### Product/Service News

| Signal | Positive | Negative |
|--------|----------|----------|
| Launches | On time, well-received | Delayed, issues |
| Reviews | Positive feedback | Negative reviews |
| Market share | Gaining | Losing |
| Pricing | Ability to raise | Forced discounts |

**Search**: `[company name] new product launch [year]`

#### Competitive News

| Signal | Positive | Negative |
|--------|----------|----------|
| Market position | Strengthening | Weakening |
| New entrants | No major threats | Disruptive competitors |
| Pricing | Rational | Price war |
| Partnerships | Strategic wins | Lost relationships |

**Search**: `[company name] competition market share [year]`

#### Regulatory/Legal News

| Signal | Positive | Negative |
|--------|----------|----------|
| Litigation | Dismissed, won | New lawsuits, losses |
| Regulation | Favorable rulings | Investigations, fines |
| Compliance | Clean record | Violations |

**Search**: `[company name] lawsuit [year]` and `[company name] SEC investigation`

#### Management News

| Signal | Positive | Negative |
|--------|----------|----------|
| Leadership | Strong hires | Key departures |
| Insider activity | Buying | Heavy selling |
| Communication | Transparent | Defensive, evasive |

**Search**: `[company name] CEO CFO executive changes`

### News Timeline Analysis

Create a timeline of significant news:

| Date | Category | Headline | Impact | Stock Reaction |
|------|----------|----------|--------|----------------|
| Dec 2024 | Earnings | Q3 beat, raised guidance | Positive | +8% |
| Nov 2024 | Product | New product delays | Negative | -5% |
| Oct 2024 | Competition | Competitor bankruptcy | Positive | +3% |

### News Search Strategy

**Recent News (1-3 months)**:
```
[ticker] news
[company name] latest news [year]
[company name] [current month] [year]
```

**Earnings Coverage**:
```
[ticker] earnings analysis
[company name] quarterly results [quarter] [year]
[ticker] earnings call highlights
```

**Deep Dive**:
```
[company name] challenges
[company name] risks
[company name] bull case bear case
```

## Industry Analysis

### Industry Position Assessment

| Factor | Questions to Answer |
|--------|---------------------|
| Market Size | How big is the TAM? Growing or shrinking? |
| Growth Rate | What's the industry CAGR? |
| Concentration | Few players or fragmented? |
| Barriers | High or low barriers to entry? |
| Cyclicality | Economic sensitivity? |
| Regulation | Heavy or light regulatory burden? |
| Disruption | Technology threats? |

### Search Queries
```
[industry] market size forecast [year]
[industry] industry outlook [year]
[industry] growth rate trends
[industry] competitive landscape
[industry] disruption risks
```

### Porter's Five Forces Quick Assessment

| Force | Assessment | Search Query |
|-------|------------|--------------|
| Rivalry | High/Medium/Low | "[industry] competition intensity" |
| New entrants | Threat level | "[industry] barriers to entry" |
| Substitutes | Threat level | "[industry] alternative products" |
| Buyer power | High/Medium/Low | "[industry] customer concentration" |
| Supplier power | High/Medium/Low | "[industry] supplier dependency" |

### Industry News Search
```
[industry] industry news [year]
[sector] tailwinds headwinds
[industry] regulatory changes [year]
[industry] technology trends
```

## Synthesizing Findings

### Bull Case vs Bear Case Framework

| Category | Bull Case | Bear Case |
|----------|-----------|-----------|
| Valuation | [Arguments stock is cheap] | [Arguments stock is expensive] |
| Growth | [Growth catalysts] | [Growth headwinds] |
| Competition | [Competitive advantages] | [Competitive threats] |
| Sentiment | [Positive indicators] | [Negative indicators] |
| Macro | [Favorable conditions] | [Unfavorable conditions] |

### Risk Assessment Matrix

| Risk Category | Probability | Impact | Mitigation |
|---------------|-------------|--------|------------|
| Competition | High/Med/Low | Severe/Moderate/Minor | How company addresses |
| Regulation | H/M/L | S/M/Mi | |
| Execution | H/M/L | S/M/Mi | |
| Macro/Economic | H/M/L | S/M/Mi | |
| Technology | H/M/L | S/M/Mi | |

### Sentiment Summary Template

```
## Sentiment Summary for [TICKER]

### Social Media Sentiment: [Bullish/Neutral/Bearish]
- Twitter: [Summary]
- Reddit: [Summary]
- StockTwits: [Summary]

### Institutional Activity: [Bullish/Neutral/Bearish]
- Recent 13F changes: [Summary]
- Insider transactions: [Summary]
- Analyst consensus: [Summary]

### News Sentiment (Past 3 Months): [Positive/Mixed/Negative]
- Key positive developments: [List]
- Key negative developments: [List]
- Upcoming catalysts: [List]

### Industry Outlook: [Favorable/Neutral/Challenging]
- Growth drivers: [List]
- Headwinds: [List]
- Company positioning: [Assessment]

### Key Themes
1. [Most important theme]
2. [Second theme]
3. [Third theme]

### Sentiment Divergences
- [Note any disagreements between retail/institutional, bull/bear]
```

## News Sources by Reliability

| Tier | Sources | Notes |
|------|---------|-------|
| Tier 1 | SEC filings, company IR, Bloomberg, Reuters | Primary sources, most reliable |
| Tier 2 | WSJ, FT, NYT business, Barron's | Quality journalism, some analysis |
| Tier 3 | CNBC, Yahoo Finance, MarketWatch | Mixed news/opinion, good for sentiment |
| Tier 4 | Seeking Alpha, Motley Fool | Opinion-heavy, varying quality |
| Tier 5 | Social media, Reddit, StockTwits | Sentiment gauge, low factual reliability |

Always verify claims from lower tiers against higher-tier sources.
