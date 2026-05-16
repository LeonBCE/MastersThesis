# Stance Model Validation Report

## Date: 2026-03-19

## Purpose

To validate whether the zero-shot NLI stance detection model tested in the thesis pipeline could classify the political direction of tweets well enough to be used in later modelling. Stance was treated as a candidate diagnostic variable rather than a final SEM input. If the model misclassifies individual tweets, aggregated state-level stance scores may not measure political direction in a defensible way.

## Background

### The full pipeline (from Topic_Modelling_4.ipynb)

1. **Data**: ~4.3 million English tweets from a large political Twitter dataset, filtered to those geolocated to US states
2. **Topic modelling**: BERTopic with 12 guided seed topic lists (2 per macro theme), producing topics that were mapped to 6 macro themes: Economy, Immigration, Foreign Policy, Social Issues, Democracy & Law, Domestic Policy
3. **Stance detection**: Each tweet scored using DeBERTa-v3-base-mnli-fever-anli (an NLI model) with two hypotheses:
   - Conservative: "This text expresses conservative right-wing political views."
   - Progressive: "This text expresses progressive left-wing political views."
   - Score = P(entailment|conservative hypothesis) - P(entailment|progressive hypothesis)
   - Range: [-1 progressive, +1 conservative]
4. **Aggregation**: Per state, stance = mean of all tweet-level stance scores within each topic
5. **Normalization**: Min-max to [0, 1] per column
6. **PLS-SEM**: Structural equation models predicting republican_pct from salience and stance composites

### Why validate?

In PLS-SEM modelling, Democracy_Law_stance dominated all models with weights >1.0 in the SENTIMENT composite and was the single largest predictor of Republican vote share. Removing it dropped R² by 33%. This raised concerns about whether the stance model was actually measuring political stance or something else (e.g., partisan identity, linguistic patterns, or noise that happens to correlate with voting at the aggregate level).

## Method

### Test design

85 diagnostic political tweets were generated with AI support to cover the main topic areas used in the pipeline. Each tweet was assigned an expected classification in the AI-assisted validation file:
- **Expected political lean**: conservative, progressive, or neutral/ambiguous
- **Expected emotion**: angry, positive, neutral, or sarcastic

The validation set was designed as a diagnostic stress test rather than as a full human-coded benchmark. Tweets were selected to cover:
- Clearly partisan statements from both sides
- Factual/neutral statements about political topics
- Sarcastic and ambiguous tweets
- All 6 macro themes and 12 sub-topics

### Models tested

1. **Stance model** (same as pipeline): MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
   - Zero-shot NLI entailment with conservative/progressive hypotheses
   - Output: stance score from -1 (progressive) to +1 (conservative)

2. **Sentiment model** (alternative): cardiffnlp/twitter-roberta-base-sentiment-latest
   - Purpose-built Twitter sentiment classifier
   - Output: score from -1 (negative emotion) to +1 (positive emotion)

### Code

See `test_stance_vs_sentiment.py` in this directory. Run with:
```
pip install transformers torch pandas scipy
python test_stance_vs_sentiment.py
```

## Results

### Stance model performance

#### Overall accuracy

| Metric | Value |
|--------|-------|
| Exact match (3-class) | 15/85 (17.6%) |
| Direction correct (non-neutral tweets only) | 27/72 (37.5%) |
| Expected accuracy from random guessing | 50% |

**The stance model did not classify political direction reliably in this diagnostic test.**

#### Mean stance by expected political lean

| Expected lean | Mean stance score | Expected direction |
|---------------|-------------------|-------------------|
| Conservative (n=36) | **-0.107** | Should be positive |
| Progressive (n=36) | **+0.023** | Should be negative |
| Neutral (n=13) | +0.041 | Should be ~0 |

**The scores are inverted.** Conservative tweets average negative (progressive), and progressive tweets average slightly positive (conservative).

#### Mean stance by topic

| Topic | Mean stance | Notes |
|-------|------------|-------|
| DemLaw_Courts | -0.244 | All tweets scored progressive regardless of content |
| SocialIssues_Race | -0.204 | Same problem |
| DomesticPolicy_Covid | -0.096 | Slight progressive bias |
| DemLaw_Felon | -0.075 | Near zero, undifferentiated |
| Immigration | -0.074 | Near zero |
| SocialIssues_Guns | -0.020 | Near zero |
| ForeignPolicy_China | +0.001 | Near zero (no signal) |
| ForeignPolicy_Israel | +0.004 | Near zero (no signal) |
| Economy | +0.059 | Slight conservative bias |
| SocialIssues_Abortion | +0.083 | Slight conservative bias |
| SocialIssues_Climate | +0.200 | Moderate conservative bias |
| DomesticPolicy_Healthcare | +0.197 | Moderate conservative bias |

Most topics produce scores clustered near zero (model is uncertain), with the few strong scores often directionally wrong.

#### Worst misclassifications — Democracy_Law

| Tweet | Expected | Stance score |
|-------|----------|-------------|
| "The Supreme Court is protecting our constitutional rights. Thank God for conservative justices." | conservative | **-0.971** (maximally progressive!) |
| "DEI is just racism with extra steps. Hire based on merit." | conservative | **-0.980** |
| "Build the wall. Finish it. Now." | conservative | **-0.757** |
| "Our judges are upholding law and order. This is what America needs." | conservative | **-0.507** |
| "Medicare for all is the only moral healthcare system." | progressive | **+0.990** (maximally conservative!) |
| "Electric vehicles are the future whether you like it or not." | progressive | **+0.982** |
| "Bidenomics is working. Unemployment is at record lows." | progressive | **+0.736** |

#### Interpretation

The NLI model did not recover political stance in a reliable way. It appears to be responding to:
- **Policy advocacy / assertiveness** → scored as "conservative" regardless of direction
- **Mentioning the word "conservative" or right-wing figures** → often scored as "progressive" (the model may interpret the topic as the stance)
- **Most tweets** → scored near zero (model is uncertain about political entailment)

### Sentiment model performance

#### Accuracy for emotional valence

| Expected emotion | Mean sentiment | Correct? |
|-----------------|---------------|----------|
| Angry (n=51) | **-0.762** | Yes |
| Neutral (n=19) | **-0.344** | Somewhat (slightly negative bias) |
| Positive (n=13) | **+0.622** | Yes |
| Sarcastic (n=2) | -0.019 | Yes (ambiguous, near zero) |

**The sentiment model correctly captures emotional tone.**

#### Sentiment by political lean

| Expected lean | Mean sentiment |
|---------------|---------------|
| Conservative | -0.474 |
| Neutral | -0.523 |
| Progressive | -0.374 |

Sentiment does NOT distinguish political direction — all groups are similarly negative (political tweets are generally angry). This confirms sentiment and stance measure different things.

### Correlation between stance and sentiment

| Level | Pearson r | p-value |
|-------|----------|---------|
| Overall | +0.150 | 0.171 |

**Low and non-significant correlation.** Stance and sentiment measure different constructs, so sentiment would capture a separate dimension of discourse rather than replacing stance directly.

## Implications for the thesis

### Why Democracy_Law_stance dominated the PLS-SEM models

The stance model did not work reliably at the individual tweet level, but Democracy_Law_stance still correlated strongly with Republican vote share at the state level (r = 0.533). This apparent contradiction is explained by:

1. **Democracy_Law is the largest topic category** (407,675 tweets). With thousands of tweets per state, even tiny systematic biases in the model compound into a consistent state-level signal.

2. **The signal likely reflects partisan composition, not stance.** Blue states (California, New York) produce more tweets and more aggressive criticism of Trump/courts, so the model's noisy scores average slightly more negative. Red states produce fewer, less critical tweets, so scores average less negative. The cross-state variation in stance is really measuring "how much does this state's Twitter population criticize Trump," which is a proxy for partisanship itself.

3. **This makes the model partially tautological.** States where Twitter users "sound more conservative" vote more conservative, but that is because the stance proxy is measuring the same underlying phenomenon (partisan lean) rather than an independent predictor.

### Sub-topic cancellation problem

When micro-topics are averaged into macro themes, opposing signals cancel:
- Social Issues: Guns stance (r=+0.469 with GOP) cancels with Climate stance (r=-0.110)
- Domestic Policy: Healthcare stance (mean=+0.04, conservative) cancels with Covid stance (mean=-0.03, progressive)

### Recommendations

1. **Report the limitation.** The stance model does not reliably classify individual tweets' political direction. State-level aggregations show correlation with voting but likely measure partisan composition rather than independent discourse characteristics.

2. **Consider sentiment as an alternative or complement.** Sentiment correctly captures emotional tone and is uncorrelated with stance, offering a different analytical dimension ("emotional temperature of discourse → voting" rather than "political lean → voting").

3. **Consider a purpose-built political stance classifier** rather than zero-shot NLI if stance detection is retained.

4. **Present models with and without Democracy_Law** to show robustness (M6 R²=0.554 vs M6b R²=0.373).

## Files

- `test_stance_vs_sentiment.py` — Test script (generates tweets, runs both models, produces analysis)
- `stance_vs_sentiment_test_results.csv` — Full per-tweet results with all scores
- `stance_validation_report.md` — This report
