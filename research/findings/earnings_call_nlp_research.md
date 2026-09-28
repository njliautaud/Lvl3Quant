I now have comprehensive data across all seven areas. Let me compile the findings.

Here is my structured research report on using LLMs/NLP to analyze earnings call transcripts for trading signals.

---

## 1. WHAT SPECIFICALLY WORKS

**Q&A section is more predictive than prepared remarks.** This is the single most robust finding across the literature. Multiple studies confirm it:

- **Van Mook (2025, Tilburg University thesis using GPT-4o)**: Q&A sentiment coefficient was 0.0227 vs. 0.0098 for prepared remarks when predicting CAR[0,+1]. The Q&A section produced higher adjusted R-squared (0.0540 vs 0.0507). This is because prepared remarks are scripted and optimized, while Q&A responses are more spontaneous and reveal genuine uncertainty.

- **Price, Doran, Peterson & Bliss (2012, Journal of Banking & Finance)**: Foundational paper with 2,800+ transcripts showing "conference call tone dominates earnings surprises over the 60 trading days following the call." The Q&A portion had incremental explanatory power for PEAD, concentrated in non-dividend-paying firms (higher cash flow uncertainty).

- **Brockman, Li & Price (CFA Institute FAJ)**: Institutional investors weigh analyst tone MORE heavily than management tone. Analysts do most of the talking and asking -- investors listen to analyst tone as an independent signal.

**What matters in the Q&A specifically:**
- Tone gap between prepared remarks (usually sunny) and Q&A answers (where management tone becomes more negative under questioning)
- Analyst question hostility/skepticism -- probing questions signal concerns
- Management evasions, deflections, and topic changes
- Response length and specificity (short vague answers = red flag)

**Management communication style is persistent and predictive:**
- **Dzielinski, Wagner & Zeckhauser (Harvard, 2017/2021)**: Introduced "CEO Clarity" measure. CEOs who frequently use words like "approximately," "probably," "maybe" are "fuzzy" speakers. Markets respond MORE strongly to earnings news from clear-speaking CEOs. When a firm appoints a clearer-speaking CEO, Tobin's Q increases.

---

## 2. ACADEMIC EVIDENCE (2024-2026 PAPERS)

**Key papers with quantitative results:**

| Paper | Year | Method | Key Result |
|-------|------|--------|------------|
| Luo (arXiv 2608.04200) | 2026 | FinBERT, multiple models across 5 datasets | FinBERT achieved 88.4% classification accuracy, BUT rank IC was only 0.0143 at 1-day horizon. **None of the 28 model-horizon tests remained significant after statistical corrections.** Critical finding: good sentiment classification does NOT equal profitable trading. |
| Kirtac & Germano (arXiv 2412.19245) | 2024 | OPT, BERT, FinBERT on 965K news articles (2010-2023) | OPT: 74.4% accuracy, Sharpe 3.05. BERT: 72.5%, Sharpe 2.11. FinBERT: 72.2%, Sharpe 2.07. Loughran-McDonald dictionary: 50.1%, Sharpe 1.23. |
| Siala et al. (arXiv 2602.00086) | 2026 | DeBERTa, FinBERT, ensemble | DeBERTa: 75% accuracy. Ensemble of 3 models: ~80% accuracy. |
| Van Mook (Tilburg thesis) | 2025 | GPT-4o sentiment on S&P 500 earnings calls | GPT-4o sentiment significantly predicts CAR[0,+1] even after controlling for earnings surprise. Q&A sentiment strongest predictor. Long-short portfolio generated statistically significant abnormal returns. |
| Kubica et al. (arXiv 2505.16090) | 2025 | Benchmarking Copilot, ChatGPT, Gemini on earnings transcripts | LLMs "often struggle with the nuanced, strategically ambiguous language found in earnings call transcripts." |
| Cao et al. "ECC Analyzer" (arXiv 2404.18470) | 2024 | LLM with RAG + multimodal (text + audio) | Hierarchical extraction from earnings calls outperforms traditional approaches for volatility prediction. |
| Zhu et al. (arXiv 2607.28496) | 2026 | LLaMA-3.1-70B structured extraction | Combining FinBERT sentiment with 6 structural features (event type, impact scope, time horizon, confidence) yielded F1=0.600 vs 0.576 for sentiment alone. 53.5% disagreement rate shows they capture orthogonal information. |
| Hadlock et al. (EMNLP FinNLP 2025) | 2025 | BART vs FinBERT for PEAD prediction | Encoder-decoder (BART) showed superior drift magnitude detection. Combining 3-day early market signal with text improved all models. |
| Koval et al. (ACL 2023) | 2023 | Long-document classification for earnings surprise prediction | Demonstrated "reasonable accuracy well above random chance" predicting future earnings surprises from conference call text alone. |

**Critical warning from Luo (2026):** The most rigorous recent study found that NONE of the sentiment-to-return relationships survived proper statistical correction (Newey-West inference + false discovery rate). This means many reported results in this space may be spurious. The signal exists but is weak and easily overstated.

---

## 3. DATA SOURCES

**Cheapest to most expensive:**

| Source | Price | Coverage | Format | Q&A Split |
|--------|-------|----------|--------|-----------|
| **Motley Fool** (scrape) | Free | Major US companies | HTML | Yes (manual parse) |
| **SEC EDGAR** (8-K filings) | Free | All public filers, but not all file transcripts | Text/HTML | Variable |
| **Seeking Alpha** | Free (basic), $239/yr (premium) | 4,500+ companies/quarter | HTML | Yes |
| **Financial Modeling Prep API** | Free tier available, $29/mo paid | Broad US coverage | JSON | Basic |
| **EarningsCall.biz** | Pay-per-use, Python/JS SDK | 9,000+ US companies | JSON | Yes (speaker segmented) |
| **Apify Earnings Call Scraper** | ~$13/1,000 transcripts | Multi-source (Motley Fool, SA, EDGAR) | Structured JSON | Yes (auto Q&A parse + metric extraction) |
| **Finnhub** | Free tier available | Broad | JSON | Limited |
| **Quartr** | Enterprise pricing | 14,500+ companies, 65 markets | Proprietary | Yes |
| **AlphaSense** | $2,000+/yr | Enterprise | Proprietary | Yes |

**Best MVP path:** Financial Modeling Prep API at $29/month gives you structured JSON transcripts with earnings surprise data, earnings calendar, and historical prices all from one API. The Apify scraper is excellent for batch collection with automatic Q&A parsing and forward-looking statement extraction at ~$0.013 per transcript.

---

## 4. PRACTICAL IMPLEMENTATION -- SIMPLEST MVP

**The proven GPT-4o prompt from Van Mook's thesis (actual prompt that produced publishable results):**

> "Act like a hedge fund analyst with expertise in financial NLP, investment analysis, and sentiment-driven quantitative modelling. You specialize in carefully reviewing corporate earnings call transcripts to extract forward-looking sentiment scores that reflect management's tone, strategic direction, and risk signals. These scores will later be used to support short-term trading and investment decisions."
>
> Instructions: Process each transcript by dividing it into [prepared remarks] and [Q&A], then score sentiment on each section separately.

**MVP Architecture:**
1. **Data:** FMP API to pull transcripts + earnings surprise data + stock prices
2. **Processing:** Claude API (or local LLM) to score each transcript on multiple dimensions
3. **Signal generation:** Combine scores with earnings surprise magnitude
4. **Execution:** Use as filter on existing strategies or standalone PEAD trade

**What to extract per transcript (based on practitioner best practices):**
- Sentiment score (-1 to +1) for prepared remarks and Q&A separately
- Hedge rate: count of uncertainty words ("uncertain," "volatile," "hope," "approximately," "may," "believe," "probably," "could") as percentage of total words
- Q&A friction score: number of non-answers, deflections, topic changes
- Guidance specificity: vague/widened/withdrawn vs. concrete numeric guidance
- Confidence delta: change in tone vs. prior quarter's call
- Red flags: excessive non-GAAP metrics, withdrawn guidance, leadership changes

**Cost estimate for Claude API:** A typical earnings call transcript is 5,000-8,000 words. At Claude Sonnet pricing, scoring ~500 transcripts per quarter would cost roughly $15-25 per earnings season. Very affordable for an MVP.

---

## 5. SPECIFIC FEATURES THAT PREDICT (BEYOND SIMPLE SENTIMENT)

**Uncertainty/hedging words (Harvard/Loughran-McDonald research):**
- Top uncertainty words in earnings calls: "approximately" (38% of uncertainty word count in presentations), "believe," "may" in prepared remarks. "Probably," "could," "believe" in Q&A answers.
- The top 25 uncertainty words (only 8.4% of the 297-word dictionary) account for 80% of total uncertainty word count. A very concentrated signal.
- Higher uncertainty word usage correlates with increased post-earnings volatility and greater analyst forecast dispersion.

**CEO clarity (Dzielinski et al.):**
- CEO clarity is a PERSONAL STYLE trait, not driven by business fundamentals. It persists across different business conditions.
- Clear-speaking CEOs get stronger market reactions to earnings news.
- Appointing a clearer CEO increases Tobin's Q and improves analyst recommendations.

**Tone manipulation detection:**
- Hopman (2021, Tilburg thesis): Found "novel evidence for managerial tone manipulation" -- managers actively avoid negative words from the Loughran-McDonald dictionary. FinBERT catches this better than dictionary methods because it understands context.
- Managers who manipulate tone in prepared remarks get caught in Q&A when forced to respond spontaneously.

**Structured features beyond sentiment (Zhu et al. 2026):**
- Event type, impact scope, temporal horizon, semantic confidence -- each contributes 14-21% of feature importance.
- Compressing everything to a single sentiment score loses substantial information. Multi-dimensional extraction is significantly better (F1 0.600 vs 0.576).

**Manager sentiment index (Jiang et al., Journal of Financial Economics 2019):**
- Aggregated managerial textual tone is a NEGATIVE predictor of future aggregate market returns (monthly R-squared of 9.75% in-sample, 8.38% out-of-sample). When managers are collectively optimistic, the market tends to underperform going forward.

---

## 6. INTEGRATION WITH YOUR EXISTING SYSTEM

Given your sector ETF dip-buying engines and options strategies, here is how earnings call NLP signals would best integrate:

**Option A -- FILTER on existing dip-buy signals (RECOMMENDED for MVP):**
- When your dip-buy engine identifies a sector ETF opportunity, check if any major holdings just reported earnings
- Use NLP sentiment to filter: only enter dip-buy trades where the underlying earnings tone supports the position (positive Q&A sentiment for longs, negative for shorts)
- This avoids trading into post-earnings drift that works against you

**Option B -- Standalone PEAD strategy:**
- During earnings season (4 windows per year), systematically score all transcripts
- Go long top-quintile sentiment surprise stocks, short bottom-quintile
- Hold 4-12 weeks per the drift literature
- Can use options (calls on positive surprise, puts on negative) for defined risk

**Option C -- Options strategy enhancement:**
- Pre-earnings: Use historical call tone analysis to predict which companies are likely to surprise (forward-looking language analysis)
- Post-earnings: Use transcript sentiment to determine whether to hold through the drift or close options positions early
- Wheel strategy modifier: Avoid selling puts on stocks where most recent earnings call showed high uncertainty/hedging language

**Best integration pattern:** Use as a MODIFIER on existing strategies, not standalone. The academic evidence says the signal exists but is weak and fragile when tested rigorously. It works best as one factor in a multi-factor framework, adding 2-5% improvement to existing strategies rather than carrying trades alone.

---

## 7. POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD)

**Is PEAD still alive?** Yes, but diminished.

**Current state (from Closelook Lab analysis, April 2026, and Wikipedia/academic sources):**
- PEAD has been documented since 1968 (Ball & Brown). Top vs bottom earnings surprise quintile historically produced ~13% annualized spread.
- Garfinkel, Hribar & Hsiao (2024): Long top SUE decile, short bottom SUE decile generates 5.1% risk-adjusted return over 3 months (~20% annualized).
- Historical range of PEAD returns: 8.76% to 43.08% annualized across studies.
- For LARGE-CAP stocks, some researchers argue PEAD has been non-existent since ~2006. For mid-caps and small-caps, it persists.
- Magnitude has declined as markets became more efficient and more arbitrageurs target the anomaly.

**Duration:** Classical PEAD persists for ~60 trading days post-announcement. The bulk of the immediate repricing happens in T+0 to T+3, but the drift continues for weeks.

**Real-world 2024 example (Closelook):**
- Rubrik (RBRK), December 2024 earnings beat: +32.8% CAR in T-1 to T+3, then continued to +15.0% CAR at T+63 (after giving back some gains). Mid-cap with restricted float amplified the drift.

**Can NLP predict which surprises will DRIFT vs. MEAN-REVERT?**

This is the key question, and the answer is: **Yes, partially.**

- **Price et al. (2012)**: Q&A tone has significant explanatory power for PEAD magnitude. Transcripts where Q&A sentiment was strongly positive showed larger and more sustained drift.
- **QuantPedia strategy (2022)**: Implemented NLP-enhanced PEAD using Brain Language Metrics on Earnings Call Transcripts. Best results: sort by sentiment surprise (current sentiment minus mean of prior 8 quarters), go long top tercile, short bottom tercile, 4-week holding period, 12 quarters of history. The NLP-enhanced version "clearly outperformed since 2019" vs vanilla PEAD.
- **Hadlock et al. (EMNLP 2025)**: Combining transcript text analysis with 3-day early price signal improved PEAD detection. BART encoder-decoder architecture outperformed encoder-only (FinBERT) for drift magnitude.

**When PEAD FAILS (important for risk management):**
- Large-cap, high-analyst-coverage stocks: too much efficiency, drift gets arbitraged quickly
- Extreme market stress: macro overwhelms individual stock signals
- When surprise was fully anticipated by options market (priced into IV)
- Multi-dimensional surprises where the beat/miss is ambiguous (e.g., revenue beat but margin miss)

**NLP signals that predict DRIFT continuation:**
- Strong Q&A sentiment aligned with the surprise direction
- Low hedging/uncertainty language in management answers
- Analyst tone aligned with surprise (analysts also bullish on beats)
- "Sentiment surprise" -- current call tone is significantly different from prior quarters (novel positive/negative information)
- Structural factors: low float, thin analyst coverage, mid-cap size -- these amplify PEAD because limits-to-arbitrage are stronger

---

## BOTTOM LINE RECOMMENDATIONS

1. **Start with FMP API ($29/mo) + Claude API** for transcript data + scoring. Total cost under $50/month.

2. **Focus on Q&A section sentiment**, not prepared remarks. This is where the alpha lives.

3. **Extract multi-dimensional features**, not just a single sentiment number. Hedge rate, confidence delta, guidance specificity, and friction score each add independent information.

4. **Use as a FILTER on existing strategies first**, not standalone. The signal is real but weak -- it improves a multi-factor system by a few percentage points, not a standalone trading system.

5. **For PEAD trades specifically:** Score transcripts, combine with earnings surprise magnitude, and focus on mid-cap stocks with restricted float/thin coverage where the anomaly is strongest. Hold 4-12 weeks.

6. **Be skeptical of high accuracy claims.** The most rigorous study (Luo 2026) found that NO sentiment model produced returns that survived proper statistical correction. Many reported results in this space suffer from data mining bias. Treat this as an edge enhancer, not an edge source.

7. **Key caution from Kubica et al. (2025):** LLMs "often struggle with the nuanced, strategically ambiguous language found in earnings call transcripts." Management is trained to use deliberately ambiguous language. Simple sentiment scoring will miss a lot. The more sophisticated your extraction (multi-dimensional, change-over-time, Q&A-focused), the better.