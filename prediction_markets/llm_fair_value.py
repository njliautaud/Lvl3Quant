"""
LLM-Powered Fair Value Estimator for Prediction Markets

Uses Claude (Haiku for fast, Sonnet for deep) to estimate true probabilities
for Polymarket and Kalshi markets, enabling edge detection vs market prices.

Architecture:
  - Fast mode: single prompt → probability estimate (Haiku, <1s)
  - Deep mode: multi-step reasoning with evidence gathering (Sonnet, ~5s)
  - File-based cache with 4-hour TTL
  - Calibration log for tracking prediction accuracy (Brier score)
  - Batch scanning with category-based context grouping

Usage:
  python llm_fair_value.py --market "Will the Fed cut rates in March 2026?" --price 0.35
  python llm_fair_value.py --scan-polymarket --limit 20
  python llm_fair_value.py --calibrate          # Show calibration stats
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("llm_fair_value")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "llm_fair_value_cache.json"
CALIBRATION_FILE = DATA_DIR / "calibration_log.json"

# Cache TTL: 4 hours
CACHE_TTL_SECONDS = 4 * 60 * 60

# Model selection
MODEL_FAST = "claude-haiku-4-5"    # $0.25/M tokens — quick estimates
MODEL_DEEP = "claude-sonnet-4-5"   # $3/M tokens — deep analysis

# Rate limiting
MIN_INTERVAL_FAST = 0.3    # seconds between fast calls
MIN_INTERVAL_DEEP = 1.0    # seconds between deep calls

# Market efficiency prior: markets are ~90% efficient
# This shrinks our estimate toward the market price
MARKET_EFFICIENCY = 0.90


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _market_cache_key(question: str, end_date: str = "") -> str:
    """Deterministic cache key from market question."""
    raw = f"{question.strip().lower()}|{end_date}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=str)


def _cache_get(cache: dict, key: str) -> Optional[dict]:
    """Return cached entry if it exists and is not stale."""
    entry = cache.get(key)
    if not entry:
        return None
    age = time.time() - entry.get("cached_at", 0)
    if age > CACHE_TTL_SECONDS:
        return None
    return entry


def _load_calibration() -> dict:
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE) as f:
                return json.load(f)
        except Exception:
            return {"predictions": []}
    return {"predictions": []}


def _save_calibration(cal: dict) -> None:
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(cal, f, indent=2, default=str)


def _categorize_market(question: str) -> str:
    """Classify a market into a broad category for context grouping."""
    q = question.lower()
    if any(kw in q for kw in ["fed", "fomc", "interest rate", "rate cut", "rate hike", "powell", "dot plot"]):
        return "monetary_policy"
    if any(kw in q for kw in ["cpi", "inflation", "pce", "deflation"]):
        return "inflation"
    if any(kw in q for kw in ["gdp", "recession", "growth", "unemployment", "nfp", "jobs"]):
        return "macro_growth"
    if any(kw in q for kw in ["btc", "bitcoin", "ethereum", "eth", "crypto", "defi", "etf", "spot"]):
        return "crypto"
    if any(kw in q for kw in ["trump", "congress", "senate", "election", "vote", "president", "tariff", "sanction"]):
        return "political"
    if any(kw in q for kw in ["spx", "s&p", "nasdaq", "dow", "stock", "equity", "market crash", "bear market"]):
        return "equities"
    if any(kw in q for kw in ["oil", "gas", "gold", "silver", "commodity", "energy"]):
        return "commodities"
    if any(kw in q for kw in ["war", "conflict", "nato", "ukraine", "russia", "china", "taiwan"]):
        return "geopolitical"
    return "general"


# ---------------------------------------------------------------------------
# Category-specific context blocks injected into prompts
# ---------------------------------------------------------------------------

CATEGORY_CONTEXT = {
    "monetary_policy": """
MONETARY POLICY CONTEXT (March 2026):
- Fed funds rate: 4.25-4.50% (paused since Dec 2025 cut)
- Next FOMC: March 17-18, 2026. CME FedWatch: ~15% cut probability
- Fed dot plot (Dec 2025): 2 cuts projected for 2026
- Inflation: Core PCE ~2.6% (above 2% target)
- Labor market: Unemployment ~4.1%, non-farm payrolls averaging ~150K/month
- Fed guidance: Data-dependent, no urgency to cut. Risk = tariff inflation pass-through
- Base rate for 'rate cut at single meeting': ~20% when CME is at 15% (priced slightly low historically)
""",
    "inflation": """
INFLATION CONTEXT (March 2026):
- CPI (Jan 2026): +3.0% YoY headline, +3.3% core
- PCE (Dec 2025): +2.6% core — Fed's preferred measure, still above target
- Shelter inflation: Declining slowly (~5% YoY), biggest drag pulling toward 2%
- Tariff risk: New tariffs in 2025-2026 adding ~0.3-0.5% to goods inflation
- Base rate for 'CPI above X%': Use current trend + 0.1% per month uncertainty
""",
    "macro_growth": """
MACRO GROWTH CONTEXT (March 2026):
- GDP Q4 2025: +2.3% annualized
- Q1 2026 tracking: GDPNow ~+1.8% (uncertainty: ±1.5%)
- Unemployment: 4.1%, stable. Job openings declining but no sudden spike
- Recession probability (next 12 months): ~20% per major forecasters
- Base rate for 'US recession in 2026': ~20-25% (inverted yield curve recently normalizing)
""",
    "crypto": """
CRYPTO CONTEXT (March 2026):
- BTC: ~$85,000-95,000 range (volatile)
- ETH: ~$2,200-2,800 range
- Crypto ETF flows: Generally positive since Jan 2024 approvals
- Regulatory environment: Friendlier in 2025-2026 vs prior years
- Base rate for crypto events: HIGH volatility, wide confidence intervals. Markets often
  50-60% efficient on crypto directional questions (more LLM edge available)
- BTC halving was April 2024 — 12-18 month bull cycle historically follows
""",
    "political": """
POLITICAL CONTEXT (March 2026):
- US President: Donald Trump (second term, started Jan 2025)
- Congress: Republican majority (slim) in both chambers
- Key policies: Tariffs (25% on Canada/Mexico, 10% on China baseline + escalation)
- Political prediction markets: Polymarket historically well-calibrated on US politics
- Base rate: For binary political outcomes, markets are ~85% efficient. Expect 1-3% edge max.
""",
    "equities": """
EQUITIES CONTEXT (March 2026):
- SPX: ~5800-6000 range (volatile, tariff uncertainty)
- VIX: ~20-25 (elevated vs historical average of 19)
- Earnings: S&P 500 2025 EPS growth ~8-10%
- PE ratio: ~20-22x forward (slightly above historical average)
- Market crash base rate (>20% drawdown in 12 months): ~15%
""",
    "commodities": """
COMMODITIES CONTEXT (March 2026):
- Oil (WTI): ~$70-75/barrel (OPEC+ production cuts partially offsetting US production)
- Gold: ~$2,600-2,700/oz (safe haven demand elevated)
- Natural gas: ~$3.50 MCF (normalized from 2022 spike)
""",
    "geopolitical": """
GEOPOLITICAL CONTEXT (March 2026):
- Ukraine-Russia: Ceasefire negotiations ongoing, outcome uncertain
- Taiwan Strait: Tension elevated, invasion probability (~5% 12-month base rate)
- Middle East: Ongoing instability, oil supply risks
- Base rate: Geopolitical binary events are notoriously hard to price. Markets tend
  to underprice tail risks in geopolitical scenarios.
""",
    "general": """
GENERAL CONTEXT (March 2026):
- Use base rates from historical frequency of similar events
- Be conservative about edge — prediction markets aggregate information well
- Key date: March 3, 2026
""",
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

FAST_PROMPT_TEMPLATE = """You are a professional prediction market analyst. Estimate the TRUE probability for this market.

MARKET: {question}
END DATE: {end_date}
CURRENT MARKET PRICE: {market_price:.3f} (implies {market_pct:.1f}% probability)
CATEGORY: {category}

{category_context}

INSTRUCTIONS:
1. State the base rate briefly (1 sentence): how often has this type of event occurred historically?
2. Assess current evidence (1 sentence): what does current data suggest?
3. Apply market efficiency discount: prediction markets are ~90% efficient, so your edge is small.
4. Output your probability estimate as a single float between 0 and 1.

RESPONSE FORMAT (JSON only, no markdown):
{{
  "base_rate": "X% historically for this type of event because ...",
  "current_evidence": "Current data suggests X because ...",
  "market_efficiency_note": "Market at {market_price:.3f} is roughly fair / slightly over / slightly under because ...",
  "probability": 0.XX,
  "confidence": "low|medium|high",
  "edge_vs_market": "X% edge in direction Y"
}}"""


DEEP_PROMPT_TEMPLATE = """You are a professional prediction market analyst performing deep analysis.

MARKET: {question}
END DATE: {end_date}
CURRENT MARKET PRICE: {market_price:.3f} (implies {market_pct:.1f}% probability)
CATEGORY: {category}

{category_context}

{news_context}

Perform a thorough multi-step analysis:

STEP 1 - BASE RATE ANALYSIS:
What is the historical base rate for this exact type of event?
Cite specific numbers (e.g., "Fed cuts at 3 of last 8 meetings = 37.5%")

STEP 2 - CURRENT EVIDENCE:
What specific data points support or contradict this outcome?
- List 3-5 concrete data points with numbers where possible
- Weight their relevance

STEP 3 - BAYESIAN UPDATE:
Starting from the base rate, apply Bayesian reasoning:
- How does the current evidence shift the probability?
- What is your raw probability estimate before market efficiency?

STEP 4 - MARKET EFFICIENCY ADJUSTMENT:
Prediction markets aggregate information efficiently.
- Markets are ~90% efficient: final estimate = 0.10 * your_raw + 0.90 * market_price
- Apply this shrinkage
- Any reasons the market might be systematically biased here?

STEP 5 - FINAL ESTIMATE:
Provide final probability and confidence interval.

RESPONSE FORMAT (JSON only, no markdown):
{{
  "base_rate": "X% because ...",
  "evidence_for": ["point 1", "point 2", "point 3"],
  "evidence_against": ["point 1", "point 2"],
  "raw_probability": 0.XX,
  "market_efficiency_note": "...",
  "probability": 0.XX,
  "confidence_interval": [0.XX, 0.XX],
  "confidence": "low|medium|high",
  "edge_vs_market": "X% edge",
  "reasoning_summary": "2-3 sentence summary"
}}"""


# ---------------------------------------------------------------------------
# LLMFairValueEstimator
# ---------------------------------------------------------------------------

class LLMFairValueEstimator:
    """
    Estimates true probability for prediction markets using Claude.

    Fast mode: Haiku, single prompt, ~0.3s, cheap ($0.25/M tokens)
    Deep mode: Sonnet, multi-step with evidence, ~3-5s, more expensive ($3/M tokens)

    Cache: file-based, 4-hour TTL. Same question within 4h reuses result.
    Rate limiting: enforced per-model.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        self._has_anthropic = False

        # Try to initialize Anthropic client
        if self.api_key:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                self._has_anthropic = True
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("anthropic package not installed. Run: pip install anthropic")
            except Exception as e:
                logger.warning(f"Anthropic init failed: {e}")
        else:
            logger.warning(
                "ANTHROPIC_API_KEY not set. LLM fair value unavailable. "
                "Using simple heuristic fallback."
            )

        # Cache
        self._cache = _load_cache()

        # Rate limiting state
        self._last_fast_call = 0.0
        self._last_deep_call = 0.0

        # Calibration
        self._calibration = _load_calibration()

    # ------------------------------------------------------------------
    # Core estimation methods
    # ------------------------------------------------------------------

    def estimate_fast(self, market: dict) -> dict:
        """
        Quick single-prompt estimate using Haiku.

        Args:
            market: dict with at minimum 'question', optionally 'end_date', 'yes_price'/_yes_price

        Returns:
            dict: {probability, confidence, reasoning, model, cached, ...}
        """
        question = market.get("question", "")
        if not question:
            return self._fallback(market, "no question provided")

        end_date = (
            market.get("end_date")
            or market.get("endDate", "")
            or market.get("end_date_iso", "")
        )
        if end_date:
            end_date = str(end_date)[:10]

        market_price = (
            market.get("yes_price")
            or market.get("_yes_price")
            or market.get("market_price")
            or 0.5
        )

        cache_key = _market_cache_key(question, end_date)
        cached = _cache_get(self._cache, cache_key)
        if cached:
            logger.debug(f"Cache hit for: {question[:50]}")
            result = dict(cached)
            result["cached"] = True
            return result

        if not self._has_anthropic:
            return self._fallback(market, "no API key")

        category = _categorize_market(question)
        ctx = CATEGORY_CONTEXT.get(category, CATEGORY_CONTEXT["general"])

        prompt = FAST_PROMPT_TEMPLATE.format(
            question=question,
            end_date=end_date or "N/A",
            market_price=market_price,
            market_pct=market_price * 100,
            category=category.replace("_", " ").title(),
            category_context=ctx.strip(),
        )

        # Rate limiting
        self._rate_limit(fast=True)

        try:
            response = self.client.messages.create(
                model=MODEL_FAST,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw_text = response.content[0].text.strip()
            parsed = self._parse_json_response(raw_text)

            prob = float(parsed.get("probability", market_price))
            prob = max(0.01, min(0.99, prob))

            # Apply market efficiency shrinkage
            prob_shrunk = self._apply_efficiency_shrinkage(prob, market_price)

            result = {
                "probability": round(prob_shrunk, 4),
                "probability_raw": round(prob, 4),
                "market_price": market_price,
                "edge": round(prob_shrunk - market_price, 4),
                "confidence": parsed.get("confidence", "medium"),
                "base_rate": parsed.get("base_rate", ""),
                "current_evidence": parsed.get("current_evidence", ""),
                "market_efficiency_note": parsed.get("market_efficiency_note", ""),
                "category": category,
                "model": MODEL_FAST,
                "mode": "fast",
                "question": question,
                "end_date": end_date,
                "cached": False,
                "cached_at": time.time(),
                "estimated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Cache and return
            self._cache[cache_key] = result
            _save_cache(self._cache)

            # Log to calibration
            self._log_prediction(question, end_date, prob_shrunk, market_price, category)

            return result

        except Exception as e:
            logger.error(f"Fast estimate failed for '{question[:50]}': {e}")
            return self._fallback(market, str(e))

    def estimate_deep(self, market: dict, news_context: str = "") -> dict:
        """
        Deep multi-step analysis using Sonnet.

        Args:
            market: dict with 'question', 'end_date', 'yes_price'
            news_context: Optional string with recent news/data gathered externally

        Returns:
            dict: full analysis with probability, CI, reasoning chain
        """
        question = market.get("question", "")
        if not question:
            return self._fallback(market, "no question provided")

        end_date = (
            market.get("end_date")
            or market.get("endDate", "")
            or market.get("end_date_iso", "")
        )
        if end_date:
            end_date = str(end_date)[:10]

        market_price = (
            market.get("yes_price")
            or market.get("_yes_price")
            or market.get("market_price")
            or 0.5
        )

        # Deep mode uses its own cache namespace
        cache_key = "deep_" + _market_cache_key(question, end_date)
        cached = _cache_get(self._cache, cache_key)
        if cached:
            result = dict(cached)
            result["cached"] = True
            return result

        if not self._has_anthropic:
            return self._fallback(market, "no API key")

        category = _categorize_market(question)
        ctx = CATEGORY_CONTEXT.get(category, CATEGORY_CONTEXT["general"])

        news_section = ""
        if news_context:
            news_section = f"RECENT NEWS/DATA:\n{news_context}\n"

        prompt = DEEP_PROMPT_TEMPLATE.format(
            question=question,
            end_date=end_date or "N/A",
            market_price=market_price,
            market_pct=market_price * 100,
            category=category.replace("_", " ").title(),
            category_context=ctx.strip(),
            news_context=news_section,
        )

        self._rate_limit(fast=False)

        try:
            response = self.client.messages.create(
                model=MODEL_DEEP,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw_text = response.content[0].text.strip()
            parsed = self._parse_json_response(raw_text)

            prob = float(parsed.get("probability", market_price))
            prob = max(0.01, min(0.99, prob))

            # Deep mode: also extract raw before shrinkage
            raw_prob = float(parsed.get("raw_probability", prob))
            raw_prob = max(0.01, min(0.99, raw_prob))

            # Market efficiency shrinkage already included in the prompt
            # but apply a mild additional shrink to be conservative
            prob_shrunk = self._apply_efficiency_shrinkage(prob, market_price, alpha=0.05)

            ci = parsed.get("confidence_interval", [
                max(0.01, prob_shrunk - 0.05),
                min(0.99, prob_shrunk + 0.05),
            ])

            result = {
                "probability": round(prob_shrunk, 4),
                "probability_raw": round(raw_prob, 4),
                "probability_llm": round(prob, 4),
                "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
                "market_price": market_price,
                "edge": round(prob_shrunk - market_price, 4),
                "confidence": parsed.get("confidence", "medium"),
                "base_rate": parsed.get("base_rate", ""),
                "evidence_for": parsed.get("evidence_for", []),
                "evidence_against": parsed.get("evidence_against", []),
                "market_efficiency_note": parsed.get("market_efficiency_note", ""),
                "reasoning_summary": parsed.get("reasoning_summary", ""),
                "category": category,
                "model": MODEL_DEEP,
                "mode": "deep",
                "question": question,
                "end_date": end_date,
                "cached": False,
                "cached_at": time.time(),
                "estimated_at": datetime.now(timezone.utc).isoformat(),
            }

            self._cache[cache_key] = result
            _save_cache(self._cache)
            self._log_prediction(question, end_date, prob_shrunk, market_price, category)

            return result

        except Exception as e:
            logger.error(f"Deep estimate failed for '{question[:50]}': {e}")
            return self._fallback(market, str(e))

    # ------------------------------------------------------------------
    # Batch scanning
    # ------------------------------------------------------------------

    def estimate_batch(self, markets: list, mode: str = "fast") -> list:
        """
        Estimate fair values for multiple markets efficiently.
        Groups by category for shared context awareness.

        Args:
            markets: list of market dicts
            mode: "fast" (Haiku) or "deep" (Sonnet)

        Returns:
            list of result dicts (same order as input)
        """
        if not markets:
            return []

        # Group by category for logging
        by_category: dict[str, list] = {}
        for m in markets:
            cat = _categorize_market(m.get("question", ""))
            by_category.setdefault(cat, []).append(m)

        logger.info(
            f"Batch estimate: {len(markets)} markets across "
            f"{len(by_category)} categories. Mode: {mode}"
        )

        results = []
        for i, market in enumerate(markets):
            try:
                if mode == "deep":
                    result = self.estimate_deep(market)
                else:
                    result = self.estimate_fast(market)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Progress: {i+1}/{len(markets)}")

            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                results.append(self._fallback(market, str(e)))

        return results

    # ------------------------------------------------------------------
    # Integration with PolymarketScanner.scan()
    # ------------------------------------------------------------------

    def as_fair_value_fn(self, mode: str = "fast"):
        """
        Returns a callable suitable as the fair_value_fn parameter
        in PolymarketScanner.scan().

        Usage:
            estimator = LLMFairValueEstimator()
            fv_fn = estimator.as_fair_value_fn("fast")
            opps = scanner.scan(fair_value_fn=fv_fn)

        Returns:
            callable(market_dict) -> float
        """
        def _fn(market: dict) -> float:
            if mode == "deep":
                result = self.estimate_deep(market)
            else:
                result = self.estimate_fast(market)
            return result["probability"]
        return _fn

    # Convenience: call the estimator directly as a function
    def __call__(self, market: dict) -> float:
        """Allows using the estimator directly as fair_value_fn."""
        return self.estimate_fast(market)["probability"]

    # ------------------------------------------------------------------
    # Web search integration
    # ------------------------------------------------------------------

    def gather_news_context(self, question: str, max_chars: int = 1500) -> str:
        """
        Gather recent news/data for a market topic.
        Returns a text string to inject into the deep estimation prompt.

        This uses a simple web search approach. For production, integrate
        with a news API (NewsAPI, Perplexity, etc.).

        Currently returns structured search suggestions — replace with
        actual API calls when available.
        """
        category = _categorize_market(question)

        # Build search query keywords
        q_lower = question.lower()
        search_terms = []

        if "fed" in q_lower or "fomc" in q_lower or "rate" in q_lower:
            search_terms.append("Federal Reserve rate decision March 2026")
            search_terms.append("CME FedWatch probability March 2026")
        elif "cpi" in q_lower or "inflation" in q_lower:
            search_terms.append("CPI inflation data February 2026")
        elif "btc" in q_lower or "bitcoin" in q_lower:
            search_terms.append("Bitcoin price March 2026")
        elif "gdp" in q_lower or "recession" in q_lower:
            search_terms.append("US GDP forecast 2026 recession probability")
        else:
            # Extract key nouns from question
            words = [w for w in question.split() if len(w) > 4 and w[0].isupper()]
            search_terms.append(" ".join(words[:4]) + " 2026")

        # Attempt web search via requests (simple scraping)
        # In production: replace with actual news API
        context_lines = [f"[Search context for: {question[:60]}]"]
        context_lines.append(f"Search queries attempted: {search_terms}")
        context_lines.append(
            "Note: Real-time web search not connected. "
            "Using embedded category context instead. "
            "For live data, integrate with NewsAPI or Perplexity."
        )

        return "\n".join(context_lines)[:max_chars]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def log_resolution(self, question: str, end_date: str, outcome: float) -> dict:
        """
        Record the actual outcome for a previously predicted market.

        Args:
            question: Market question (used to find prediction)
            end_date: End date string
            outcome: 1.0 if YES resolved, 0.0 if NO resolved

        Returns:
            dict with updated Brier score stats
        """
        cal = _load_calibration()
        predictions = cal.get("predictions", [])

        # Find matching prediction
        cache_key = _market_cache_key(question, end_date)
        matched = [p for p in predictions if p.get("cache_key") == cache_key]

        if not matched:
            logger.warning(f"No prediction found for: {question[:50]}")
            return {"error": "no matching prediction"}

        pred = matched[-1]  # Most recent
        predicted_prob = pred.get("probability", 0.5)
        brier = (predicted_prob - outcome) ** 2

        # Update calibration record
        for p in predictions:
            if p.get("cache_key") == cache_key and p.get("resolved_at") is None:
                p["outcome"] = outcome
                p["brier_score"] = round(brier, 6)
                p["resolved_at"] = datetime.now(timezone.utc).isoformat()
                p["correct"] = (outcome > 0.5 and predicted_prob > 0.5) or \
                               (outcome < 0.5 and predicted_prob < 0.5)

        cal["predictions"] = predictions
        _save_calibration(cal)

        logger.info(
            f"Resolution logged: '{question[:40]}' → {outcome} "
            f"(predicted {predicted_prob:.3f}, Brier={brier:.4f})"
        )

        return self.get_calibration_stats()

    def get_calibration_stats(self) -> dict:
        """
        Compute calibration statistics from the log.

        Returns:
            dict with overall Brier score, accuracy, count, etc.
        """
        cal = _load_calibration()
        resolved = [
            p for p in cal.get("predictions", [])
            if p.get("outcome") is not None
        ]

        if not resolved:
            return {
                "resolved_count": 0,
                "brier_score": None,
                "accuracy": None,
                "message": "No resolved markets yet.",
            }

        brier_scores = [p["brier_score"] for p in resolved if "brier_score" in p]
        correct = [p for p in resolved if p.get("correct")]

        # Calibration by bucket (10 buckets of width 0.1)
        buckets = {}
        for p in resolved:
            prob = p.get("probability", 0.5)
            bucket = round(min(0.9, max(0.0, round(prob * 10) / 10)), 1)
            if bucket not in buckets:
                buckets[bucket] = {"sum_outcomes": 0, "count": 0}
            buckets[bucket]["sum_outcomes"] += p.get("outcome", 0)
            buckets[bucket]["count"] += 1

        calibration_table = {
            str(k): {
                "predicted": k,
                "actual": round(v["sum_outcomes"] / v["count"], 3) if v["count"] else None,
                "count": v["count"],
            }
            for k, v in sorted(buckets.items())
        }

        return {
            "resolved_count": len(resolved),
            "total_predictions": len(cal.get("predictions", [])),
            "mean_brier_score": round(sum(brier_scores) / len(brier_scores), 6) if brier_scores else None,
            "accuracy": round(len(correct) / len(resolved), 4) if resolved else None,
            "calibration_table": calibration_table,
            "best_brier_possible": 0.0,
            "random_brier": 0.25,
            "note": "Lower Brier score = better. Random = 0.25. Perfect = 0.0.",
        }

    # ------------------------------------------------------------------
    # Standalone scan of Polymarket
    # ------------------------------------------------------------------

    def scan_polymarket(self, limit: int = 20, mode: str = "fast",
                        min_edge: float = 0.02) -> list:
        """
        Full pipeline: fetch Polymarket markets, estimate fair values, find edge.

        Args:
            limit: Max markets to analyze
            mode: "fast" or "deep"
            min_edge: Minimum |edge| to include in results

        Returns:
            list of opportunity dicts sorted by |edge|
        """
        # Import here to avoid circular dependency
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from polymarket_client import PolymarketScanner
        except ImportError:
            logger.error("Could not import PolymarketScanner")
            return []

        scanner = PolymarketScanner()
        logger.info(f"Fetching top {limit} Polymarket financial markets...")

        markets = scanner.get_active_markets(limit=limit * 3)  # Over-fetch, then filter
        financial = scanner.filter_financial(markets)
        tradeable = scanner.filter_tradeable(financial)[:limit]

        logger.info(f"Analyzing {len(tradeable)} tradeable markets with LLM...")
        results = self.estimate_batch(tradeable, mode=mode)

        opportunities = []
        for market, est in zip(tradeable, results):
            prob = est["probability"]
            mkt_price = market["_yes_price"]
            edge = prob - mkt_price

            if abs(edge) < min_edge:
                continue

            # Calculate net edge after Polymarket 2% taker fee
            taker_fee = 0.02 * mkt_price * (1 - mkt_price)
            net_edge = abs(edge) - taker_fee

            if net_edge <= 0:
                continue

            opp = {
                "question": market.get("question", "")[:100],
                "end_date": (market.get("endDate") or market.get("end_date_iso") or "")[:10],
                "volume": market.get("_volume", 0),
                "market_price": round(mkt_price, 4),
                "fair_value": prob,
                "raw_probability": est.get("probability_raw", prob),
                "edge": round(edge, 4),
                "net_edge": round(net_edge, 4),
                "action": "BUY_YES" if edge > 0 else "BUY_NO",
                "confidence": est.get("confidence", "medium"),
                "category": est.get("category", "general"),
                "base_rate": est.get("base_rate", ""),
                "reasoning": est.get("reasoning_summary", est.get("current_evidence", "")),
                "cached": est.get("cached", False),
                "model": est.get("model", "heuristic"),
            }
            opportunities.append(opp)

        opportunities.sort(key=lambda x: x["net_edge"], reverse=True)
        logger.info(f"Found {len(opportunities)} opportunities with edge > {min_edge:.2f}")
        return opportunities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_efficiency_shrinkage(
        self, llm_prob: float, market_price: float, alpha: float = 0.10
    ) -> float:
        """
        Shrink LLM estimate toward market price.
        markets are ~90% efficient → our update is 10% of the gap.

        alpha: fraction of the update to keep (default 0.10 = 10% update)
        """
        return market_price + alpha * (llm_prob - market_price)

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code fences."""
        # Strip markdown fences if present
        if "```" in text:
            lines = text.split("\n")
            in_block = False
            json_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or not any(c in text for c in ["{", "}"]):
                    json_lines.append(line)
            text = "\n".join(json_lines).strip()

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from response: {text[:200]}")
            # Try to extract just the probability
            import re
            m = re.search(r'"probability"\s*:\s*([0-9.]+)', text)
            if m:
                return {"probability": float(m.group(1))}
            return {}

    def _fallback(self, market: dict, reason: str) -> dict:
        """
        Graceful degradation when LLM is unavailable.
        Returns market price as fair value (no edge expected).
        """
        mkt_price = (
            market.get("yes_price")
            or market.get("_yes_price")
            or market.get("market_price")
            or 0.5
        )
        question = market.get("question", "")
        logger.debug(f"Fallback for '{question[:40]}': {reason}")

        return {
            "probability": float(mkt_price),
            "probability_raw": float(mkt_price),
            "market_price": float(mkt_price),
            "edge": 0.0,
            "confidence": "low",
            "base_rate": "",
            "current_evidence": "",
            "category": _categorize_market(question),
            "model": "heuristic_fallback",
            "mode": "fallback",
            "question": question,
            "fallback_reason": reason,
            "cached": False,
            "estimated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _rate_limit(self, fast: bool = True) -> None:
        """Enforce minimum interval between API calls."""
        min_interval = MIN_INTERVAL_FAST if fast else MIN_INTERVAL_DEEP
        last = self._last_fast_call if fast else self._last_deep_call

        elapsed = time.time() - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        if fast:
            self._last_fast_call = time.time()
        else:
            self._last_deep_call = time.time()

    def _log_prediction(
        self, question: str, end_date: str,
        probability: float, market_price: float, category: str
    ) -> None:
        """Record prediction for later calibration."""
        cal = self._calibration
        cal.setdefault("predictions", [])

        cache_key = _market_cache_key(question, end_date)

        # Avoid duplicate entries for same market within same session
        existing = [p for p in cal["predictions"] if p.get("cache_key") == cache_key
                    and p.get("resolved_at") is None]
        if existing:
            return

        cal["predictions"].append({
            "cache_key": cache_key,
            "question": question[:120],
            "end_date": end_date,
            "probability": probability,
            "market_price": market_price,
            "edge": round(probability - market_price, 4),
            "category": category,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "outcome": None,
            "brier_score": None,
            "resolved_at": None,
        })

        # Keep at most 1000 predictions
        cal["predictions"] = cal["predictions"][-1000:]
        self._calibration = cal
        _save_calibration(cal)

    def clear_cache(self) -> int:
        """Clear all cached estimates. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache = {}
        _save_cache(self._cache)
        logger.info(f"Cleared {count} cache entries")
        return count

    def cache_stats(self) -> dict:
        """Return statistics about the current cache."""
        cache = _load_cache()
        now = time.time()
        valid = sum(1 for v in cache.values() if now - v.get("cached_at", 0) < CACHE_TTL_SECONDS)
        return {
            "total_entries": len(cache),
            "valid_entries": valid,
            "expired_entries": len(cache) - valid,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "cache_file": str(CACHE_FILE),
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_estimator: Optional[LLMFairValueEstimator] = None


def get_estimator() -> LLMFairValueEstimator:
    """Get or create the module-level estimator singleton."""
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = LLMFairValueEstimator()
    return _default_estimator


def estimate_market(market: dict, mode: str = "fast") -> dict:
    """Convenience wrapper — estimate a single market."""
    est = get_estimator()
    if mode == "deep":
        return est.estimate_deep(market)
    return est.estimate_fast(market)


def estimate_probability(question: str, market_price: float,
                         end_date: str = "", mode: str = "fast") -> float:
    """
    Simple interface: question + price → probability float.
    Suitable as fair_value_fn in PolymarketScanner.scan() via lambda.
    """
    market = {"question": question, "yes_price": market_price, "end_date": end_date}
    result = estimate_market(market, mode=mode)
    return result["probability"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_result(result: dict) -> str:
    """Format an estimation result for CLI display."""
    lines = [
        f"Market:    {result.get('question', '')[:80]}",
        f"End date:  {result.get('end_date', 'N/A')}",
        f"Category:  {result.get('category', 'N/A').replace('_', ' ').title()}",
        f"",
        f"Market price (implied prob): {result.get('market_price', 0):.3f} "
        f"({result.get('market_price', 0)*100:.1f}%)",
        f"LLM fair value:              {result.get('probability', 0):.3f} "
        f"({result.get('probability', 0)*100:.1f}%)",
        f"Edge vs market:              {result.get('edge', 0):+.4f} "
        f"({result.get('edge', 0)*100:+.2f}%)",
        f"Confidence:                  {result.get('confidence', 'N/A')}",
        f"Model:                       {result.get('model', 'N/A')}",
        f"Cached:                      {result.get('cached', False)}",
    ]

    if result.get("base_rate"):
        lines.extend(["", f"Base rate:  {result['base_rate']}"])
    if result.get("current_evidence"):
        lines.extend([f"Evidence:   {result['current_evidence']}"])
    if result.get("reasoning_summary"):
        lines.extend([f"Reasoning:  {result['reasoning_summary']}"])
    if result.get("evidence_for"):
        lines.extend(["", "Evidence FOR:"])
        for e in result["evidence_for"]:
            lines.append(f"  + {e}")
    if result.get("evidence_against"):
        lines.extend(["Evidence AGAINST:"])
        for e in result["evidence_against"]:
            lines.append(f"  - {e}")
    if result.get("confidence_interval"):
        ci = result["confidence_interval"]
        lines.append(f"\n90% confidence interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
    if result.get("market_efficiency_note"):
        lines.extend(["", f"Efficiency: {result['market_efficiency_note']}"])
    if result.get("fallback_reason"):
        lines.extend(["", f"[FALLBACK: {result['fallback_reason']}]"])

    return "\n".join(lines)


def _format_opportunities(opps: list) -> str:
    """Format a list of opportunities for CLI display."""
    if not opps:
        return "No opportunities found (all markets appear fairly priced)."

    lines = [
        f"{'='*70}",
        f"LLM FAIR VALUE SCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"{'='*70}",
        f"{len(opps)} opportunities found\n",
    ]

    for i, opp in enumerate(opps, 1):
        action_indicator = "^" if opp["action"] == "BUY_YES" else "v"
        cached_marker = " [cached]" if opp.get("cached") else ""
        lines.extend([
            f"{i}. {opp['question'][:72]}",
            f"   End: {opp.get('end_date', 'N/A')} | Vol: ${opp.get('volume', 0)/1e6:.1f}M | "
            f"Category: {opp.get('category', '').replace('_', ' ')}",
            f"   Market: {opp['market_price']:.3f} → FV: {opp['fair_value']:.3f} "
            f"| Edge: {opp['edge']:+.4f} → Net: {opp['net_edge']:+.4f} "
            f"| {action_indicator} {opp['action']}{cached_marker}",
        ])
        if opp.get("base_rate"):
            lines.append(f"   Base rate: {opp['base_rate'][:80]}")
        if opp.get("reasoning"):
            lines.append(f"   Reasoning: {opp['reasoning'][:80]}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered fair value estimator for prediction markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_fair_value.py --market "Will the Fed cut rates in March 2026?" --price 0.35
  python llm_fair_value.py --market "Will BTC reach 100k by June 2026?" --price 0.42 --deep
  python llm_fair_value.py --scan-polymarket --limit 20
  python llm_fair_value.py --scan-polymarket --limit 10 --deep --min-edge 0.03
  python llm_fair_value.py --calibrate
  python llm_fair_value.py --cache-stats
  python llm_fair_value.py --clear-cache
        """,
    )
    parser.add_argument("--market", type=str, help="Market question to estimate")
    parser.add_argument("--price", type=float, default=0.5, help="Current market YES price (0-1)")
    parser.add_argument("--end-date", type=str, default="", help="Market end date (YYYY-MM-DD)")
    parser.add_argument("--deep", action="store_true", help="Use deep analysis (Sonnet)")
    parser.add_argument("--scan-polymarket", action="store_true", help="Scan Polymarket for opportunities")
    parser.add_argument("--limit", type=int, default=20, help="Number of markets to scan")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge threshold")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--calibrate", action="store_true", help="Show calibration statistics")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--clear-cache", action="store_true", help="Clear estimate cache")
    parser.add_argument("--resolve", type=str, help="Resolve a market (format: 'question|outcome')")

    args = parser.parse_args()

    estimator = LLMFairValueEstimator()

    # --- Clear cache ---
    if args.clear_cache:
        n = estimator.clear_cache()
        print(f"Cleared {n} cached entries.")
        return

    # --- Cache stats ---
    if args.cache_stats:
        stats = estimator.cache_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Cache file: {stats['cache_file']}")
            print(f"Total entries: {stats['total_entries']}")
            print(f"Valid (not expired): {stats['valid_entries']}")
            print(f"Expired: {stats['expired_entries']}")
            print(f"TTL: {stats['ttl_seconds']//3600}h")
        return

    # --- Calibration stats ---
    if args.calibrate:
        stats = estimator.get_calibration_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("=== CALIBRATION STATISTICS ===")
            print(f"Total predictions logged: {stats.get('total_predictions', 0)}")
            print(f"Resolved markets: {stats.get('resolved_count', 0)}")
            if stats.get("mean_brier_score") is not None:
                print(f"Mean Brier score: {stats['mean_brier_score']:.6f} "
                      f"(random=0.25, perfect=0.0)")
                print(f"Accuracy (directionally correct): {stats.get('accuracy', 0):.1%}")
                if stats.get("calibration_table"):
                    print("\nCalibration table (predicted vs actual):")
                    print(f"  {'Predicted':>12} {'Actual':>8} {'Count':>6}")
                    for bucket, row in stats["calibration_table"].items():
                        actual = f"{row['actual']:.3f}" if row["actual"] is not None else "N/A"
                        print(f"  {float(bucket):>11.1%} {actual:>8} {row['count']:>6}")
            else:
                print(stats.get("message", "No resolved data yet."))
        return

    # --- Resolve a market ---
    if args.resolve:
        parts = args.resolve.split("|")
        if len(parts) != 2:
            print("Error: --resolve format is 'question|outcome' where outcome is 0 or 1")
            sys.exit(1)
        question, outcome_str = parts
        try:
            outcome = float(outcome_str)
        except ValueError:
            print(f"Error: outcome must be 0 or 1, got '{outcome_str}'")
            sys.exit(1)
        result = estimator.log_resolution(question.strip(), "", outcome)
        print(json.dumps(result, indent=2))
        return

    # --- Scan Polymarket ---
    if args.scan_polymarket:
        mode = "deep" if args.deep else "fast"
        opps = estimator.scan_polymarket(
            limit=args.limit,
            mode=mode,
            min_edge=args.min_edge,
        )
        if args.json:
            print(json.dumps(opps, indent=2, default=str))
        else:
            print(_format_opportunities(opps))
        return

    # --- Single market estimate ---
    if args.market:
        market = {
            "question": args.market,
            "yes_price": args.price,
            "end_date": args.end_date,
        }
        mode = "deep" if args.deep else "fast"

        if mode == "deep":
            # Gather news context first
            news = estimator.gather_news_context(args.market)
            result = estimator.estimate_deep(market, news_context=news)
        else:
            result = estimator.estimate_fast(market)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("=" * 70)
            print(_format_result(result))
            print("=" * 70)
        return

    # No args: show help
    parser.print_help()


if __name__ == "__main__":
    main()
