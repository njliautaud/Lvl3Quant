#!/usr/bin/env python3
"""
LLM Edge Scanner — Find prediction market mispricings using logical reasoning.

Edge Sources:
  1. Sports matchup logic — rankings, form, head-to-head don't match odds
  2. Contrarian reasoning — consensus diverges from logical analysis
  3. Anchoring bias — round numbers, recency bias in market pricing
  4. Category expertise — politics (polling models), crypto (on-chain data), macro (base rates)
  5. Multi-outcome inconsistency — probabilities in related markets don't add up logically
  6. Time decay mispricing — markets not properly discounting time-to-resolution
  7. News lag — LLM can process news implications faster than market adjusts

Architecture:
  - Scans ALL Polymarket markets (not just financial)
  - Groups by category for efficient batch analysis
  - Uses Haiku for first-pass screening ($0.25/M — cheap)
  - Escalates high-edge opportunities to Sonnet for deep analysis
  - Tracks calibration over time to measure actual edge

Usage:
    python llm_edge_scanner.py --scan                  # Full scan
    python llm_edge_scanner.py --scan --category sports # Category-specific
    python llm_edge_scanner.py --analyze "market question" --price 0.20
    python llm_edge_scanner.py --calibrate             # Show historical accuracy
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("llm_edge_scanner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from polymarket_client import PolymarketScanner, TAKER_FEE

SCAN_RESULTS_FILE = DATA_DIR / "llm_edge_scan.json"
OPPORTUNITIES_FILE = DATA_DIR / "llm_opportunities.json"

# ---------------------------------------------------------------------------
# Category detection — broader than financial-only
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "sports": [
        "nba", "nfl", "mlb", "nhl", "premier league", "champions league",
        "world cup", "super bowl", "playoffs", "championship", "finals",
        "game", "match", "win", "beat", "score", "mvp", "all-star",
        "team", "series", "tournament", "grand slam", "olympics", "ufc",
        "fight", "boxing", "f1", "formula", "race", "tennis", "golf",
        "masters", "wimbledon", "stanley cup", "world series",
    ],
    "politics": [
        "president", "election", "vote", "congress", "senate", "governor",
        "democrat", "republican", "poll", "primary", "nomination", "cabinet",
        "impeach", "legislation", "bill", "veto", "supreme court",
        "trump", "biden", "desantis", "newsom",
    ],
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "defi", "nft",
        "solana", "sol", "token", "blockchain", "halving", "etf",
        "binance", "coinbase",
    ],
    "macro": [
        "fed", "fomc", "interest rate", "inflation", "cpi", "gdp",
        "recession", "unemployment", "tariff", "treasury", "bond",
        "dollar", "gold", "oil", "commodity",
    ],
    "tech": [
        "ai", "openai", "google", "apple", "tesla", "nvidia", "microsoft",
        "spacex", "launch", "ipo", "acquisition", "merger",
    ],
    "entertainment": [
        "oscar", "grammy", "emmy", "movie", "film", "album", "song",
        "streaming", "netflix", "disney", "box office", "tv show",
    ],
    "science": [
        "climate", "nasa", "space", "vaccine", "fda", "drug", "trial",
        "discovery", "earthquake", "hurricane", "weather",
    ],
}


def categorize_market(question: str) -> str:
    """Classify a market question into a category."""
    q = question.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in q)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Edge-specific prompts — tailored reasoning per category
# ---------------------------------------------------------------------------

SPORTS_REASONING_PROMPT = """You are a sharp sports bettor analyzing prediction market odds.

MARKET: {question}
CURRENT ODDS: {yes_pct:.0f}% YES / {no_pct:.0f}% NO (price: {yes_price:.3f})
VOLUME: ${volume:,.0f}
END DATE: {end_date}

ANALYZE FOR MISPRICING:

1. **Rankings & Form**: If teams/players are mentioned, what are their current rankings?
   - A #2 ranked team vs #1 should NOT be 80/20 — more like 55/45 or 60/40 at most
   - Recent form (last 5-10 games) matters more than season record

2. **Head-to-Head**: Historical matchup data. Some teams consistently over/underperform vs specific opponents.

3. **Home/Away**: Home advantage is typically 3-5% in most sports. Is it priced in?

4. **Injuries/Suspensions**: Key player absences can shift odds 5-15%.

5. **Public Bias**: Public tends to overbet favorites, name brands, and recent winners.
   Contrarian value often exists on underdogs.

6. **Line Value**: Compare to traditional sportsbook odds if you know them.
   Polymarket odds are often less efficient than Vegas.

GIVE YOUR ANALYSIS:
- What should the TRUE probability be?
- Is the market overpricing or underpricing?
- How confident are you (1-10)?
- What's the expected edge in cents?

RESPOND IN JSON:
{{"probability": 0.XX, "edge_cents": X.X, "confidence": X, "direction": "BUY_YES|BUY_NO|NO_EDGE", "reasoning": "2-3 sentences"}}"""

POLITICS_REASONING_PROMPT = """You are a political analyst evaluating prediction market odds.

MARKET: {question}
CURRENT ODDS: {yes_pct:.0f}% YES / {no_pct:.0f}% NO (price: {yes_price:.3f})
VOLUME: ${volume:,.0f}
END DATE: {end_date}

ANALYZE FOR MISPRICING:

1. **Base Rate**: How often has this type of political event occurred historically?
   Use concrete numbers (e.g., "incumbents win 70% of gubernatorial races").

2. **Polling Data**: What do aggregated polls suggest? Apply historical polling error margins.
   - Polls typically have 3-4% error
   - Systemic bias: polls underestimated Republicans in 2016/2020 by ~2-3%

3. **Structural Factors**: Incumbency advantage, partisan lean, gerrymandering, turnout models.

4. **Known Biases in Prediction Markets**:
   - Markets tend to overreact to recent news/polls
   - Long-shot bias: unlikely events are slightly overpriced
   - Celebrity/name recognition bias: famous candidates get overpriced

5. **Timeline**: How far until resolution? More uncertainty = wider fair value range.

RESPOND IN JSON:
{{"probability": 0.XX, "edge_cents": X.X, "confidence": X, "direction": "BUY_YES|BUY_NO|NO_EDGE", "reasoning": "2-3 sentences"}}"""

GENERAL_REASONING_PROMPT = """You are an expert analyst evaluating prediction market odds for logical mispricings.

MARKET: {question}
CURRENT ODDS: {yes_pct:.0f}% YES / {no_pct:.0f}% NO (price: {yes_price:.3f})
CATEGORY: {category}
VOLUME: ${volume:,.0f}
END DATE: {end_date}

ANALYZE FOR MISPRICING:

1. **Base Rate**: What is the historical frequency of this type of event?
2. **Current Evidence**: What specific facts support or contradict the market price?
3. **Logical Consistency**: Does the price make logical sense given what we know?
4. **Market Biases**:
   - Recency bias (overweighting recent events)
   - Anchoring (stuck on round numbers or initial pricing)
   - Favorite-longshot bias (longshots overpriced, favorites underpriced)
   - Narrative bias (compelling stories get overpriced)
5. **Time Value**: How does time-to-resolution affect fair value?

KEY RULE: Markets are ~85-90% efficient. You need a STRONG logical reason to deviate.
Only flag edge > 3 cents (after 2% taker fee).

RESPOND IN JSON:
{{"probability": 0.XX, "edge_cents": X.X, "confidence": X, "direction": "BUY_YES|BUY_NO|NO_EDGE", "reasoning": "2-3 sentences"}}"""

CRYPTO_REASONING_PROMPT = """You are a crypto analyst evaluating prediction market odds.

MARKET: {question}
CURRENT ODDS: {yes_pct:.0f}% YES / {no_pct:.0f}% NO (price: {yes_price:.3f})
VOLUME: ${volume:,.0f}
END DATE: {end_date}

ANALYZE FOR MISPRICING:

1. **Price History & Volatility**: BTC daily vol ~3-5%. Use this to estimate probability of price targets.
   - For "BTC above $X by date Y": calculate required move as % and convert to probability
   - Use log-normal distribution: P(BTC > target) ≈ Φ(-(ln(target/current) - μt) / (σ√t))

2. **On-Chain Signals**: ETF flows, exchange reserves, miner behavior, whale accumulation.

3. **Macro Correlation**: Fed policy, dollar strength, risk-on/risk-off sentiment.

4. **Market Biases in Crypto**:
   - Crypto prediction markets attract retail speculators → more mispricings
   - Overreaction to short-term price moves
   - Narrative-driven pricing ("halving cycle" may already be priced in)
   - Round number anchoring ($100K, $50K, etc.)

5. **Regulatory Calendar**: SEC decisions, legislation, enforcement actions.

RESPOND IN JSON:
{{"probability": 0.XX, "edge_cents": X.X, "confidence": X, "direction": "BUY_YES|BUY_NO|NO_EDGE", "reasoning": "2-3 sentences"}}"""

MACRO_REASONING_PROMPT = """You are a macro economist evaluating prediction market odds on economic events.

MARKET: {question}
CURRENT ODDS: {yes_pct:.0f}% YES / {no_pct:.0f}% NO (price: {yes_price:.3f})
VOLUME: ${volume:,.0f}
END DATE: {end_date}

ANALYZE FOR MISPRICING:

1. **Data-Driven Base Rate**: Use actual economic data and forecasts.
   - For Fed decisions: CME FedWatch is the benchmark. Polymarket should match ±2%.
   - For inflation: Use current trend ± forecast uncertainty bands.
   - For GDP: Use GDPNow/Blue Chip consensus ± 1.5% standard error.
   - For recession: ~20% 12-month probability per professional forecasters.

2. **Market Efficiency**: Macro markets on Polymarket tend to be well-priced (85-95% efficient).
   Edge is smaller here — only flag if >3% divergence from data-driven estimate.

3. **Calendar Effects**: Events near FOMC/CPI release dates have compressed uncertainty.

4. **Known Biases**:
   - Markets overweight recent data points (e.g., one hot CPI print)
   - Tail risk underpricing (recession/financial crisis scenarios)
   - Tariff uncertainty systematically underpriced since 2025

RESPOND IN JSON:
{{"probability": 0.XX, "edge_cents": X.X, "confidence": X, "direction": "BUY_YES|BUY_NO|NO_EDGE", "reasoning": "2-3 sentences"}}"""


def get_prompt_for_category(category: str) -> str:
    """Get the appropriate reasoning prompt for a category."""
    prompts = {
        "sports": SPORTS_REASONING_PROMPT,
        "politics": POLITICS_REASONING_PROMPT,
        "crypto": CRYPTO_REASONING_PROMPT,
        "macro": MACRO_REASONING_PROMPT,
    }
    return prompts.get(category, GENERAL_REASONING_PROMPT)


# ---------------------------------------------------------------------------
# LLM Edge Scanner
# ---------------------------------------------------------------------------

class LLMEdgeScanner:
    """
    Scans Polymarket for mispricings using LLM reasoning.

    Two-pass architecture:
    1. Fast screening (Groq/Llama 3.3 70B) — cheap, fast, scan everything
    2. Deep analysis (Claude CLI or Groq) — more thorough for candidates with edge

    Supports multiple backends:
    - Groq (default): Llama 3.3 70B, fast and nearly free
    - Claude CLI: Uses local Claude Code installation
    - Anthropic API: Direct API calls (needs ANTHROPIC_API_KEY)
    """

    MODEL_GROQ = "llama-3.3-70b-versatile"
    MODEL_GROQ_FAST = "llama-3.1-8b-instant"
    MIN_EDGE_SCREEN = 0.03                  # 3 cents minimum edge from screening
    MIN_EDGE_DEEP = 0.02                    # 2 cents after deep (tighter, more confident)
    MIN_CONFIDENCE = 5                      # 1-10 scale, minimum to act
    MIN_VOLUME = 50_000                     # Minimum market volume (USD)

    def __init__(self, provider: str = "anthropic"):
        """
        Initialize with a provider: "anthropic" (default), "groq", or "claude_cli".
        Primary: Anthropic Haiku. Falls back to Groq on failure.
        """
        self.provider = provider
        self.scanner = PolymarketScanner()
        self._groq_key = os.environ.get("GROQ_API_KEY")
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        # Load keys from .env or config
        env_file = THIS_DIR.parents[1] / ".env"
        if env_file.exists():
            try:
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    v = v.strip().strip("'\"")
                    if k.strip() == "GROQ_API_KEY" and not self._groq_key:
                        self._groq_key = v
                    elif k.strip() == "ANTHROPIC_API_KEY" and not self._anthropic_key:
                        self._anthropic_key = v
            except Exception:
                pass

        # Auto-select best available provider
        if provider == "anthropic" and not self._anthropic_key:
            if self._groq_key:
                self.provider = "groq"
                logger.info("No ANTHROPIC_API_KEY, falling back to Groq")
            else:
                self.provider = "claude_cli"
                logger.info("No API keys, falling back to Claude CLI")
        elif provider == "groq" and not self._groq_key:
            if self._anthropic_key:
                self.provider = "anthropic"
                logger.info("No GROQ_API_KEY, falling back to Anthropic")
            else:
                self.provider = "claude_cli"
                logger.info("No GROQ_API_KEY, falling back to Claude CLI")

        logger.info(f"LLM Edge Scanner initialized with provider: {self.provider}")

    def _call_llm(self, prompt: str, deep: bool = False) -> dict:
        """Call LLM and parse JSON response. Primary: Anthropic Haiku. Fallback: Groq."""
        raw = ""
        try:
            if self.provider == "anthropic":
                try:
                    raw = self._call_anthropic(prompt, deep)
                except Exception as e:
                    logger.warning(f"Anthropic failed, trying Groq fallback: {e}")
                    if self._groq_key:
                        raw = self._call_groq(prompt, deep)
                    else:
                        raise
            elif self.provider == "groq":
                try:
                    raw = self._call_groq(prompt, deep)
                except Exception as e:
                    logger.warning(f"Groq failed, trying Anthropic fallback: {e}")
                    if self._anthropic_key:
                        raw = self._call_anthropic(prompt, deep)
                    else:
                        raise
            elif self.provider == "claude_cli":
                raw = self._call_claude_cli(prompt)
            else:
                return {"error": f"unknown provider: {self.provider}"}

            if not raw:
                return {"error": "empty_response"}

            # Parse JSON from response
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)

        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            logger.warning(f"Failed to parse LLM response: {raw[:200]}")
            return {"error": "parse_failed", "raw": raw[:500]}
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"error": str(e)}

    def _call_groq(self, prompt: str, deep: bool = False) -> str:
        """Call Groq API (OpenAI-compatible). Uses requests to avoid Cloudflare blocks."""
        import requests as _requests
        model = self.MODEL_GROQ if deep else self.MODEL_GROQ
        resp = _requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_claude_cli(self, prompt: str) -> str:
        """Call Claude via CLI (local installation)."""
        import subprocess
        import platform
        # On Windows, use claude.cmd; on Unix, use claude
        claude_cmd = "claude.cmd" if platform.system() == "Windows" else "claude"
        result = subprocess.run(
            [claude_cmd, "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=120,
            shell=(platform.system() == "Windows"),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {result.stderr[:200]}")
        return result.stdout.strip()

    def _call_anthropic(self, prompt: str, deep: bool = False) -> str:
        """Call Anthropic API directly."""
        import urllib.request
        model = "claude-sonnet-4-5-20250514" if deep else "claude-haiku-4-5-20251001"
        body = json.dumps({
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": self._anthropic_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data["content"][0]["text"]

    def screen_market(self, market: dict) -> dict:
        """
        Fast screening pass on a single market.
        Returns edge assessment.
        """
        question = market.get("question", "")
        prices = market.get("_parsed_prices", [])
        if not prices:
            from polymarket_client import PolymarketScanner as PS
            prices = PS()._parse_prices(market.get("outcomePrices", []))

        yes_price = prices[0] if prices else 0.5
        no_price = 1 - yes_price
        volume = float(market.get("volumeNum", 0) or 0)
        end_date = market.get("endDate", market.get("end_date", "N/A"))
        if end_date and len(str(end_date)) > 10:
            end_date = str(end_date)[:10]

        category = categorize_market(question)
        prompt_template = get_prompt_for_category(category)

        prompt = prompt_template.format(
            question=question,
            yes_price=yes_price,
            yes_pct=yes_price * 100,
            no_pct=no_price * 100,
            volume=volume,
            end_date=end_date or "N/A",
            category=category,
        )

        result = self._call_llm(prompt)

        if "error" in result:
            return {
                "question": question,
                "category": category,
                "market_price": yes_price,
                "error": result["error"],
                "has_edge": False,
            }

        prob = float(result.get("probability", yes_price))
        prob = max(0.01, min(0.99, prob))
        edge_raw = prob - yes_price

        # Calculate edge after fees
        # Taker fee on Polymarket: 2% * price * (1 - price)
        if edge_raw > 0:  # BUY YES
            fee = TAKER_FEE * yes_price * (1 - yes_price)
            edge_net = edge_raw - fee
        elif edge_raw < 0:  # BUY NO (equivalent to selling YES)
            fee = TAKER_FEE * no_price * (1 - no_price)
            edge_net = abs(edge_raw) - fee
        else:
            edge_net = 0
            fee = 0

        confidence = int(result.get("confidence", 5))
        direction = result.get("direction", "NO_EDGE")

        has_edge = (
            abs(edge_net) >= self.MIN_EDGE_SCREEN
            and confidence >= self.MIN_CONFIDENCE
            and direction != "NO_EDGE"
        )

        return {
            "question": question[:120],
            "category": category,
            "market_price": round(yes_price, 4),
            "llm_probability": round(prob, 4),
            "edge_raw": round(edge_raw, 4),
            "edge_net": round(edge_net, 4),
            "fee": round(fee, 4),
            "confidence": confidence,
            "direction": direction,
            "reasoning": result.get("reasoning", ""),
            "volume": volume,
            "end_date": str(end_date),
            "has_edge": has_edge,
            "model": self.provider,
            "market_slug": market.get("slug", market.get("market_slug", "")),
            "condition_id": market.get("conditionId", ""),
        }

    def deep_analyze(self, market: dict, screen_result: dict) -> dict:
        """
        Deep analysis pass on a screened candidate.
        Uses Sonnet for more thorough reasoning.
        """
        question = market.get("question", "")
        yes_price = screen_result["market_price"]
        no_price = 1 - yes_price
        volume = screen_result["volume"]
        end_date = screen_result["end_date"]
        category = screen_result["category"]

        # Enhanced prompt with screening context
        prompt_template = get_prompt_for_category(category)
        base_prompt = prompt_template.format(
            question=question,
            yes_price=yes_price,
            yes_pct=yes_price * 100,
            no_pct=no_price * 100,
            volume=volume,
            end_date=end_date,
            category=category,
        )

        enhanced = (
            f"{base_prompt}\n\n"
            f"SCREENING PASS FOUND: {screen_result['edge_raw']*100:.1f}% raw edge "
            f"(confidence {screen_result['confidence']}/10). "
            f"Screening reasoning: {screen_result['reasoning']}\n\n"
            f"Now do a DEEPER analysis. Challenge the screening result. "
            f"Consider what could go wrong. Be more conservative."
        )

        result = self._call_llm(enhanced, deep=True)

        if "error" in result:
            return {**screen_result, "deep_error": result["error"]}

        prob = float(result.get("probability", yes_price))
        prob = max(0.01, min(0.99, prob))
        edge_raw = prob - yes_price

        if edge_raw > 0:
            fee = TAKER_FEE * yes_price * (1 - yes_price)
            edge_net = edge_raw - fee
        elif edge_raw < 0:
            fee = TAKER_FEE * no_price * (1 - no_price)
            edge_net = abs(edge_raw) - fee
        else:
            edge_net = 0
            fee = 0

        confidence = int(result.get("confidence", 5))
        direction = result.get("direction", "NO_EDGE")

        return {
            **screen_result,
            "deep_probability": round(prob, 4),
            "deep_edge_raw": round(edge_raw, 4),
            "deep_edge_net": round(edge_net, 4),
            "deep_confidence": confidence,
            "deep_direction": direction,
            "deep_reasoning": result.get("reasoning", ""),
            "deep_model": self.provider,
            "confirmed_edge": (
                abs(edge_net) >= self.MIN_EDGE_DEEP
                and confidence >= self.MIN_CONFIDENCE
                and direction != "NO_EDGE"
            ),
        }

    def full_scan(self, limit: int = 200, category_filter: str = None,
                  deep_threshold: int = 10) -> dict:
        """
        Full two-pass scan of Polymarket.

        Args:
            limit: Max markets to fetch
            category_filter: Only scan this category (e.g., "sports")
            deep_threshold: Max number of candidates to run deep analysis on

        Returns:
            dict with scan results, opportunities, and stats
        """
        logger.info(f"Starting full scan (limit={limit}, category={category_filter})")
        scan_start = time.time()

        # Fetch all active markets
        markets = self.scanner.get_active_markets(limit=limit)
        logger.info(f"Fetched {len(markets)} active markets")

        # Parse prices and filter tradeable
        tradeable = []
        for m in markets:
            prices = self.scanner._parse_prices(m.get("outcomePrices", []))
            if not prices:
                continue
            yes_price = prices[0]
            vol = float(m.get("volumeNum", 0) or 0)
            if 0.05 <= yes_price <= 0.95 and vol >= self.MIN_VOLUME:
                m["_parsed_prices"] = prices
                m["_yes_price"] = yes_price

                cat = categorize_market(m.get("question", ""))
                if category_filter and cat != category_filter:
                    continue
                tradeable.append(m)

        logger.info(f"Tradeable markets: {len(tradeable)}")

        has_llm = (
            (self.provider == "groq" and self._groq_key)
            or (self.provider == "anthropic" and self._anthropic_key)
            or self.provider == "claude_cli"
        )
        if not has_llm:
            return {
                "error": "No ANTHROPIC_API_KEY set. Cannot run LLM screening.",
                "tradeable_count": len(tradeable),
                "markets_sample": [
                    {
                        "question": m.get("question", "")[:100],
                        "price": m.get("_yes_price", 0),
                        "volume": float(m.get("volumeNum", 0) or 0),
                        "category": categorize_market(m.get("question", "")),
                    }
                    for m in tradeable[:20]
                ],
            }

        # Pass 1: Fast screening
        screened = []
        candidates = []
        for i, m in enumerate(tradeable):
            result = self.screen_market(m)
            screened.append(result)

            if result.get("has_edge"):
                candidates.append((m, result))
                logger.info(
                    f"  CANDIDATE: {result['question'][:60]} | "
                    f"edge={result['edge_net']:.3f} | conf={result['confidence']} | "
                    f"{result['direction']}"
                )

            if (i + 1) % 20 == 0:
                logger.info(f"  Screened {i+1}/{len(tradeable)}, candidates: {len(candidates)}")

            time.sleep(0.3)  # Rate limiting

        logger.info(f"Screening complete: {len(candidates)} candidates from {len(tradeable)} markets")

        # Pass 2: Deep analysis on top candidates
        candidates.sort(key=lambda x: abs(x[1]["edge_net"]), reverse=True)
        deep_results = []

        for m, screen in candidates[:deep_threshold]:
            logger.info(f"Deep analyzing: {screen['question'][:60]}")
            deep = self.deep_analyze(m, screen)
            deep_results.append(deep)
            time.sleep(1.0)  # Rate limiting for Sonnet

        # Filter confirmed opportunities
        opportunities = [d for d in deep_results if d.get("confirmed_edge")]

        # Category breakdown
        cat_counts = {}
        for s in screened:
            cat = s.get("category", "general")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        scan_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - scan_start, 1),
            "total_markets": len(markets),
            "tradeable": len(tradeable),
            "candidates_screened": len(candidates),
            "deep_analyzed": len(deep_results),
            "confirmed_opportunities": len(opportunities),
            "category_breakdown": cat_counts,
            "opportunities": opportunities,
            "all_candidates": [c[1] for c in candidates],
            "screening_summary": {
                "avg_edge": round(
                    sum(abs(s.get("edge_net", 0)) for s in screened if s.get("has_edge"))
                    / max(1, len(candidates)), 4
                ),
                "categories_with_edge": list(set(
                    s["category"] for s in screened if s.get("has_edge")
                )),
            },
        }

        # Save results
        with open(SCAN_RESULTS_FILE, "w") as f:
            json.dump(scan_result, f, indent=2, default=str)

        if opportunities:
            with open(OPPORTUNITIES_FILE, "w") as f:
                json.dump(opportunities, f, indent=2, default=str)

        return scan_result

    def format_discord_report(self, scan: dict) -> str:
        """Format scan results for Discord."""
        lines = [
            f"**LLM EDGE SCAN** ({datetime.now().strftime('%H:%M ET')})",
            f"Markets: {scan.get('total_markets', 0)} total, "
            f"{scan.get('tradeable', 0)} tradeable",
            f"Candidates: {scan.get('candidates_screened', 0)} -> "
            f"**{scan.get('confirmed_opportunities', 0)} confirmed**",
            f"Duration: {scan.get('duration_seconds', 0):.0f}s",
        ]

        cats = scan.get("category_breakdown", {})
        if cats:
            lines.append(f"Categories: {', '.join(f'{k}:{v}' for k, v in sorted(cats.items()))}")

        opps = scan.get("opportunities", [])
        if opps:
            lines.append("\n**Opportunities:**")
            for opp in opps[:10]:
                edge = opp.get("deep_edge_net", opp.get("edge_net", 0))
                conf = opp.get("deep_confidence", opp.get("confidence", 0))
                direction = opp.get("deep_direction", opp.get("direction", "?"))
                lines.append(
                    f"  • {opp['question'][:55]}\n"
                    f"    {direction} @ {opp['market_price']:.2f} | "
                    f"edge: {edge:.3f} | conf: {conf}/10"
                )
                reasoning = opp.get("deep_reasoning", opp.get("reasoning", ""))
                if reasoning:
                    lines.append(f"    _{reasoning[:80]}_")
        else:
            all_cands = scan.get("all_candidates", [])
            if all_cands:
                lines.append("\n**Top Candidates (not confirmed after deep analysis):**")
                for c in all_cands[:5]:
                    lines.append(
                        f"  • {c['question'][:55]} | "
                        f"edge: {c['edge_net']:.3f} | {c['direction']}"
                    )
            else:
                lines.append("\nNo opportunities found. Markets appear fairly priced.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dry-run scan (no API key needed — shows what WOULD be analyzed)
# ---------------------------------------------------------------------------

def dry_run_scan(limit: int = 100, category_filter: str = None):
    """
    Scan Polymarket without LLM calls — just categorize and show what we'd analyze.
    Useful for testing without API key.
    """
    scanner = PolymarketScanner()
    markets = scanner.get_active_markets(limit=limit)

    tradeable = []
    for m in markets:
        prices = scanner._parse_prices(m.get("outcomePrices", []))
        if not prices:
            continue
        yes_price = prices[0]
        vol = float(m.get("volumeNum", 0) or 0)
        if 0.05 <= yes_price <= 0.95 and vol >= 50_000:
            cat = categorize_market(m.get("question", ""))
            if category_filter and cat != category_filter:
                continue
            tradeable.append({
                "question": m.get("question", "")[:100],
                "price": round(yes_price, 3),
                "volume": vol,
                "category": cat,
                "slug": m.get("slug", ""),
                "end_date": str(m.get("endDate", ""))[:10],
            })

    # Group by category
    by_cat = {}
    for t in tradeable:
        by_cat.setdefault(t["category"], []).append(t)

    print(f"\n{'='*70}")
    print(f"  POLYMARKET DRY-RUN SCAN — {len(tradeable)} tradeable markets")
    print(f"{'='*70}")

    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        print(f"\n  [{cat.upper()}] — {len(items)} markets")
        # Sort by most "interesting" (furthest from 50%)
        items.sort(key=lambda x: abs(x["price"] - 0.5))
        for item in items[:8]:
            print(
                f"    {item['price']:.0%} | ${item['volume']:>12,.0f} | "
                f"{item['question'][:60]}"
            )

    print(f"\n{'='*70}")
    print(f"  Total: {len(tradeable)} markets across {len(by_cat)} categories")
    print(f"  Set ANTHROPIC_API_KEY to enable LLM screening")
    print(f"{'='*70}\n")

    return {"tradeable": tradeable, "by_category": by_cat}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Edge Scanner for Prediction Markets")
    parser.add_argument("--scan", action="store_true", help="Full LLM scan")
    parser.add_argument("--dry-run", action="store_true", help="Scan without LLM (show markets)")
    parser.add_argument("--category", type=str, default=None,
                       help="Filter by category (sports, politics, crypto, macro, etc.)")
    parser.add_argument("--limit", type=int, default=200, help="Max markets to fetch")
    parser.add_argument("--analyze", type=str, default=None,
                       help="Analyze a specific market question")
    parser.add_argument("--price", type=float, default=None,
                       help="Current market price (for --analyze)")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_scan(limit=args.limit, category_filter=args.category)
        return

    if args.analyze:
        scanner = LLMEdgeScanner()
        market = {
            "question": args.analyze,
            "outcomePrices": json.dumps([args.price or 0.5, 1 - (args.price or 0.5)]),
            "volumeNum": 100000,
            "endDate": "2026-12-31",
        }
        result = scanner.screen_market(market)
        print(json.dumps(result, indent=2))
        return

    if args.scan:
        scanner = LLMEdgeScanner()
        result = scanner.full_scan(limit=args.limit, category_filter=args.category)
        print(scanner.format_discord_report(result))
        return

    # Default: dry run
    dry_run_scan(limit=args.limit, category_filter=args.category)


if __name__ == "__main__":
    main()
