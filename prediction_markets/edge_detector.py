#!/usr/bin/env python3
"""
Edge Detection Engine for Polymarket prediction markets.

Combines multiple edge-detection strategies:
  1. Momentum — consistent price movement in one direction (3+ data points)
  2. Mean Reversion — price spiked and is reverting to mean
  3. Volume Anomaly — sudden volume spike on a market
  4. LLM Mispricing — compare market odds to LLM-estimated probability (Groq)

Each opportunity is scored 0-100. Only edges scored 70+ are actionable.

Usage:
    python -m trading_agents.prediction_markets.edge_detector --scan
    python -m trading_agents.prediction_markets.edge_detector --scan --min-score 80
    python -m trading_agents.prediction_markets.edge_detector --analyze SLUG
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("edge_detector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PRICE_HISTORY_DIR = DATA_DIR / "price_history"
PRICE_HISTORY_DIR.mkdir(exist_ok=True)

EDGE_RESULTS_FILE = DATA_DIR / "edge_detector_results.json"
EDGE_LOG_FILE = DATA_DIR / "edge_detector.jsonl"

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from polymarket_client import PolymarketScanner, TAKER_FEE

# Load .env for API keys
ENV_FILE = THIS_DIR.parents[1] / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip("'\"")
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v


# ---------------------------------------------------------------------------
# Price History Helpers
# ---------------------------------------------------------------------------

def load_price_history(slug: str, hours: float = 48) -> list:
    """Load price history for a market within the last N hours."""
    history_file = PRICE_HISTORY_DIR / f"{slug[:80]}.jsonl"
    if not history_file.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    history = []
    with open(history_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ts = datetime.fromisoformat(entry["t"])
                if ts >= cutoff:
                    history.append(entry)
            except Exception:
                continue
    return history


def record_price(slug: str, yes_price: float, volume: float = 0):
    """Record a price snapshot."""
    history_file = PRICE_HISTORY_DIR / f"{slug[:80]}.jsonl"
    entry = {
        "t": datetime.now(timezone.utc).isoformat(),
        "p": round(yes_price, 4),
        "v": round(volume, 0),
    }
    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Edge Strategy 1: Momentum
# ---------------------------------------------------------------------------

def detect_momentum(history: list, current_price: float) -> dict:
    """
    Detect momentum: price moving consistently in one direction.
    Requires 3+ consecutive data points in the same direction.

    Returns:
        dict with signal, direction, score (0-100), and details
    """
    if len(history) < 5:
        return {"signal": "INSUFFICIENT_DATA", "score": 0}

    prices = [h["p"] for h in history]
    recent = prices[-min(20, len(prices)):]
    changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

    if not changes:
        return {"signal": "NO_DATA", "score": 0}

    up_count = sum(1 for c in changes if c > 0.001)
    down_count = sum(1 for c in changes if c < -0.001)
    total_move = recent[-1] - recent[0]

    # Count consecutive moves in same direction
    max_consecutive_up = 0
    max_consecutive_down = 0
    current_streak_up = 0
    current_streak_down = 0

    for c in changes:
        if c > 0.001:
            current_streak_up += 1
            current_streak_down = 0
            max_consecutive_up = max(max_consecutive_up, current_streak_up)
        elif c < -0.001:
            current_streak_down += 1
            current_streak_up = 0
            max_consecutive_down = max(max_consecutive_down, current_streak_down)
        else:
            current_streak_up = 0
            current_streak_down = 0

    # Score based on consistency + magnitude + consecutive moves
    consistency = max(up_count, down_count) / len(changes) if changes else 0
    magnitude = abs(total_move)

    # Need at least 3 consecutive moves and >2% total move
    if max_consecutive_up >= 3 and total_move > 0.02:
        # Score: 40 for direction consistency, 30 for magnitude, 30 for streak
        score = int(
            min(40, consistency * 50)
            + min(30, magnitude * 300)
            + min(30, max_consecutive_up * 6)
        )
        return {
            "signal": "MOMENTUM_UP",
            "direction": "BUY_YES",
            "score": min(100, score),
            "total_move": round(total_move, 4),
            "consistency": round(consistency, 2),
            "consecutive": max_consecutive_up,
            "edge_estimate": round(total_move * 0.3, 4),
            "trend_strength": round(consistency * magnitude * 100, 2),
        }

    elif max_consecutive_down >= 3 and total_move < -0.02:
        score = int(
            min(40, consistency * 50)
            + min(30, magnitude * 300)
            + min(30, max_consecutive_down * 6)
        )
        return {
            "signal": "MOMENTUM_DOWN",
            "direction": "BUY_NO",
            "score": min(100, score),
            "total_move": round(total_move, 4),
            "consistency": round(consistency, 2),
            "consecutive": max_consecutive_down,
            "edge_estimate": round(abs(total_move) * 0.3, 4),
            "trend_strength": round(consistency * magnitude * 100, 2),
        }

    return {
        "signal": "NO_MOMENTUM",
        "score": 0,
        "total_move": round(total_move, 4),
        "consistency": round(consistency, 2),
    }


# ---------------------------------------------------------------------------
# Edge Strategy 2: Mean Reversion
# ---------------------------------------------------------------------------

def detect_mean_reversion(history: list, current_price: float) -> dict:
    """
    Detect mean reversion: price spiked from its average and should revert.

    Score based on z-score magnitude and historical reversion rate.
    """
    if len(history) < 10:
        return {"signal": "INSUFFICIENT_DATA", "score": 0}

    prices = [h["p"] for h in history]
    avg = sum(prices) / len(prices)
    std = (sum((p - avg) ** 2 for p in prices) / len(prices)) ** 0.5

    if std < 0.005:
        return {"signal": "LOW_VOL", "score": 0, "avg": round(avg, 4), "std": round(std, 4)}

    z_score = (current_price - avg) / std if std > 0 else 0
    deviation_pct = abs(current_price - avg) / avg if avg > 0 else 0

    # Score: 50 for z-score, 30 for deviation magnitude, 20 for data quality
    data_quality = min(20, len(history) // 5)  # More data = higher confidence

    if z_score > 1.5:
        # Price too high -> BUY NO (expect reversion down)
        z_score_component = min(50, int(abs(z_score) * 15))
        deviation_component = min(30, int(deviation_pct * 300))
        score = z_score_component + deviation_component + data_quality

        return {
            "signal": "MEAN_REVERT_SHORT",
            "direction": "BUY_NO",
            "score": min(100, score),
            "z_score": round(z_score, 2),
            "avg": round(avg, 4),
            "current": round(current_price, 4),
            "deviation_pct": round(deviation_pct, 4),
            "expected_revert": round(avg + std * 0.5, 4),
            "edge_estimate": round(current_price - avg, 4),
            "data_points": len(history),
        }

    elif z_score < -1.5:
        # Price too low -> BUY YES (expect reversion up)
        z_score_component = min(50, int(abs(z_score) * 15))
        deviation_component = min(30, int(deviation_pct * 300))
        score = z_score_component + deviation_component + data_quality

        return {
            "signal": "MEAN_REVERT_LONG",
            "direction": "BUY_YES",
            "score": min(100, score),
            "z_score": round(z_score, 2),
            "avg": round(avg, 4),
            "current": round(current_price, 4),
            "deviation_pct": round(deviation_pct, 4),
            "expected_revert": round(avg - std * 0.5, 4),
            "edge_estimate": round(avg - current_price, 4),
            "data_points": len(history),
        }

    return {
        "signal": "NO_SIGNAL",
        "score": 0,
        "z_score": round(z_score, 2),
        "avg": round(avg, 4),
    }


# ---------------------------------------------------------------------------
# Edge Strategy 3: Volume Anomaly
# ---------------------------------------------------------------------------

def detect_volume_anomaly(history: list, current_volume: float) -> dict:
    """
    Detect sudden volume increases that may indicate information asymmetry.
    A volume spike often precedes a price move.
    """
    if len(history) < 10:
        return {"signal": "INSUFFICIENT_DATA", "score": 0}

    volumes = [h.get("v", 0) for h in history if h.get("v", 0) > 0]
    if len(volumes) < 5:
        return {"signal": "NO_VOLUME_DATA", "score": 0}

    avg_vol = sum(volumes) / len(volumes)
    if avg_vol <= 0:
        return {"signal": "ZERO_VOLUME", "score": 0}

    vol_ratio = current_volume / avg_vol

    # Check if recent price is moving with volume
    recent_prices = [h["p"] for h in history[-5:]]
    price_trend = recent_prices[-1] - recent_prices[0] if len(recent_prices) >= 2 else 0

    if vol_ratio >= 2.0:
        # Volume is 2x+ the average - something is happening
        score = min(100, int(vol_ratio * 15 + abs(price_trend) * 200))

        direction = "BUY_YES" if price_trend > 0.01 else "BUY_NO" if price_trend < -0.01 else "NEUTRAL"

        return {
            "signal": "VOLUME_SPIKE",
            "direction": direction,
            "score": min(100, score),
            "vol_ratio": round(vol_ratio, 2),
            "avg_volume": round(avg_vol, 0),
            "current_volume": round(current_volume, 0),
            "price_trend": round(price_trend, 4),
            "edge_estimate": round(abs(price_trend) * 0.5, 4) if direction != "NEUTRAL" else 0,
        }

    return {
        "signal": "NORMAL_VOLUME",
        "score": 0,
        "vol_ratio": round(vol_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Edge Strategy 4: LLM Mispricing (Anthropic Haiku, Groq fallback)
# ---------------------------------------------------------------------------

def _call_anthropic_haiku(prompt: str, api_key: str) -> str:
    """Call Anthropic Haiku API. Returns raw text response."""
    import urllib.request
    body = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data["content"][0]["text"]


def _call_groq_llm(prompt: str, api_key: str) -> str:
    """Call Groq API. Returns raw text response."""
    import requests as _requests
    resp = None
    for attempt in range(3):
        resp = _requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        if resp.status_code == 429:
            wait = (2 ** attempt) * 3
            logger.debug(f"Groq rate limited, waiting {wait}s (attempt {attempt+1}/3)")
            time.sleep(wait)
            continue
        break
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _load_api_keys() -> tuple:
    """Load Anthropic and Groq API keys from env/.env file."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    if not anthropic_key or not groq_key:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.exists():
            try:
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    v = v.strip().strip("'\"")
                    if k.strip() == "ANTHROPIC_API_KEY" and not anthropic_key:
                        anthropic_key = v
                    elif k.strip() == "GROQ_API_KEY" and not groq_key:
                        groq_key = v
            except Exception:
                pass
    return anthropic_key, groq_key


def detect_llm_mispricing(
    question: str,
    yes_price: float,
    volume: float,
    end_date: str,
    category: str = "general",
) -> dict:
    """
    Use LLM to estimate fair probability and compare to market price.
    Primary: Anthropic Haiku (fast, reliable). Fallback: Groq (Llama 3.3 70B).
    """
    anthropic_key, groq_key = _load_api_keys()
    if not anthropic_key and not groq_key:
        return {"signal": "NO_API_KEY", "score": 0}

    prompt = f"""You are evaluating prediction market odds for mispricing.

MARKET: {question}
CURRENT ODDS: {yes_price*100:.0f}% YES / {(1-yes_price)*100:.0f}% NO
VOLUME: ${volume:,.0f}
END DATE: {end_date}
CATEGORY: {category}

Analyze whether the market price is fair. Consider:
1. Base rates for this type of event
2. Current evidence supporting/contradicting the price
3. Known biases (recency, anchoring, longshot)
4. Time to resolution

Markets are ~85-90% efficient. Only flag if you have strong logical reason.
Taker fee is 2% * price * (1-price), typically 0.2-0.5%.

RESPOND IN JSON ONLY:
{{"probability": 0.XX, "confidence": X, "direction": "BUY_YES|BUY_NO|NO_EDGE", "reasoning": "1-2 sentences"}}"""

    try:
        raw = None
        # Primary: Anthropic Haiku
        if anthropic_key:
            try:
                raw = _call_anthropic_haiku(prompt, anthropic_key)
                logger.debug("LLM mispricing: used Anthropic Haiku")
            except Exception as e:
                logger.warning(f"Anthropic Haiku failed, falling back to Groq: {e}")
        # Fallback: Groq
        if raw is None and groq_key:
            try:
                raw = _call_groq_llm(prompt, groq_key)
                logger.debug("LLM mispricing: used Groq fallback")
            except Exception as e:
                logger.warning(f"Groq fallback also failed: {e}")
                return {"signal": "LLM_ERROR", "score": 0, "error": str(e)}
        if raw is None:
            return {"signal": "LLM_ERROR", "score": 0, "error": "All LLM providers failed"}

        # Parse JSON
        import re
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                return {"signal": "PARSE_ERROR", "score": 0, "raw": raw[:200]}

        prob = float(result.get("probability", yes_price))
        prob = max(0.01, min(0.99, prob))
        confidence = int(result.get("confidence", 5))
        direction = result.get("direction", "NO_EDGE")
        reasoning = result.get("reasoning", "")

        edge_raw = abs(prob - yes_price)
        fee = TAKER_FEE * yes_price * (1 - yes_price)
        edge_net = edge_raw - fee

        if edge_net <= 0.01 or direction == "NO_EDGE":
            return {
                "signal": "NO_LLM_EDGE",
                "score": 0,
                "llm_probability": round(prob, 4),
                "market_price": round(yes_price, 4),
                "reasoning": reasoning,
            }

        # Score: 40 for edge size, 30 for confidence, 30 for edge/fee ratio
        edge_score = min(40, int(edge_net * 400))
        conf_score = min(30, confidence * 3)
        ratio_score = min(30, int((edge_net / max(0.001, fee)) * 10))
        score = edge_score + conf_score + ratio_score

        return {
            "signal": "LLM_MISPRICING",
            "direction": direction,
            "score": min(100, score),
            "llm_probability": round(prob, 4),
            "market_price": round(yes_price, 4),
            "edge_raw": round(edge_raw, 4),
            "edge_net": round(edge_net, 4),
            "fee": round(fee, 4),
            "confidence": confidence,
            "reasoning": reasoning,
        }

    except Exception as e:
        logger.warning(f"LLM mispricing check failed: {e}")
        return {"signal": "LLM_ERROR", "score": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# Category Detection
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "sports": ["nba", "nfl", "mlb", "nhl", "game", "match", "win", "beat", "championship",
               "playoffs", "ufc", "fight", "race", "tennis", "golf"],
    "politics": ["president", "election", "vote", "congress", "senate", "trump", "democrat",
                 "republican", "governor", "poll", "primary"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "token"],
    "macro": ["fed", "fomc", "interest rate", "inflation", "cpi", "gdp", "recession",
              "tariff", "treasury", "bond"],
    "tech": ["ai", "openai", "google", "apple", "tesla", "nvidia", "microsoft", "spacex"],
}


def categorize_market(question: str) -> str:
    q = question.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in q)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ---------------------------------------------------------------------------
# Composite Edge Detector
# ---------------------------------------------------------------------------

class EdgeDetector:
    """
    Composite edge detection engine.

    Runs all 4 strategies on each market and combines scores.
    Returns opportunities sorted by composite score.
    """

    MIN_SCORE = 70  # Minimum score to act on
    MIN_VOLUME = 50_000  # Minimum market volume (USD)

    def __init__(self, use_llm: bool = True, min_score: int = 70):
        self.scanner = PolymarketScanner()
        self.use_llm = use_llm
        self.min_score = min_score

    def analyze_market(
        self,
        market: dict,
        history: list = None,
    ) -> dict:
        """
        Run all edge detection strategies on a single market.

        Returns:
            dict with per-strategy results and composite score
        """
        slug = market.get("slug", "")
        question = market.get("question", "")
        prices = self.scanner._parse_prices(market.get("outcomePrices", []))
        if not prices:
            return {"slug": slug, "score": 0, "error": "no_prices"}

        yes_price = prices[0]
        volume = float(market.get("volumeNum", 0) or 0)
        end_date = str(market.get("endDate", ""))[:10]
        category = categorize_market(question)

        # Load price history if not provided
        if history is None:
            history = load_price_history(slug, hours=48)

        # Record current price
        record_price(slug, yes_price, volume)

        # Run strategies
        strategies = {}

        # 1. Momentum
        mom = detect_momentum(history, yes_price)
        strategies["momentum"] = mom

        # 2. Mean Reversion
        mr = detect_mean_reversion(history, yes_price)
        strategies["mean_reversion"] = mr

        # 3. Volume Anomaly
        va = detect_volume_anomaly(history, volume)
        strategies["volume_anomaly"] = va

        # 4. LLM Mispricing (if enabled and API key available)
        # Pre-screen: only call LLM if at least one other strategy has a signal,
        # or if market has high volume (top candidate). This reduces API calls by ~80%.
        other_signals = sum(1 for s in [mom, mr, va] if s.get("score", 0) > 0)
        high_volume = volume > 500_000
        if self.use_llm and (other_signals > 0 or high_volume):
            llm = detect_llm_mispricing(question, yes_price, volume, end_date, category)
            strategies["llm_mispricing"] = llm
        else:
            strategies["llm_mispricing"] = {"signal": "SKIPPED" if self.use_llm else "DISABLED", "score": 0}

        # Composite score: weighted average of active signals
        # LLM gets highest weight (40%), momentum + MR (25% each), volume (10%)
        weights = {
            "llm_mispricing": 0.40,
            "momentum": 0.25,
            "mean_reversion": 0.25,
            "volume_anomaly": 0.10,
        }

        weighted_score = 0
        total_weight = 0
        active_signals = []
        best_direction = "NO_EDGE"
        best_edge = 0

        for strategy_name, result in strategies.items():
            s = result.get("score", 0)
            if s > 0:
                w = weights.get(strategy_name, 0.1)
                weighted_score += s * w
                total_weight += w
                active_signals.append(strategy_name)

                edge = result.get("edge_estimate", result.get("edge_net", 0))
                if edge > best_edge:
                    best_edge = edge
                    best_direction = result.get("direction", "NO_EDGE")

        composite_score = int(weighted_score / max(0.01, total_weight)) if total_weight > 0 else 0

        # Bonus for multiple agreeing signals
        if len(active_signals) >= 2:
            # Check if directions agree
            directions = [strategies[s].get("direction", "NEUTRAL") for s in active_signals]
            buy_yes = sum(1 for d in directions if d == "BUY_YES")
            buy_no = sum(1 for d in directions if d == "BUY_NO")
            if buy_yes >= 2 or buy_no >= 2:
                composite_score = min(100, composite_score + 10)  # Agreement bonus

        # Get token IDs for trading
        tokens = market.get("clobTokenIds", market.get("tokens", []))
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except Exception:
                tokens = []

        yes_token = None
        no_token = None
        if isinstance(tokens, list) and tokens:
            if isinstance(tokens[0], dict):
                for t in tokens:
                    if t.get("outcome") == "Yes":
                        yes_token = t.get("token_id")
                    elif t.get("outcome") == "No":
                        no_token = t.get("token_id")
            else:
                yes_token = tokens[0] if len(tokens) > 0 else None
                no_token = tokens[1] if len(tokens) > 1 else None

        return {
            "slug": slug,
            "question": question[:120],
            "category": category,
            "yes_price": round(yes_price, 4),
            "volume": volume,
            "end_date": end_date,
            "composite_score": composite_score,
            "direction": best_direction,
            "edge_estimate": round(best_edge, 4),
            "active_signals": active_signals,
            "strategies": strategies,
            "yes_token": yes_token,
            "no_token": no_token,
            "history_points": len(history),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def scan(self, limit: int = 200, min_score: int = None) -> dict:
        """
        Full scan: fetch markets, analyze all, return sorted opportunities.
        """
        min_score = min_score or self.min_score
        logger.info(f"Edge scan starting (limit={limit}, min_score={min_score})")
        scan_start = time.time()

        markets = self.scanner.get_active_markets(limit=limit)
        logger.info(f"Fetched {len(markets)} markets")

        results = []
        opportunities = []

        for i, m in enumerate(markets):
            prices = self.scanner._parse_prices(m.get("outcomePrices", []))
            if not prices:
                continue

            yes_price = prices[0]
            vol = float(m.get("volumeNum", 0) or 0)

            # Filter: tradeable range + minimum volume
            if not (0.05 <= yes_price <= 0.95) or vol < self.MIN_VOLUME:
                continue

            analysis = self.analyze_market(m)
            results.append(analysis)

            if analysis["composite_score"] >= min_score:
                opportunities.append(analysis)
                logger.info(
                    f"  EDGE: {analysis['question'][:50]} | "
                    f"score={analysis['composite_score']} | "
                    f"{analysis['direction']} | "
                    f"signals={analysis['active_signals']}"
                )

            if (i + 1) % 50 == 0:
                logger.info(f"  Analyzed {i+1} markets, {len(opportunities)} edges found")

            # Rate limit LLM calls (Groq free = 30 RPM, need ~2.5s between calls)
            llm_signal = analysis.get("strategies", {}).get("llm_mispricing", {}).get("signal", "")
            if self.use_llm and llm_signal not in ("DISABLED", "NO_API_KEY", "SKIPPED"):
                time.sleep(2.5)

        # Sort by composite score
        opportunities.sort(key=lambda x: x["composite_score"], reverse=True)

        scan_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - scan_start, 1),
            "total_markets": len(markets),
            "analyzed": len(results),
            "opportunities_found": len(opportunities),
            "min_score_used": min_score,
            "llm_enabled": self.use_llm,
            "opportunities": opportunities,
        }

        # Save results
        with open(EDGE_RESULTS_FILE, "w") as f:
            json.dump(scan_result, f, indent=2, default=str)

        # Log to JSONL
        log_entry = {
            "timestamp": scan_result["timestamp"],
            "duration_s": scan_result["duration_seconds"],
            "analyzed": len(results),
            "edges": len(opportunities),
            "top_score": opportunities[0]["composite_score"] if opportunities else 0,
        }
        with open(EDGE_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(
            f"Scan complete: {len(results)} analyzed, "
            f"{len(opportunities)} opportunities (score >= {min_score})"
        )
        return scan_result

    def format_discord_report(self, scan: dict) -> str:
        """Format scan results for Discord."""
        opps = scan.get("opportunities", [])
        lines = [
            f"**EDGE DETECTOR SCAN** ({datetime.now().strftime('%H:%M ET')})",
            f"Analyzed: {scan.get('analyzed', 0)} markets | "
            f"Edges found: {scan.get('opportunities_found', 0)} | "
            f"Duration: {scan.get('duration_seconds', 0):.0f}s",
        ]

        if opps:
            lines.append("")
            for opp in opps[:8]:
                score = opp["composite_score"]
                emoji = "🔴" if score >= 90 else "🟠" if score >= 80 else "🟡"
                signals = ", ".join(opp["active_signals"])
                lines.append(
                    f"{emoji} **{opp['question'][:55]}**\n"
                    f"  Score: {score}/100 | {opp['direction']} @ {opp['yes_price']:.2f} | "
                    f"Edge: {opp['edge_estimate']:.3f} | Signals: {signals}"
                )
        else:
            lines.append("\nNo high-confidence edges found. Markets appear fairly priced.")

        return "\n".join(lines)

    def get_actionable_edges(self, min_score: int = None) -> list:
        """
        Load most recent scan results and return actionable edges.
        Used by paper_trader and live_trader.
        """
        min_score = min_score or self.min_score
        if not EDGE_RESULTS_FILE.exists():
            return []

        try:
            with open(EDGE_RESULTS_FILE) as f:
                data = json.load(f)

            # Check freshness (within last 2 hours)
            ts = datetime.fromisoformat(data["timestamp"])
            if datetime.now(timezone.utc) - ts > timedelta(hours=2):
                logger.warning("Edge results are stale (>2h old). Run a fresh scan.")
                return []

            return [
                opp for opp in data.get("opportunities", [])
                if opp.get("composite_score", 0) >= min_score
            ]
        except Exception as e:
            logger.warning(f"Failed to load edge results: {e}")
            return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Edge Detection Engine for Polymarket")
    parser.add_argument("--scan", action="store_true", help="Full edge detection scan")
    parser.add_argument("--analyze", type=str, help="Analyze a specific market slug")
    parser.add_argument("--limit", type=int, default=200, help="Max markets to scan")
    parser.add_argument("--min-score", type=int, default=70, help="Minimum edge score (0-100)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM analysis")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    if args.analyze:
        detector = EdgeDetector(use_llm=not args.no_llm, min_score=args.min_score)
        # Resolve the slug to a market
        scanner = PolymarketScanner()
        markets = scanner.get_active_markets(limit=500)
        target = None
        for m in markets:
            if m.get("slug", "") == args.analyze:
                target = m
                break
        if not target:
            print(f"Market not found: {args.analyze}")
            return

        result = detector.analyze_market(target)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"\nEdge Analysis: {result['question']}")
            print(f"  Score: {result['composite_score']}/100 | Direction: {result['direction']}")
            print(f"  Price: {result['yes_price']:.3f} | Volume: ${result['volume']:,.0f}")
            print(f"  Active signals: {', '.join(result['active_signals']) or 'None'}")
            for name, strat in result["strategies"].items():
                sig = strat.get("signal", "?")
                s = strat.get("score", 0)
                if s > 0:
                    print(f"    [{name}] {sig} (score={s}): {strat.get('reasoning', strat.get('edge_estimate', ''))}")
        return

    if args.scan or True:  # Default to scan
        detector = EdgeDetector(use_llm=not args.no_llm, min_score=args.min_score)
        result = detector.scan(limit=args.limit, min_score=args.min_score)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(detector.format_discord_report(result))


if __name__ == "__main__":
    main()
