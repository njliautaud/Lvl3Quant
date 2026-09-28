"""
Cross-Platform Arbitrage Scanner — Kalshi vs Polymarket

Identifies events that exist on BOTH Kalshi and Polymarket but are priced
differently, then calculates the net arbitrage edge after accounting for
both platforms' fee structures.

Supported overlapping categories:
  1. FOMC rate decisions — Kalshi brackets, Polymarket yes/no
  2. Inflation (CPI) — both have brackets/yes-no
  3. Bitcoin price targets — Kalshi BTCUSD, Polymarket BTC markets
  4. Presidential / political elections — both platforms
  5. GDP growth — both platforms
  6. Unemployment — both platforms

Fee schedule (as of 2026):
  Kalshi taker: 3.5% coefficient for SPX/equity markets, 7% for standard.
                Fee = ceil(coeff * contracts * price * (1-price) * 100) / 100
  Polymarket taker: 2% flat.
                    Fee = 0.02 * price * (1-price)

An arbitrage opportunity is only flagged when:
    net_edge = |p_kalshi - p_poly| - kalshi_fee - poly_fee > 0

Fuzzy matching:
  Event matching uses text similarity (character n-gram Jaccard) rather than
  exact string comparison, so "Will the Fed cut rates by 25bp at the March
  2026 FOMC?" matches "Fed Funds rate ≤ 4.25% after March 18 FOMC meeting".

Usage:
    python cross_platform_arb.py --scan
    python cross_platform_arb.py --fomc
    python cross_platform_arb.py --btc
    python cross_platform_arb.py --report
"""

import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("cross_platform_arb")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# API endpoints
# ============================================================
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_LIVE_BASE = "https://trading-api.kalshi.com/trade-api/v2"

# ============================================================
# Fee constants
# ============================================================
# Kalshi taker fee coefficient (fee = ceil(coeff * N * p * (1-p) * 100) / 100 per contract)
KALSHI_FEE_SPX = 0.035          # 3.5% for SPX/equity markets
KALSHI_FEE_STANDARD = 0.07      # 7% for all other markets
KALSHI_FEE_FOMC = 0.07          # FOMC / rate markets are standard
KALSHI_FEE_BTC = 0.07           # Crypto markets are standard

# Polymarket taker fee (fee = 0.02 * p * (1-p))
POLY_TAKER_FEE = 0.02

# Strategy weight (from strategy_engine.py)
STRATEGY_WEIGHT = 0.05

# Minimum net edge to flag an opportunity
MIN_NET_EDGE = 0.005     # 0.5 cents

# Fuzzy match threshold (Jaccard similarity on character 3-grams)
FUZZY_MATCH_THRESHOLD = 0.18


# ============================================================
# HTTP helper
# ============================================================

def _get(url: str, params: dict = None, headers: dict = None, retries: int = 3) -> object:
    """HTTP GET with exponential backoff retry."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=12)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 1.5 ** attempt
                logger.debug(f"HTTP error (attempt {attempt + 1}): {e}. Retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                logger.warning(f"HTTP failed: {url} — {e}")
                return None
    return None


def _parse_prices(prices_raw) -> list:
    """Parse outcomePrices from Gamma API (may be JSON string or list)."""
    if not prices_raw:
        return []
    if isinstance(prices_raw, str):
        try:
            prices_raw = json.loads(prices_raw)
        except (json.JSONDecodeError, ValueError):
            return []
    try:
        return [float(p) for p in prices_raw]
    except (ValueError, TypeError):
        return []


# ============================================================
# Fee calculators
# ============================================================

def kalshi_taker_fee_per_contract(price: float, is_spx: bool = False) -> float:
    """
    Kalshi per-contract taker fee (in dollars, for one $1-max contract).
    Fee = ceil(coeff * 1 * price * (1-price) * 100) / 100
    """
    coeff = KALSHI_FEE_SPX if is_spx else KALSHI_FEE_STANDARD
    return math.ceil(coeff * price * (1 - price) * 100) / 100


def poly_taker_fee(price: float) -> float:
    """
    Polymarket per-contract taker fee.
    Fee = 0.02 * price * (1-price)
    """
    return POLY_TAKER_FEE * price * (1 - price)


def net_arb_edge(
    kalshi_yes_price: float,
    poly_yes_price: float,
    kalshi_is_spx: bool = False,
) -> dict:
    """
    Calculate the net arbitrage edge after both platforms' fees.

    The arbitrage structure:
      - If kalshi_yes_price > poly_yes_price:
          BUY YES on Polymarket + BUY NO on Kalshi
          (both should resolve together, pay both takers)
      - If poly_yes_price > kalshi_yes_price:
          BUY YES on Kalshi + BUY NO on Polymarket

    Args:
        kalshi_yes_price: Kalshi YES price (0-1)
        poly_yes_price: Polymarket YES price (0-1)
        kalshi_is_spx: Whether Kalshi market is SPX-category (lower fee)

    Returns:
        dict with gross_edge, kalshi_fee, poly_fee, net_edge, direction, action_kalshi, action_poly
    """
    gross_edge = abs(kalshi_yes_price - poly_yes_price)

    if kalshi_yes_price > poly_yes_price:
        # Kalshi is more expensive — buy YES on Polymarket (cheap), buy NO on Kalshi
        direction = "kalshi_overpriced"
        kalshi_entry = 1 - kalshi_yes_price  # NO price on Kalshi
        poly_entry = poly_yes_price
        action_kalshi = "BUY_NO"
        action_poly = "BUY_YES"
    else:
        # Polymarket is more expensive — buy YES on Kalshi, buy NO on Polymarket
        direction = "poly_overpriced"
        kalshi_entry = kalshi_yes_price
        poly_entry = 1 - poly_yes_price  # NO price on Polymarket
        action_kalshi = "BUY_YES"
        action_poly = "BUY_NO"

    k_fee = kalshi_taker_fee_per_contract(kalshi_entry, is_spx=kalshi_is_spx)
    p_fee = poly_taker_fee(poly_entry)
    total_fees = k_fee + p_fee
    net_edge = gross_edge - total_fees

    return {
        "gross_edge": round(gross_edge, 4),
        "kalshi_fee": round(k_fee, 4),
        "poly_fee": round(p_fee, 4),
        "total_fees": round(total_fees, 4),
        "net_edge_after_fees": round(net_edge, 4),
        "direction": direction,
        "action_kalshi": action_kalshi,
        "action_poly": action_poly,
        "kalshi_entry_price": round(kalshi_entry, 4),
        "poly_entry_price": round(poly_entry, 4),
        "is_profitable": net_edge > MIN_NET_EDGE,
    }


# ============================================================
# Fuzzy text matching
# ============================================================

def _trigrams(text: str) -> set:
    """Generate character 3-grams from text."""
    t = re.sub(r"[^a-z0-9 ]", "", text.lower())
    # Also pad for edge n-grams
    t = f"  {t}  "
    return {t[i:i+3] for i in range(len(t) - 2)}


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity on character 3-grams of two strings."""
    tg_a = _trigrams(a)
    tg_b = _trigrams(b)
    if not tg_a or not tg_b:
        return 0.0
    intersection = tg_a & tg_b
    union = tg_a | tg_b
    return len(intersection) / len(union)


def _normalise_question(q: str) -> str:
    """Lowercase and strip common noise for better fuzzy matching."""
    q = q.lower()
    # Remove dates in common formats that differ between platforms
    q = re.sub(r"\b(january|february|march|april|may|june|july|august|september|"
               r"october|november|december)\b", "month", q)
    q = re.sub(r"\b20\d{2}\b", "year", q)
    q = re.sub(r"\b\d{1,2}/\d{1,2}\b", "", q)
    # Normalise number formats
    q = re.sub(r"\$([0-9,]+)", lambda m: m.group(1).replace(",", ""), q)
    return q.strip()


def find_best_match(
    question: str,
    candidates: list,
    key: str = "question",
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> Optional[dict]:
    """
    Find the best fuzzy match for `question` among `candidates`.

    Args:
        question: The query question text.
        candidates: List of dicts, each with a text field named `key`.
        key: The field name holding question text in each candidate.
        threshold: Minimum Jaccard similarity to consider a match.

    Returns:
        The best-matching candidate dict, or None if no match exceeds threshold.
    """
    q_norm = _normalise_question(question)
    best_score = 0.0
    best_match = None

    for c in candidates:
        c_text = c.get(key, "")
        c_norm = _normalise_question(c_text)
        score = jaccard_similarity(q_norm, c_norm)
        if score > best_score:
            best_score = score
            best_match = c

    if best_score >= threshold:
        return best_match
    return None


# ============================================================
# Kalshi market fetcher (REST, no auth needed for reading)
# ============================================================

class KalshiMarketReader:
    """
    Read-only access to Kalshi market data via their public REST API.
    No authentication required for market discovery.
    """

    def __init__(self, mode: str = "demo"):
        self.base = KALSHI_DEMO_BASE if mode == "demo" else KALSHI_LIVE_BASE

    def get_events(self, limit: int = 100, series_ticker: str = None) -> list:
        """Fetch events from Kalshi. Optionally filter by series ticker prefix."""
        params = {"limit": limit, "status": "open"}
        if series_ticker:
            params["series_ticker"] = series_ticker

        data = _get(f"{self.base}/events", params=params)
        if data is None:
            return []
        return data.get("events", [])

    def get_markets(self, event_ticker: str) -> list:
        """Fetch all markets for a given event ticker."""
        data = _get(f"{self.base}/events/{event_ticker}")
        if data is None:
            return []
        markets = data.get("markets", [])
        return markets

    def get_market(self, ticker: str) -> Optional[dict]:
        """Fetch a single market by ticker."""
        data = _get(f"{self.base}/markets/{ticker}")
        if data is None:
            return None
        return data.get("market", data)

    def search_markets(self, keywords: list, limit: int = 100) -> list:
        """
        Search for markets matching any of the given keywords.
        Kalshi's public API doesn't support full-text search, so we fetch
        a broad set and filter locally.
        """
        all_events = self.get_events(limit=limit)
        results = []
        for event in all_events:
            title = (event.get("title") or event.get("event_ticker") or "").lower()
            if any(kw.lower() in title for kw in keywords):
                results.append(event)
        return results


# ============================================================
# Polymarket market fetcher
# ============================================================

class PolymarketMarketReader:
    """Read-only access to Polymarket Gamma API."""

    def __init__(self):
        self.gamma_base = GAMMA_API

    def get_markets(self, limit: int = 200, query: str = None) -> list:
        """Fetch active markets, optionally filtered by query."""
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "order": "volumeNum",
            "ascending": "false",
        }
        if query:
            params["_c"] = query

        data = _get(f"{self.gamma_base}/markets", params=params)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        return data.get("data", []) if isinstance(data, dict) else []

    def filter_by_keywords(self, markets: list, keywords: list) -> list:
        """Filter markets whose question contains at least one keyword."""
        result = []
        for m in markets:
            q = (m.get("question") or "").lower()
            if any(kw.lower() in q for kw in keywords):
                result.append(m)
        return result

    def yes_price(self, market: dict) -> Optional[float]:
        """Extract YES price from a market dict."""
        prices = _parse_prices(market.get("outcomePrices", []))
        return prices[0] if prices else None


# ============================================================
# Main CrossPlatformArbitrage class
# ============================================================

class CrossPlatformArbitrage:
    """
    Scans for the same event priced differently on Kalshi and Polymarket.

    Matching is fuzzy (Jaccard on character 3-grams) to handle differences
    in question phrasing between the two platforms.

    The arbitrage is only flagged when the net edge after BOTH platforms'
    taker fees is positive.
    """

    # Event category templates — define what to look for on each platform
    EVENT_CATEGORIES = [
        {
            "name": "fomc_rate",
            "description": "FOMC Federal Funds Rate Decision",
            "kalshi_keywords": ["fomc", "fed funds", "interest rate", "rate decision"],
            "poly_keywords": ["fomc", "federal reserve", "interest rate", "fed funds",
                              "rate cut", "rate hike", "hold"],
            "kalshi_is_spx": False,
        },
        {
            "name": "cpi_inflation",
            "description": "CPI / Inflation Data",
            "kalshi_keywords": ["cpi", "inflation", "consumer price"],
            "poly_keywords": ["cpi", "inflation", "consumer price index"],
            "kalshi_is_spx": False,
        },
        {
            "name": "bitcoin_price",
            "description": "Bitcoin Price Target",
            "kalshi_keywords": ["bitcoin", "btc", "btcusd"],
            "poly_keywords": ["bitcoin", "btc", "will bitcoin", "will btc"],
            "kalshi_is_spx": False,
        },
        {
            "name": "gdp_growth",
            "description": "GDP Growth Rate",
            "kalshi_keywords": ["gdp", "gross domestic", "growth rate"],
            "poly_keywords": ["gdp", "gross domestic product", "growth"],
            "kalshi_is_spx": False,
        },
        {
            "name": "unemployment",
            "description": "Unemployment Rate",
            "kalshi_keywords": ["unemployment", "jobless", "nonfarm"],
            "poly_keywords": ["unemployment", "jobless claims", "nonfarm payroll", "jobs"],
            "kalshi_is_spx": False,
        },
        {
            "name": "presidential_election",
            "description": "Presidential / Major Elections",
            "kalshi_keywords": ["president", "election", "elect", "2028"],
            "poly_keywords": ["president", "election", "win the presidency", "elected president"],
            "kalshi_is_spx": False,
        },
        {
            "name": "recession",
            "description": "US Recession Probability",
            "kalshi_keywords": ["recession", "nber", "economic contraction"],
            "poly_keywords": ["recession", "nber recession", "economic recession"],
            "kalshi_is_spx": False,
        },
    ]

    def __init__(self, kalshi_mode: str = "demo"):
        self.kalshi = KalshiMarketReader(mode=kalshi_mode)
        self.poly = PolymarketMarketReader()
        self._cache_kalshi: dict = {}  # category -> list of events
        self._cache_poly: dict = {}    # category -> list of markets

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def _fetch_kalshi_for_category(self, category: dict) -> list:
        """Fetch and cache Kalshi events for a category."""
        name = category["name"]
        if name not in self._cache_kalshi:
            events = self.kalshi.search_markets(
                keywords=category["kalshi_keywords"],
                limit=200,
            )
            self._cache_kalshi[name] = events
        return self._cache_kalshi[name]

    def _fetch_poly_for_category(self, category: dict) -> list:
        """Fetch and cache Polymarket markets for a category."""
        name = category["name"]
        if name not in self._cache_poly:
            all_markets = self.poly.get_markets(limit=300)
            filtered = self.poly.filter_by_keywords(all_markets, category["poly_keywords"])
            self._cache_poly[name] = filtered
        return self._cache_poly[name]

    def _extract_kalshi_yes_price(self, event: dict) -> Optional[float]:
        """
        Extract a YES-equivalent price from a Kalshi event.

        Kalshi events may contain multiple markets (bracket contracts).
        For fuzzy matching purposes, we use the first YES market price.

        Post March 5, 2026: _dollars fields (FixedPointDollars strings) are primary.
        Legacy integer-cent fields have been removed from the API.
        """
        # Try _dollars fields first (FixedPointDollars strings like "0.5600")
        for price_field in ["yes_bid_dollars", "yes_ask_dollars", "last_price_dollars",
                            "yes_bid", "yes_ask", "yes_price", "last_price"]:
            val = event.get(price_field)
            if val is not None:
                try:
                    p = float(val)
                    # Legacy cent fields (0-100) may still appear in cached data — normalise
                    if p > 1.0:
                        p /= 100.0
                    if 0 < p < 1:
                        return p
                except (TypeError, ValueError):
                    continue

        # Try markets list inside event
        markets = event.get("markets", [])
        for m in markets:
            for field in ["yes_bid_dollars", "yes_ask_dollars", "yes_bid", "yes_ask"]:
                val = m.get(field)
                if val is not None:
                    try:
                        p = float(val)
                        if p > 1.0:
                            p /= 100.0
                        if 0 < p < 1:
                            return p
                    except (TypeError, ValueError):
                        continue

        return None

    def _build_arb_opportunity(
        self,
        category: dict,
        kalshi_event: dict,
        poly_market: dict,
        kalshi_yes: float,
        poly_yes: float,
        similarity: float,
    ) -> Optional[dict]:
        """Build an arbitrage opportunity dict if edge is positive after fees."""
        edge_info = net_arb_edge(
            kalshi_yes_price=kalshi_yes,
            poly_yes_price=poly_yes,
            kalshi_is_spx=category.get("kalshi_is_spx", False),
        )

        if not edge_info["is_profitable"]:
            return None

        kalshi_title = (
            kalshi_event.get("title")
            or kalshi_event.get("event_ticker")
            or kalshi_event.get("series_ticker")
            or "?"
        )
        poly_question = poly_market.get("question", "?")
        poly_volume = float(poly_market.get("volumeNum", 0) or 0)
        poly_end = (
            poly_market.get("endDate")
            or poly_market.get("endDateIso")
            or poly_market.get("end_date_iso")
            or ""
        )[:10]

        return {
            "strategy": "cross_platform_arb",
            "weight": STRATEGY_WEIGHT,
            "platform": "kalshi+polymarket",
            "category": category["name"],
            "category_description": category["description"],
            "kalshi_question": kalshi_title[:100],
            "kalshi_ticker": kalshi_event.get("event_ticker", ""),
            "poly_question": poly_question[:100],
            "poly_slug": (
                poly_market.get("slug")
                or poly_market.get("marketSlug")
                or poly_market.get("market_slug", "")
            ),
            "kalshi_yes_price": round(kalshi_yes, 4),
            "poly_yes_price": round(poly_yes, 4),
            "gross_edge": edge_info["gross_edge"],
            "kalshi_fee": edge_info["kalshi_fee"],
            "poly_fee": edge_info["poly_fee"],
            "total_fees": edge_info["total_fees"],
            "net_edge_after_fees": edge_info["net_edge_after_fees"],
            "direction": edge_info["direction"],
            "action_kalshi": edge_info["action_kalshi"],
            "action_poly": edge_info["action_poly"],
            "kalshi_entry_price": edge_info["kalshi_entry_price"],
            "poly_entry_price": edge_info["poly_entry_price"],
            "match_similarity": round(similarity, 3),
            "poly_volume": round(poly_volume, 0),
            "poly_end_date": poly_end,
            "action": (
                f"{edge_info['action_kalshi']} on Kalshi | "
                f"{edge_info['action_poly']} on Polymarket"
            ),
            "fee_note": (
                f"Kalshi taker: {edge_info['kalshi_fee']:.4f} | "
                f"Poly taker: {edge_info['poly_fee']:.4f} | "
                f"Total: {edge_info['total_fees']:.4f}"
            ),
        }

    # ------------------------------------------------------------------
    # Matching engine
    # ------------------------------------------------------------------

    def find_overlapping_events(self) -> list:
        """
        Identify matching events that exist on BOTH Kalshi and Polymarket.

        Uses fuzzy question matching (Jaccard similarity on character 3-grams).

        Returns:
            list of match dicts: {category, kalshi_event, poly_market, similarity}
        """
        logger.info("Finding overlapping events between Kalshi and Polymarket...")
        matches = []

        for category in self.EVENT_CATEGORIES:
            logger.info(f"  Category: {category['name']}")

            kalshi_events = self._fetch_kalshi_for_category(category)
            poly_markets = self._fetch_poly_for_category(category)

            logger.info(f"    Kalshi events: {len(kalshi_events)} | Poly markets: {len(poly_markets)}")

            if not kalshi_events or not poly_markets:
                continue

            for k_event in kalshi_events:
                k_title = (
                    k_event.get("title")
                    or k_event.get("event_ticker")
                    or ""
                )

                # Find best Polymarket match
                k_norm = _normalise_question(k_title)
                best_score = 0.0
                best_poly = None

                for p_market in poly_markets:
                    p_question = p_market.get("question", "")
                    score = jaccard_similarity(k_norm, _normalise_question(p_question))
                    if score > best_score:
                        best_score = score
                        best_poly = p_market

                if best_score >= FUZZY_MATCH_THRESHOLD and best_poly is not None:
                    matches.append({
                        "category": category["name"],
                        "category_description": category["description"],
                        "kalshi_event": k_event,
                        "poly_market": best_poly,
                        "similarity": round(best_score, 3),
                    })

            time.sleep(0.2)  # Rate-limit courtesy delay between categories

        logger.info(f"Total overlapping events found: {len(matches)}")
        return matches

    def calculate_arb_edge(
        self,
        kalshi_price: float,
        poly_price: float,
        fees: dict = None,
        kalshi_is_spx: bool = False,
    ) -> dict:
        """
        Compute net arbitrage edge for a given pair of prices.

        This is a standalone method — useful for computing edge on any pair
        without going through the full scan pipeline.

        Args:
            kalshi_price: Kalshi YES price (0-1)
            poly_price: Polymarket YES price (0-1)
            fees: Optional fee overrides dict with 'kalshi_coeff' and 'poly_rate' keys.
            kalshi_is_spx: Whether Kalshi market is SPX category (3.5% vs 7%)

        Returns:
            dict with gross_edge, fees, net_edge_after_fees, direction, action recommendations.
        """
        if fees:
            # Allow caller to override fee coefficients
            k_coeff = fees.get("kalshi_coeff", KALSHI_FEE_SPX if kalshi_is_spx else KALSHI_FEE_STANDARD)
            p_rate = fees.get("poly_rate", POLY_TAKER_FEE)

            gross_edge = abs(kalshi_price - poly_price)
            if kalshi_price > poly_price:
                direction = "kalshi_overpriced"
                k_entry = 1 - kalshi_price
                p_entry = poly_price
                action_kalshi = "BUY_NO"
                action_poly = "BUY_YES"
            else:
                direction = "poly_overpriced"
                k_entry = kalshi_price
                p_entry = 1 - poly_price
                action_kalshi = "BUY_YES"
                action_poly = "BUY_NO"

            k_fee = math.ceil(k_coeff * k_entry * (1 - k_entry) * 100) / 100
            p_fee = p_rate * p_entry * (1 - p_entry)
            total_fees = k_fee + p_fee
            net_edge = gross_edge - total_fees

            return {
                "gross_edge": round(gross_edge, 4),
                "kalshi_fee": round(k_fee, 4),
                "poly_fee": round(p_fee, 4),
                "total_fees": round(total_fees, 4),
                "net_edge_after_fees": round(net_edge, 4),
                "direction": direction,
                "action_kalshi": action_kalshi,
                "action_poly": action_poly,
                "kalshi_entry_price": round(k_entry, 4),
                "poly_entry_price": round(p_entry, 4),
                "is_profitable": net_edge > MIN_NET_EDGE,
                "fee_override": True,
            }

        # Use default fee schedule
        return net_arb_edge(kalshi_price, poly_price, kalshi_is_spx=kalshi_is_spx)

    # ------------------------------------------------------------------
    # Full scan
    # ------------------------------------------------------------------

    def scan_cross_platform(self) -> list:
        """
        Full cross-platform arbitrage scan.

        1. Find overlapping events (fuzzy match)
        2. Extract prices from both platforms
        3. Calculate net edge after both fees
        4. Return profitable opportunities sorted by net edge

        Returns:
            list of opportunity dicts sorted by net_edge_after_fees descending.
        """
        logger.info("=" * 55)
        logger.info("CROSS-PLATFORM ARBITRAGE SCAN")
        logger.info("=" * 55)

        matches = self.find_overlapping_events()
        opportunities = []

        for match in matches:
            category = next(
                (c for c in self.EVENT_CATEGORIES if c["name"] == match["category"]),
                None,
            )
            if category is None:
                continue

            k_event = match["kalshi_event"]
            p_market = match["poly_market"]

            # Extract prices
            kalshi_yes = self._extract_kalshi_yes_price(k_event)
            poly_yes = self.poly.yes_price(p_market)

            if kalshi_yes is None or poly_yes is None:
                logger.debug(
                    f"Missing price: Kalshi={kalshi_yes}, Poly={poly_yes} "
                    f"for {k_event.get('event_ticker', '?')}"
                )
                continue

            opp = self._build_arb_opportunity(
                category=category,
                kalshi_event=k_event,
                poly_market=p_market,
                kalshi_yes=kalshi_yes,
                poly_yes=poly_yes,
                similarity=match["similarity"],
            )
            if opp:
                opportunities.append(opp)

        opportunities.sort(key=lambda x: x["net_edge_after_fees"], reverse=True)
        logger.info(f"Profitable arbitrage opportunities: {len(opportunities)}")
        return opportunities

    # ------------------------------------------------------------------
    # Category-specific convenience scans
    # ------------------------------------------------------------------

    def _scan_category(self, category_name: str) -> list:
        """Run scan for a single named category."""
        category = next(
            (c for c in self.EVENT_CATEGORIES if c["name"] == category_name), None
        )
        if category is None:
            logger.error(f"Unknown category: {category_name}")
            return []

        kalshi_events = self._fetch_kalshi_for_category(category)
        poly_markets = self._fetch_poly_for_category(category)

        logger.info(
            f"{category_name}: {len(kalshi_events)} Kalshi events, "
            f"{len(poly_markets)} Poly markets"
        )

        opportunities = []

        for k_event in kalshi_events:
            k_title = k_event.get("title") or k_event.get("event_ticker") or ""

            for p_market in poly_markets:
                p_question = p_market.get("question", "")
                sim = jaccard_similarity(
                    _normalise_question(k_title),
                    _normalise_question(p_question),
                )

                if sim < FUZZY_MATCH_THRESHOLD:
                    continue

                kalshi_yes = self._extract_kalshi_yes_price(k_event)
                poly_yes = self.poly.yes_price(p_market)

                if kalshi_yes is None or poly_yes is None:
                    continue

                opp = self._build_arb_opportunity(
                    category=category,
                    kalshi_event=k_event,
                    poly_market=p_market,
                    kalshi_yes=kalshi_yes,
                    poly_yes=poly_yes,
                    similarity=sim,
                )
                if opp:
                    opportunities.append(opp)

        opportunities.sort(key=lambda x: x["net_edge_after_fees"], reverse=True)
        return opportunities

    def scan_fomc(self) -> list:
        """Scan FOMC rate decision markets only."""
        logger.info("Scanning FOMC cross-platform arbitrage...")
        return self._scan_category("fomc_rate")

    def scan_bitcoin(self) -> list:
        """Scan Bitcoin price target markets only."""
        logger.info("Scanning Bitcoin cross-platform arbitrage...")
        return self._scan_category("bitcoin_price")

    def scan_cpi(self) -> list:
        """Scan CPI/inflation markets only."""
        logger.info("Scanning CPI cross-platform arbitrage...")
        return self._scan_category("cpi_inflation")

    def scan_elections(self) -> list:
        """Scan presidential/major election markets only."""
        logger.info("Scanning election cross-platform arbitrage...")
        return self._scan_category("presidential_election")

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    def format_report(self, opportunities: list) -> str:
        """Format arbitrage opportunities as a human-readable report."""
        lines = [
            "=" * 65,
            "CROSS-PLATFORM ARBITRAGE REPORT",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 65,
        ]

        if not opportunities:
            lines.append("\nNo profitable cross-platform arbitrage found.")
            lines.append(
                "\nNote: Profitable arb after fees requires |K-P| > kalshi_fee + poly_fee."
            )
            return "\n".join(lines)

        lines.append(f"\n{len(opportunities)} profitable opportunities:\n")

        for i, o in enumerate(opportunities[:20], 1):
            net = o["net_edge_after_fees"]
            gross = o["gross_edge"]
            lines.extend([
                f"{i:2d}. [{o['category_description']}] Match similarity: {o['match_similarity']:.0%}",
                f"    Kalshi: \"{o['kalshi_question'][:65]}\"",
                f"    Poly:   \"{o['poly_question'][:65]}\"",
                f"    Prices: Kalshi YES={o['kalshi_yes_price']:.3f} | Poly YES={o['poly_yes_price']:.3f}",
                f"    Gross edge: {gross:.4f} | Fees: {o['total_fees']:.4f} | NET: {net:+.4f}",
                f"    Action: {o['action']}",
                f"    Fee breakdown: {o['fee_note']}",
                f"    Poly volume: ${o['poly_volume']/1e3:.0f}K | End: {o['poly_end_date']}",
                "",
            ])

        # Summary
        total_edge = sum(o["net_edge_after_fees"] for o in opportunities)
        avg_edge = total_edge / len(opportunities)
        lines.extend([
            "-" * 65,
            f"Total opportunities: {len(opportunities)}",
            f"Average net edge: {avg_edge:.4f}",
            f"Best opportunity: {opportunities[0]['net_edge_after_fees']:.4f} "
            f"({opportunities[0]['category_description']})",
        ])

        return "\n".join(lines)

    def save_results(self, opportunities: list) -> Path:
        """Save scan results to the data directory."""
        filename = DATA_DIR / f"cross_platform_arb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_opportunities": len(opportunities),
            "opportunities": opportunities,
            "scan_config": {
                "min_net_edge": MIN_NET_EDGE,
                "fuzzy_match_threshold": FUZZY_MATCH_THRESHOLD,
                "fee_schedule": {
                    "kalshi_standard": KALSHI_FEE_STANDARD,
                    "kalshi_spx": KALSHI_FEE_SPX,
                    "polymarket_taker": POLY_TAKER_FEE,
                },
            },
        }
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"Results saved to {filename}")
        return filename


# ============================================================
# Module-level convenience functions (for strategy_engine.py integration)
# ============================================================

def scan_all_arb(kalshi_mode: str = "demo") -> list:
    """
    Convenience function: run full cross-platform scan.
    Called by strategy_engine.py scan_cross_platform().
    """
    arb = CrossPlatformArbitrage(kalshi_mode=kalshi_mode)
    return arb.scan_cross_platform()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Platform Arbitrage Scanner")
    parser.add_argument("--scan", action="store_true", help="Full cross-platform scan")
    parser.add_argument("--fomc", action="store_true", help="FOMC markets only")
    parser.add_argument("--btc", action="store_true", help="Bitcoin markets only")
    parser.add_argument("--cpi", action="store_true", help="CPI/inflation markets only")
    parser.add_argument("--elections", action="store_true", help="Election markets only")
    parser.add_argument("--report", action="store_true", help="Generate formatted report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="Kalshi API mode (default: demo)",
    )
    parser.add_argument(
        "--edge-test",
        nargs=2,
        type=float,
        metavar=("KALSHI_YES", "POLY_YES"),
        help="Compute edge for given price pair (e.g., --edge-test 0.65 0.55)",
    )
    args = parser.parse_args()

    # Quick edge calculator mode
    if args.edge_test:
        k_price, p_price = args.edge_test
        result = net_arb_edge(k_price, p_price, kalshi_is_spx=False)
        print("\nEdge Calculation:")
        print(f"  Kalshi YES: {k_price:.4f} | Poly YES: {p_price:.4f}")
        print(f"  Gross edge: {result['gross_edge']:.4f}")
        print(f"  Kalshi fee: {result['kalshi_fee']:.4f}")
        print(f"  Poly fee:   {result['poly_fee']:.4f}")
        print(f"  Total fees: {result['total_fees']:.4f}")
        print(f"  Net edge:   {result['net_edge_after_fees']:+.4f}")
        print(f"  Direction:  {result['direction']}")
        print(f"  Action:     {result['action_kalshi']} Kalshi | {result['action_poly']} Polymarket")
        print(f"  Profitable: {'YES' if result['is_profitable'] else 'NO'}")
        exit(0)

    arb = CrossPlatformArbitrage(kalshi_mode=args.mode)
    opps = []

    if args.scan or not any([args.fomc, args.btc, args.cpi, args.elections]):
        opps = arb.scan_cross_platform()
    if args.fomc:
        opps = arb.scan_fomc()
    if args.btc:
        opps = arb.scan_bitcoin()
    if args.cpi:
        opps = arb.scan_cpi()
    if args.elections:
        opps = arb.scan_elections()

    if args.json:
        print(json.dumps(opps, indent=2, default=str))
    elif args.report or opps:
        print(arb.format_report(opps))

    if opps:
        path = arb.save_results(opps)
        print(f"\nResults saved: {path}")
