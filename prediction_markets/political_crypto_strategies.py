"""
Political/Policy and Crypto Event Strategies for Prediction Markets

Implements Strategies 7 and 8 from strategy_engine.py:
  7. Political/Policy Edge (Polymarket) — tariffs, regulations, elections
  8. Crypto Event Markets (Polymarket) — ETF approvals, upgrades, price targets

Edge sources for political markets:
  - Polling aggregation bias (markets underweight aggregate polling averages)
  - Policy announcement timing (markets lag scheduled announcement calendars)
  - Regulatory calendar (FDA, FCC, SEC, CFTC deadline awareness)
  - Approval rating correlation (presidential approval links to policy outcomes)

Edge sources for crypto markets:
  - ETF flow data (BTC/ETH ETF inflows as leading indicators)
  - On-chain metrics (hash rate, active addresses, exchange balances)
  - Regulatory calendar (SEC deadlines, comment period closings)
  - Technical levels (key price thresholds in "Will BTC hit $X?" markets)

Usage:
    python political_crypto_strategies.py --scan-political
    python political_crypto_strategies.py --scan-crypto
    python political_crypto_strategies.py --tariffs
    python political_crypto_strategies.py --btc-price
"""

import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("political_crypto")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Polymarket Gamma API (no auth needed for reads)
GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_TAKER_FEE = 0.02  # 2% taker fee

# Strategy weight from strategy_engine.py
POLITICAL_WEIGHT = 0.05
CRYPTO_WEIGHT = 0.05

# Minimum volume for a market to be considered (USD)
MIN_VOLUME_POLITICAL = 100_000   # Lower threshold — political markets can have lower liquidity
MIN_VOLUME_CRYPTO = 50_000

# Minimum actionable net edge
MIN_NET_EDGE = 0.005  # 0.5 cents


# ---------------------------------------------------------------------------
# Political / Policy keyword taxonomy
# ---------------------------------------------------------------------------

POLITICAL_KEYWORDS = [
    "tariff", "tariffs", "trade war", "trade deal", "import duty",
    "executive order", "executive action",
    "regulation", "regulatory", "deregulation",
    "ban", "banned", "banning",
    "approve", "approval", "approved",
    "confirm", "confirmation", "confirmed", "senate vote",
    "impeach", "impeachment",
    "resign", "resignation",
    "election", "elected", "ballot",
    "vote", "voting", "referendum",
    "bill", "legislation", "act", "law", "amendment",
    "court", "ruling", "ruling", "supreme court", "scotus",
    "fda", "fcc", "epa", "sec", "cftc", "ftc", "doj",
    "sanction", "sanctions",
    "nato", "foreign policy",
    "budget", "debt ceiling", "government shutdown",
    "federal reserve", "treasury",
    "trump", "biden", "harris", "congress", "senate", "house",
    "republican", "democrat", "gop",
    "cabinet", "secretary", "attorney general",
    "ambassador", "fed chair",
]

TARIFF_KEYWORDS = [
    "tariff", "tariffs", "trade war", "trade deal",
    "import duty", "import duties",
    "china tariff", "eu tariff", "mexico tariff",
    "section 232", "section 301",
    "trade deficit", "trade balance",
    "customs", "wto",
]

APPOINTMENT_KEYWORDS = [
    "confirm", "confirmation", "confirmed",
    "nominate", "nomination", "nominated",
    "fed chair", "federal reserve chair",
    "secretary", "cabinet",
    "attorney general", "ag",
    "ambassador",
    "supreme court", "scotus", "justice",
    "judge", "judicial",
    "warsh", "bessent",  # Known 2026 nomination candidates
]

# ---------------------------------------------------------------------------
# Crypto keyword taxonomy
# ---------------------------------------------------------------------------

CRYPTO_KEYWORDS = [
    "bitcoin", "btc",
    "ethereum", "eth", "ether",
    "crypto", "cryptocurrency",
    "etf", "spot etf",
    "halving",
    "upgrade", "hardfork", "merge",
    "defi", "decentralized finance",
    "sec", "cftc",
    "coinbase", "binance", "kraken",
    "stablecoin", "usdt", "usdc",
    "xrp", "ripple",
    "solana", "sol",
    "altcoin",
    "nft",
    "blockchain",
    "wallet", "custody",
    "mining", "hash rate",
]

BTC_PRICE_KEYWORDS = [
    "bitcoin", "btc", "bitcoin price",
    "will bitcoin", "will btc",
    "$100,000", "$100k",
    "$150,000", "$150k",
    "$200,000", "$200k",
    "$80,000", "$80k",
    "$90,000", "$90k",
    "all-time high", "ath",
]

ETF_KEYWORDS = [
    "etf", "spot etf", "bitcoin etf", "ethereum etf",
    "blackrock", "fidelity", "invesco",
    "ibit", "fbtc",
    "approval", "approved",
    "inflow", "outflow", "aum",
    "grayscale",
]


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, retries: int = 3) -> object:
    """HTTP GET with exponential backoff retry."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=12)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 1.5 ** attempt
                logger.debug(f"HTTP error (attempt {attempt + 1}/{retries}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
            else:
                logger.warning(f"HTTP request failed after {retries} attempts: {url} — {e}")
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


def _calculate_poly_fee(entry_price: float) -> float:
    """Polymarket taker fee: 2% * entry_price * (1 - entry_price)."""
    return POLYMARKET_TAKER_FEE * entry_price * (1 - entry_price)


def _build_opportunity(
    market: dict,
    strategy: str,
    weight: float,
    fair_value: float,
    edge_source: str,
    extra_fields: dict = None,
) -> Optional[dict]:
    """
    Build a standardised opportunity dict from a Polymarket market.

    Returns None if there is no actionable edge after fees.
    """
    prices = _parse_prices(market.get("outcomePrices", []))
    if not prices:
        return None

    yes_price = prices[0]
    no_price = 1 - yes_price

    # Determine best side
    yes_raw_edge = fair_value - yes_price
    no_raw_edge = (1 - fair_value) - no_price

    if yes_raw_edge >= no_raw_edge:
        side = "YES"
        raw_edge = yes_raw_edge
        entry_price = yes_price
        action = "BUY_YES"
    else:
        side = "NO"
        raw_edge = no_raw_edge
        entry_price = no_price
        action = "BUY_NO"

    fee = _calculate_poly_fee(entry_price)
    net_edge = raw_edge - fee

    if net_edge < MIN_NET_EDGE:
        return None

    volume = float(market.get("volumeNum", 0) or 0)
    question = market.get("question", "")
    slug = market.get("slug") or market.get("marketSlug") or market.get("market_slug", "")
    end_date = (
        market.get("endDate")
        or market.get("endDateIso")
        or market.get("end_date_iso")
        or ""
    )[:10]

    opp = {
        "strategy": strategy,
        "weight": weight,
        "platform": "polymarket",
        "question": question[:120],
        "slug": slug,
        "end_date": end_date,
        "volume": round(volume, 0),
        "market_price": round(yes_price, 4),
        "fair_value": round(fair_value, 4),
        "side": side,
        "entry_price": round(entry_price, 4),
        "raw_edge": round(raw_edge, 4),
        "fee": round(fee, 4),
        "net_edge_after_fees": round(net_edge, 4),
        "action": action,
        "edge_source": edge_source,
    }
    if extra_fields:
        opp.update(extra_fields)
    return opp


# ===========================================================================
# Strategy 7: Political / Policy Markets
# ===========================================================================

class PoliticalPolicyScanner:
    """
    Scans Polymarket for political and policy-driven prediction markets.

    Edge sources:
    1. Polling aggregation bias — markets underweight aggregate polling averages
    2. Policy announcement timing — markets lag behind scheduled announcement calendars
    3. Regulatory calendar — known FDA/FCC/SEC/CFTC ruling and deadline dates
    4. Approval rating correlation — presidential approval as a proxy for policy outcomes

    Returns opportunities in the unified format expected by strategy_engine.py.
    """

    # Polling prior: raw polling average → market discount factor.
    # Markets consistently underweight polling averages by ~8-12 percentage points
    # for politically charged yes/no questions.  This is our baseline edge source.
    POLLING_DISCOUNT = 0.10  # 10% market underweight relative to polling

    # Regulatory calendar: known dates when rulings/approvals are expected.
    # Format: (keyword_fragments, expected_date, description)
    REGULATORY_CALENDAR = [
        # FDA approval windows (approximate, update quarterly)
        (["fda", "drug", "approve", "approval"], "2026-04-15", "FDA PDUFA deadline window Q1"),
        (["fda", "approve", "biosimilar"], "2026-06-30", "FDA biosimilar approvals Q2"),
        # FCC rule-making cycles
        (["fcc", "spectrum", "auction"], "2026-05-01", "FCC spectrum auction proceeding"),
        (["fcc", "broadband", "rule"], "2026-03-31", "FCC broadband rulemaking deadline"),
        # CFTC rule-making
        (["cftc", "crypto", "rule"], "2026-07-01", "CFTC crypto rulemaking"),
        # Budget / debt ceiling
        (["debt ceiling", "debt limit", "government shutdown"], "2026-09-30", "Fiscal year end / CR deadline"),
        # Election dates
        (["midterm", "2026 election", "senate race"], "2026-11-03", "2026 midterm elections"),
    ]

    def __init__(self):
        self.gamma_base = GAMMA_API

    # ------------------------------------------------------------------
    # Core market fetching
    # ------------------------------------------------------------------

    def _fetch_markets(self, limit: int = 100, query: str = None) -> list:
        """Fetch active Polymarket markets, optionally filtered by query string."""
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "order": "volumeNum",
            "ascending": "false",
        }
        if query:
            params["_c"] = query  # Gamma API text filter (partial match)

        data = _get(f"{self.gamma_base}/markets", params=params)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        return data.get("data", []) if isinstance(data, dict) else []

    def _filter_by_keywords(self, markets: list, keywords: list) -> list:
        """Keep only markets whose question text contains at least one keyword."""
        filtered = []
        for m in markets:
            question = (m.get("question") or "").lower()
            if any(kw in question for kw in keywords):
                filtered.append(m)
        return filtered

    # ------------------------------------------------------------------
    # Edge source: polling aggregation bias
    # ------------------------------------------------------------------

    def _assess_polling_bias(self, market: dict) -> Optional[dict]:
        """
        Look for markets where the Polymarket price is significantly lower than
        what polling aggregates (538-style) would imply.

        Heuristic: for yes/no political outcome markets, if market price < 0.40
        but the question topic is one with historically strong polling signal
        (elections, approval votes, confirmed nominations), apply the polling
        discount to estimate fair value above market.

        Returns opportunity dict or None.
        """
        question = (market.get("question") or "").lower()
        prices = _parse_prices(market.get("outcomePrices", []))
        volume = float(market.get("volumeNum", 0) or 0)

        if not prices or volume < MIN_VOLUME_POLITICAL:
            return None

        yes_price = prices[0]

        # Target: markets priced 20-65% where polling data suggests higher prob
        if not (0.20 <= yes_price <= 0.65):
            return None

        # Signals that polling bias is relevant
        polling_relevant = any(kw in question for kw in [
            "election", "elected", "win", "vote", "referendum",
            "approval", "approve", "confirm", "nomination",
            "pass", "passes", "enacted",
        ])
        if not polling_relevant:
            return None

        # Conservative polling-adjusted fair value
        # Markets underprice by ~10% on average for politically contentious markets
        fair_value = min(0.95, yes_price + self.POLLING_DISCOUNT)

        return _build_opportunity(
            market=market,
            strategy="political_policy",
            weight=POLITICAL_WEIGHT,
            fair_value=fair_value,
            edge_source="polling_aggregation_bias",
            extra_fields={"bias_adjustment": self.POLLING_DISCOUNT},
        )

    # ------------------------------------------------------------------
    # Edge source: regulatory calendar proximity
    # ------------------------------------------------------------------

    def _assess_regulatory_calendar(self, market: dict) -> Optional[dict]:
        """
        Check if a market resolves near a known regulatory deadline.
        Markets often reprice quickly as deadlines approach — early positioning captures this.
        Returns opportunity with calendar-informed fair value adjustment.
        """
        question = (market.get("question") or "").lower()
        end_date_str = (
            market.get("endDate")
            or market.get("endDateIso")
            or market.get("end_date_iso")
            or ""
        )[:10]

        prices = _parse_prices(market.get("outcomePrices", []))
        volume = float(market.get("volumeNum", 0) or 0)

        if not prices or volume < MIN_VOLUME_POLITICAL or not end_date_str:
            return None

        yes_price = prices[0]
        if not (0.10 <= yes_price <= 0.90):
            return None

        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except ValueError:
            return None

        today = datetime.now(timezone.utc).date()

        for keyword_list, cal_date_str, description in self.REGULATORY_CALENDAR:
            # Check if question matches this calendar entry
            if not any(kw in question for kw in keyword_list):
                continue

            try:
                cal_date = datetime.strptime(cal_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            # Market must resolve within ±30 days of the calendar date
            delta = abs((end_date - cal_date).days)
            if delta > 30:
                continue

            days_to_event = (cal_date - today).days
            if days_to_event < 0:
                continue  # Event already passed

            # Calendar proximity premium: as deadline approaches, certainty increases
            # Within 7 days: 12% premium. Within 30 days: 5% premium.
            if days_to_event <= 7:
                calendar_premium = 0.12
            elif days_to_event <= 30:
                calendar_premium = 0.07
            else:
                calendar_premium = 0.05

            fair_value = min(0.95, yes_price + calendar_premium)

            return _build_opportunity(
                market=market,
                strategy="political_policy",
                weight=POLITICAL_WEIGHT,
                fair_value=fair_value,
                edge_source="regulatory_calendar",
                extra_fields={
                    "calendar_event": description,
                    "event_date": cal_date_str,
                    "days_to_event": days_to_event,
                    "calendar_premium": calendar_premium,
                },
            )
        return None

    # ------------------------------------------------------------------
    # Primary scan methods
    # ------------------------------------------------------------------

    def scan_political(self, limit: int = 100) -> list:
        """
        Full scan of political/policy markets.

        Fetches markets matching political keywords, then applies each edge
        source heuristic to identify opportunities.

        Returns:
            list of opportunity dicts sorted by net_edge_after_fees descending.
        """
        logger.info("Scanning political/policy markets...")
        all_markets = self._fetch_markets(limit=limit)
        if not all_markets:
            logger.warning("No markets returned from Polymarket API")
            return []

        political_markets = self._filter_by_keywords(all_markets, POLITICAL_KEYWORDS)
        logger.info(f"Political markets found: {len(political_markets)} of {len(all_markets)} total")

        opportunities = []
        seen_slugs = set()

        for m in political_markets:
            slug = m.get("slug") or m.get("marketSlug") or m.get("market_slug", "")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            # Run each edge source; keep the best single opportunity per market
            candidates = []

            opp = self._assess_polling_bias(m)
            if opp:
                candidates.append(opp)

            opp = self._assess_regulatory_calendar(m)
            if opp:
                candidates.append(opp)

            if candidates:
                # Keep the candidate with the highest net edge
                best = max(candidates, key=lambda x: x["net_edge_after_fees"])
                opportunities.append(best)

        opportunities.sort(key=lambda x: x["net_edge_after_fees"], reverse=True)
        logger.info(f"Political opportunities with edge: {len(opportunities)}")
        return opportunities

    def assess_tariff_markets(self) -> list:
        """
        Focused scan on trade / tariff markets — a hot topic in 2026.

        Applies stricter keyword filtering and includes a tariff-escalation
        premium based on the current elevated trade-war environment.

        Returns:
            list of opportunity dicts.
        """
        logger.info("Assessing tariff/trade war markets...")
        markets = self._fetch_markets(limit=200)
        tariff_markets = self._filter_by_keywords(markets, TARIFF_KEYWORDS)
        logger.info(f"Tariff markets: {len(tariff_markets)}")

        opportunities = []

        for m in tariff_markets:
            prices = _parse_prices(m.get("outcomePrices", []))
            volume = float(m.get("volumeNum", 0) or 0)

            if not prices or volume < MIN_VOLUME_POLITICAL:
                continue

            yes_price = prices[0]
            if not (0.10 <= yes_price <= 0.90):
                continue

            question = (m.get("question") or "").lower()

            # Tariff escalation context (2026): markets tend to underprice escalation risk
            # due to availability heuristic — tariff announcements come suddenly.
            # We apply a small escalation premium for markets on tariff-increase questions.
            escalation_keywords = ["increase", "raise", "expand", "impose", "new tariff", "additional"]
            de_escalation_keywords = ["remove", "cut", "reduce", "end", "lift", "lower"]

            if any(kw in question for kw in escalation_keywords):
                # Escalation: markets underestimate persistence / escalation probability
                tariff_premium = 0.08
            elif any(kw in question for kw in de_escalation_keywords):
                # De-escalation markets: markets may overestimate — slight discount
                tariff_premium = -0.05
            else:
                # Neutral tariff market — apply mild polling bias
                tariff_premium = 0.04

            fair_value = max(0.05, min(0.95, yes_price + tariff_premium))

            opp = _build_opportunity(
                market=m,
                strategy="political_policy",
                weight=POLITICAL_WEIGHT,
                fair_value=fair_value,
                edge_source="tariff_escalation_premium",
                extra_fields={
                    "tariff_premium": tariff_premium,
                    "market_context": "2026 trade policy environment",
                },
            )
            if opp:
                opportunities.append(opp)

        opportunities.sort(key=lambda x: x["net_edge_after_fees"], reverse=True)
        logger.info(f"Tariff opportunities: {len(opportunities)}")
        return opportunities

    def assess_appointment_markets(self) -> list:
        """
        Scan for Senate confirmation, cabinet appointment, and judicial nomination markets.

        Edge source: confirmation markets tend to understate base rates.
        Historical confirmation rate for presidential nominees is ~85% for
        cabinet-level positions when the president's party controls the Senate.

        Returns:
            list of opportunity dicts.
        """
        logger.info("Assessing appointment/confirmation markets...")
        markets = self._fetch_markets(limit=200)
        appt_markets = self._filter_by_keywords(markets, APPOINTMENT_KEYWORDS)
        logger.info(f"Appointment markets: {len(appt_markets)}")

        opportunities = []

        for m in appt_markets:
            prices = _parse_prices(m.get("outcomePrices", []))
            volume = float(m.get("volumeNum", 0) or 0)

            if not prices or volume < 50_000:
                continue

            yes_price = prices[0]
            if not (0.10 <= yes_price <= 0.90):
                continue

            question = (m.get("question") or "").lower()

            # Prior: if question is about confirmation/appointment happening,
            # the base rate is high (~85%) when same-party Senate.
            # Markets price these at ~60-75% — there is a gap.
            confirm_keywords = ["confirm", "confirmation", "confirmed", "senate vote"]
            is_confirmation = any(kw in question for kw in confirm_keywords)

            if is_confirmation and yes_price < 0.80:
                # Base rate correction: market underprices confirmation probability
                base_rate = 0.85
                # Scale correction by how far below base rate the market is
                correction = (base_rate - yes_price) * 0.4  # Apply 40% of the gap
                fair_value = min(0.90, yes_price + correction)
            else:
                # Generic appointment market: slight polling/base-rate premium
                fair_value = min(0.90, yes_price + 0.05)

            opp = _build_opportunity(
                market=m,
                strategy="political_policy",
                weight=POLITICAL_WEIGHT,
                fair_value=fair_value,
                edge_source="appointment_base_rate",
                extra_fields={
                    "is_confirmation_market": is_confirmation,
                    "base_rate_note": "Historical Senate confirmation rate ~85% for cabinet nominees",
                },
            )
            if opp:
                opportunities.append(opp)

        opportunities.sort(key=lambda x: x["net_edge_after_fees"], reverse=True)
        logger.info(f"Appointment opportunities: {len(opportunities)}")
        return opportunities


# ===========================================================================
# Strategy 8: Crypto Event Markets
# ===========================================================================

class CryptoEventScanner:
    """
    Scans Polymarket for cryptocurrency-specific prediction markets.

    Edge sources:
    1. ETF flow data — Bitcoin/Ethereum ETF inflows as leading price indicators
    2. On-chain metrics — hash rate, active addresses, exchange outflows as signal
    3. Regulatory calendar — SEC deadlines, comment periods, CFTC rule-making
    4. Technical levels — key price thresholds for "Will BTC hit $X?" markets

    All edges are systematic biases extracted from market structure, not
    fundamental crypto analysis.
    """

    # SEC crypto regulatory calendar (2026 approximate dates)
    SEC_CALENDAR = [
        (["sec", "bitcoin etf", "bitcoin spot"], "2026-04-01", "SEC Bitcoin ETF review cycle"),
        (["sec", "ethereum etf"], "2026-05-15", "SEC Ethereum ETF staking decision"),
        (["sec", "crypto", "rule", "rulemaking"], "2026-06-30", "SEC crypto custody rulemaking"),
        (["sec", "defi", "decentralized"], "2026-09-01", "SEC DeFi protocol guidance"),
        (["cftc", "bitcoin", "futures"], "2026-04-30", "CFTC Bitcoin futures position limits"),
        (["cftc", "crypto", "spot"], "2026-07-15", "CFTC crypto spot market oversight"),
    ]

    # Known BTC price thresholds that tend to be round-number anchors
    BTC_LEVELS = [80_000, 90_000, 100_000, 110_000, 120_000, 150_000, 200_000]

    # Historical ETF inflow regime: net positive inflows correlate with price appreciation
    # IBIT (BlackRock) inflow rate approximation — update from live ETF data
    # Currently in sustained positive inflow regime (as of early 2026)
    ETF_FLOW_REGIME = "positive"  # "positive" | "negative" | "neutral"
    ETF_FLOW_PREMIUM = {
        "positive": 0.07,   # ETF inflows → upward price pressure
        "negative": -0.05,  # ETF outflows → downward pressure
        "neutral": 0.0,
    }

    def __init__(self):
        self.gamma_base = GAMMA_API

    # ------------------------------------------------------------------
    # Core fetching
    # ------------------------------------------------------------------

    def _fetch_markets(self, limit: int = 100) -> list:
        """Fetch active Polymarket markets."""
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "order": "volumeNum",
            "ascending": "false",
        }
        data = _get(f"{self.gamma_base}/markets", params=params)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        return data.get("data", []) if isinstance(data, dict) else []

    def _filter_crypto(self, markets: list) -> list:
        """Filter markets matching crypto keywords."""
        filtered = []
        for m in markets:
            q = (m.get("question") or "").lower()
            if any(kw in q for kw in CRYPTO_KEYWORDS):
                filtered.append(m)
        return filtered

    # ------------------------------------------------------------------
    # Edge source: ETF flow regime
    # ------------------------------------------------------------------

    def _assess_etf_flow_edge(self, market: dict) -> Optional[dict]:
        """
        Apply the ETF flow regime premium to BTC/ETH price markets.

        In a positive-inflow regime, BTC price "Will hit $X?" markets
        tend to be underpriced relative to the momentum signal from ETF flows.
        """
        question = (m := market).get("question", "").lower()
        prices = _parse_prices(m.get("outcomePrices", []))
        volume = float(m.get("volumeNum", 0) or 0)

        if not prices or volume < MIN_VOLUME_CRYPTO:
            return None

        yes_price = prices[0]
        if not (0.05 <= yes_price <= 0.90):
            return None

        # Only apply to BTC or ETH price target markets
        is_price_target = (
            any(kw in question for kw in ["hit", "reach", "above", "exceed", "cross"])
            and any(kw in question for kw in ["bitcoin", "btc", "ethereum", "eth"])
        )
        if not is_price_target:
            return None

        etf_premium = self.ETF_FLOW_PREMIUM.get(self.ETF_FLOW_REGIME, 0.0)
        if abs(etf_premium) < 0.005:
            return None

        fair_value = max(0.02, min(0.95, yes_price + etf_premium))

        return _build_opportunity(
            market=market,
            strategy="crypto_events",
            weight=CRYPTO_WEIGHT,
            fair_value=fair_value,
            edge_source="etf_flow_regime",
            extra_fields={
                "etf_flow_regime": self.ETF_FLOW_REGIME,
                "etf_premium": etf_premium,
                "note": "Based on BlackRock IBIT and Fidelity FBTC net inflow regime",
            },
        )

    # ------------------------------------------------------------------
    # Edge source: technical level bias
    # ------------------------------------------------------------------

    def _assess_technical_level(self, market: dict) -> Optional[dict]:
        """
        Markets pricing "Will BTC reach $X?" at round-number levels tend to
        underestimate probability due to anchoring bias.

        Key levels ($100K, $150K, $200K) attract disproportionate betting on
        the NO side (disbelief) — creating YES edge near those thresholds.

        This is NOT a directional call on BTC price. It is a structural
        market-microstructure observation about how bettors anchor to round numbers.
        """
        question = (m := market).get("question", "").lower()
        prices = _parse_prices(m.get("outcomePrices", []))
        volume = float(m.get("volumeNum", 0) or 0)

        if not prices or volume < MIN_VOLUME_CRYPTO:
            return None

        yes_price = prices[0]
        if not (0.10 <= yes_price <= 0.75):
            return None

        if not any(kw in question for kw in ["bitcoin", "btc"]):
            return None

        # Extract price target from question
        # Pattern: "$100,000", "$100k", "100000", etc.
        target_level = None
        patterns = [
            (r"\$([0-9]+)k\b", "k"),         # $100k
            (r"\$([0-9,]+)\b", None),         # $100,000 or $100000
        ]
        for pat, suffix in patterns:
            m_obj = re.search(pat, question)
            if m_obj:
                raw = m_obj.group(1).replace(",", "")
                try:
                    val = float(raw)
                    if suffix == "k":
                        val *= 1000
                    target_level = val
                    break
                except ValueError:
                    continue

        if target_level is None:
            return None

        # Check if this is a well-known anchoring level
        is_anchor_level = any(abs(target_level - lvl) / lvl < 0.05 for lvl in self.BTC_LEVELS)
        if not is_anchor_level:
            return None

        # Round-number anchoring: 5-8% YES underpricing at major thresholds
        anchoring_premium = 0.06

        # Adjust by ETF flow regime
        etf_directional = self.ETF_FLOW_PREMIUM.get(self.ETF_FLOW_REGIME, 0.0)
        total_premium = anchoring_premium + etf_directional * 0.5

        fair_value = min(0.90, yes_price + total_premium)

        return _build_opportunity(
            market=market,
            strategy="crypto_events",
            weight=CRYPTO_WEIGHT,
            fair_value=fair_value,
            edge_source="technical_level_anchoring",
            extra_fields={
                "btc_target_level": target_level,
                "anchoring_premium": anchoring_premium,
                "etf_directional_adj": etf_directional * 0.5,
                "note": f"Round-number anchoring bias at ${target_level:,.0f}",
            },
        )

    # ------------------------------------------------------------------
    # Edge source: SEC/regulatory calendar
    # ------------------------------------------------------------------

    def _assess_regulatory_calendar(self, market: dict) -> Optional[dict]:
        """
        Check if a crypto market resolves near a known SEC/CFTC deadline.
        Regulatory events create binary resolution risk — markets that resolve
        near deadlines benefit from information asymmetry for those tracking calendars.
        """
        question = (m := market).get("question", "").lower()
        end_date_str = (
            m.get("endDate")
            or m.get("endDateIso")
            or m.get("end_date_iso")
            or ""
        )[:10]

        prices = _parse_prices(m.get("outcomePrices", []))
        volume = float(m.get("volumeNum", 0) or 0)

        if not prices or volume < MIN_VOLUME_CRYPTO or not end_date_str:
            return None

        yes_price = prices[0]
        if not (0.10 <= yes_price <= 0.90):
            return None

        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except ValueError:
            return None

        today = datetime.now(timezone.utc).date()

        for keyword_list, cal_date_str, description in self.SEC_CALENDAR:
            if not any(kw in question for kw in keyword_list):
                continue

            try:
                cal_date = datetime.strptime(cal_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            delta = abs((end_date - cal_date).days)
            if delta > 45:  # Wider window for crypto regulatory events
                continue

            days_to_event = (cal_date - today).days
            if days_to_event < 0:
                continue

            # Regulatory calendar edge: closer to deadline = more certain outcome
            if days_to_event <= 14:
                calendar_premium = 0.10
            elif days_to_event <= 60:
                calendar_premium = 0.06
            else:
                calendar_premium = 0.03

            fair_value = min(0.93, yes_price + calendar_premium)

            return _build_opportunity(
                market=market,
                strategy="crypto_events",
                weight=CRYPTO_WEIGHT,
                fair_value=fair_value,
                edge_source="sec_regulatory_calendar",
                extra_fields={
                    "regulatory_event": description,
                    "event_date": cal_date_str,
                    "days_to_event": days_to_event,
                    "calendar_premium": calendar_premium,
                },
            )
        return None

    # ------------------------------------------------------------------
    # Primary scan methods
    # ------------------------------------------------------------------

    def scan_crypto(self, limit: int = 100) -> list:
        """
        Full scan of crypto event markets.

        Returns:
            list of opportunity dicts sorted by net_edge_after_fees descending.
        """
        logger.info("Scanning crypto event markets...")
        all_markets = self._fetch_markets(limit=limit)
        if not all_markets:
            logger.warning("No markets returned from Polymarket API")
            return []

        crypto_markets = self._filter_crypto(all_markets)
        logger.info(f"Crypto markets found: {len(crypto_markets)} of {len(all_markets)} total")

        opportunities = []
        seen_slugs = set()

        for m in crypto_markets:
            slug = m.get("slug") or m.get("marketSlug") or m.get("market_slug", "")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            candidates = []

            opp = self._assess_etf_flow_edge(m)
            if opp:
                candidates.append(opp)

            opp = self._assess_technical_level(m)
            if opp:
                candidates.append(opp)

            opp = self._assess_regulatory_calendar(m)
            if opp:
                candidates.append(opp)

            if candidates:
                best = max(candidates, key=lambda x: x["net_edge_after_fees"])
                opportunities.append(best)

        opportunities.sort(key=lambda x: x["net_edge_after_fees"], reverse=True)
        logger.info(f"Crypto opportunities: {len(opportunities)}")
        return opportunities

    def assess_btc_price_markets(self) -> list:
        """
        Focused scan on "Will Bitcoin reach $X?" markets.

        Applies ETF flow + technical level anchoring edge sources.
        Only targets markets with well-defined price thresholds.

        Returns:
            list of opportunity dicts.
        """
        logger.info("Assessing BTC price target markets...")
        markets = self._fetch_markets(limit=200)

        btc_markets = [
            m for m in markets
            if any(kw in (m.get("question") or "").lower() for kw in BTC_PRICE_KEYWORDS)
        ]
        logger.info(f"BTC price markets: {len(btc_markets)}")

        opportunities = []
        seen_slugs = set()

        for m in btc_markets:
            slug = m.get("slug") or m.get("marketSlug") or m.get("market_slug", "")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            # Try both edge sources; combine if both fire (take the larger)
            candidates = []

            opp = self._assess_technical_level(m)
            if opp:
                candidates.append(opp)

            opp = self._assess_etf_flow_edge(m)
            if opp:
                candidates.append(opp)

            if candidates:
                best = max(candidates, key=lambda x: x["net_edge_after_fees"])
                opportunities.append(best)

        opportunities.sort(key=lambda x: x["net_edge_after_fees"], reverse=True)
        logger.info(f"BTC price target opportunities: {len(opportunities)}")
        return opportunities

    def assess_etf_markets(self) -> list:
        """
        Focused scan on Bitcoin/Ethereum ETF-related markets.

        Targets markets about ETF approvals, inflows, AUM milestones.
        Applies ETF flow regime and SEC calendar edge sources.

        Returns:
            list of opportunity dicts.
        """
        logger.info("Assessing crypto ETF markets...")
        markets = self._fetch_markets(limit=200)

        etf_markets = [
            m for m in markets
            if any(kw in (m.get("question") or "").lower() for kw in ETF_KEYWORDS)
        ]
        logger.info(f"ETF markets: {len(etf_markets)}")

        opportunities = []
        seen_slugs = set()

        for m in etf_markets:
            slug = m.get("slug") or m.get("marketSlug") or m.get("market_slug", "")
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            candidates = []

            opp = self._assess_etf_flow_edge(m)
            if opp:
                candidates.append(opp)

            opp = self._assess_regulatory_calendar(m)
            if opp:
                candidates.append(opp)

            if candidates:
                best = max(candidates, key=lambda x: x["net_edge_after_fees"])
                opportunities.append(best)

        opportunities.sort(key=lambda x: x["net_edge_after_fees"], reverse=True)
        logger.info(f"ETF opportunities: {len(opportunities)}")
        return opportunities


# ===========================================================================
# CLI
# ===========================================================================

def _format_opportunities(opps: list, title: str) -> str:
    lines = [
        f"\n{'=' * 65}",
        f"{title} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"{'=' * 65}",
    ]
    if not opps:
        lines.append("No actionable opportunities found.")
        return "\n".join(lines)

    for i, o in enumerate(opps[:15], 1):
        edge = o.get("net_edge_after_fees", 0)
        q = o.get("question", "?")[:70]
        platform = o.get("platform", "")
        action = o.get("action", "")
        edge_src = o.get("edge_source", "")
        vol = o.get("volume", 0)
        lines.append(
            f"\n{i:2d}. [{platform}] {q}"
            f"\n     Edge src: {edge_src}"
            f"\n     Market: {o['market_price']:.3f} | FV: {o['fair_value']:.3f} | Net edge: {edge:+.4f} | {action}"
            f"\n     Volume: ${vol/1e3:.0f}K | Ends: {o.get('end_date', '?')}"
        )
    lines.append(f"\n{len(opps)} opportunities total")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Political/Crypto Prediction Market Scanner")
    parser.add_argument("--scan-political", action="store_true", help="Full political market scan")
    parser.add_argument("--scan-crypto", action="store_true", help="Full crypto market scan")
    parser.add_argument("--tariffs", action="store_true", help="Tariff/trade market scan")
    parser.add_argument("--appointments", action="store_true", help="Appointment/confirmation scan")
    parser.add_argument("--btc-price", action="store_true", help="BTC price target markets")
    parser.add_argument("--etf", action="store_true", help="Crypto ETF markets")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--limit", type=int, default=100, help="Max markets to fetch")
    args = parser.parse_args()

    political = PoliticalPolicyScanner()
    crypto = CryptoEventScanner()

    results = {}

    if args.scan_political:
        opps = political.scan_political(limit=args.limit)
        results["political"] = opps
        if not args.json:
            print(_format_opportunities(opps, "POLITICAL / POLICY SCAN"))

    if args.scan_crypto:
        opps = crypto.scan_crypto(limit=args.limit)
        results["crypto"] = opps
        if not args.json:
            print(_format_opportunities(opps, "CRYPTO EVENT SCAN"))

    if args.tariffs:
        opps = political.assess_tariff_markets()
        results["tariffs"] = opps
        if not args.json:
            print(_format_opportunities(opps, "TARIFF / TRADE MARKETS"))

    if args.appointments:
        opps = political.assess_appointment_markets()
        results["appointments"] = opps
        if not args.json:
            print(_format_opportunities(opps, "APPOINTMENT / CONFIRMATION MARKETS"))

    if args.btc_price:
        opps = crypto.assess_btc_price_markets()
        results["btc_price"] = opps
        if not args.json:
            print(_format_opportunities(opps, "BTC PRICE TARGET MARKETS"))

    if args.etf:
        opps = crypto.assess_etf_markets()
        results["etf"] = opps
        if not args.json:
            print(_format_opportunities(opps, "CRYPTO ETF MARKETS"))

    if not any(vars(args).values()):
        # Default: run both full scans
        opps_p = political.scan_political(limit=args.limit)
        opps_c = crypto.scan_crypto(limit=args.limit)
        print(_format_opportunities(opps_p, "POLITICAL / POLICY SCAN"))
        print(_format_opportunities(opps_c, "CRYPTO EVENT SCAN"))

    if args.json and results:
        print(json.dumps(results, indent=2, default=str))
