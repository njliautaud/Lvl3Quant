"""
Polymarket CLOB Client for Prediction Markets Trading

Uses the Gamma API for market discovery (no auth needed) and
the CLOB API for orderbooks and trading.

Features:
- Market scanning with financial/macro filters
- LLM-powered fair value estimation
- Edge detection vs current prices
- Order placement via py-clob-client (requires API keys)

No auth required for reading market data.
Set POLYMARKET_API_KEY / POLYMARKET_SECRET / POLYMARKET_PASSPHRASE / POLYMARKET_PRIVATE_KEY
for trading.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger("polymarket_client")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Fee constants
TAKER_FEE = 0.02   # 2% taker fee
MAKER_FEE = 0.0    # 0% maker fee


# ---------------------------------------------------------------------------
# Rate Limiter (max 10 requests/second)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, max_per_second: float = 10.0):
        self.max_per_second = max_per_second
        self.min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


_rate_limiter = RateLimiter(max_per_second=10.0)


# ---------------------------------------------------------------------------
# Proxy-aware session factory
# ---------------------------------------------------------------------------

def create_proxy_session(
    proxy_host: str = None,
    proxy_port: int = None,
    proxy_type: str = None,
    proxy_url: str = None,
) -> requests.Session:
    """
    Create a requests.Session that routes through a SOCKS5/HTTP proxy.

    Proxy config priority:
      1. Explicit proxy_url parameter
      2. Explicit host/port/type parameters
      3. POLYMARKET_PROXY env var (full URL)
      4. POLYMARKET_PROXY_HOST + POLYMARKET_PROXY_PORT + POLYMARKET_PROXY_TYPE env vars
      5. No proxy (direct connection)
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })

    # Resolve proxy URL
    resolved_proxy = proxy_url or os.environ.get("POLYMARKET_PROXY")

    if not resolved_proxy:
        host = proxy_host or os.environ.get("POLYMARKET_PROXY_HOST")
        port = proxy_port or os.environ.get("POLYMARKET_PROXY_PORT")
        ptype = proxy_type or os.environ.get("POLYMARKET_PROXY_TYPE", "socks5h")

        if host and port:
            resolved_proxy = f"{ptype}://{host}:{port}"

    if resolved_proxy:
        session.proxies.update({
            "http": resolved_proxy,
            "https": resolved_proxy,
        })
        logger.info(f"Proxy configured: {resolved_proxy.split('@')[-1]}")

    return session

# Categories of interest for financial trading
FINANCIAL_KEYWORDS = [
    "fed", "federal reserve", "interest rate", "fomc", "powell",
    "inflation", "cpi", "pce", "gdp", "recession", "unemployment",
    "btc", "bitcoin", "ethereum", "crypto", "defi",
    "spx", "s&p", "nasdaq", "dow jones", "stock market", "stocks",
    "tariff", "trade war", "dollar", "treasury", "yield", "bond",
    "gold", "oil", "energy", "commodities",
    "china", "economy", "economic",
    "trump", "congress", "senate", "budget", "deficit",
    "warsh", "bessent", "fed chair",
]

# Markets in the "sweet spot" for trading: price 10-90% (genuine uncertainty)
MIN_PRICE = 0.08
MAX_PRICE = 0.92

# Minimum volume threshold (USD)
MIN_VOLUME = 500_000


def _get(url: str, params: dict = None, retries: int = 3, session: requests.Session = None) -> dict:
    """HTTP GET with rate limiting and exponential backoff retry."""
    _rate_limiter.wait()
    getter = session or requests
    for attempt in range(retries):
        try:
            resp = getter.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                backoff = min(30, (2 ** attempt) + 0.5)
                logger.debug(f"Retry {attempt+1}/{retries} for {url}: {e} (backoff {backoff:.1f}s)")
                time.sleep(backoff)
                continue
            raise
    return {}


class PolymarketScanner:
    """
    Scans Polymarket for financial/macro opportunities.
    No authentication required for reading data.
    """

    def __init__(self):
        self.gamma_base = GAMMA_API
        self.clob_base = CLOB_API

    # ------------------------------------------------------------------
    # Market Discovery
    # ------------------------------------------------------------------

    def get_active_markets(self, limit: int = 200, order: str = "volumeNum") -> list:
        """
        Fetch active markets ordered by volume.
        Returns raw market dicts from Gamma API.

        If limit > 100, paginates automatically using the offset parameter.
        The Gamma API returns at most 100 results per request.
        """
        PAGE_SIZE = 100
        all_markets = []
        offset = 0

        while len(all_markets) < limit:
            fetch_size = min(PAGE_SIZE, limit - len(all_markets))
            data = _get(
                f"{self.gamma_base}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": fetch_size,
                    "offset": offset,
                    "order": order,
                    "ascending": "false",
                },
            )

            if isinstance(data, list):
                page = data
            elif isinstance(data, dict):
                page = data.get("data", data) if not isinstance(data.get("data"), dict) else []
                if isinstance(page, dict):
                    page = []
            else:
                page = []

            if not page:
                break

            all_markets.extend(page)
            offset += len(page)

            # If we got fewer than requested, there are no more
            if len(page) < fetch_size:
                break

            # Small delay between pages to be polite
            if len(all_markets) < limit:
                time.sleep(0.2)

        return all_markets

    def filter_financial(self, markets: list) -> list:
        """Filter markets matching financial/macro keywords."""
        result = []
        for m in markets:
            q = m.get("question", "").lower()
            if any(kw in q for kw in FINANCIAL_KEYWORDS):
                result.append(m)
        return result

    def _parse_prices(self, prices_raw) -> list:
        """Parse outcomePrices which may be a JSON string or list."""
        if not prices_raw:
            return []
        if isinstance(prices_raw, str):
            try:
                import json as _json
                prices_raw = _json.loads(prices_raw)
            except Exception:
                return []
        try:
            return [float(p) for p in prices_raw]
        except (ValueError, TypeError):
            return []

    def filter_tradeable(self, markets: list) -> list:
        """
        Keep only markets with genuine uncertainty (price 5-95%)
        and sufficient volume.
        """
        tradeable = []
        for m in markets:
            prices = self._parse_prices(m.get("outcomePrices", []))
            if not prices:
                continue

            yes_price = prices[0] if prices else 0.5
            vol = float(m.get("volumeNum", 0))

            if (
                MIN_PRICE <= yes_price <= MAX_PRICE
                and vol >= MIN_VOLUME
            ):
                m["_yes_price"] = yes_price
                m["_volume"] = vol
                tradeable.append(m)
        return tradeable

    def get_orderbook(self, token_id: str) -> dict:
        """
        Get CLOB orderbook for a specific token.
        Returns {bids: [{price, size}], asks: [{price, size}]}
        """
        try:
            data = _get(f"{self.clob_base}/book", params={"token_id": token_id})
            return data
        except Exception as e:
            logger.warning(f"Orderbook fetch failed for {token_id}: {e}")
            return {}

    def get_midpoint(self, token_id: str) -> float | None:
        """Get midpoint price for a token."""
        try:
            data = _get(f"{self.clob_base}/midpoint", params={"token_id": token_id})
            mid = data.get("mid")
            return float(mid) if mid is not None else None
        except Exception:
            return None

    def get_spread(self, token_id: str) -> dict:
        """Get bid/ask spread for a token."""
        try:
            data = _get(f"{self.clob_base}/spread", params={"token_id": token_id})
            return data
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Combinatorial Mispricing Detection
    # ------------------------------------------------------------------

    def find_mispricings(self, markets: list) -> list:
        """
        Find logical contradictions in multi-outcome Polymarket events.

        Only considers negRisk markets grouped by negRiskMarketID, which
        represent mutually exclusive outcomes (e.g., "Who wins the 2028
        election?").  Slug-prefix grouping was removed — it produced false
        positives by grouping correlated (non-exclusive) markets.

        Returns list of dicts: {type, group, probability_sum, direction,
                                deviation, edge_per_contract, markets}
        """
        # Group by negRiskMarketID (only proper multi-outcome events)
        groups = {}
        for m in markets:
            if not m.get("negRisk", False):
                continue
            nrid = m.get("negRiskMarketID", "")
            if not nrid:
                continue
            groups.setdefault(nrid, []).append(m)

        mispricings = []
        for nrid, group_markets in groups.items():
            if len(group_markets) < 2:
                continue

            # Sum YES prices across the mutually exclusive group
            total_prob = 0.0
            valid_count = 0
            for gm in group_markets:
                prices = self._parse_prices(gm.get("outcomePrices", []))
                if prices:
                    total_prob += prices[0]
                    valid_count += 1

            if valid_count < 2 or total_prob < 0.1:
                continue

            deviation = total_prob - 1.0
            # Only flag OVERPRICED groups (sum > 1.0).
            # "Underpriced" groups (sum < 1.0) are almost always just
            # incomplete data — we don't see all markets in the group.
            # Need meaningful deviation (>3%) to overcome fees.
            if deviation > 0.03:
                direction = "overpriced"
                edge_per = deviation / valid_count
                # Maker fee is 0%, taker fee is 2% per leg
                fee_per = TAKER_FEE * 0.25  # Avg fee at mid-price
                net_edge = edge_per - fee_per

                if net_edge > 0:
                    # Use first market's question as group label
                    label = group_markets[0].get("question", nrid[:20])[:50]
                    mispricings.append({
                        "type": "probability_sum",
                        "group": label,
                        "negRiskMarketID": nrid,
                        "probability_sum": round(total_prob, 4),
                        "direction": direction,
                        "deviation": round(deviation, 4),
                        "edge_per_contract": round(net_edge, 4),
                        "markets": valid_count,
                    })

        mispricings.sort(key=lambda x: x["edge_per_contract"], reverse=True)
        return mispricings

    # ------------------------------------------------------------------
    # Price Analysis
    # ------------------------------------------------------------------

    def calculate_edge(
        self,
        fair_value: float,
        market_price: float,
        side: str = "YES",
    ) -> dict:
        """
        Calculate edge on a trade.

        Args:
            fair_value: Our estimate of true probability (0-1)
            market_price: Current market YES price (0-1)
            side: 'YES' or 'NO' - which side to trade

        Returns:
            dict with edge, net_edge, action, entry_price
        """
        if side == "YES":
            # Buying YES at market_price, fair value is fair_value
            raw_edge = fair_value - market_price
            entry_price = market_price
        else:
            # Buying NO at (1 - market_price), fair value is (1 - fair_value)
            raw_edge = (1 - fair_value) - (1 - market_price)
            entry_price = 1 - market_price

        # Taker fee reduces edge
        fee = TAKER_FEE * entry_price * (1 - entry_price)
        net_edge = raw_edge - fee

        return {
            "side": side,
            "entry_price": round(entry_price, 4),
            "fair_value": round(fair_value, 4),
            "raw_edge": round(raw_edge, 4),
            "fee": round(fee, 4),
            "net_edge": round(net_edge, 4),
            "action": (
                f"BUY_{side}" if net_edge > 0 else "NO_EDGE"
            ),
        }

    def best_edge(self, fair_value: float, market_price: float) -> dict:
        """Return the better of YES or NO edge."""
        yes_edge = self.calculate_edge(fair_value, market_price, "YES")
        no_edge = self.calculate_edge(fair_value, market_price, "NO")
        return yes_edge if yes_edge["net_edge"] >= no_edge["net_edge"] else no_edge

    # ------------------------------------------------------------------
    # Full Scan
    # ------------------------------------------------------------------

    def scan(
        self,
        limit: int = 200,
        min_net_edge: float = 0.02,
        fair_value_fn=None,
    ) -> list:
        """
        Full scan pipeline:
        1. Fetch top markets by volume
        2. Filter for financial topics
        3. Filter for tradeable (uncertain) markets
        4. Score edge using fair_value_fn
        5. Return sorted opportunities

        Args:
            limit: Max markets to fetch
            min_net_edge: Minimum net edge to include
            fair_value_fn: Callable(market_dict) -> float (0-1)
                           If None, uses market price as fair value (no edge expected)

        Returns:
            List of opportunity dicts sorted by net_edge descending
        """
        logger.info("Scanning Polymarket for financial opportunities...")
        markets = self.get_active_markets(limit=limit)
        logger.info(f"Fetched {len(markets)} markets")

        financial = self.filter_financial(markets)
        logger.info(f"Financial markets: {len(financial)}")

        tradeable = self.filter_tradeable(financial)
        logger.info(f"Tradeable (price 8-92%, vol >${MIN_VOLUME/1e6:.1f}M): {len(tradeable)}")

        opportunities = []
        for m in tradeable:
            yes_price = m["_yes_price"]
            question = m.get("question", "")

            # Get fair value
            if fair_value_fn:
                try:
                    fv = fair_value_fn(m)
                except Exception as e:
                    logger.warning(f"Fair value fn failed for '{question[:40]}': {e}")
                    fv = yes_price  # Fallback: no edge
            else:
                fv = yes_price  # No model = no edge

            edge = self.best_edge(fv, yes_price)

            if edge["net_edge"] < min_net_edge:
                continue

            # Get live orderbook spread
            tokens = m.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
            token_id = yes_token.get("token_id") if yes_token else None

            opp = {
                "source": "polymarket",
                "question": question,
                "market_slug": m.get("slug", m.get("marketSlug", m.get("market_slug", ""))),
                "end_date": m.get("endDate", m.get("end_date_iso", ""))[:10],
                "volume": m["_volume"],
                "market_price": yes_price,
                "fair_value": fv,
                "token_id": token_id,
                **edge,
            }
            opportunities.append(opp)

        opportunities.sort(key=lambda x: x["net_edge"], reverse=True)
        logger.info(f"Opportunities with edge >{min_net_edge}: {len(opportunities)}")
        return opportunities


class PolymarketTrader:
    """
    Polymarket trading client. Requires API credentials for order placement.
    Uses py-clob-client for authenticated operations.
    """

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        api_passphrase: str = None,
        private_key: str = None,
        mode: str = "readonly",
    ):
        self.mode = mode  # "readonly" or "live"
        self.api_key = api_key or os.environ.get("POLYMARKET_API_KEY")
        self.api_secret = api_secret or os.environ.get("POLYMARKET_SECRET")
        self.api_passphrase = api_passphrase or os.environ.get("POLYMARKET_PASSPHRASE")
        self.private_key = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY")

        self.scanner = PolymarketScanner()
        self.client = None

        # Risk limits
        self.max_position_per_market = 500   # USDC
        self.max_total_exposure = 5000       # USDC
        self.min_net_edge = 0.02             # 2 cents minimum

        # State
        self.positions = {}
        self.trade_log = []

        if mode == "live" and self.private_key:
            self._init_client()
        else:
            logger.info(
                f"PolymarketTrader in {mode} mode. "
                f"Credentials: {'set' if self.api_key else 'NOT SET'}"
            )

    def _init_client(self):
        """Initialize py-clob-client for trading."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = None
            if self.api_key:
                creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    api_passphrase=self.api_passphrase,
                )

            self.client = ClobClient(
                host=CLOB_API,
                chain_id=CHAIN_ID,
                key=self.private_key,
                creds=creds,
            )
            logger.info("Polymarket CLOB client initialized")
        except Exception as e:
            logger.warning(f"CLOB client init failed: {e}")
            self.client = None

    def scan_for_opportunities(
        self,
        fair_value_fn=None,
        min_net_edge: float = None,
    ) -> list:
        """
        Wrapper around PolymarketScanner.scan with risk filtering.
        """
        min_edge = min_net_edge or self.min_net_edge
        return self.scanner.scan(
            min_net_edge=min_edge,
            fair_value_fn=fair_value_fn,
        )

    def get_portfolio_summary(self) -> dict:
        """Get current positions and P&L. Requires authentication."""
        if not self.client:
            return {"error": "Not authenticated", "mode": self.mode}
        try:
            balance = self.client.get_balance()
            positions = self.client.get_positions()
            return {
                "balance_usdc": balance,
                "positions": positions,
                "trade_count": len(self.trade_log),
            }
        except Exception as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket scanner")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Min net edge")
    parser.add_argument("--limit", type=int, default=200, help="Markets to scan")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    scanner = PolymarketScanner()
    opps = scanner.scan(limit=args.limit, min_net_edge=args.min_edge)

    if args.json:
        print(json.dumps(opps))
    else:
        print(f"\n{'='*70}")
        print(f"POLYMARKET SCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*70}")
        if not opps:
            print("No opportunities found (all markets fairly priced)")
        for o in opps:
            print(
                f"\n{o['question'][:70]}"
                f"\n  End: {o['end_date']} | Vol: ${o['volume']/1e6:.1f}M"
                f"\n  Market: {o['market_price']:.3f} | FV: {o['fair_value']:.3f}"
                f"\n  Edge: {o['raw_edge']:+.3f} → Net: {o['net_edge']:+.3f} | {o['action']}"
            )
        print(f"\n{len(opps)} opportunities total")


if __name__ == "__main__":
    main()
