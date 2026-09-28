"""
Prediction Markets Historical Data Collector

Collects and stores historical snapshots from Polymarket and Kalshi for:
- Backtesting strategy performance
- Calibration (our predictions vs actual outcomes)
- Economic calendar event tracking

Data is stored in JSONL format for streaming-friendly appends.

Usage:
    python data_collector.py --polymarket-snapshot  # Collect one Polymarket snapshot
    python data_collector.py --kalshi-brackets      # Collect today's Kalshi brackets
    python data_collector.py --resolved             # Pull recently resolved markets
    python data_collector.py --calendar             # Print upcoming economic events
    python data_collector.py --all                  # Run all collection tasks
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("data_collector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# --- Paths ------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- API Constants ----------------------------------------------------------

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_LIVE_BASE = "https://trading-api.kalshi.com/trade-api/v2"

# Polymarket financial keyword filter (same as polymarket_client.py)
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

# 2026 Economic Calendar — all key macro dates
# Source: Federal Reserve, BLS, BEA official calendars
ECONOMIC_CALENDAR_2026 = [
    # ---- FOMC meetings (announcement on second day) ----
    {"event": "FOMC", "date": "2026-01-29", "description": "January FOMC meeting — rate decision 2:00 PM ET"},
    {"event": "FOMC", "date": "2026-03-18", "description": "March FOMC meeting — rate decision 2:00 PM ET"},
    {"event": "FOMC", "date": "2026-05-06", "description": "May FOMC meeting — rate decision 2:00 PM ET"},
    {"event": "FOMC", "date": "2026-06-17", "description": "June FOMC meeting — rate decision + SEP projections 2:00 PM ET"},
    {"event": "FOMC", "date": "2026-07-29", "description": "July FOMC meeting — rate decision 2:00 PM ET"},
    {"event": "FOMC", "date": "2026-09-16", "description": "September FOMC meeting — rate decision + SEP projections 2:00 PM ET"},
    {"event": "FOMC", "date": "2026-10-28", "description": "October FOMC meeting — rate decision 2:00 PM ET"},
    {"event": "FOMC", "date": "2026-12-16", "description": "December FOMC meeting — rate decision + SEP projections 2:00 PM ET"},

    # ---- CPI releases (BLS, 8:30 AM ET) ----
    # Approximate dates — BLS releases ~2 weeks after reference month end
    {"event": "CPI", "date": "2026-01-14", "description": "CPI December 2025 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-02-11", "description": "CPI January 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-03-11", "description": "CPI February 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-04-10", "description": "CPI March 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-05-13", "description": "CPI April 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-06-10", "description": "CPI May 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-07-14", "description": "CPI June 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-08-12", "description": "CPI July 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-09-10", "description": "CPI August 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-10-14", "description": "CPI September 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-11-12", "description": "CPI October 2026 — BLS 8:30 AM ET"},
    {"event": "CPI", "date": "2026-12-10", "description": "CPI November 2026 — BLS 8:30 AM ET"},

    # ---- NFP releases (BLS, first Friday of month, 8:30 AM ET) ----
    {"event": "NFP", "date": "2026-01-09", "description": "NFP December 2025 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-02-06", "description": "NFP January 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-03-06", "description": "NFP February 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-04-03", "description": "NFP March 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-05-08", "description": "NFP April 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-06-05", "description": "NFP May 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-07-10", "description": "NFP June 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-08-07", "description": "NFP July 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-09-04", "description": "NFP August 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-10-02", "description": "NFP September 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-11-06", "description": "NFP October 2026 — BLS 8:30 AM ET"},
    {"event": "NFP", "date": "2026-12-04", "description": "NFP November 2026 — BLS 8:30 AM ET"},

    # ---- GDP releases (BEA, advance estimate ~1 month after quarter end) ----
    {"event": "GDP", "date": "2026-01-29", "description": "GDP Q4 2025 advance estimate — BEA 8:30 AM ET"},
    {"event": "GDP", "date": "2026-04-29", "description": "GDP Q1 2026 advance estimate — BEA 8:30 AM ET"},
    {"event": "GDP", "date": "2026-07-30", "description": "GDP Q2 2026 advance estimate — BEA 8:30 AM ET"},
    {"event": "GDP", "date": "2026-10-29", "description": "GDP Q3 2026 advance estimate — BEA 8:30 AM ET"},

    # ---- PCE releases (Fed's preferred inflation gauge, end of month) ----
    {"event": "PCE", "date": "2026-01-30", "description": "PCE December 2025 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-02-27", "description": "PCE January 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-03-27", "description": "PCE February 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-04-30", "description": "PCE March 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-05-29", "description": "PCE April 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-06-26", "description": "PCE May 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-07-31", "description": "PCE June 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-08-28", "description": "PCE July 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-09-25", "description": "PCE August 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-10-30", "description": "PCE September 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-11-25", "description": "PCE October 2026 — BEA 8:30 AM ET"},
    {"event": "PCE", "date": "2026-12-23", "description": "PCE November 2026 — BEA 8:30 AM ET"},
]


# ---------------------------------------------------------------------------
# HTTP Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, headers: dict = None,
         retries: int = 3, backoff: float = 1.5) -> dict:
    """HTTP GET with exponential backoff retry."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = backoff ** attempt
                logger.warning(f"GET {url} failed (attempt {attempt + 1}/{retries}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                logger.error(f"GET {url} failed after {retries} attempts: {e}")
                raise
    return {}


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _write_json(path: Path, data) -> None:
    """Write/overwrite a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# MarketDataCollector
# ---------------------------------------------------------------------------

class MarketDataCollector:
    """
    Collects historical market data from Polymarket and Kalshi.

    All collected data is written to the `data/` subdirectory as JSONL or JSON.
    JSONL files are append-only: safe to call multiple times per day.
    """

    def __init__(
        self,
        kalshi_mode: str = "demo",
        kalshi_api_key: str = None,
        kalshi_private_key_path: str = None,
    ):
        self.kalshi_mode = kalshi_mode
        self.kalshi_base = KALSHI_LIVE_BASE if kalshi_mode == "live" else KALSHI_DEMO_BASE
        self.kalshi_api_key = kalshi_api_key or os.environ.get("KALSHI_API_KEY")
        self.kalshi_private_key_path = (
            kalshi_private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        )
        logger.info(
            f"MarketDataCollector initialized — Kalshi mode: {kalshi_mode}, "
            f"API key: {'set' if self.kalshi_api_key else 'NOT SET'}"
        )

    # -----------------------------------------------------------------------
    # Internal: Polymarket helpers
    # -----------------------------------------------------------------------

    def _parse_prices(self, prices_raw) -> list:
        """Parse outcomePrices which may be a JSON string or list."""
        if not prices_raw:
            return []
        if isinstance(prices_raw, str):
            try:
                prices_raw = json.loads(prices_raw)
            except Exception:
                return []
        try:
            return [float(p) for p in prices_raw]
        except (ValueError, TypeError):
            return []

    def _is_financial(self, question: str) -> bool:
        """Return True if the market question matches financial keywords."""
        q = question.lower()
        return any(kw in q for kw in FINANCIAL_KEYWORDS)

    def _polymarket_snapshot_record(self, market: dict, timestamp: str) -> dict:
        """Build a standardised snapshot record from a Gamma API market dict."""
        prices = self._parse_prices(market.get("outcomePrices", []))
        yes_price = prices[0] if prices else None

        end_date_raw = market.get("endDate") or market.get("end_date_iso") or ""
        end_date = end_date_raw[:10] if end_date_raw else None

        return {
            "timestamp": timestamp,
            "question": market.get("question", ""),
            "market_slug": (
                market.get("slug")
                or market.get("marketSlug")
                or market.get("market_slug")
                or ""
            ),
            "condition_id": market.get("conditionId", market.get("condition_id", "")),
            "yes_price": yes_price,
            "volume": float(market.get("volumeNum", 0) or 0),
            "liquidity": float(market.get("liquidityNum", 0) or 0),
            "end_date": end_date,
            "resolved": bool(market.get("closed") or market.get("resolved")),
            "resolution": None,  # Populated by collect_resolved_markets
        }

    # -----------------------------------------------------------------------
    # Polymarket: snapshot
    # -----------------------------------------------------------------------

    def collect_polymarket_snapshot(self, limit: int = 200) -> list:
        """
        Fetch one snapshot of the top Polymarket financial markets by volume
        and append to today's JSONL file.

        Returns:
            List of snapshot records that were written.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = DATA_DIR / f"polymarket_snapshots_{today}.jsonl"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(f"Collecting Polymarket snapshot (limit={limit})...")

        try:
            data = _get(
                f"{GAMMA_API}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "order": "volumeNum",
                    "ascending": "false",
                },
            )
        except Exception as e:
            logger.error(f"Polymarket snapshot fetch failed: {e}")
            return []

        markets = data if isinstance(data, list) else data.get("data", [])
        logger.info(f"Fetched {len(markets)} markets from Gamma API")

        records = []
        for m in markets:
            question = m.get("question", "")
            if not self._is_financial(question):
                continue

            record = self._polymarket_snapshot_record(m, timestamp)
            if record["yes_price"] is None:
                continue  # Skip binary markets with no price data

            _append_jsonl(out_path, record)
            records.append(record)

        logger.info(
            f"Snapshot written: {len(records)} financial markets → {out_path}"
        )
        return records

    # -----------------------------------------------------------------------
    # Polymarket: resolved markets
    # -----------------------------------------------------------------------

    def collect_resolved_markets(self, days_back: int = 30) -> list:
        """
        Fetch recently resolved Polymarket markets with their outcomes.
        Appends to `data/polymarket_resolved.jsonl`.

        Args:
            days_back: How many days of resolutions to pull.

        Returns:
            List of resolved market records written.
        """
        out_path = DATA_DIR / "polymarket_resolved.jsonl"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            f"Collecting resolved Polymarket markets (last {days_back} days)..."
        )

        # Gamma API supports filtering by closed=true
        try:
            data = _get(
                f"{GAMMA_API}/markets",
                params={
                    "closed": "true",
                    "limit": 500,
                    "order": "volumeNum",
                    "ascending": "false",
                },
            )
        except Exception as e:
            logger.error(f"Resolved markets fetch failed: {e}")
            return []

        markets = data if isinstance(data, list) else data.get("data", [])
        logger.info(f"Fetched {len(markets)} closed markets from Gamma API")

        records = []
        for m in markets:
            question = m.get("question", "")
            if not self._is_financial(question):
                continue

            # Check within our time window
            end_date_raw = (
                m.get("endDate") or m.get("end_date_iso") or ""
            )
            if end_date_raw and end_date_raw[:19] < cutoff_str[:19]:
                continue  # Resolved before our window

            # Extract resolution
            resolution = None
            # Gamma API: outcomePrices for resolved markets are [1.0, 0.0] or [0.0, 1.0]
            prices = self._parse_prices(m.get("outcomePrices", []))
            if prices and len(prices) >= 2:
                if prices[0] >= 0.99:
                    resolution = "YES"
                elif prices[0] <= 0.01:
                    resolution = "NO"
            # Also check explicit resolution field
            res_raw = m.get("resolution") or m.get("resolvedOutcome")
            if res_raw:
                res_str = str(res_raw).strip().upper()
                if res_str in ("YES", "1", "TRUE", "WIN"):
                    resolution = "YES"
                elif res_str in ("NO", "0", "FALSE", "LOSS"):
                    resolution = "NO"

            record = self._polymarket_snapshot_record(m, timestamp)
            record["resolved"] = True
            record["resolution"] = resolution
            record["end_date"] = end_date_raw[:10] if end_date_raw else None

            _append_jsonl(out_path, record)
            records.append(record)

        logger.info(
            f"Resolved markets written: {len(records)} → {out_path}"
        )
        return records

    # -----------------------------------------------------------------------
    # Polymarket: calibration dataset
    # -----------------------------------------------------------------------

    def build_calibration_dataset(self) -> list:
        """
        Pair our daily snapshots with actual resolutions to build a
        calibration dataset.

        For each snapshot where the market has since resolved, we record:
        - our_price_at_snapshot: the YES price we observed
        - actual_resolution: YES or NO
        - horizon_days: days between snapshot and resolution

        Reads all `polymarket_snapshots_*.jsonl` and `polymarket_resolved.jsonl`.
        Writes `data/calibration_dataset.json`.

        Returns:
            List of calibration records.
        """
        resolved_path = DATA_DIR / "polymarket_resolved.jsonl"
        out_path = DATA_DIR / "calibration_dataset.json"

        # Load resolved markets keyed by condition_id or market_slug
        resolved_index: dict = {}
        if resolved_path.exists():
            with open(resolved_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = rec.get("condition_id") or rec.get("market_slug")
                    if key and rec.get("resolution"):
                        resolved_index[key] = rec

        logger.info(f"Loaded {len(resolved_index)} resolved markets for calibration")

        # Load all snapshot files
        snapshot_files = sorted(DATA_DIR.glob("polymarket_snapshots_*.jsonl"))
        calibration: list = []

        for snap_file in snapshot_files:
            with open(snap_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        snap = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    key = snap.get("condition_id") or snap.get("market_slug")
                    if not key:
                        continue

                    resolved = resolved_index.get(key)
                    if not resolved or not resolved.get("resolution"):
                        continue

                    # Calculate horizon in days
                    try:
                        snap_dt = datetime.fromisoformat(
                            snap["timestamp"].replace("Z", "+00:00")
                        )
                        end_str = resolved.get("end_date")
                        if end_str:
                            end_dt = datetime.fromisoformat(end_str + "T00:00:00+00:00")
                            horizon_days = max(0, (end_dt - snap_dt).days)
                        else:
                            horizon_days = None
                    except Exception:
                        horizon_days = None

                    calibration.append({
                        "question": snap.get("question", ""),
                        "market_slug": snap.get("market_slug", ""),
                        "snapshot_timestamp": snap.get("timestamp"),
                        "our_price_at_snapshot": snap.get("yes_price"),
                        "volume_at_snapshot": snap.get("volume"),
                        "end_date": resolved.get("end_date"),
                        "actual_resolution": resolved["resolution"],
                        "horizon_days": horizon_days,
                        "correct": (
                            (snap.get("yes_price", 0.5) >= 0.5 and resolved["resolution"] == "YES")
                            or (snap.get("yes_price", 0.5) < 0.5 and resolved["resolution"] == "NO")
                        ),
                    })

        # Write calibration dataset
        _write_json(out_path, calibration)
        logger.info(
            f"Calibration dataset built: {len(calibration)} records → {out_path}"
        )

        # Print quick summary
        if calibration:
            correct = sum(1 for r in calibration if r.get("correct"))
            logger.info(
                f"Directional accuracy: {correct}/{len(calibration)} = "
                f"{correct/len(calibration):.1%}"
            )

        return calibration

    # -----------------------------------------------------------------------
    # Internal: Kalshi helpers
    # -----------------------------------------------------------------------

    def _kalshi_headers(self) -> dict:
        """Build auth headers for Kalshi REST calls. Returns empty dict if no key."""
        # Full RSA-PSS signing is handled by the kalshi-python SDK.
        # For read-only data collection we don't need authentication on public endpoints.
        headers = {"Content-Type": "application/json"}
        return headers

    def _parse_kalshi_bracket(self, ticker: str, subtitle: str) -> tuple:
        """
        Parse floor and cap from a Kalshi SPX bracket.

        Subtitle examples:
          '5800 to 5825'
          '5875 or above'
          '5750 or below'
        """
        if not subtitle:
            # Fall back to ticker parsing: INXD-26MAR03-B5800-5825
            try:
                parts = ticker.split("-B")[1].split("-")
                return float(parts[0].replace(",", "")), float(parts[1].replace(",", ""))
            except Exception:
                return 0.0, 0.0

        subtitle_lower = subtitle.lower()
        try:
            if "or above" in subtitle_lower:
                val = float(subtitle.split()[0].replace(",", ""))
                return val, float("inf")
            elif "or below" in subtitle_lower:
                val = float(subtitle.split()[0].replace(",", ""))
                return float("-inf"), val
            elif " to " in subtitle:
                parts = subtitle.split(" to ")
                return float(parts[0].replace(",", "")), float(parts[1].replace(",", ""))
        except Exception as e:
            logger.warning(f"Could not parse bracket subtitle '{subtitle}': {e}")

        return 0.0, 0.0

    # -----------------------------------------------------------------------
    # Kalshi: bracket snapshots
    # -----------------------------------------------------------------------

    def collect_kalshi_brackets(self, date: str = None) -> list:
        """
        Collect SPX bracket (INXD) prices from Kalshi for a given date.
        Appends to `data/kalshi_brackets_{YYYY-MM-DD}.jsonl`.

        Args:
            date: 'YYYY-MM-DD' format. Defaults to today.

        Returns:
            List of bracket snapshot records written.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        out_path = DATA_DIR / f"kalshi_brackets_{date}.jsonl"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Kalshi event ticker format: INXD-YYMMMDD (e.g. INXD-26MAR03)
        dt = datetime.strptime(date, "%Y-%m-%d")
        event_ticker = f"INXD-{dt.strftime('%y%b%d').upper()}"

        logger.info(
            f"Collecting Kalshi brackets for {date} (event: {event_ticker})..."
        )

        # Attempt public REST endpoint (no auth required for market data)
        url = f"{self.kalshi_base}/events/{event_ticker}"
        try:
            data = _get(url, headers=self._kalshi_headers())
        except Exception as e:
            logger.error(f"Kalshi bracket fetch failed for {event_ticker}: {e}")
            return []

        event = data.get("event", data)
        markets_list = event.get("markets", [])

        # If the event response embeds market objects, use them.
        # Otherwise, fetch each market individually.
        if markets_list and isinstance(markets_list[0], str):
            # List of ticker strings — fetch each
            full_markets = []
            for ticker in markets_list:
                try:
                    market_data = _get(
                        f"{self.kalshi_base}/markets/{ticker}",
                        headers=self._kalshi_headers(),
                    )
                    full_markets.append(market_data.get("market", market_data))
                    time.sleep(0.15)  # Rate limit courtesy
                except Exception as e:
                    logger.warning(f"Failed to fetch market {ticker}: {e}")
            markets_list = full_markets

        records = []
        for market in markets_list:
            ticker = market.get("ticker", "")
            subtitle = market.get("subtitle", "")
            floor, cap = self._parse_kalshi_bracket(ticker, subtitle)

            # Prices: prefer _dollars fields (subpenny precision, FixedPointDollars strings).
            # Legacy integer-cent fields removed after March 5, 2026; fallback kept for
            # processing historical cached data only.
            def _price(dollars_field, cents_field):
                v = market.get(dollars_field)
                if v is not None:
                    return float(v)
                v = market.get(cents_field)
                if v is not None:
                    c = float(v)
                    return c / 100.0 if c > 1.0 else c
                return None

            record = {
                "timestamp": timestamp,
                "event_ticker": event_ticker,
                "ticker": ticker,
                "subtitle": subtitle,
                "floor": floor if floor != float("-inf") else None,
                "cap": cap if cap != float("inf") else None,
                "yes_bid": _price("yes_bid_dollars", "yes_bid"),
                "yes_ask": _price("yes_ask_dollars", "yes_ask"),
                "no_bid": _price("no_bid_dollars", "no_bid"),
                "no_ask": _price("no_ask_dollars", "no_ask"),
                "volume": market.get("volume"),
                "open_interest": market.get("open_interest"),
                "status": market.get("status"),
                "spx_at_snapshot": None,  # Caller should enrich this if available
                "resolved": market.get("status") in ("finalized", "determined"),
                "resolution": None,  # Populated by collect_bracket_resolutions
            }

            _append_jsonl(out_path, record)
            records.append(record)

        logger.info(
            f"Kalshi brackets written: {len(records)} contracts → {out_path}"
        )
        return records

    def enrich_bracket_spx(self, records: list, spx_price: float) -> list:
        """
        Add the SPX price at snapshot time to bracket records (in-memory only).
        Call this after collect_kalshi_brackets if you have the current SPX price.

        Args:
            records: List of records from collect_kalshi_brackets.
            spx_price: Current SPX index price.

        Returns:
            Same list with 'spx_at_snapshot' populated.
        """
        for r in records:
            r["spx_at_snapshot"] = spx_price
        return records

    # -----------------------------------------------------------------------
    # Kalshi: bracket resolutions
    # -----------------------------------------------------------------------

    def collect_bracket_resolutions(self, days_back: int = 30) -> list:
        """
        Fetch resolved Kalshi SPX bracket contracts with their outcomes.
        Appends to `data/kalshi_resolved.jsonl`.

        Iterates over past trading days and fetches settled INXD events.

        Args:
            days_back: How many calendar days to look back.

        Returns:
            List of resolution records written.
        """
        out_path = DATA_DIR / "kalshi_resolved.jsonl"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            f"Collecting Kalshi bracket resolutions (last {days_back} days)..."
        )

        today = datetime.now()
        records = []

        for i in range(1, days_back + 1):
            day = today - timedelta(days=i)
            # Skip weekends — SPX doesn't trade Sat/Sun
            if day.weekday() >= 5:
                continue

            date_str = day.strftime("%Y-%m-%d")
            event_ticker = f"INXD-{day.strftime('%y%b%d').upper()}"

            try:
                data = _get(
                    f"{self.kalshi_base}/events/{event_ticker}",
                    headers=self._kalshi_headers(),
                )
            except Exception as e:
                logger.debug(f"No data for {event_ticker}: {e}")
                time.sleep(0.3)
                continue

            event = data.get("event", data)
            markets_list = event.get("markets", [])

            if not markets_list:
                time.sleep(0.3)
                continue

            # Fetch individual markets if only tickers returned
            if markets_list and isinstance(markets_list[0], str):
                full_markets = []
                for ticker in markets_list:
                    try:
                        md = _get(
                            f"{self.kalshi_base}/markets/{ticker}",
                            headers=self._kalshi_headers(),
                        )
                        full_markets.append(md.get("market", md))
                        time.sleep(0.1)
                    except Exception:
                        pass
                markets_list = full_markets

            for market in markets_list:
                status = market.get("status", "")
                if status not in ("finalized", "determined", "settled"):
                    continue

                ticker = market.get("ticker", "")
                subtitle = market.get("subtitle", "")
                floor, cap = self._parse_kalshi_bracket(ticker, subtitle)

                # Parse resolution
                resolution = None
                result = market.get("result") or market.get("yes_sub_title") or ""
                if result:
                    result_lower = str(result).lower()
                    if result_lower in ("yes", "true", "1", "winner"):
                        resolution = "YES"
                    elif result_lower in ("no", "false", "0", "loser"):
                        resolution = "NO"

                record = {
                    "timestamp": timestamp,
                    "event_ticker": event_ticker,
                    "ticker": ticker,
                    "subtitle": subtitle,
                    "date": date_str,
                    "floor": floor if floor != float("-inf") else None,
                    "cap": cap if cap != float("inf") else None,
                    "final_yes_price": market.get("last_price_dollars")
                        or (market.get("last_price", 0) / 100 if market.get("last_price") else None),
                    "volume": market.get("volume"),
                    "open_interest": market.get("open_interest"),
                    "resolved": True,
                    "resolution": resolution,
                }

                _append_jsonl(out_path, record)
                records.append(record)

            time.sleep(0.3)  # Polite rate limiting

        logger.info(
            f"Kalshi resolutions written: {len(records)} contracts → {out_path}"
        )
        return records

    # -----------------------------------------------------------------------
    # Economic calendar
    # -----------------------------------------------------------------------

    def get_upcoming_events(self, days_ahead: int = 30) -> list:
        """
        Return upcoming macro events from the 2026 economic calendar.

        Args:
            days_ahead: How many days forward to look.

        Returns:
            Sorted list of event dicts within the window.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        cutoff_dt = datetime.now() + timedelta(days=days_ahead)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d")

        upcoming = [
            e for e in ECONOMIC_CALENDAR_2026
            if today_str <= e["date"] <= cutoff_str
        ]
        upcoming.sort(key=lambda x: x["date"])
        return upcoming

    def save_calendar(self) -> Path:
        """
        Write the full 2026 economic calendar to
        `data/economic_calendar_2026.json`.

        Returns:
            Path to the written file.
        """
        out_path = DATA_DIR / "economic_calendar_2026.json"
        _write_json(out_path, ECONOMIC_CALENDAR_2026)
        logger.info(
            f"Economic calendar saved: {len(ECONOMIC_CALENDAR_2026)} events → {out_path}"
        )
        return out_path

    def print_upcoming_events(self, days_ahead: int = 30) -> None:
        """Print a formatted table of upcoming events."""
        events = self.get_upcoming_events(days_ahead)
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"\n{'='*70}")
        print(f"UPCOMING MACRO EVENTS — next {days_ahead} days from {today}")
        print(f"{'='*70}")
        if not events:
            print("No events in this window.")
        for e in events:
            delta = (
                datetime.strptime(e["date"], "%Y-%m-%d") - datetime.now()
            ).days
            print(f"  {e['date']} (T-{delta:2d}d)  [{e['event']:<5}]  {e['description']}")
        print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prediction Markets Data Collector"
    )
    parser.add_argument(
        "--polymarket-snapshot",
        action="store_true",
        help="Collect one Polymarket snapshot",
    )
    parser.add_argument(
        "--kalshi-brackets",
        action="store_true",
        help="Collect today's Kalshi SPX brackets",
    )
    parser.add_argument(
        "--resolved",
        action="store_true",
        help="Collect recently resolved markets from both platforms",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Build calibration dataset from snapshots + resolutions",
    )
    parser.add_argument(
        "--calendar",
        action="store_true",
        help="Print upcoming economic events and save calendar JSON",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all collection tasks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max markets to fetch from Polymarket (default: 200)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Days back for resolved market collection (default: 30)",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=30,
        help="Days ahead for economic calendar display (default: 30)",
    )
    parser.add_argument(
        "--spx",
        type=float,
        default=None,
        help="Current SPX price (enriches bracket snapshot records)",
    )
    parser.add_argument(
        "--kalshi-mode",
        default="demo",
        choices=["demo", "live"],
        help="Kalshi API mode (default: demo)",
    )

    args = parser.parse_args()

    collector = MarketDataCollector(kalshi_mode=args.kalshi_mode)

    run_all = args.all
    did_something = False

    if run_all or args.polymarket_snapshot:
        records = collector.collect_polymarket_snapshot(limit=args.limit)
        print(f"Polymarket snapshot: {len(records)} records written")
        did_something = True

    if run_all or args.kalshi_brackets:
        records = collector.collect_kalshi_brackets()
        if args.spx:
            records = collector.enrich_bracket_spx(records, args.spx)
        print(f"Kalshi brackets: {len(records)} records written")
        did_something = True

    if run_all or args.resolved:
        poly_records = collector.collect_resolved_markets(days_back=args.days_back)
        kalshi_records = collector.collect_bracket_resolutions(days_back=args.days_back)
        print(
            f"Resolved: {len(poly_records)} Polymarket, "
            f"{len(kalshi_records)} Kalshi records written"
        )
        did_something = True

    if run_all or args.calibration:
        cal = collector.build_calibration_dataset()
        print(f"Calibration dataset: {len(cal)} records written")
        did_something = True

    if run_all or args.calendar:
        collector.save_calendar()
        collector.print_upcoming_events(days_ahead=args.days_ahead)
        did_something = True

    if not did_something:
        parser.print_help()


if __name__ == "__main__":
    main()
