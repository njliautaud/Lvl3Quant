#!/usr/bin/env python3
"""
Insider Signal Scorer
Scores insider buying activity and generates trading signals.
Academic evidence: insider BUYING during dips is the strongest predictor of future returns.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Paths
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
FILINGS_DIR = BASE_DIR / "data" / "insider_filings"
STATE_DIR = BASE_DIR / "state"
LOG_DIR = BASE_DIR / "logs" / "insider_signals"
OUTPUT_FILE = STATE_DIR / "insider_signals.json"
UNIVERSE_FILE = BASE_DIR / "data" / "quality_universe.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scorer.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Signal thresholds
CLUSTER_MIN_INSIDERS = 3
CLUSTER_WINDOW_DAYS = 14
DIP_THRESHOLD_PCT = 5.0  # Stock must be >5% below 20-day high
LARGE_PURCHASE_USD = 500_000
LARGE_PURCHASE_MULTIPLIER = 10  # 10x typical purchase size

# Role confidence base scores
ROLE_BASE_CONFIDENCE = {
    "CEO": 80,
    "CFO": 80,
    "COO": 70,
    "CTO": 65,
    "VP": 55,
    "Director": 60,
    "Other": 40,
}

# Transaction codes to include (open market purchases only)
VALID_PURCHASE_CODES = {"P"}

# Transaction codes to exclude
EXCLUDED_CODES = {"G", "J", "W", "Z", "F"}  # Gifts, other, will, trust, tax withholding


def load_universe() -> list[str]:
    with open(UNIVERSE_FILE) as f:
        return json.load(f)["tickers"]


def load_transactions() -> list[dict]:
    """Load all parsed insider transactions."""
    master_file = FILINGS_DIR / "all_transactions.json"
    if master_file.exists():
        with open(master_file) as f:
            return json.load(f)

    # Fallback: load per-ticker files
    all_txns = []
    for f in FILINGS_DIR.glob("*_form4.json"):
        with open(f) as fh:
            all_txns.extend(json.load(fh))
    return all_txns


def get_price_data(ticker: str) -> dict | None:
    """
    Get recent price data for a ticker to calculate dip metrics.
    Uses Yahoo Finance v8 API (no key needed).
    Returns dict with current_price, high_20d, pct_from_high.
    """
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"range": "1mo", "interval": "1d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if not resp.ok:
            return None

        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None

        quotes = result[0]
        highs = quotes.get("indicators", {}).get("quote", [{}])[0].get("high", [])
        closes = quotes.get("indicators", {}).get("quote", [{}])[0].get("close", [])

        # Filter None values
        valid_highs = [h for h in highs if h is not None]
        valid_closes = [c for c in closes if c is not None]

        if not valid_highs or not valid_closes:
            return None

        current_price = valid_closes[-1]
        high_20d = max(valid_highs[-20:]) if len(valid_highs) >= 20 else max(valid_highs)
        pct_from_high = ((high_20d - current_price) / high_20d) * 100

        return {
            "current_price": round(current_price, 2),
            "high_20d": round(high_20d, 2),
            "pct_from_high": round(pct_from_high, 2),
            "is_dipping": pct_from_high >= DIP_THRESHOLD_PCT,
        }
    except Exception as e:
        log.warning(f"Price fetch failed for {ticker}: {e}")
        return None


def compute_typical_purchase_size(transactions: list[dict], ticker: str) -> float:
    """Compute median purchase value for a ticker to identify unusually large buys."""
    values = [
        t["total_value"]
        for t in transactions
        if t["ticker"] == ticker and t["transaction_code"] == "P" and t["total_value"] > 0
    ]
    if not values:
        return 100_000  # Default assumption
    values.sort()
    mid = len(values) // 2
    return values[mid] if len(values) % 2 == 1 else (values[mid - 1] + values[mid]) / 2


def detect_cluster_buys(purchases: list[dict], window_days: int = CLUSTER_WINDOW_DAYS) -> list[dict]:
    """
    Detect cluster buys: 3+ different insiders buying within a window.
    This is the STRONGEST insider signal per academic literature.
    """
    clusters = []

    # Group by ticker
    by_ticker = defaultdict(list)
    for p in purchases:
        by_ticker[p["ticker"]].append(p)

    for ticker, txns in by_ticker.items():
        # Sort by date
        txns.sort(key=lambda x: x["transaction_date"])

        # Sliding window to find clusters
        for i, anchor in enumerate(txns):
            anchor_date = datetime.strptime(anchor["transaction_date"], "%Y-%m-%d")
            window_end = anchor_date + timedelta(days=window_days)

            window_txns = []
            unique_insiders = set()
            for t in txns:
                t_date = datetime.strptime(t["transaction_date"], "%Y-%m-%d")
                if anchor_date <= t_date <= window_end:
                    window_txns.append(t)
                    unique_insiders.add(t["insider_name"])

            if len(unique_insiders) >= CLUSTER_MIN_INSIDERS:
                total_value = sum(t["total_value"] for t in window_txns)
                clusters.append({
                    "ticker": ticker,
                    "signal_type": "CLUSTER_BUY",
                    "num_insiders": len(unique_insiders),
                    "insiders": list(unique_insiders),
                    "window_start": anchor["transaction_date"],
                    "window_end": min(t["transaction_date"] for t in window_txns
                                     if t["insider_name"] == list(unique_insiders)[-1]),
                    "total_value": round(total_value, 2),
                    "transactions": window_txns,
                    "confidence": min(95, 85 + (len(unique_insiders) - 3) * 5),
                })

    # Deduplicate overlapping clusters (keep the one with most insiders)
    if clusters:
        clusters.sort(key=lambda x: (-x["num_insiders"], x["window_start"]))
        deduped = []
        seen_windows = set()
        for c in clusters:
            key = (c["ticker"], c["window_start"][:7])  # Dedup by ticker + month
            if key not in seen_windows:
                seen_windows.add(key)
                deduped.append(c)
        clusters = deduped

    return clusters


def score_individual_purchases(purchases: list[dict], all_transactions: list[dict]) -> list[dict]:
    """Score individual insider purchases based on role, size, and market context."""
    signals = []
    tickers = set(p["ticker"] for p in purchases)

    # Get price data for dip detection
    price_cache = {}
    for ticker in tickers:
        price_cache[ticker] = get_price_data(ticker)

    for p in purchases:
        ticker = p["ticker"]
        role = p.get("role_bucket", "Other")
        value = p.get("total_value", 0)

        if value <= 0:
            continue

        # Base confidence from role
        base_conf = ROLE_BASE_CONFIDENCE.get(role, 40)

        # Dip amplifier
        price_data = price_cache.get(ticker)
        is_dip = False
        dip_pct = 0
        if price_data and price_data["is_dipping"]:
            is_dip = True
            dip_pct = price_data["pct_from_high"]
            # Deeper dip = stronger signal
            base_conf += min(10, int(dip_pct / 2))

        # Large purchase amplifier
        typical_size = compute_typical_purchase_size(all_transactions, ticker)
        is_large = value >= LARGE_PURCHASE_USD or (typical_size > 0 and value >= typical_size * LARGE_PURCHASE_MULTIPLIER)
        if is_large:
            base_conf += 10

        # Cap confidence
        confidence = min(95, base_conf)

        # Determine signal type
        if role in ("CEO", "CFO") and is_dip:
            signal_type = "C_SUITE_DIP_BUY"
        elif role in ("CEO", "CFO"):
            signal_type = "C_SUITE_BUY"
        elif role == "Director" and is_dip:
            signal_type = "DIRECTOR_DIP_BUY"
        elif role == "Director":
            signal_type = "DIRECTOR_BUY"
        elif is_dip:
            signal_type = "INSIDER_DIP_BUY"
        else:
            signal_type = "INSIDER_BUY"

        # Only keep signals with meaningful confidence
        if confidence >= 50:
            signal = {
                "ticker": ticker,
                "insider_name": p["insider_name"],
                "title": p["title"],
                "role_bucket": role,
                "shares": p["shares"],
                "price": p["price"],
                "total_value": p["total_value"],
                "transaction_date": p["transaction_date"],
                "filing_date": p["filing_date"],
                "signal_type": signal_type,
                "confidence": confidence,
                "is_dip": is_dip,
                "dip_pct": dip_pct,
                "is_large_purchase": is_large,
                "current_price": price_data["current_price"] if price_data else None,
            }
            signals.append(signal)

    return signals


def run(lookback_days: int = 90):
    """Main scoring pipeline."""
    log.info("=== Insider Signal Scorer ===")

    all_transactions = load_transactions()
    if not all_transactions:
        log.warning("No transactions found. Run insider_filing_fetcher.py first.")
        return []

    log.info(f"Loaded {len(all_transactions)} total transactions")

    # Filter to purchases only
    purchases = [
        t for t in all_transactions
        if t.get("transaction_code") in VALID_PURCHASE_CODES
        and t.get("transaction_code") not in EXCLUDED_CODES
    ]
    log.info(f"Found {len(purchases)} open-market purchases")

    # Filter to recent purchases for signal generation
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    recent_purchases = [p for p in purchases if p.get("transaction_date", "") >= cutoff]
    log.info(f"Recent purchases (last {lookback_days} days): {len(recent_purchases)}")

    # 1. Detect cluster buys (strongest signal)
    cluster_signals = detect_cluster_buys(recent_purchases)
    log.info(f"Cluster buy signals: {len(cluster_signals)}")

    # 2. Score individual purchases
    individual_signals = score_individual_purchases(recent_purchases, all_transactions)
    log.info(f"Individual buy signals: {len(individual_signals)}")

    # Combine and sort by confidence
    all_signals = []

    for cs in cluster_signals:
        all_signals.append({
            "ticker": cs["ticker"],
            "signal_type": cs["signal_type"],
            "confidence": cs["confidence"],
            "num_insiders": cs["num_insiders"],
            "insiders": cs["insiders"],
            "total_value": cs["total_value"],
            "window_start": cs["window_start"],
            "description": f"{cs['num_insiders']} insiders bought ${cs['total_value']:,.0f} worth within {CLUSTER_WINDOW_DAYS} days",
        })

    for sig in individual_signals:
        all_signals.append({
            "ticker": sig["ticker"],
            "signal_type": sig["signal_type"],
            "confidence": sig["confidence"],
            "insider_name": sig["insider_name"],
            "title": sig["title"],
            "role_bucket": sig["role_bucket"],
            "shares": sig["shares"],
            "price": sig["price"],
            "total_value": sig["total_value"],
            "transaction_date": sig["transaction_date"],
            "is_dip": sig["is_dip"],
            "dip_pct": sig.get("dip_pct", 0),
            "is_large_purchase": sig["is_large_purchase"],
            "current_price": sig.get("current_price"),
            "description": f"{sig['insider_name']} ({sig['role_bucket']}) bought {sig['shares']:,.0f} shares @ ${sig['price']:.2f} = ${sig['total_value']:,.0f}",
        })

    all_signals.sort(key=lambda x: -x["confidence"])

    # Save output
    output = {
        "generated_at": datetime.now().isoformat(),
        "lookback_days": lookback_days,
        "total_signals": len(all_signals),
        "cluster_signals": len(cluster_signals),
        "individual_signals": len(individual_signals),
        "signals": all_signals,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Saved {len(all_signals)} signals to {OUTPUT_FILE}")

    # Print summary
    print("\n=== TOP INSIDER SIGNALS ===\n")
    for sig in all_signals[:20]:
        emoji = "***" if sig["confidence"] >= 85 else "**" if sig["confidence"] >= 70 else "*"
        print(f"{emoji} [{sig['confidence']}%] {sig['ticker']} - {sig['signal_type']}: {sig['description']}")

    return all_signals


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score insider buying signals")
    parser.add_argument("--lookback", type=int, default=90, help="Days to look back for signals (default: 90)")
    args = parser.parse_args()
    run(lookback_days=args.lookback)
