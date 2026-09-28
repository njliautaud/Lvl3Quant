"""Standalone Polymarket scanner — called by the JS prediction markets agent."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from polymarket_client import PolymarketScanner


def scan(min_net_edge=0.02, limit=200, fair_values=None):
    """
    Scan Polymarket for financial opportunities.

    Args:
        min_net_edge: Minimum net edge threshold (0-1)
        limit: Max markets to fetch
        fair_values: dict of {market_slug: fair_value} from LLM analysis

    Returns:
        list of opportunity dicts
    """
    scanner = PolymarketScanner()

    fv_fn = None
    if fair_values:
        def fv_fn(market):
            slug = market.get("slug", market.get("marketSlug", market.get("market_slug", "")))
            return fair_values.get(slug, market["_yes_price"])

    return scanner.scan(limit=limit, min_net_edge=min_net_edge, fair_value_fn=fv_fn)


def get_financial_markets(limit=300):
    """Get financial markets including uncertain multi-outcome events."""
    scanner = PolymarketScanner()

    # Get all markets (more pages to find uncertain ones)
    markets = scanner.get_active_markets(limit=limit)
    financial = scanner.filter_financial(markets)

    # Also get ALL active markets and look for uncertain ones
    all_markets = scanner.get_active_markets(limit=500)
    financial_all = scanner.filter_financial(all_markets)

    # Combine and deduplicate
    seen = set()
    combined = []
    for m in financial_all + financial:
        slug = m.get("slug", m.get("marketSlug", m.get("market_slug", "")))
        if slug not in seen:
            seen.add(slug)
            combined.append(m)

    result = []
    for m in combined:
        prices = scanner._parse_prices(m.get("outcomePrices", []))
        slug = m.get("slug", m.get("marketSlug", m.get("market_slug", "")))
        result.append({
            "question": m.get("question", ""),
            "slug": slug,
            "end_date": (m.get("endDate") or m.get("endDateIso") or m.get("end_date_iso") or "")[:10],
            "volume": float(m.get("volumeNum", m.get("volume", 0))),
            "yes_price": prices[0] if prices else None,
            "no_price": prices[1] if len(prices) > 1 else None,
            "tags": [t.get("label", t) if isinstance(t, dict) else t for t in m.get("tags", [])],
        })
    return sorted(result, key=lambda x: x["volume"], reverse=True)


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"

    if cmd == "markets":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        result = get_financial_markets(limit=limit)
        print(json.dumps(result))

    elif cmd == "scan":
        min_edge = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 200

        # If fair values provided as JSON arg
        fair_values = None
        if len(sys.argv) > 4:
            try:
                fair_values = json.loads(sys.argv[4])
            except json.JSONDecodeError:
                pass

        opps = scan(min_net_edge=min_edge, limit=limit, fair_values=fair_values)
        print(json.dumps(opps))

    else:
        print(json.dumps({"error": f"Unknown command: {cmd}. Use 'scan' or 'markets'"}))
