"""
Unified Prediction Market Strategy Engine

Scans across ALL edge sources, not just FOMC:
1. SPX Bracket Volatility Edge (Kalshi) — our vol model vs market prices
2. Favorite-Longshot Bias (Kalshi/Polymarket) — sell overpriced longshots
3. Combinatorial Mispricings (Polymarket) — logical contradictions
4. Cross-Platform Arbitrage — same event, different prices
5. Event-Driven Macro (Kalshi+Polymarket) — FOMC, CPI, NFP, GDP
6. Stale Quote Exploitation (Polymarket) — post-500ms delay removal
7. Political/Policy Edge (Polymarket) — tariffs, regulations, elections
8. Crypto Event Markets (Polymarket) — ETF approvals, upgrades

Usage:
    python strategy_engine.py --scan-all          # Full scan across all strategies
    python strategy_engine.py --scan-kalshi       # Kalshi-only scan
    python strategy_engine.py --scan-polymarket    # Polymarket-only scan
    python strategy_engine.py --report            # Generate opportunity report
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("strategy_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Add parent to path for local imports when running standalone
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

# Import local modules
from kalshi_client import BracketPricer, VolModel, KalshiTrader, calculate_taker_fee
from fomc_tracker import FOMCTracker

try:
    from polymarket_client import PolymarketScanner
    HAS_POLYMARKET = True
except ImportError:
    HAS_POLYMARKET = False

# New strategy modules (v2)
try:
    from political_crypto_strategies import PoliticalPolicyScanner, CryptoEventScanner
    HAS_POLITICAL_CRYPTO = True
except ImportError:
    HAS_POLITICAL_CRYPTO = False

try:
    from cross_platform_arb import CrossPlatformArbitrage
    HAS_CROSS_PLATFORM = True
except ImportError:
    HAS_CROSS_PLATFORM = False

try:
    from llm_fair_value import LLMFairValueEstimator
    HAS_LLM_FV = True
except ImportError:
    HAS_LLM_FV = False

try:
    from data_collector import MarketDataCollector
    HAS_COLLECTOR = True
except ImportError:
    HAS_COLLECTOR = False

# Strategy weight/priority (higher = more capital allocation)
STRATEGY_WEIGHTS = {
    "spx_bracket_vol": 0.30,      # Our strongest edge: vol model IC=0.644
    "fomc_divergence": 0.20,      # NBER-validated edge
    "event_driven_macro": 0.15,   # CPI/NFP/GDP brackets
    "favorite_longshot": 0.10,    # Statistical bias
    "combinatorial_arb": 0.10,    # Logical mispricings
    "cross_platform_arb": 0.05,   # Same event, different platforms
    "political_policy": 0.05,     # LLM-powered policy analysis
    "crypto_events": 0.05,        # Crypto market events
}


class StrategyEngine:
    """Unified engine that runs all prediction market strategies."""

    def __init__(self, mode: str = "demo", use_llm: bool = False):
        self.mode = mode
        self.pricer = BracketPricer(use_fat_tails=True, df=5.0)
        self.vol_model = VolModel()
        self.fomc_tracker = FOMCTracker()
        self.kalshi_trader = KalshiTrader(mode=mode)
        self.polymarket_scanner = PolymarketScanner() if HAS_POLYMARKET else None

        # New v2 modules
        self.political_scanner = PoliticalPolicyScanner() if HAS_POLITICAL_CRYPTO else None
        self.crypto_scanner = CryptoEventScanner() if HAS_POLITICAL_CRYPTO else None
        self.cross_platform_arb = CrossPlatformArbitrage(kalshi_mode=mode) if HAS_CROSS_PLATFORM else None
        self.llm_estimator = LLMFairValueEstimator() if (HAS_LLM_FV and use_llm) else None
        self.data_collector = MarketDataCollector() if HAS_COLLECTOR else None

        self.results_file = DATA_DIR / f"strategy_scan_{datetime.now().strftime('%Y%m%d')}.json"
        self.all_opportunities = []

    # ------------------------------------------------------------------
    # Strategy 1: SPX Bracket Volatility Edge
    # ------------------------------------------------------------------
    def scan_spx_brackets(self, current_spx: float, predicted_vol: float,
                          hours_to_close: float) -> list:
        """
        Price SPX brackets using our vol model with fat-tail correction.
        Our vol model has IC=0.644 at 30min — this is our strongest edge.
        """
        logger.info("Scanning SPX brackets...")
        opportunities = self.kalshi_trader.scan_for_opportunities(
            current_spx=current_spx,
            predicted_vol=predicted_vol,
            hours_to_close=hours_to_close,
        )

        for opp in opportunities:
            opp["strategy"] = "spx_bracket_vol"
            opp["weight"] = STRATEGY_WEIGHTS["spx_bracket_vol"]
            opp["platform"] = "kalshi"

        logger.info(f"SPX brackets: {len(opportunities)} opportunities")
        return opportunities

    # ------------------------------------------------------------------
    # Strategy 2: FOMC Divergence
    # ------------------------------------------------------------------
    def scan_fomc_divergence(self) -> list:
        """
        Monitor Kalshi vs CME FedWatch divergence.
        Trade convergence when divergence > 15%.
        """
        logger.info("Scanning FOMC divergence...")
        result = self.fomc_tracker.run_scan()
        rec = result["recommendation"]

        opportunities = []
        if rec["action"] in ("TRADE", "MONITOR"):
            opportunities.append({
                "strategy": "fomc_divergence",
                "weight": STRATEGY_WEIGHTS["fomc_divergence"],
                "platform": "kalshi",
                "meeting": rec["meeting"],
                "days_to_meeting": rec["days_to_meeting"],
                "divergence": rec["divergence"],
                "direction": rec["direction"],
                "signal_strength": rec["signal_strength"],
                "recommended_size_pct": rec["recommended_size_pct"],
                "action": rec["action"],
                "net_edge_after_fees": rec["divergence"] * 0.5,  # Conservative estimate
            })

        logger.info(f"FOMC: {len(opportunities)} opportunities (next meeting in {self.fomc_tracker.days_to_next_fomc()} days)")
        return opportunities

    # ------------------------------------------------------------------
    # Strategy 3: Event-Driven Macro
    # ------------------------------------------------------------------
    def scan_macro_events(self) -> list:
        """
        Scan for CPI, NFP, GDP bracket opportunities.
        Price brackets using our vol model + event vol premium.
        """
        logger.info("Scanning macro events...")
        opportunities = []

        # Check upcoming macro events
        events = [
            {"name": "CPI", "vol_premium": 1.3, "keywords": ["cpi", "inflation"]},
            {"name": "NFP", "vol_premium": 1.2, "keywords": ["nfp", "jobs", "employment"]},
            {"name": "GDP", "vol_premium": 1.1, "keywords": ["gdp", "growth"]},
            {"name": "PCE", "vol_premium": 1.15, "keywords": ["pce", "spending"]},
            {"name": "FOMC", "vol_premium": 1.5, "keywords": ["fomc", "fed", "rate"]},
        ]

        if self.polymarket_scanner:
            # Fetch markets once, then filter locally per event type
            markets = self.polymarket_scanner.get_active_markets(limit=50)
            for event in events:
                for m in markets:
                    question = m.get("question", "").lower()
                    if not any(k in question for k in event["keywords"]):
                        continue

                    prices = self.polymarket_scanner._parse_prices(m.get("outcomePrices", []))
                    if not prices:
                        continue

                    yes_price = prices[0]
                    volume = float(m.get("volumeNum", 0) or 0)

                    if 0.10 <= yes_price <= 0.90 and volume > 100000:
                        opportunities.append({
                            "strategy": "event_driven_macro",
                            "weight": STRATEGY_WEIGHTS["event_driven_macro"],
                            "platform": "polymarket",
                            "event_type": event["name"],
                            "question": m.get("question", "")[:100],
                            "market_price": yes_price,
                            "volume": volume,
                            "vol_premium": event["vol_premium"],
                            "action": "MONITOR",
                            "net_edge_after_fees": 0,  # Need LLM fair value
                        })

        logger.info(f"Macro events: {len(opportunities)} markets found")
        return opportunities

    # ------------------------------------------------------------------
    # Strategy 4: Favorite-Longshot Bias
    # ------------------------------------------------------------------
    def scan_longshot_bias(self) -> list:
        """
        Find overpriced longshot contracts.
        Academic research: contracts priced <$0.10 win only ~2% (priced at 5-10%).
        Sell NO on these = buy YES at 90-95 cents = collect the bias.

        ONLY for non-financial markets (sports, entertainment, politics).
        Financial tail risk is real — don't fade financial longshots.

        Improvements over naive 2.5x blanket factor:
        1. Category-specific overpricing factors (sports > entertainment > politics)
        2. Time decay: longshots closer to resolution are more accurately priced
        3. Volume-weighted efficiency: higher volume = more efficient pricing = less edge
        """
        logger.info("Scanning for longshot bias...")
        opportunities = []

        if not self.polymarket_scanner:
            return opportunities

        markets = self.polymarket_scanner.get_active_markets(limit=200)

        # Category-specific overpricing factors (academic literature calibration)
        # Sports: strongest longshot bias (casual bettors love underdogs)
        # Entertainment/pop culture: moderate bias (novelty markets, less sophisticated)
        # Politics: weakest bias (more informed participants, polling anchors)
        CATEGORY_OVERPRICING = {
            "sports": 3.0,        # Sports longshots overpriced by ~3x
            "entertainment": 2.5, # Entertainment/pop culture by ~2.5x
            "politics": 2.0,      # Political longshots by ~2x (sharper bettors)
            "other": 2.2,         # Default conservative estimate
        }

        # NOTE: Order matters — politics is checked FIRST to avoid false positives.
        # "Will AOC win the 2028 presidential nomination?" has "win" but is politics,
        # not sports. Politics keywords are checked before sports keywords.
        POLITICS_KEYWORDS_LONGSHOT = ["election", "president", "presidential", "governor",
                                      "senator", "congress", "party", "nominee", "candidate",
                                      "vote", "primary", "caucus", "ballot", "democrat",
                                      "republican", "gop", "cabinet", "impeach", "minister",
                                      "parliament", "legislation"]
        SPORTS_KEYWORDS = ["champion", "mvp", "playoff", "super bowl", "world cup",
                           "game", "match", "tournament", "season", "nfl", "nba", "mlb",
                           "nhl", "ufc", "boxing", "f1", "tennis", "golf", "fifa",
                           "stanley cup", "grand slam", "olympics", "medal"]
        ENTERTAINMENT_KEYWORDS = ["oscar", "grammy", "emmy", "award", "movie", "album",
                                  "song", "netflix", "streaming", "box office", "celebrity",
                                  "reality tv", "show"]

        for m in markets:
            question = (m.get("question", "") or "").lower()
            prices = self.polymarket_scanner._parse_prices(m.get("outcomePrices", []))
            volume = float(m.get("volumeNum", 0) or 0)

            if not prices or volume < 100000:
                continue

            yes_price = prices[0]

            # Only target extreme longshots (price < 10 cents)
            if yes_price > 0.10:
                continue

            # SKIP financial markets — tail risk is real
            financial_keywords = ["spx", "s&p", "fed", "interest rate", "gdp", "cpi",
                                 "recession", "stock", "market crash", "bitcoin", "btc",
                                 "ethereum", "crypto", "tariff", "treasury", "bond",
                                 "inflation", "unemployment"]
            if any(kw in question for kw in financial_keywords):
                continue

            # --- 1. Determine category and overpricing factor ---
            # Check politics FIRST — many political questions contain "win"
            # which would false-positive as sports otherwise.
            if any(kw in question for kw in POLITICS_KEYWORDS_LONGSHOT):
                category = "politics"
            elif any(kw in question for kw in SPORTS_KEYWORDS):
                category = "sports"
            elif any(kw in question for kw in ENTERTAINMENT_KEYWORDS):
                category = "entertainment"
            else:
                category = "other"

            base_overpricing = CATEGORY_OVERPRICING[category]

            # --- 2. Time decay: closer to resolution = more accurate pricing ---
            end_date_str = (
                m.get("endDate") or m.get("endDateIso") or m.get("end_date_iso") or ""
            )[:10]
            days_to_resolution = None
            time_decay_factor = 1.0  # No adjustment by default
            if end_date_str:
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    end_dt = _dt.strptime(end_date_str, "%Y-%m-%d")
                    days_to_resolution = (end_dt - _dt.now()).days
                    if days_to_resolution is not None and days_to_resolution > 0:
                        # Markets become more efficient closer to resolution.
                        # >90 days: full overpricing factor applies
                        # 30-90 days: reduce overpricing by 15%
                        # 7-30 days: reduce by 30%
                        # <7 days: reduce by 50% (market has converged significantly)
                        if days_to_resolution < 7:
                            time_decay_factor = 0.50
                        elif days_to_resolution < 30:
                            time_decay_factor = 0.70
                        elif days_to_resolution < 90:
                            time_decay_factor = 0.85
                        # else 1.0 (full factor)
                except (ValueError, TypeError):
                    pass

            # --- 3. Volume-weighted efficiency: higher volume = tighter pricing ---
            # $100K-$500K: less efficient, full bias
            # $500K-$2M: moderately efficient, 85% of bias
            # $2M+: very efficient, 70% of bias
            if volume > 2_000_000:
                volume_efficiency_factor = 0.70
            elif volume > 500_000:
                volume_efficiency_factor = 0.85
            else:
                volume_efficiency_factor = 1.0

            # Combine: effective overpricing = base * time_decay * volume_efficiency
            effective_overpricing = base_overpricing * time_decay_factor * volume_efficiency_factor

            # Don't bother if the effective overpricing is less than 1.5x
            # (insufficient edge after fees)
            if effective_overpricing < 1.5:
                continue

            implied_prob = yes_price
            estimated_actual = implied_prob / effective_overpricing

            edge = implied_prob - estimated_actual
            no_price = 1 - yes_price
            fee = 0.02 * no_price * (1 - no_price)  # Taker fee on NO side
            net_edge = edge - fee

            if net_edge > 0.005:
                opportunities.append({
                    "strategy": "favorite_longshot",
                    "weight": STRATEGY_WEIGHTS["favorite_longshot"],
                    "platform": "polymarket",
                    "question": m.get("question", "")[:100],
                    "market_price": yes_price,
                    "estimated_actual_prob": round(estimated_actual, 4),
                    "action": "BUY_NO",  # Equivalent to selling the longshot
                    "entry_price": round(no_price, 4),
                    "edge": round(edge, 4),
                    "net_edge_after_fees": round(net_edge, 4),
                    "volume": volume,
                    "category": category,
                    "overpricing_factor": round(effective_overpricing, 2),
                    "time_decay_factor": round(time_decay_factor, 2),
                    "volume_efficiency_factor": round(volume_efficiency_factor, 2),
                    "days_to_resolution": days_to_resolution,
                    "risk_note": "Max loss = no_price per contract if longshot hits",
                })

        opportunities.sort(key=lambda x: x["net_edge_after_fees"], reverse=True)
        logger.info(f"Longshot bias: {len(opportunities)} opportunities")
        return opportunities

    # ------------------------------------------------------------------
    # Strategy 5: Combinatorial Mispricings
    # ------------------------------------------------------------------
    def scan_combinatorial(self) -> list:
        """
        Find logical contradictions across related Polymarket markets.
        E.g., P(A) + P(B) + P(C) > 1.0 for mutually exclusive outcomes.
        """
        logger.info("Scanning for combinatorial mispricings...")
        if not self.polymarket_scanner:
            return []

        markets = self.polymarket_scanner.get_active_markets(limit=500)
        mispricings = self.polymarket_scanner.find_mispricings(markets)

        opportunities = []
        for mp in mispricings:
            if mp["edge_per_contract"] > 0.01:  # >1 cent edge per contract
                opportunities.append({
                    "strategy": "combinatorial_arb",
                    "weight": STRATEGY_WEIGHTS["combinatorial_arb"],
                    "platform": "polymarket",
                    "type": mp["type"],
                    "group": mp["group"][:80],
                    "probability_sum": mp["probability_sum"],
                    "direction": mp["direction"],
                    "net_edge_after_fees": mp["edge_per_contract"],
                    "action": "SELL" if mp["direction"] == "overpriced" else "BUY",
                    "markets_count": mp["markets"],
                })

        logger.info(f"Combinatorial: {len(opportunities)} mispricings")
        return opportunities

    # ------------------------------------------------------------------
    # Strategy 6: Cross-Platform Arbitrage
    # ------------------------------------------------------------------
    def scan_cross_platform(self) -> list:
        """
        Find the same event priced differently on Kalshi vs Polymarket.
        Uses fuzzy matching to identify overlapping events.
        """
        logger.info("Scanning cross-platform arbitrage...")
        if not self.cross_platform_arb:
            logger.info("Cross-platform module not available")
            return []

        try:
            opps = self.cross_platform_arb.scan_cross_platform()
            for opp in opps:
                opp["strategy"] = "cross_platform_arb"
                opp["weight"] = STRATEGY_WEIGHTS["cross_platform_arb"]
            logger.info(f"Cross-platform: {len(opps)} arbitrage opportunities")
            return opps
        except Exception as e:
            logger.error(f"Cross-platform scan failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Strategy 7: Political/Policy Edge
    # ------------------------------------------------------------------
    def scan_political_policy(self) -> list:
        """
        Scan for political/policy markets with exploitable biases.
        Covers tariffs, regulations, elections, appointments.
        """
        logger.info("Scanning political/policy markets...")
        if not self.political_scanner:
            logger.info("Political scanner not available")
            return []

        try:
            opps = self.political_scanner.scan_political(limit=100)
            for opp in opps:
                opp.setdefault("strategy", "political_policy")
                opp.setdefault("weight", STRATEGY_WEIGHTS["political_policy"])
            logger.info(f"Political/policy: {len(opps)} opportunities")
            return opps
        except Exception as e:
            logger.error(f"Political scan failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Strategy 8: Crypto Event Markets
    # ------------------------------------------------------------------
    def scan_crypto_events(self) -> list:
        """
        Scan crypto-specific markets for edge.
        ETF flows, technical levels, regulatory calendar.
        """
        logger.info("Scanning crypto event markets...")
        if not self.crypto_scanner:
            logger.info("Crypto scanner not available")
            return []

        try:
            opps = self.crypto_scanner.scan_crypto(limit=100)
            for opp in opps:
                opp.setdefault("strategy", "crypto_events")
                opp.setdefault("weight", STRATEGY_WEIGHTS["crypto_events"])
            logger.info(f"Crypto events: {len(opps)} opportunities")
            return opps
        except Exception as e:
            logger.error(f"Crypto scan failed: {e}")
            return []

    # ------------------------------------------------------------------
    # LLM-Enhanced Polymarket Scan
    # ------------------------------------------------------------------
    def scan_llm_enhanced(self, limit: int = 30) -> list:
        """
        Use LLM fair value estimation to find mispriced Polymarket markets.
        This is expensive (API calls) so use sparingly.
        """
        if not self.llm_estimator or not self.polymarket_scanner:
            return []

        logger.info(f"Running LLM-enhanced scan on top {limit} markets...")
        try:
            fair_value_fn = self.llm_estimator.as_fair_value_fn("fast")
            opps = self.polymarket_scanner.scan(
                limit=limit,
                min_net_edge=0.02,
                fair_value_fn=fair_value_fn,
            )
            for opp in opps:
                opp["strategy"] = "llm_fair_value"
                opp["weight"] = 0.15  # Medium-high weight for LLM picks
                opp["platform"] = "polymarket"
            logger.info(f"LLM-enhanced: {len(opps)} opportunities with edge")
            return opps
        except Exception as e:
            logger.error(f"LLM-enhanced scan failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Data Collection (run periodically to build history)
    # ------------------------------------------------------------------
    def collect_snapshot(self):
        """Collect market data snapshot for historical analysis."""
        if not self.data_collector:
            return
        try:
            self.data_collector.collect_polymarket_snapshot()
            self.data_collector.collect_kalshi_brackets()
            logger.info("Market data snapshot collected")
        except Exception as e:
            logger.error(f"Snapshot collection failed: {e}")

    # ------------------------------------------------------------------
    # Full Scan
    # ------------------------------------------------------------------
    def scan_all(self, current_spx: float = 5900.0, predicted_vol: float = 0.18,
                 hours_to_close: float = 4.0, include_llm: bool = False) -> dict:
        """Run all strategies and return unified opportunity list."""
        logger.info("=" * 60)
        logger.info("UNIFIED PREDICTION MARKET SCAN (v2 — 8 strategies)")
        logger.info("=" * 60)

        all_opps = []

        # Collect snapshot for historical data (non-blocking)
        self.collect_snapshot()

        # Run each strategy
        strategies = [
            ("spx_bracket_vol", lambda: self.scan_spx_brackets(current_spx, predicted_vol, hours_to_close)),
            ("fomc_divergence", self.scan_fomc_divergence),
            ("event_driven_macro", self.scan_macro_events),
            ("favorite_longshot", self.scan_longshot_bias),
            ("combinatorial_arb", self.scan_combinatorial),
            ("cross_platform_arb", self.scan_cross_platform),
            ("political_policy", self.scan_political_policy),
            ("crypto_events", self.scan_crypto_events),
        ]

        # Optionally add LLM-enhanced scan (expensive, off by default)
        if include_llm and self.llm_estimator:
            strategies.append(("llm_fair_value", lambda: self.scan_llm_enhanced(limit=30)))

        strategy_results = {}
        for name, fn in strategies:
            try:
                opps = fn()
                all_opps.extend(opps)
                strategy_results[name] = {
                    "count": len(opps),
                    "total_edge": sum(o.get("net_edge_after_fees", 0) for o in opps),
                }
            except Exception as e:
                logger.error(f"Strategy {name} failed: {e}")
                strategy_results[name] = {"count": 0, "error": str(e)}

        # Sort by weighted edge (edge * strategy weight)
        all_opps.sort(
            key=lambda x: x.get("net_edge_after_fees", 0) * x.get("weight", 0.1),
            reverse=True,
        )

        self.all_opportunities = all_opps

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_opportunities": len(all_opps),
            "strategies": strategy_results,
            "top_opportunities": all_opps[:20],
            "summary": self._generate_summary(all_opps, strategy_results),
        }

        # Save to file
        with open(self.results_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {self.results_file}")

        return result

    def _generate_summary(self, opps: list, strategy_results: dict) -> dict:
        """Generate a summary of the scan."""
        if not opps:
            return {
                "verdict": "NO_OPPORTUNITIES",
                "message": "All markets appear fairly priced. No actionable edge found.",
                "total_estimated_daily_edge": 0,
            }

        actionable = [o for o in opps if o.get("net_edge_after_fees", 0) > 0.005]
        total_edge = sum(o.get("net_edge_after_fees", 0) for o in actionable)
        avg_edge = total_edge / len(actionable) if actionable else 0

        best_strategy = max(strategy_results.items(),
                          key=lambda x: x[1].get("total_edge", 0))[0]

        return {
            "verdict": "OPPORTUNITIES_FOUND" if actionable else "MARGINAL",
            "actionable_count": len(actionable),
            "total_estimated_edge": round(total_edge, 4),
            "avg_edge_per_trade": round(avg_edge, 4),
            "best_strategy": best_strategy,
            "platforms": {
                "kalshi": len([o for o in actionable if o.get("platform") == "kalshi"]),
                "polymarket": len([o for o in actionable if o.get("platform") == "polymarket"]),
            },
            "message": f"{len(actionable)} actionable opportunities across {len(set(o.get('strategy') for o in actionable))} strategies. "
                       f"Best edge: {best_strategy.replace('_', ' ')}.",
        }

    def format_report(self, result: dict) -> str:
        """Format scan results as a readable report."""
        lines = [
            "**PREDICTION MARKET SCAN REPORT**",
            f"Time: {result['timestamp'][:19]} UTC",
            f"Total opportunities: {result['total_opportunities']}",
            "",
            "**Strategy Breakdown:**",
        ]

        for name, data in result["strategies"].items():
            status = f"{data['count']} opps" if data['count'] > 0 else "none"
            if "error" in data:
                status = f"ERROR: {data['error'][:50]}"
            lines.append(f"  {name.replace('_', ' ').title()}: {status}")

        summary = result.get("summary", {})
        lines.extend([
            "",
            f"**Verdict:** {summary.get('verdict', 'N/A')}",
            f"**Message:** {summary.get('message', 'N/A')}",
        ])

        if result.get("top_opportunities"):
            lines.extend(["", "**Top Opportunities:**"])
            for i, opp in enumerate(result["top_opportunities"][:5], 1):
                edge = opp.get("net_edge_after_fees", 0)
                strategy = opp.get("strategy", "unknown").replace("_", " ")
                platform = opp.get("platform", "")
                action = opp.get("action", "")
                desc = opp.get("question", opp.get("ticker", opp.get("group", "?")))[:60]
                lines.append(f"  {i}. [{platform}] {desc}")
                lines.append(f"     Strategy: {strategy} | Edge: {edge:.4f} | Action: {action}")

        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prediction Market Strategy Engine v2")
    parser.add_argument("--scan-all", action="store_true", help="Run all 8+ strategies")
    parser.add_argument("--scan-kalshi", action="store_true", help="Kalshi strategies only")
    parser.add_argument("--scan-polymarket", action="store_true", help="Polymarket strategies only")
    parser.add_argument("--scan-political", action="store_true", help="Political/policy only")
    parser.add_argument("--scan-crypto", action="store_true", help="Crypto events only")
    parser.add_argument("--scan-arb", action="store_true", help="Cross-platform arbitrage only")
    parser.add_argument("--with-llm", action="store_true", help="Include LLM fair value scan (costs API $)")
    parser.add_argument("--collect", action="store_true", help="Collect market data snapshot")
    parser.add_argument("--report", action="store_true", help="Generate formatted report")
    parser.add_argument("--spx", type=float, default=5900, help="Current SPX price")
    parser.add_argument("--vol", type=float, default=0.18, help="Predicted annualized vol")
    parser.add_argument("--hours", type=float, default=4.0, help="Hours to market close")
    args = parser.parse_args()

    engine = StrategyEngine(mode="demo", use_llm=args.with_llm)

    if args.collect:
        engine.collect_snapshot()
        print("Snapshot collected.")
    elif args.scan_political:
        opps = engine.scan_political_policy()
        print(f"Political/Policy: {len(opps)} opportunities")
        for o in opps[:10]:
            print(f"  {o.get('question', o.get('ticker', '?'))[:60]} | Edge: {o.get('net_edge_after_fees', 0):.4f}")
    elif args.scan_crypto:
        opps = engine.scan_crypto_events()
        print(f"Crypto Events: {len(opps)} opportunities")
        for o in opps[:10]:
            print(f"  {o.get('question', o.get('ticker', '?'))[:60]} | Edge: {o.get('net_edge_after_fees', 0):.4f}")
    elif args.scan_arb:
        opps = engine.scan_cross_platform()
        print(f"Cross-Platform Arb: {len(opps)} opportunities")
        for o in opps[:10]:
            print(f"  {o.get('question', o.get('event', '?'))[:60]} | Net Edge: {o.get('net_edge_after_fees', 0):.4f}")
    else:
        result = engine.scan_all(
            current_spx=args.spx,
            predicted_vol=args.vol,
            hours_to_close=args.hours,
            include_llm=args.with_llm,
        )
        print(engine.format_report(result))
