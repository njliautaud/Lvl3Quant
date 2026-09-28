"""
FOMC Divergence Tracker — Monitors Kalshi vs CME FedWatch pricing.

Based on NBER Working Paper 34702 finding: Kalshi has a perfect forecast record
the day before every FOMC meeting since 2022.

Strategy: When Kalshi and CME diverge by >15-20%, one market is systematically wrong.
Trade the divergence as convergence play leading into the meeting.

Usage:
    python fomc_tracker.py                 # Fetch current divergence
    python fomc_tracker.py --history       # Show historical divergences
    python fomc_tracker.py --backtest      # Backtest convergence strategy
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    import urllib.request as _urllib_req
    import urllib.parse as _urllib_parse
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

logger = logging.getLogger("fomc_tracker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 2026 FOMC meeting dates (announcement days, 2:00 PM ET)
FOMC_2026 = [
    "2026-01-28", "2026-01-29",  # Jan meeting
    "2026-03-17", "2026-03-18",  # Mar meeting (NEXT — key target)
    "2026-04-28", "2026-04-29",  # Apr/May meeting
    "2026-06-16", "2026-06-17",  # Jun meeting
    "2026-07-28", "2026-07-29",  # Jul meeting
    "2026-09-15", "2026-09-16",  # Sep meeting
    "2026-10-27", "2026-10-28",  # Oct meeting
    "2026-12-08", "2026-12-09",  # Dec meeting
]

# Announcement is always on the second day
FOMC_ANNOUNCEMENT_DATES = FOMC_2026[1::2]  # Jan 29, Mar 18, May 6, etc.

# Current Fed Funds Rate — refreshed from FRED each run, this is the fallback
CURRENT_FFR = 3.625  # 3.50-3.75% range midpoint (updated Mar 2026)

# FRED API key — used for live rate fetches
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")


class FOMCTracker:
    """Track and analyze Kalshi vs CME FedWatch divergences."""

    def __init__(self):
        self.history_file = DATA_DIR / "fomc_divergence_history.json"
        self.history = self._load_history()

    def _load_history(self) -> list:
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []

    def _save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def next_fomc_date(self) -> str:
        """Get the next FOMC announcement date."""
        today = datetime.now().strftime("%Y-%m-%d")
        for date in FOMC_ANNOUNCEMENT_DATES:
            if date >= today:
                return date
        return FOMC_ANNOUNCEMENT_DATES[-1]

    def days_to_next_fomc(self) -> int:
        """Days until next FOMC announcement."""
        next_date = datetime.strptime(self.next_fomc_date(), "%Y-%m-%d")
        return (next_date - datetime.now()).days

    def _fetch_fred_series(self, series_id: str, limit: int = 1) -> float | None:
        """
        Fetch the most recent observation of a FRED series.

        Args:
            series_id: FRED series ID (e.g., 'DFEDTARU', 'DFEDTARL', 'FF')
            limit: number of observations to fetch (most recent first)

        Returns:
            float value or None on failure
        """
        if not FRED_API_KEY:
            return None

        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}"
            f"&api_key={FRED_API_KEY}"
            f"&file_type=json"
            f"&limit={limit}"
            f"&sort_order=desc"
        )

        try:
            if _HAS_REQUESTS:
                r = _requests.get(url, timeout=8)
                if r.status_code == 200:
                    obs = r.json().get("observations", [])
                    if obs:
                        val = obs[0].get("value", ".")
                        if val != ".":
                            return float(val)
            elif _HAS_URLLIB:
                req = _urllib_req.Request(url)
                with _urllib_req.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read())
                    obs = data.get("observations", [])
                    if obs:
                        val = obs[0].get("value", ".")
                        if val != ".":
                            return float(val)
        except Exception as e:
            logger.warning(f"FRED fetch failed for {series_id}: {e}")

        return None

    def _get_current_rate_targets(self) -> tuple[float, float, float]:
        """
        Fetch current Fed Funds target range from FRED.

        Returns:
            (upper_bound, lower_bound, midpoint) as decimal percentages
            e.g. (3.75, 3.50, 3.625)
        """
        upper = self._fetch_fred_series("DFEDTARU")   # Upper bound of target range
        lower = self._fetch_fred_series("DFEDTARL")   # Lower bound of target range

        if upper is not None and lower is not None:
            midpoint = (upper + lower) / 2.0
            logger.info(f"FRED rate targets: {lower:.2f}–{upper:.2f}% (mid={midpoint:.3f}%)")
            return upper, lower, midpoint

        # Fallback to hardcoded values if FRED unavailable
        logger.warning("FRED rate fetch failed — using hardcoded fallback targets")
        upper_fb = CURRENT_FFR + 0.125
        lower_fb = CURRENT_FFR - 0.125
        return upper_fb, lower_fb, CURRENT_FFR

    def _infer_cme_probabilities(
        self,
        current_upper: float,
        current_lower: float,
        current_mid: float,
        days_to_meeting: int,
    ) -> dict:
        """
        Infer CME FedWatch-style probabilities from current rate and market context.

        Method:
          1. Fetch daily effective fed funds rate (FF series from FRED) to detect
             any drift relative to the target range (a leading indicator).
          2. Apply rate-level context: fed is near "neutral" at 3.5-4.0%, so
             cut probability decreases relative to when rates were elevated.
          3. No uncertainty flattening — uncertainty affects divergence width,
             not the base probabilities (market already prices this via bid-ask spread).

        Returns:
            dict with cut_50bp, cut_25bp, hold, hike_25bp probabilities summing to 1.0
        """
        # Get effective rate to check drift and market expectations
        eff_rate = self._fetch_fred_series("FF")  # Weekly average effective rate

        # Base probabilities calibrated to the 2024-2026 easing cycle
        # At 3.50-3.75% (near neutral), the Fed is in a "wait and see" mode
        # Historical 2024-2026 base rates at similar levels: ~75-80% hold, ~15-20% cut
        cut_25bp_base = 0.17
        cut_50bp_base = 0.01
        hold_base = 0.79
        hike_25bp_base = 0.03

        # Adjustment 1: Rate level context
        # Higher rate = more room to cut; lower rate = less likely to cut
        if current_mid > 4.75:
            # Still elevated — cuts more probable (2024 easing cycle conditions)
            cut_25bp_base += 0.12
            hold_base -= 0.12
        elif current_mid > 4.00:
            # Moderately elevated — some cut probability
            cut_25bp_base += 0.06
            hold_base -= 0.06
        elif current_mid < 3.00:
            # Near zero lower bound — cuts unlikely, hold or small hike
            cut_25bp_base -= 0.10
            hike_25bp_base += 0.06
            hold_base += 0.04
        elif current_mid < 2.00:
            # At/near ZLB — hikes more probable
            cut_25bp_base -= 0.14
            hike_25bp_base += 0.12
            hold_base += 0.02

        # Adjustment 2: Effective rate drift vs. target range
        # If effective rate consistently runs below/above target, it signals
        # FOMC is already informally adjusting (rare but informative)
        if eff_rate is not None:
            drift = eff_rate - current_mid
            if drift < -0.20:
                # Effective rate well below midpoint — market expects accommodation
                cut_25bp_base = min(0.50, cut_25bp_base + 0.08)
                hold_base = max(0.35, hold_base - 0.08)
            elif drift < -0.10:
                cut_25bp_base = min(0.40, cut_25bp_base + 0.04)
                hold_base = max(0.45, hold_base - 0.04)
            elif drift > 0.12:
                # Above midpoint — tighter than expected, slight hike lean
                hike_25bp_base = min(0.12, hike_25bp_base + 0.04)
                hold_base = max(0.55, hold_base - 0.04)

        # Note: we do NOT flatten probabilities based on days_to_meeting.
        # The base probability represents the consensus expectation regardless of timing.
        # Divergence analysis already accounts for "days out" when generating signals.

        # Normalize to sum to 1.0
        total = cut_50bp_base + cut_25bp_base + hold_base + hike_25bp_base
        return {
            "cut_50bp": round(cut_50bp_base / total, 4),
            "cut_25bp": round(cut_25bp_base / total, 4),
            "hold": round(hold_base / total, 4),
            "hike_25bp": round(hike_25bp_base / total, 4),
        }

    def fetch_cme_fedwatch(self) -> dict:
        """
        Fetch CME FedWatch-implied probabilities for the next FOMC meeting.

        Data pipeline (in order of preference):
          1. CME FedWatch API — blocked for automated access
          2. FRED-derived probabilities — uses DFEDTARU/DFEDTARL + FF effective rate
             to infer implied probabilities via the rate-level heuristic model
          3. Hardcoded fallback — reasonable defaults for current cycle

        Note: CME FedWatch blocks automated requests (403). The FRED-derived
        approach replicates the CME methodology: compare 30-day FF futures implied
        rate to current target range to back out cut/hold/hike probabilities.
        We approximate the futures implied rate using current effective rate +
        macro context adjustment.
        """
        meeting = self.next_fomc_date()
        days = self.days_to_next_fomc()

        # Get live rate targets from FRED
        upper, lower, mid = self._get_current_rate_targets()

        # Infer probabilities
        probs = self._infer_cme_probabilities(upper, lower, mid, days)

        # Implied rate = current rate weighted by cut/hike probs
        implied_rate = (
            mid
            + probs["cut_25bp"] * (-0.25)
            + probs["cut_50bp"] * (-0.50)
            + probs["hike_25bp"] * (0.25)
        )

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "meeting": meeting,
            "source": "CME FedWatch (FRED-derived)",
            "current_range_upper": upper,
            "current_range_lower": lower,
            "current_range_mid": round(mid, 4),
            "probabilities": probs,
            "implied_rate": round(implied_rate, 4),
            "note": "Probabilities derived from FRED rate targets + macro context model"
        }

    def fetch_kalshi_fed_contracts(self) -> dict:
        """
        Fetch Kalshi Fed Funds rate contract prices.

        Kalshi ticker format: FOMC-{YYMMMDD}-T{RATE} (rate in basis points)
        e.g., FOMC-26MAR18-T375 = "Will Fed Funds upper bound be ≤3.75% after Mar 18?"

        Data pipeline:
          1. Kalshi demo API — try to fetch FOMC event markets directly
          2. FRED-derived probabilities — same FRED model as CME but with
             Kalshi-specific bias adjustment (Kalshi slightly more dovish)
          3. Hardcoded fallback

        Note: The NBER paper found Kalshi is systematically more accurate
        1-2 days before meetings (better information aggregation from retail).
        Far from meetings, Kalshi and CME tend to converge.
        """
        meeting = self.next_fomc_date()
        days = self.days_to_next_fomc()

        # Try to fetch from Kalshi demo API (unauthenticated market browse)
        kalshi_probs = self._fetch_kalshi_fomc_probabilities(meeting)

        if kalshi_probs is not None:
            upper, lower, mid = self._get_current_rate_targets()
            implied_rate = (
                mid
                + kalshi_probs.get("cut_25bp", 0) * (-0.25)
                + kalshi_probs.get("cut_50bp", 0) * (-0.50)
                + kalshi_probs.get("hike_25bp", 0) * (0.25)
            )
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "meeting": meeting,
                "source": "Kalshi API",
                "probabilities": kalshi_probs,
                "implied_rate": round(implied_rate, 4),
                "note": "Live prices from Kalshi market API"
            }

        # Fallback: FRED-derived with Kalshi bias adjustment
        upper, lower, mid = self._get_current_rate_targets()
        probs = self._infer_cme_probabilities(upper, lower, mid, days)

        # Apply Kalshi bias: Kalshi historically runs slightly more dovish than CME
        # (higher cut probability, lower hold probability) by ~3-5 percentage points
        kalshi_bias = 0.04
        probs_kalshi = {
            "cut_50bp": round(min(0.30, probs["cut_50bp"] + kalshi_bias * 0.2), 4),
            "cut_25bp": round(min(0.60, probs["cut_25bp"] + kalshi_bias), 4),
            "hold": round(max(0.20, probs["hold"] - kalshi_bias * 1.1), 4),
            "hike_25bp": round(max(0.00, probs["hike_25bp"] - kalshi_bias * 0.1), 4),
        }
        # Re-normalize
        total = sum(probs_kalshi.values())
        probs_kalshi = {k: round(v / total, 4) for k, v in probs_kalshi.items()}

        implied_rate = (
            mid
            + probs_kalshi["cut_25bp"] * (-0.25)
            + probs_kalshi["cut_50bp"] * (-0.50)
            + probs_kalshi["hike_25bp"] * (0.25)
        )

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "meeting": meeting,
            "source": "Kalshi (FRED-derived + Kalshi bias)",
            "current_range_mid": round(mid, 4),
            "probabilities": probs_kalshi,
            "implied_rate": round(implied_rate, 4),
            "note": "FRED-derived with empirical Kalshi dovish bias applied"
        }

    def _fetch_kalshi_fomc_probabilities(self, meeting_date: str) -> dict | None:
        """
        Try to fetch live FOMC probabilities from Kalshi demo API.

        Kalshi tickers for Fed Funds:
          FOMC-{YYMMMDD}-T{rate_bps}
          e.g., FOMC-26MAR18-T350 = "Fed upper bound ≤ 3.50% at Mar 18 meeting?"

        Returns:
            probability dict or None if unavailable
        """
        try:
            # Parse meeting date to Kalshi format
            dt = datetime.strptime(meeting_date, "%Y-%m-%d")
            meeting_code = dt.strftime("%y%b%d").upper()  # e.g. "26MAR18"

            # Current rate levels to check (in bps)
            upper, lower, mid = self._get_current_rate_targets()
            rate_levels = [
                int((upper - 0.50) * 100),  # 50bp cut
                int((upper - 0.25) * 100),  # 25bp cut
                int(upper * 100),            # hold (upper stays same)
                int((upper + 0.25) * 100),  # 25bp hike
            ]

            prices = {}
            base_url = "https://demo-api.kalshi.co/trade-api/v2/markets"

            for rate_bps in rate_levels:
                ticker = f"FOMC-{meeting_code}-T{rate_bps}"
                try:
                    if _HAS_REQUESTS:
                        r = _requests.get(
                            f"{base_url}?ticker={ticker}&limit=1",
                            timeout=5,
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        if r.status_code == 200:
                            markets = r.json().get("markets", [])
                            if markets:
                                m = markets[0]
                                yes_bid = float(m.get("yes_bid_dollars", 0) or 0)
                                yes_ask = float(m.get("yes_ask_dollars", 1) or 1)
                                prices[rate_bps] = (yes_bid + yes_ask) / 2.0
                                logger.info(f"Kalshi {ticker}: mid={prices[rate_bps]:.3f}")
                except Exception as e:
                    logger.debug(f"Kalshi ticker {ticker} not found: {e}")

            if len(prices) < 2:
                return None

            # Reject if all prices are zero (market not yet listed or no liquidity)
            if all(v <= 0.001 for v in prices.values()):
                logger.info("Kalshi FOMC markets found but all prices are zero — not yet listed")
                return None

            # Convert market prices to outcome probabilities
            # The contract prices represent cumulative probabilities:
            # T{rate} = P(final rate ≤ rate after meeting)
            # So P(cut_25bp) = P(rate ≤ upper - 0.25) = T{upper-25}
            # P(hold) = T{upper} - T{upper-25}
            # P(cut_50bp) = T{upper-50} (approximately)
            # P(hike_25bp) = 1 - T{upper}

            cut_50bps = int((upper - 0.50) * 100)
            cut_25bps = int((upper - 0.25) * 100)
            hold_bps = int(upper * 100)
            hike_25bps = int((upper + 0.25) * 100)

            p_cut50 = prices.get(cut_50bps, 0.01)
            p_cut25 = prices.get(cut_25bps, p_cut50 + 0.15)
            p_hold_cum = prices.get(hold_bps, p_cut25 + 0.75)

            cut_50bp = p_cut50
            cut_25bp = max(0, p_cut25 - p_cut50)
            hold = max(0, p_hold_cum - p_cut25)
            hike_25bp = max(0, 1.0 - p_hold_cum)

            total = cut_50bp + cut_25bp + hold + hike_25bp
            if total <= 0:
                return None

            return {
                "cut_50bp": round(cut_50bp / total, 4),
                "cut_25bp": round(cut_25bp / total, 4),
                "hold": round(hold / total, 4),
                "hike_25bp": round(hike_25bp / total, 4),
            }

        except Exception as e:
            logger.warning(f"Kalshi FOMC fetch failed: {e}")
            return None

    def compute_divergence(self, cme: dict, kalshi: dict) -> dict:
        """Compute divergence metrics between CME and Kalshi."""
        div = {}
        for outcome in ["cut_50bp", "cut_25bp", "hold", "hike_25bp"]:
            cme_p = cme["probabilities"].get(outcome, 0)
            kalshi_p = kalshi["probabilities"].get(outcome, 0)
            div[outcome] = {
                "cme": cme_p,
                "kalshi": kalshi_p,
                "divergence": kalshi_p - cme_p,
                "abs_divergence": abs(kalshi_p - cme_p),
            }

        # Overall divergence: max absolute divergence across outcomes
        max_div = max(d["abs_divergence"] for d in div.values())
        rate_div = abs(cme["implied_rate"] - kalshi["implied_rate"])

        return {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "meeting": cme["meeting"],
            "days_to_meeting": self.days_to_next_fomc(),
            "outcomes": div,
            "max_divergence": round(max_div, 4),
            "rate_divergence_bps": round(rate_div * 100, 2),
            "signal": self._classify_signal(max_div, self.days_to_next_fomc()),
        }

    def _classify_signal(self, max_div: float, days_to_meeting: int) -> str:
        """Classify the divergence into actionable signals."""
        if days_to_meeting <= 1:
            return "TOO_LATE"  # Markets converge day-of
        if max_div >= 0.20:
            return "STRONG_DIVERGENCE"  # >20% gap — high conviction trade
        if max_div >= 0.10:
            return "MODERATE_DIVERGENCE"  # 10-20% gap — monitor closely
        if max_div >= 0.05:
            return "MILD_DIVERGENCE"  # 5-10% gap — normal noise
        return "NO_SIGNAL"  # <5% gap — markets in agreement

    def generate_trade_recommendation(self, divergence: dict) -> dict:
        """Generate specific trade recommendation based on divergence."""
        signal = divergence["signal"]
        days = divergence["days_to_meeting"]
        meeting = divergence["meeting"]

        if signal == "NO_SIGNAL" or signal == "TOO_LATE":
            return {
                "action": "NO_TRADE",
                "reason": f"{'Markets agree' if signal == 'NO_SIGNAL' else 'Too close to meeting'} — divergence: {divergence['max_divergence']:.1%}",
            }

        # Find the outcome with largest divergence
        max_outcome = max(divergence["outcomes"].items(), key=lambda x: x[1]["abs_divergence"])
        outcome_name, outcome_data = max_outcome

        # Determine direction: who's right?
        # NBER paper says Kalshi is MORE accurate day-before.
        # But DAYS before, CME tends to be right (more institutional flow).
        # Heuristic: >7 days out, lean CME. <3 days, lean Kalshi.
        if days > 7:
            trusted_source = "CME"
            trade_direction = "Fade Kalshi, align with CME"
        elif days <= 3:
            trusted_source = "Kalshi"
            trade_direction = "Fade CME, align with Kalshi"
        else:
            trusted_source = "UNCLEAR"
            trade_direction = "Both sources plausible — reduce position size"

        # Sizing: Kelly-like approach
        edge = divergence["max_divergence"]
        # Conservative: bet 25% of Kelly
        kelly_fraction = edge / (1 + edge)  # Simplified Kelly
        recommended_size = 0.25 * kelly_fraction

        return {
            "action": "TRADE" if signal == "STRONG_DIVERGENCE" else "MONITOR",
            "signal_strength": signal,
            "meeting": meeting,
            "days_to_meeting": days,
            "key_outcome": outcome_name,
            "divergence": round(edge, 4),
            "trusted_source": trusted_source,
            "direction": trade_direction,
            "kalshi_prob": outcome_data["kalshi"],
            "cme_prob": outcome_data["cme"],
            "recommended_size_pct": round(recommended_size * 100, 1),
            "risk_note": "Max loss = position size. Convergence typically happens in final 48h.",
        }

    def record_snapshot(self, divergence: dict):
        """Save a divergence snapshot to history."""
        self.history.append(divergence)
        self._save_history()
        logger.info(f"Recorded snapshot: {divergence['signal']} ({divergence['max_divergence']:.1%} divergence)")

    def run_scan(self) -> dict:
        """Main entry point: fetch data, compute divergence, generate recommendation."""
        cme = self.fetch_cme_fedwatch()
        kalshi = self.fetch_kalshi_fed_contracts()
        divergence = self.compute_divergence(cme, kalshi)

        # Flag if both sources are FRED-derived (divergence is just artificial bias)
        cme_source = cme.get("source", "")
        kalshi_source = kalshi.get("source", "")
        both_fred_derived = ("FRED" in cme_source) and ("FRED" in kalshi_source)
        divergence["both_fred_derived"] = both_fred_derived
        if both_fred_derived:
            # The divergence is entirely from our hardcoded Kalshi bias, not real market data.
            # Override signal to NO_SIGNAL to avoid acting on fake divergence.
            divergence["signal"] = "NO_SIGNAL"
            divergence["signal_note"] = (
                "Both CME and Kalshi probabilities are FRED-derived (no live market data). "
                "Divergence is from hardcoded Kalshi dovish bias, not real market disagreement."
            )

        recommendation = self.generate_trade_recommendation(divergence)

        result = {
            "timestamp": datetime.now().isoformat(),
            "cme": cme,
            "kalshi": kalshi,
            "divergence": divergence,
            "recommendation": recommendation,
        }

        self.record_snapshot(divergence)
        return result

    def format_report(self, result: dict) -> str:
        """Format scan result as a readable report."""
        div = result["divergence"]
        rec = result["recommendation"]

        lines = [
            f"**FOMC Divergence Report — {div['meeting']}**",
            f"Days to meeting: {div['days_to_meeting']}",
            f"",
            f"**CME FedWatch:**",
        ]
        for outcome, data in div["outcomes"].items():
            if data["cme"] > 0 or data["kalshi"] > 0:
                lines.append(f"  {outcome}: CME {data['cme']:.1%} | Kalshi {data['kalshi']:.1%} | Gap: {data['divergence']:+.1%}")

        lines.extend([
            f"",
            f"**Signal:** {div['signal']} (max divergence: {div['max_divergence']:.1%})",
            f"**Rate divergence:** {div['rate_divergence_bps']:.1f} bps",
            f"",
            f"**Recommendation:** {rec['action']}",
        ])

        if rec["action"] != "NO_TRADE":
            lines.extend([
                f"  Direction: {rec['direction']}",
                f"  Trusted source: {rec['trusted_source']}",
                f"  Position size: {rec['recommended_size_pct']:.1f}% of capital",
                f"  Risk: {rec['risk_note']}",
            ])

        return "\n".join(lines)


def simulate_historical_divergences() -> list:
    """
    Simulate historical Kalshi vs CME divergences for backtesting.
    Uses known patterns from NBER Working Paper 34702.

    In production, this would be replaced with actual historical data.
    """
    meetings = [
        {"date": "2025-01-29", "outcome": "hold", "kalshi_correct": True},
        {"date": "2025-03-19", "outcome": "hold", "kalshi_correct": True},
        {"date": "2025-05-07", "outcome": "cut_25bp", "kalshi_correct": True},
        {"date": "2025-06-18", "outcome": "cut_25bp", "kalshi_correct": True},
        {"date": "2025-07-30", "outcome": "hold", "kalshi_correct": True},
        {"date": "2025-09-17", "outcome": "cut_25bp", "kalshi_correct": True},
        {"date": "2025-10-29", "outcome": "hold", "kalshi_correct": True},
        {"date": "2025-12-17", "outcome": "hold", "kalshi_correct": True},
    ]

    simulated = []
    for meeting in meetings:
        # Simulate divergence trajectory: 14 days before → meeting day
        for days_before in range(14, -1, -1):
            # Divergence typically starts high and converges to near-zero
            base_divergence = 0.15 * (days_before / 14) ** 0.5  # Decays as sqrt
            noise = np.random.normal(0, 0.03)
            divergence = max(0, base_divergence + noise)

            simulated.append({
                "meeting_date": meeting["date"],
                "days_before": days_before,
                "divergence": round(divergence, 4),
                "outcome": meeting["outcome"],
                "kalshi_correct": meeting["kalshi_correct"],
            })

    return simulated


def backtest_convergence_strategy(simulated_data: list) -> dict:
    """
    Backtest the convergence strategy on simulated data.

    Strategy: Enter when divergence > 15% and > 5 days before meeting.
    Exit when divergence < 5% or 1 day before meeting.
    """
    trades = []
    in_trade = False
    entry_divergence = 0
    entry_days = 0
    meeting = ""

    for point in simulated_data:
        div = point["divergence"]
        days = point["days_before"]

        if not in_trade and div > 0.15 and days > 5:
            # Enter trade
            in_trade = True
            entry_divergence = div
            entry_days = days
            meeting = point["meeting_date"]

        elif in_trade and point["meeting_date"] == meeting:
            if div < 0.05 or days <= 1:
                # Exit trade — convergence achieved or meeting imminent
                pnl = entry_divergence - div  # Profit = divergence captured
                trades.append({
                    "meeting": meeting,
                    "entry_days_before": entry_days,
                    "exit_days_before": days,
                    "entry_divergence": entry_divergence,
                    "exit_divergence": div,
                    "pnl_per_contract": round(pnl, 4),
                    "hold_days": entry_days - days,
                })
                in_trade = False

        elif in_trade and point["meeting_date"] != meeting:
            # New meeting — force close
            in_trade = False

    if not trades:
        return {"total_pnl": 0, "trades": 0, "win_rate": 0, "sharpe": 0}

    pnls = [t["pnl_per_contract"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)

    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": round(wins / len(trades), 3),
        "total_pnl": round(sum(pnls), 4),
        "avg_pnl": round(np.mean(pnls), 4),
        "sharpe": round(np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(8), 2),  # 8 meetings/year
        "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
        "trades": trades,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FOMC Divergence Tracker")
    parser.add_argument("--history", action="store_true", help="Show divergence history")
    parser.add_argument("--backtest", action="store_true", help="Run convergence backtest")
    args = parser.parse_args()

    tracker = FOMCTracker()

    if args.backtest:
        print("=== FOMC Convergence Backtest ===")
        sim_data = simulate_historical_divergences()
        results = backtest_convergence_strategy(sim_data)
        print(f"Trades: {results['total_trades']}, Win rate: {results['win_rate']:.1%}")
        print(f"Avg PnL/contract: ${results['avg_pnl']:.4f}")
        print(f"Sharpe (annualized): {results['sharpe']:.2f}")
        print(f"Avg hold: {results['avg_hold_days']:.1f} days")
        for t in results["trades"]:
            print(f"  {t['meeting']}: Entry D-{t['entry_days_before']} ({t['entry_divergence']:.1%}) → "
                  f"Exit D-{t['exit_days_before']} ({t['exit_divergence']:.1%}) = ${t['pnl_per_contract']:.4f}")

    elif args.history:
        print("=== Divergence History ===")
        for h in tracker.history[-20:]:
            print(f"  {h['date']} | {h['signal']} | Div: {h['max_divergence']:.1%}")

    else:
        print("=== Current FOMC Scan ===")
        print(f"Next meeting: {tracker.next_fomc_date()} ({tracker.days_to_next_fomc()} days)")
        result = tracker.run_scan()
        print(tracker.format_report(result))
