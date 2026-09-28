#!/usr/bin/env python3
"""
Earnings Signal Generator
==========================
Takes NLP scores from earnings_nlp_scorer.py and generates trading signals.

Core logic:
  - Bullish NLP + stock dropped post-earnings → BUY (overreaction)
  - Bearish NLP + stock rallied post-earnings → SELL/PUT (market hasn't caught up)
  - Score magnitude + price reaction magnitude → signal confidence

Outputs signals to /home/jupiter/Lvl3Quant/state/earnings_nlp_signals.json

Usage:
    python earnings_signal_generator.py                  # Generate all signals
    python earnings_signal_generator.py --ticker AAPL    # Single ticker
    python earnings_signal_generator.py --dry-run        # Show signals without saving
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

# Paths
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
SCORES_DIR = BASE_DIR / "data" / "earnings_transcripts" / "scores"
SIGNALS_FILE = BASE_DIR / "state" / "earnings_nlp_signals.json"
LOG_DIR = BASE_DIR / "logs" / "earnings_nlp"

LOG_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "signal_generator.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Signal thresholds
NLP_THRESHOLD = 0.3          # Min composite score magnitude to generate signal
PRICE_MOVE_THRESHOLD = 0.03  # 3% post-earnings price move
SIGNAL_EXPIRY_DAYS = 14      # Signals expire after 14 days


def get_post_earnings_price_move(ticker: str, earnings_date: str) -> dict | None:
    """Get the stock price move around the earnings date."""
    try:
        dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        start = dt - timedelta(days=5)
        end = min(dt + timedelta(days=10), datetime.now())

        # Handle BRK-B for yfinance
        yf_ticker = ticker.replace("-", "-")

        stock = yf.Ticker(yf_ticker)
        hist = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if hist.empty or len(hist) < 3:
            log.warning(f"{ticker}: Not enough price history around {earnings_date}")
            return None

        # Find the closest trading day to earnings date
        hist.index = hist.index.tz_localize(None)
        pre_dates = hist.index[hist.index <= dt]
        post_dates = hist.index[hist.index > dt]

        if len(pre_dates) == 0 or len(post_dates) == 0:
            # Earnings might be very recent, use last available day as pre
            pre_close = hist["Close"].iloc[0]
            post_close = hist["Close"].iloc[-1]
            pre_date = hist.index[0]
            post_date = hist.index[-1]
        else:
            pre_date = pre_dates[-1]
            post_date = post_dates[0]
            pre_close = hist.loc[pre_date, "Close"]
            post_close = hist.loc[post_date, "Close"]

        pct_change = (post_close - pre_close) / pre_close
        current_price = hist["Close"].iloc[-1]

        # Also check current price vs pre-earnings
        total_move = (current_price - pre_close) / pre_close

        return {
            "pre_earnings_close": round(float(pre_close), 2),
            "post_earnings_close": round(float(post_close), 2),
            "current_price": round(float(current_price), 2),
            "gap_pct": round(float(pct_change), 4),
            "total_move_pct": round(float(total_move), 4),
            "pre_date": pre_date.strftime("%Y-%m-%d"),
            "post_date": post_date.strftime("%Y-%m-%d"),
        }
    except Exception as e:
        log.error(f"{ticker}: Error fetching price data: {e}")
        return None


def compute_signal_confidence(composite_score: float, price_move: float) -> int:
    """Compute signal confidence 0-100 based on score magnitude and price divergence."""
    # Score magnitude contribution (0-50)
    score_mag = abs(composite_score)
    score_confidence = min(score_mag / 0.8, 1.0) * 50  # 0.8+ score = max 50

    # Price divergence contribution (0-50)
    # Higher divergence between NLP and price = higher confidence
    price_mag = abs(price_move)
    divergence = 0
    if (composite_score > 0 and price_move < 0) or (composite_score < 0 and price_move > 0):
        # Divergence case — this is what we're looking for
        divergence = (score_mag + price_mag) / 2
    else:
        # Aligned case — lower confidence (market already priced it in)
        divergence = 0

    price_confidence = min(divergence / 0.10, 1.0) * 50  # 10%+ divergence = max 50

    return int(min(score_confidence + price_confidence, 100))


def generate_signal(score_data: dict) -> dict | None:
    """Generate a trading signal from NLP score + price action."""
    ticker = score_data.get("ticker", "")
    composite = score_data.get("composite_score", 0.0)
    filing_date = score_data.get("filing_date", "")

    if score_data.get("status") != "scored":
        log.info(f"{ticker}: Skipping — status={score_data.get('status')}")
        return None

    if abs(composite) < 0.05:
        log.info(f"{ticker}: Composite score too weak ({composite:.3f}), no signal")
        return None

    # Get price action
    price_data = get_post_earnings_price_move(ticker, filing_date)
    if not price_data:
        log.warning(f"{ticker}: Could not get price data")
        return None

    gap_pct = price_data["gap_pct"]
    total_move = price_data["total_move_pct"]

    # Signal logic
    signal = None
    rationale = ""

    if composite > NLP_THRESHOLD and total_move < -PRICE_MOVE_THRESHOLD:
        # Bullish NLP + stock dropped → BUY (overreaction)
        signal = "BUY"
        rationale = (
            f"Bullish earnings sentiment ({composite:+.3f}) but stock dropped {total_move:.1%}. "
            f"Potential post-earnings overreaction. Consider calls or equity."
        )
    elif composite < -NLP_THRESHOLD and total_move > PRICE_MOVE_THRESHOLD:
        # Bearish NLP + stock rallied → SELL/PUT
        signal = "SELL"
        rationale = (
            f"Bearish earnings sentiment ({composite:+.3f}) but stock rallied {total_move:.1%}. "
            f"Market may not have caught the weakness. Consider puts."
        )
    elif composite > 0.5 and total_move > 0:
        # Very bullish NLP + stock already up → momentum
        signal = "BUY_MOMENTUM"
        rationale = (
            f"Very bullish earnings ({composite:+.3f}) with positive price action ({total_move:.1%}). "
            f"Strong momentum continuation. Consider equity or call spreads."
        )
    elif composite < -0.5 and total_move < 0:
        # Very bearish NLP + stock already down → continued weakness
        signal = "SELL_MOMENTUM"
        rationale = (
            f"Very bearish earnings ({composite:+.3f}) with negative price action ({total_move:.1%}). "
            f"Continued weakness expected. Consider puts or put spreads."
        )
    elif abs(composite) > NLP_THRESHOLD:
        # Signal present but weak price divergence
        direction = "BUY" if composite > 0 else "SELL"
        signal = f"{direction}_WEAK"
        rationale = (
            f"{'Bullish' if composite > 0 else 'Bearish'} sentiment ({composite:+.3f}), "
            f"price move {total_move:.1%}. Weak divergence — monitor for entry."
        )
    else:
        log.info(f"{ticker}: No actionable signal (NLP={composite:.3f}, move={total_move:.1%})")
        return None

    confidence = compute_signal_confidence(composite, total_move)

    # Check expiry
    try:
        filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
        days_since = (datetime.now() - filing_dt).days
        expired = days_since > SIGNAL_EXPIRY_DAYS
    except ValueError:
        days_since = 999
        expired = True

    result = {
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "rationale": rationale,
        "composite_score": composite,
        "composite_label": score_data.get("composite_label", ""),
        "filing_date": filing_date,
        "days_since_filing": days_since,
        "expired": expired,
        "price_data": price_data,
        "sub_scores": {
            k: v.get("score", 0) for k, v in score_data.get("scores", {}).items()
        },
        "generated_at": datetime.now().isoformat(),
    }

    action_label = "EXPIRED" if expired else "ACTIVE"
    log.info(
        f"{ticker}: {action_label} {signal} (conf={confidence}%) | "
        f"NLP={composite:+.3f} | Gap={gap_pct:.1%} | TotalMove={total_move:.1%}"
    )

    return result


def generate_all_signals(ticker_filter: str = None, dry_run: bool = False) -> list:
    """Generate signals for all scored transcripts."""
    score_files = sorted(SCORES_DIR.glob("*_score.json"))

    if ticker_filter:
        score_files = [f for f in score_files if f.name.startswith(ticker_filter)]

    log.info(f"Processing {len(score_files)} score files")

    signals = []
    for filepath in score_files:
        try:
            with open(filepath) as f:
                score_data = json.load(f)

            signal = generate_signal(score_data)
            if signal:
                signals.append(signal)
        except Exception as e:
            log.error(f"Error processing {filepath.name}: {e}")

    # Sort by confidence (highest first), then separate active vs expired
    active_signals = [s for s in signals if not s.get("expired", True)]
    expired_signals = [s for s in signals if s.get("expired", True)]

    active_signals.sort(key=lambda x: x["confidence"], reverse=True)
    expired_signals.sort(key=lambda x: x["confidence"], reverse=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "total_signals": len(signals),
        "active_signals": len(active_signals),
        "expired_signals": len(expired_signals),
        "signals": active_signals,
        "expired": expired_signals[:10],  # Keep top 10 expired for reference
    }

    if not dry_run:
        with open(SIGNALS_FILE, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"Saved signals to {SIGNALS_FILE}")

    return signals


def main():
    parser = argparse.ArgumentParser(description="Generate earnings-based trading signals")
    parser.add_argument("--ticker", type=str, help="Generate signal for single ticker")
    parser.add_argument("--dry-run", action="store_true", help="Show signals without saving")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Earnings Signal Generator Starting")

    signals = generate_all_signals(ticker_filter=args.ticker, dry_run=args.dry_run)

    if signals:
        log.info("=" * 60)
        log.info("Signal Summary:")
        for s in sorted(signals, key=lambda x: x["confidence"], reverse=True):
            status = "EXPIRED" if s.get("expired") else "ACTIVE"
            log.info(
                f"  {s['ticker']:6s} | {status:7s} | {s['signal']:15s} | "
                f"Conf={s['confidence']:3d}% | NLP={s['composite_score']:+.3f} | "
                f"Move={s['price_data']['total_move_pct']:.1%}"
            )

        active = [s for s in signals if not s.get("expired")]
        if active:
            log.info(f"\n{len(active)} ACTIVE signals ready for review")
        else:
            log.info("\nNo active signals (all expired or below threshold)")
    else:
        log.info("No signals generated. Run scorer first or check thresholds.")

    return signals


if __name__ == "__main__":
    main()
