"""
Spread Strategy Engine — selects optimal options spread structure given a signal.

Robinhood Level 2: can do vertical spreads (debit and credit).
Account size ~$750, max risk per trade = 20% = $150.

Liquidity filters enforced on BOTH legs:
  - Open Interest >= 500
  - Daily Volume >= 100
  - Bid-Ask spread <= 15% of mid price
"""

import calendar
import json
import os
import logging
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(BASE_DIR, "state")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

ACCOUNT_EQUITY = 750.0  # default, overridden from agentic_signals
MAX_RISK_PCT = 0.20
MIN_OPEN_INTEREST = 500
MIN_DAILY_VOLUME = 100
MAX_BID_ASK_SPREAD_PCT = 0.15  # 15% of mid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SpreadEngine] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "spread_strategy_engine.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard expiry selection helpers
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month/year (standard monthly expiry)."""
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])


def find_standard_expiry(ref_date: date | None = None,
                         dte_min: int = 21, dte_max: int = 45) -> date:
    """
    Find nearest standard monthly option expiry (3rd Friday) within
    dte_min..dte_max.

    RULES (liquidity-first):
      1. Prefer standard monthly (3rd Friday) within dte_min..dte_max.
      2. If none in range, pick the nearest monthly with >= 21 DTE (even
         if slightly outside dte_max — monthly liquidity beats DTE precision).
      3. Only fall back to a weekly Friday if no monthly has >= 21 DTE.
      4. NEVER return a non-Friday date.
    """
    if ref_date is None:
        ref_date = date.today()

    # Collect 3rd-Friday monthlies for the next 4 months
    monthlies: list[tuple[date, int]] = []
    for month_offset in range(0, 5):
        y = ref_date.year + (ref_date.month + month_offset - 1) // 12
        m = (ref_date.month + month_offset - 1) % 12 + 1
        tf = _third_friday(y, m)
        dte = (tf - ref_date).days
        if dte >= 7:
            monthlies.append((tf, dte))

    # 1. Standard monthly in the preferred DTE window
    in_window = [(tf, dte) for tf, dte in monthlies if dte_min <= dte <= dte_max]
    if in_window:
        return in_window[0][0]

    # 2. Nearest monthly with >= 21 DTE (slightly outside window is fine)
    viable = [(tf, dte) for tf, dte in monthlies if dte >= 21]
    if viable:
        target_dte = (dte_min + dte_max) // 2
        viable.sort(key=lambda x: abs(x[1] - target_dte))
        return viable[0][0]

    # 3. Any monthly with >= 7 DTE
    if monthlies:
        return monthlies[0][0]

    # 4. Last resort: nearest Friday with >= 21 DTE
    start = ref_date + timedelta(days=21)
    days_to_friday = (4 - start.weekday()) % 7
    if days_to_friday == 0 and start.weekday() != 4:
        days_to_friday = 7
    return start + timedelta(days=days_to_friday)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptionLeg:
    """One leg of a spread."""
    ticker: str
    option_type: str  # "call" or "put"
    strike: float
    expiry: str
    action: str  # "buy" or "sell"
    estimated_premium: float  # per-share premium (multiply by 100 for contract)
    open_interest: int = 0
    daily_volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    pricing_source: str = "bs_estimated"  # "bs_estimated" or "market"


@dataclass
class SpreadRecommendation:
    """Full spread recommendation."""
    ticker: str
    direction: str  # "bull" or "bear"
    spread_type: str  # e.g., "bull_call_spread", "bear_put_spread", etc.
    spread_category: str  # "debit" or "credit"
    legs: List[Dict[str, Any]]
    max_risk: float  # dollars at risk
    max_reward: float  # max profit in dollars
    reward_risk_ratio: float
    breakeven: float
    net_premium: float  # positive = credit received, negative = debit paid
    width: float  # strike distance
    confidence: float
    rationale: str
    liquidity_ok: bool
    pricing_source: str = "bs_estimated"  # "bs_estimated" or "market"
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Black-Scholes helpers (for synthetic option pricing when real data unavailable)
# ---------------------------------------------------------------------------

import math

def _norm_cdf(x: float) -> float:
    """Approximation of cumulative normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def estimate_option_price(
    ticker: str,
    option_type: str,
    strike: float,
    stock_price: float,
    days_to_expiry: int,
    iv: float = 0.30,
    risk_free: float = 0.05,
) -> float:
    """Estimate option price using Black-Scholes."""
    T = max(days_to_expiry / 365.0, 1 / 365.0)
    if option_type == "call":
        return bs_call(stock_price, strike, T, risk_free, iv)
    else:
        return bs_put(stock_price, strike, T, risk_free, iv)


# ---------------------------------------------------------------------------
# Liquidity filter
# ---------------------------------------------------------------------------

def check_liquidity(leg: OptionLeg) -> tuple:
    """
    Check if an option leg passes liquidity filters.
    Returns (passed: bool, reason: str).

    When we don't have real market data, we use heuristic checks based on
    strike proximity and ticker liquidity tier.
    """
    # If we have real OI/volume data
    if leg.open_interest > 0 or leg.daily_volume > 0:
        if leg.open_interest < MIN_OPEN_INTEREST:
            return False, f"OI={leg.open_interest} < {MIN_OPEN_INTEREST}"
        if leg.daily_volume < MIN_DAILY_VOLUME:
            return False, f"Volume={leg.daily_volume} < {MIN_DAILY_VOLUME}"
        if leg.bid > 0 and leg.ask > 0:
            mid = (leg.bid + leg.ask) / 2
            spread_pct = (leg.ask - leg.bid) / mid if mid > 0 else 1.0
            if spread_pct > MAX_BID_ASK_SPREAD_PCT:
                return False, f"Bid-ask spread {spread_pct:.1%} > {MAX_BID_ASK_SPREAD_PCT:.0%}"
        return True, "OK"

    # No real OI/volume data available — mark as NOT liquid so downstream
    # knows these are unverified BS-estimated prices with no confirmed OI.
    return False, "No real-time OI/volume data — liquidity unverified"


# ---------------------------------------------------------------------------
# Strike selection helpers
# ---------------------------------------------------------------------------

def get_atm_strike(stock_price: float, increment: float = 1.0) -> float:
    """Round to nearest strike increment."""
    # Most stocks use $1 increments; ETFs sometimes $0.50
    return round(stock_price / increment) * increment


def get_otm_strike(stock_price: float, direction: str, width: float = 5.0, increment: float = 1.0) -> float:
    """Get OTM strike for the short leg of a spread."""
    atm = get_atm_strike(stock_price, increment)
    if direction == "bull":
        return atm + width  # Higher strike for bull call spread short leg
    else:
        return atm - width  # Lower strike for bear put spread short leg


def determine_strike_increment(stock_price: float) -> float:
    """Determine appropriate strike increment based on stock price."""
    if stock_price < 30:
        return 0.50
    elif stock_price < 100:
        return 1.0
    elif stock_price < 300:
        return 2.50
    else:
        return 5.0


def determine_spread_width(stock_price: float, max_risk_dollars: float, increment: float) -> float:
    """
    Determine spread width that keeps max risk under limit.
    For debit spreads: max_risk = net_debit * 100
    For a width W, max debit ~ W * 100, so W <= max_risk / 100
    But we also want reasonable reward/risk, so we target 2-5 point widths.
    """
    # Max width based on risk budget
    max_width = max_risk_dollars / 100.0
    # Preferred widths by price level
    if stock_price < 30:
        preferred = 2.0
    elif stock_price < 100:
        preferred = 5.0
    elif stock_price < 300:
        preferred = 5.0
    else:
        preferred = 10.0

    width = min(preferred, max_width)
    # Snap to increment
    width = max(increment, round(width / increment) * increment)
    return width


# ---------------------------------------------------------------------------
# Spread structure selection
# ---------------------------------------------------------------------------

def select_spread(
    ticker: str,
    direction: str,
    confidence: float,
    stock_price: float,
    days_to_expiry: int = 30,
    iv: float = 0.30,
    account_equity: float = ACCOUNT_EQUITY,
) -> Optional[SpreadRecommendation]:
    """
    Given a signal, select the optimal spread structure.

    Selection logic:
    - High confidence (>0.70) + bullish -> Bull Call Spread (debit, higher reward)
    - Medium confidence (0.40-0.70) + bullish -> Bull Put Spread (credit, lower risk)
    - High confidence (>0.70) + bearish -> Bear Put Spread (debit)
    - Medium confidence (0.40-0.70) + bearish -> Bear Call Spread (credit)
    - Low confidence (<0.40) -> skip
    """
    max_risk = account_equity * MAX_RISK_PCT

    if confidence < 0.40:
        return SpreadRecommendation(
            ticker=ticker, direction=direction, spread_type="none",
            spread_category="none", legs=[], max_risk=0, max_reward=0,
            reward_risk_ratio=0, breakeven=0, net_premium=0, width=0,
            confidence=confidence, rationale="Confidence too low (<40%)",
            liquidity_ok=False, rejection_reason="Confidence below 40% threshold",
        )

    increment = determine_strike_increment(stock_price)
    atm = get_atm_strike(stock_price, increment)
    width = determine_spread_width(stock_price, max_risk, increment)

    is_high_conf = confidence > 0.70
    is_bullish = direction in ("bull", "bullish", "long")

    if is_bullish and is_high_conf:
        return _build_bull_call_spread(ticker, stock_price, atm, width, increment, days_to_expiry, iv, confidence, max_risk)
    elif is_bullish and not is_high_conf:
        return _build_bull_put_spread(ticker, stock_price, atm, width, increment, days_to_expiry, iv, confidence, max_risk)
    elif not is_bullish and is_high_conf:
        return _build_bear_put_spread(ticker, stock_price, atm, width, increment, days_to_expiry, iv, confidence, max_risk)
    else:
        return _build_bear_call_spread(ticker, stock_price, atm, width, increment, days_to_expiry, iv, confidence, max_risk)


def _build_bull_call_spread(
    ticker, stock_price, atm, width, increment, dte, iv, confidence, max_risk
) -> SpreadRecommendation:
    """Buy ATM call, sell OTM call. Debit spread."""
    long_strike = atm
    short_strike = atm + width
    expiry_str = find_standard_expiry(date.today(), 30, 45).isoformat()

    long_prem = estimate_option_price(ticker, "call", long_strike, stock_price, dte, iv)
    short_prem = estimate_option_price(ticker, "call", short_strike, stock_price, dte, iv)
    net_debit = long_prem - short_prem  # per share

    cost = net_debit * 100  # per contract
    max_profit = (width - net_debit) * 100
    rr = max_profit / cost if cost > 0 else 0
    breakeven = long_strike + net_debit

    long_leg = OptionLeg(ticker, "call", long_strike, expiry_str, "buy", long_prem)
    short_leg = OptionLeg(ticker, "call", short_strike, expiry_str, "sell", short_prem)

    liquidity_ok = True
    rejection = None
    for leg in [long_leg, short_leg]:
        ok, reason = check_liquidity(leg)
        if not ok:
            liquidity_ok = False
            rejection = f"{leg.strike} {leg.option_type}: {reason}"
            break

    if cost > max_risk:
        liquidity_ok = False
        rejection = f"Cost ${cost:.0f} exceeds max risk ${max_risk:.0f}"

    return SpreadRecommendation(
        ticker=ticker, direction="bull",
        spread_type="bull_call_spread", spread_category="debit",
        legs=[asdict(long_leg), asdict(short_leg)],
        max_risk=round(cost, 2), max_reward=round(max_profit, 2),
        reward_risk_ratio=round(rr, 2),
        breakeven=round(breakeven, 2),
        net_premium=round(-net_debit * 100, 2),  # negative = debit
        width=width, confidence=confidence,
        rationale=f"High confidence bull signal. Buy {long_strike}C / Sell {short_strike}C. "
                  f"Max risk ${cost:.0f}, max reward ${max_profit:.0f} ({rr:.1f}x).",
        liquidity_ok=liquidity_ok, pricing_source="bs_estimated",
        rejection_reason=rejection,
    )


def _build_bear_put_spread(
    ticker, stock_price, atm, width, increment, dte, iv, confidence, max_risk
) -> SpreadRecommendation:
    """Buy ATM put, sell OTM put. Debit spread."""
    long_strike = atm
    short_strike = atm - width
    expiry_str = find_standard_expiry(date.today(), 30, 45).isoformat()

    long_prem = estimate_option_price(ticker, "put", long_strike, stock_price, dte, iv)
    short_prem = estimate_option_price(ticker, "put", short_strike, stock_price, dte, iv)
    net_debit = long_prem - short_prem

    cost = net_debit * 100
    max_profit = (width - net_debit) * 100
    rr = max_profit / cost if cost > 0 else 0
    breakeven = long_strike - net_debit

    long_leg = OptionLeg(ticker, "put", long_strike, expiry_str, "buy", long_prem)
    short_leg = OptionLeg(ticker, "put", short_strike, expiry_str, "sell", short_prem)

    liquidity_ok = True
    rejection = None
    for leg in [long_leg, short_leg]:
        ok, reason = check_liquidity(leg)
        if not ok:
            liquidity_ok = False
            rejection = f"{leg.strike} {leg.option_type}: {reason}"
            break

    if cost > max_risk:
        liquidity_ok = False
        rejection = f"Cost ${cost:.0f} exceeds max risk ${max_risk:.0f}"

    return SpreadRecommendation(
        ticker=ticker, direction="bear",
        spread_type="bear_put_spread", spread_category="debit",
        legs=[asdict(long_leg), asdict(short_leg)],
        max_risk=round(cost, 2), max_reward=round(max_profit, 2),
        reward_risk_ratio=round(rr, 2),
        breakeven=round(breakeven, 2),
        net_premium=round(-net_debit * 100, 2),
        width=width, confidence=confidence,
        rationale=f"High confidence bear signal. Buy {long_strike}P / Sell {short_strike}P. "
                  f"Max risk ${cost:.0f}, max reward ${max_profit:.0f} ({rr:.1f}x).",
        liquidity_ok=liquidity_ok, pricing_source="bs_estimated",
        rejection_reason=rejection,
    )


def _build_bull_put_spread(
    ticker, stock_price, atm, width, increment, dte, iv, confidence, max_risk
) -> SpreadRecommendation:
    """Sell ATM put, buy OTM put. Credit spread."""
    short_strike = atm
    long_strike = atm - width
    expiry_str = find_standard_expiry(date.today(), 30, 45).isoformat()

    short_prem = estimate_option_price(ticker, "put", short_strike, stock_price, dte, iv)
    long_prem = estimate_option_price(ticker, "put", long_strike, stock_price, dte, iv)
    net_credit = short_prem - long_prem

    credit = net_credit * 100
    max_loss = (width - net_credit) * 100
    rr = credit / max_loss if max_loss > 0 else 0
    breakeven = short_strike - net_credit

    short_leg = OptionLeg(ticker, "put", short_strike, expiry_str, "sell", short_prem)
    long_leg = OptionLeg(ticker, "put", long_strike, expiry_str, "buy", long_prem)

    liquidity_ok = True
    rejection = None
    for leg in [short_leg, long_leg]:
        ok, reason = check_liquidity(leg)
        if not ok:
            liquidity_ok = False
            rejection = f"{leg.strike} {leg.option_type}: {reason}"
            break

    if max_loss > max_risk:
        liquidity_ok = False
        rejection = f"Max loss ${max_loss:.0f} exceeds max risk ${max_risk:.0f}"

    return SpreadRecommendation(
        ticker=ticker, direction="bull",
        spread_type="bull_put_spread", spread_category="credit",
        legs=[asdict(short_leg), asdict(long_leg)],
        max_risk=round(max_loss, 2), max_reward=round(credit, 2),
        reward_risk_ratio=round(rr, 2),
        breakeven=round(breakeven, 2),
        net_premium=round(credit, 2),
        width=width, confidence=confidence,
        rationale=f"Medium confidence bull signal. Sell {short_strike}P / Buy {long_strike}P. "
                  f"Credit ${credit:.0f}, max risk ${max_loss:.0f}.",
        liquidity_ok=liquidity_ok, pricing_source="bs_estimated",
        rejection_reason=rejection,
    )


def _build_bear_call_spread(
    ticker, stock_price, atm, width, increment, dte, iv, confidence, max_risk
) -> SpreadRecommendation:
    """Sell ATM call, buy OTM call. Credit spread."""
    short_strike = atm
    long_strike = atm + width
    expiry_str = find_standard_expiry(date.today(), 30, 45).isoformat()

    short_prem = estimate_option_price(ticker, "call", short_strike, stock_price, dte, iv)
    long_prem = estimate_option_price(ticker, "call", long_strike, stock_price, dte, iv)
    net_credit = short_prem - long_prem

    credit = net_credit * 100
    max_loss = (width - net_credit) * 100
    rr = credit / max_loss if max_loss > 0 else 0
    breakeven = short_strike + net_credit

    short_leg = OptionLeg(ticker, "call", short_strike, expiry_str, "sell", short_prem)
    long_leg = OptionLeg(ticker, "call", long_strike, expiry_str, "buy", long_prem)

    liquidity_ok = True
    rejection = None
    for leg in [short_leg, long_leg]:
        ok, reason = check_liquidity(leg)
        if not ok:
            liquidity_ok = False
            rejection = f"{leg.strike} {leg.option_type}: {reason}"
            break

    if max_loss > max_risk:
        liquidity_ok = False
        rejection = f"Max loss ${max_loss:.0f} exceeds max risk ${max_risk:.0f}"

    return SpreadRecommendation(
        ticker=ticker, direction="bear",
        spread_type="bear_call_spread", spread_category="credit",
        legs=[asdict(short_leg), asdict(long_leg)],
        max_risk=round(max_loss, 2), max_reward=round(credit, 2),
        reward_risk_ratio=round(rr, 2),
        breakeven=round(breakeven, 2),
        net_premium=round(credit, 2),
        width=width, confidence=confidence,
        rationale=f"Medium confidence bear signal. Sell {short_strike}C / Buy {long_strike}C. "
                  f"Credit ${credit:.0f}, max risk ${max_loss:.0f}.",
        liquidity_ok=liquidity_ok, pricing_source="bs_estimated",
        rejection_reason=rejection,
    )


# ---------------------------------------------------------------------------
# Process signals -> spread recommendations
# ---------------------------------------------------------------------------

def load_signals() -> Dict[str, Any]:
    """Load agentic signals from state file."""
    path = os.path.join(STATE_DIR, "agentic_signals.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning(f"Could not load signals: {e}")
        return {}


def get_stock_price_estimate(ticker: str, signal: dict) -> float:
    """
    Estimate current stock price from signal data.
    Uses recommended_strike as proxy (ATM), or falls back to heuristic.
    """
    if "recommended_strike" in signal:
        return signal["recommended_strike"]
    # Fallback: use estimated_cost to back out approximate price
    return 50.0  # generic fallback


def process_all_signals() -> List[Dict[str, Any]]:
    """Process all signals and generate spread recommendations."""
    data = load_signals()
    if not data:
        log.warning("No signals loaded")
        return []

    account_equity = data.get("account_equity", ACCOUNT_EQUITY)
    signals = data.get("signals", [])
    recommendations = []

    for sig in signals:
        ticker = sig.get("ticker", "")
        direction = sig.get("direction", "")
        confidence = sig.get("confidence_score", 0)
        stock_price = get_stock_price_estimate(ticker, sig)

        # Map direction
        mapped_dir = "bull" if direction in ("bull", "bullish", "long") else "bear"

        # Estimate IV from signal context
        iv = 0.30  # default
        if "RSI" in sig.get("reason", ""):
            # Higher IV estimate for volatile names
            iv = 0.35

        # Determine DTE from signal's recommended expiry
        dte = 30
        if "recommended_expiry" in sig:
            try:
                exp_date = datetime.strptime(sig["recommended_expiry"], "%Y-%m-%d")
                dte = max(1, (exp_date - datetime.now()).days)
            except ValueError:
                pass

        spread = select_spread(
            ticker=ticker,
            direction=mapped_dir,
            confidence=confidence,
            stock_price=stock_price,
            days_to_expiry=dte,
            iv=iv,
            account_equity=account_equity,
        )

        if spread:
            rec = asdict(spread)
            rec["source_signal"] = {
                "confirming_sources": sig.get("confirming_sources", []),
                "n_confirming": sig.get("n_confirming", 0),
                "exit_guidance": sig.get("exit_guidance", {}),
            }
            recommendations.append(rec)
            status = "PASS" if spread.liquidity_ok else f"REJECTED: {spread.rejection_reason}"
            log.info(f"{ticker} {mapped_dir} conf={confidence:.0%} -> {spread.spread_type} [{status}]")

    return recommendations


def save_recommendations(recs: List[Dict[str, Any]]) -> str:
    """Save recommendations to state file."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "total_signals": len(recs),
        "actionable": sum(1 for r in recs if r.get("liquidity_ok")),
        "recommendations": recs,
    }
    path = os.path.join(STATE_DIR, "spread_recommendations.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved {len(recs)} recommendations ({output['actionable']} actionable) to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Spread Strategy Engine — starting")
    recs = process_all_signals()
    path = save_recommendations(recs)
    log.info(f"Done. Output: {path}")

    # Print summary
    actionable = [r for r in recs if r.get("liquidity_ok")]
    print(f"\n{'='*60}")
    print(f"SPREAD RECOMMENDATIONS SUMMARY")
    print(f"{'='*60}")
    print(f"Total signals processed: {len(recs)}")
    print(f"Actionable spreads: {len(actionable)}")
    for r in actionable:
        print(f"  {r['ticker']:6s} {r['spread_type']:20s} risk=${r['max_risk']:.0f}  "
              f"reward=${r['max_reward']:.0f}  R:R={r['reward_risk_ratio']:.1f}x  "
              f"conf={r['confidence']:.0%}")
    rejected = [r for r in recs if not r.get("liquidity_ok")]
    if rejected:
        print(f"\nRejected ({len(rejected)}):")
        for r in rejected:
            print(f"  {r['ticker']:6s} — {r.get('rejection_reason', 'unknown')}")


if __name__ == "__main__":
    main()
