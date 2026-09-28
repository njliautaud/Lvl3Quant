"""
Exit Rule Engine — evaluates ALL exit conditions for a position.

Returns a prioritized list of triggered rules. The FIRST triggered rule wins.
Pure logic — no API calls, no file I/O. Takes position dict + market data dict.

Usage:
    from lib.exit_rules import evaluate_exits
    result = evaluate_exits(position, market_data)
    if result.triggered:
        print(f"EXIT: {result.rule_name} — {result.reason}")
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional
from lib.constants import (
    LV_TP_PCT, LV_SL_PCT, LV_TRAIL_ACTIVATE, LV_TRAIL_GIVEBACK,
    HV_TP_PCT, HV_SL_PCT, HV_TRAIL_ACTIVATE, HV_TRAIL_GIVEBACK,
    HV_DAY1_EARLY_EXIT, VIX_REGIME_THRESHOLD,
    SECTOR_MAX_HOLD_DAYS, DEFAULT_MAX_HOLD_DAYS,
    HV_HOLD_REDUCTION, MIN_HOLD_DAYS,
    THETA_EXIT_DTE, THESIS_BREAK_PCT,
    DEAD_MONEY_LOW, DEAD_MONEY_HIGH,
    MACRO_REGIMES,
)


@dataclass
class ExitResult:
    triggered: bool
    rule_name: str           # e.g. "take_profit", "stop_loss", "trailing_stop"
    reason: str              # Human-readable explanation
    action: str              # "close_all", "close_half", "hold"
    urgency: str             # "immediate", "eod", "next_check"
    pnl_pct: float           # Current P&L %
    details: dict            # Extra data for logging


@dataclass
class MarketData:
    """Everything the exit engine needs about current market state."""
    current_mark: float       # Current option mark/mid price
    underlying_price: float   # Current underlying price
    vix: float                # Current VIX level
    macro_regime: str         # RISK_ON, RISK_OFF, TRANSITION, etc.
    market_open_time: Optional[datetime] = None  # When market opened today


@dataclass
class Position:
    """Normalized position data."""
    ticker: str
    option_type: str          # "call" or "put"
    strike: float
    expiration: str           # YYYY-MM-DD
    entry_price: float        # Per-contract entry price
    entry_date: str           # YYYY-MM-DD
    underlying_entry_price: float  # Underlying price at entry
    quantity: int
    peak_value: float         # Highest mark since entry
    n_sources: int            # Number of confirming sources at entry (for graduated exit)
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    trailing_active: bool = False


def _trading_days_since(entry_date_str: str) -> int:
    """Count weekdays from entry_date to today (exclusive of entry day)."""
    entry = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
    today = date.today()
    count = 0
    current = entry
    while current < today:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return count


def _days_to_expiry(expiry_str: str) -> int:
    """Calendar days to expiry."""
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    return (expiry - date.today()).days


def _is_high_vol(vix: float) -> bool:
    return vix >= VIX_REGIME_THRESHOLD


def _get_params(vix: float, macro_regime: str = "RISK_ON"):
    """Get the correct exit params based on VIX regime + macro override."""
    hv = _is_high_vol(vix)
    tp_pct = HV_TP_PCT if hv else LV_TP_PCT
    sl_pct = HV_SL_PCT if hv else LV_SL_PCT
    trail_activate = HV_TRAIL_ACTIVATE if hv else LV_TRAIL_ACTIVATE
    trail_giveback = HV_TRAIL_GIVEBACK if hv else LV_TRAIL_GIVEBACK

    # Macro regime can tighten SL further
    regime_cfg = MACRO_REGIMES.get(macro_regime, {})
    sl_override = regime_cfg.get("sl_override")
    if sl_override is not None and sl_override < sl_pct:
        sl_pct = sl_override

    return tp_pct, sl_pct, trail_activate, trail_giveback


def _get_max_hold(ticker: str, vix: float) -> int:
    """Sector-specific hold period, reduced in high-vol."""
    base = SECTOR_MAX_HOLD_DAYS.get(ticker, DEFAULT_MAX_HOLD_DAYS)
    if _is_high_vol(vix):
        base = max(MIN_HOLD_DAYS, base - HV_HOLD_REDUCTION)
    return base


def evaluate_exits(position: Position, market: MarketData) -> ExitResult:
    """
    Evaluate ALL exit rules in priority order. Returns first triggered rule.
    If nothing triggers, returns a "hold" result.

    Priority order matches the prompt spec:
    (a) Take Profit
    (b) Stop Loss (with graduated exit for high-conviction)
    (c) Trailing Stop
    (d) Time Stop
    (e) Theta Decay
    (f) Thesis Break
    (g) Dead Money
    (h) Macro Regime tightening (handled via SL param override)
    """
    entry = position.entry_price
    mark = market.current_mark
    pnl_pct = (mark - entry) / entry  # Fractional, e.g. 0.15 = +15%
    pnl_pct_display = pnl_pct * 100   # For display

    tp_pct, sl_pct, trail_activate, trail_giveback = _get_params(
        market.vix, market.macro_regime
    )
    hv = _is_high_vol(market.vix)
    hold_days = _trading_days_since(position.entry_date)
    dte = _days_to_expiry(position.expiration)
    max_hold = _get_max_hold(position.ticker, market.vix)

    # ── (a) TAKE PROFIT ──
    # Skip if GTC TP order is on the books (it'll execute automatically)
    if position.tp_order_id is None:
        tp_price = entry * (1 + tp_pct)
        if mark >= tp_price:
            return ExitResult(
                triggered=True, rule_name="take_profit",
                reason=f"TP hit: mark ${mark:.2f} >= target ${tp_price:.2f} (+{pnl_pct_display:.1f}%)",
                action="close_all", urgency="immediate",
                pnl_pct=pnl_pct_display,
                details={"tp_price": tp_price, "tp_pct": tp_pct, "regime": "HV" if hv else "LV"}
            )

    # ── (b) STOP LOSS ──
    # Skip if GTC SL order is on the books
    if position.sl_order_id is None:
        sl_price = entry * (1 - sl_pct)

        # High-vol day-1 early exit
        if hv and hold_days == 0 and pnl_pct < -HV_DAY1_EARLY_EXIT:
            return ExitResult(
                triggered=True, rule_name="hv_early_exit",
                reason=f"HV day-1 early exit: {pnl_pct_display:+.1f}% loss on first day (threshold -{HV_DAY1_EARLY_EXIT*100:.0f}%)",
                action="close_all", urgency="immediate",
                pnl_pct=pnl_pct_display,
                details={"sl_price": sl_price, "hold_days": 0, "vix": market.vix}
            )

        # Check first 30 min no-stop rule
        now = datetime.now()
        in_first_30_min = False
        if market.market_open_time:
            minutes_since_open = (now - market.market_open_time).total_seconds() / 60
            in_first_30_min = minutes_since_open < 30

        if mark <= sl_price and not in_first_30_min:
            # Graduated exit for high-conviction (5+ sources)
            if position.n_sources >= 5:
                return ExitResult(
                    triggered=True, rule_name="stop_loss_graduated",
                    reason=f"SL hit (graduated): mark ${mark:.2f} <= ${sl_price:.2f} ({pnl_pct_display:+.1f}%). "
                           f"High-conviction ({position.n_sources} sources) — sell HALF, hold rest with -30% hard floor.",
                    action="close_half", urgency="immediate",
                    pnl_pct=pnl_pct_display,
                    details={"sl_price": sl_price, "hard_floor": entry * 0.70, "n_sources": position.n_sources}
                )
            else:
                return ExitResult(
                    triggered=True, rule_name="stop_loss",
                    reason=f"SL hit: mark ${mark:.2f} <= ${sl_price:.2f} ({pnl_pct_display:+.1f}%). "
                           f"Standard conviction ({position.n_sources} sources) — close all.",
                    action="close_all", urgency="immediate",
                    pnl_pct=pnl_pct_display,
                    details={"sl_price": sl_price, "n_sources": position.n_sources}
                )

    # ── (c) TRAILING STOP ──
    trail_trigger_price = entry * (1 + trail_activate)
    peak = position.peak_value

    # Update peak
    if mark > peak:
        peak = mark

    # Check if trailing should be active
    trailing_active = position.trailing_active or (peak >= trail_trigger_price)

    if trailing_active and peak > entry:
        # Trail floor = peak minus (giveback% * gain from entry)
        gain_from_entry = peak - entry
        trail_floor = peak - (gain_from_entry * trail_giveback)

        if mark < trail_floor:
            return ExitResult(
                triggered=True, rule_name="trailing_stop",
                reason=f"Trailing stop: mark ${mark:.2f} < floor ${trail_floor:.2f}. "
                       f"Peak was ${peak:.2f}, gave back {trail_giveback*100:.0f}% of ${gain_from_entry:.2f} gain.",
                action="close_all", urgency="immediate",
                pnl_pct=pnl_pct_display,
                details={"peak": peak, "trail_floor": trail_floor, "giveback": trail_giveback}
            )

    # ── (d) TIME STOP ──
    if hold_days >= max_hold:
        return ExitResult(
            triggered=True, rule_name="time_stop",
            reason=f"Time stop: held {hold_days} trading days (max {max_hold} for {position.ticker}). "
                   f"P&L {pnl_pct_display:+.1f}%.",
            action="close_all", urgency="eod",
            pnl_pct=pnl_pct_display,
            details={"hold_days": hold_days, "max_hold": max_hold, "ticker": position.ticker}
        )

    # ── (e) THETA DECAY ──
    if dte <= THETA_EXIT_DTE and pnl_pct <= 0:
        return ExitResult(
            triggered=True, rule_name="theta_decay",
            reason=f"Theta exit: {dte} DTE with {pnl_pct_display:+.1f}% P&L. "
                   f"Theta is eating premium faster than signal can work.",
            action="close_all", urgency="immediate",
            pnl_pct=pnl_pct_display,
            details={"dte": dte, "threshold_dte": THETA_EXIT_DTE}
        )

    # ── (f) THESIS BREAK ──
    if position.underlying_entry_price > 0 and market.underlying_price > 0:
        underlying_move = (market.underlying_price - position.underlying_entry_price) / position.underlying_entry_price
        # For calls, underlying dropping >5% = thesis broken
        # For puts, underlying rallying >5% = thesis broken
        if position.option_type == "call" and underlying_move < -THESIS_BREAK_PCT:
            return ExitResult(
                triggered=True, rule_name="thesis_break",
                reason=f"Thesis break: {position.ticker} dropped {underlying_move*100:+.1f}% from entry "
                       f"(${position.underlying_entry_price:.2f} -> ${market.underlying_price:.2f}). Call thesis invalid.",
                action="close_all", urgency="immediate",
                pnl_pct=pnl_pct_display,
                details={"underlying_move": underlying_move, "entry_underlying": position.underlying_entry_price}
            )
        elif position.option_type == "put" and underlying_move > THESIS_BREAK_PCT:
            return ExitResult(
                triggered=True, rule_name="thesis_break",
                reason=f"Thesis break: {position.ticker} rallied {underlying_move*100:+.1f}% from entry "
                       f"(${position.underlying_entry_price:.2f} -> ${market.underlying_price:.2f}). Put thesis invalid.",
                action="close_all", urgency="immediate",
                pnl_pct=pnl_pct_display,
                details={"underlying_move": underlying_move, "entry_underlying": position.underlying_entry_price}
            )

    # ── (g) DEAD MONEY ──
    if hold_days >= 1 and DEAD_MONEY_LOW < pnl_pct < DEAD_MONEY_HIGH:
        return ExitResult(
            triggered=True, rule_name="dead_money",
            reason=f"Dead money: held {hold_days} days, P&L {pnl_pct_display:+.1f}% (between "
                   f"{DEAD_MONEY_LOW*100:+.0f}% and {DEAD_MONEY_HIGH*100:+.0f}%). Capital better used elsewhere.",
            action="close_all", urgency="eod",
            pnl_pct=pnl_pct_display,
            details={"hold_days": hold_days}
        )

    # ── NO TRIGGER — HOLD ──
    proximity_warnings = []
    # Check proximity to each exit
    if position.tp_order_id is None:
        tp_price = entry * (1 + tp_pct)
        tp_away = (tp_price - mark) / tp_price * 100
        if 0 < tp_away <= 5:
            proximity_warnings.append(f"TP {tp_away:.1f}% away")

    if position.sl_order_id is None:
        sl_price = entry * (1 - sl_pct)
        if mark > sl_price:
            sl_away = (mark - sl_price) / mark * 100
            if sl_away <= 5:
                proximity_warnings.append(f"SL {sl_away:.1f}% away")

    if hold_days >= max_hold - 1:
        proximity_warnings.append(f"Time stop in {max_hold - hold_days}d")

    return ExitResult(
        triggered=False, rule_name="hold",
        reason=f"Holding. P&L {pnl_pct_display:+.1f}%, day {hold_days}/{max_hold}, {dte} DTE. "
               + (f"Warnings: {', '.join(proximity_warnings)}" if proximity_warnings else "No triggers near."),
        action="hold", urgency="next_check",
        pnl_pct=pnl_pct_display,
        details={
            "peak": peak, "hold_days": hold_days, "max_hold": max_hold,
            "dte": dte, "trailing_active": trailing_active,
            "proximity_warnings": proximity_warnings,
            "bracket_tp": position.tp_order_id is not None,
            "bracket_sl": position.sl_order_id is not None,
        }
    )


def compute_bracket_prices(entry_price: float, vix: float, macro_regime: str = "RISK_ON") -> dict:
    """
    Compute exact bracket order prices for a new entry.
    Returns dict with tp_limit, sl_stop, sl_limit prices.
    """
    tp_pct, sl_pct, _, _ = _get_params(vix, macro_regime)

    tp_limit = round(entry_price * (1 + tp_pct), 2)
    sl_stop = round(entry_price * (1 - sl_pct), 2)
    sl_limit = round(sl_stop * (1 - 0.05), 2)  # 5% below trigger to ensure fill

    return {
        "tp_limit_price": tp_limit,
        "tp_pct": tp_pct,
        "sl_stop_price": sl_stop,
        "sl_limit_price": sl_limit,
        "sl_pct": sl_pct,
        "regime": "HV" if _is_high_vol(vix) else "LV",
        "vix": vix,
        "macro_regime": macro_regime,
    }
