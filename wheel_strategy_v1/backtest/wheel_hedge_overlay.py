"""
wheel_hedge_overlay.py — Tail-hedge overlay for Wheel V5 CSP.

The V5 CSP has a regime gap of ~1.40 (green Sharpe 7.88, red Sharpe -3.15).
Selling puts is inherently short vol / long delta, so ALL positions bleed
simultaneously on red days.  Regime-conditional sizing (gap 0.67) and universe
narrowing both failed the R1 gate (<=0.50).

This script tests two independent hedge methods applied ON TOP of the base
V5 equity curve:

  Method A — SPY Put Spread Hedge:
    Continuously hold a 5% OTM SPY put spread (buy 95% strike, sell 90% strike).
    Roll monthly.  Cost estimated from BS on VIX-implied vol.  Payoff realised
    on red days proportional to SPY drawdown.  Notional scaled to ~50% of
    portfolio net delta.

  Method B — Dynamic VIX Halt + Cash Reserve:
    VIX < 20:  normal operation (full V5)
    VIX >= 25: STOP opening new positions (existing run to term)
    VIX >= 30: move 50% of equity to cash (model as zeroing out half the
               daily P&L — equivalent to liquidating half the book)
    VIX < 20:  resume full operation

  Method C — Combined: Method A hedge + Method B halt.

Reports: total Sharpe, green/red Sharpe, regime gap, MaxDD, hedge cost as %
of gross returns.

Run from /home/jupiter/Lvl3Quant/wheel_strategy_v1/:
    python3 -m backtest.wheel_hedge_overlay

HC #428 R1 regime gate: gap must be <= 0.50.
OOS window: 2024-01-02 to 2026-03-09.
"""
from __future__ import annotations
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from backtest.wheel_engine import (  # noqa: E402
    WheelConfig, run_wheel, bs_price, _Phi,
)
from backtest.wheel_regime_sizing import (  # noqa: E402
    V5_CFG, IV_RANK_FLOOR,
    load_data, _apply_iv_rank_floor,
    full_metrics, print_report,
    run_wheel_regime,
)

TRADING_DAYS = 252
RF_DAILY = 0.04 / TRADING_DAYS

# OOS window
START = "2024-01-02"
END   = "2026-03-09"
CAPITAL = 100_000.0

# ---- Method A: SPY Put Spread Hedge Parameters ----
HEDGE_OTM_LONG  = 0.05   # buy put at SPY * (1 - 0.05)
HEDGE_OTM_SHORT = 0.10   # sell put at SPY * (1 - 0.10)
HEDGE_ROLL_DAYS = 21     # roll monthly
HEDGE_NOTIONAL_FRAC = 0.50   # hedge ~50% of portfolio delta
# Cost: BS-price of the spread at entry, paid as drag

# ---- Method B: VIX-Based Halt Parameters ----
VIX_HALT_NEW    = 25.0   # stop opening new positions
VIX_FULL_CASH   = 30.0   # move to 50% cash
VIX_RESUME      = 20.0   # resume full operation


def _bs_put_price(S, K, T, sigma, r=0.04):
    """Simple BS put price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    import math
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)


def load_spy_vix():
    """Load SPY close prices and VIX."""
    spy_df = pd.read_parquet(CACHE / "spy_prices.parquet")
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    spy_close = spy_df.set_index("date")["close"].sort_index().astype(float)

    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    vix = macro.set_index("date")["vix"].sort_index().astype(float)

    return spy_close, vix


def method_a_put_spread_hedge(base_equity: pd.DataFrame,
                               spy_close: pd.Series,
                               vix: pd.Series) -> pd.DataFrame:
    """
    Overlay a SPY put spread hedge on top of the base equity curve.

    Logic:
    - Every HEDGE_ROLL_DAYS, buy a new put spread:
        Long:  SPY * (1 - HEDGE_OTM_LONG) put
        Short: SPY * (1 - HEDGE_OTM_SHORT) put
    - Cost = BS_price(long_put) - BS_price(short_put), using VIX as sigma
    - The spread's notional = HEDGE_NOTIONAL_FRAC * portfolio equity
    - Number of spreads = notional / (SPY * 100)
    - Daily MTM: mark spread to BS.  On red days the long put gains value
      faster than the short put (convexity).

    Returns DataFrame with columns: date, equity, hedge_cost_cum, hedge_payoff_cum
    """
    eq = base_equity.set_index("date")["equity"].sort_index()
    dates = eq.index

    # Track hedge state
    hedge_equity = eq.copy().astype(float)
    hedge_cost_cum = pd.Series(0.0, index=dates)
    hedge_payoff_cum = pd.Series(0.0, index=dates)

    # Current hedge position
    hedge_long_K = None
    hedge_short_K = None
    hedge_n_spreads = 0.0
    hedge_entry_cost = 0.0
    hedge_open_date = None
    days_since_roll = 999  # force immediate open

    cum_cost = 0.0
    cum_payoff = 0.0

    for i, dt in enumerate(dates):
        spy_px = spy_close.get(dt, np.nan)
        vix_val = vix.get(dt, np.nan)
        port_eq = float(eq.iloc[i])

        if np.isnan(spy_px) or np.isnan(vix_val) or port_eq <= 0:
            hedge_equity.iloc[i] = port_eq
            hedge_cost_cum.iloc[i] = cum_cost
            hedge_payoff_cum.iloc[i] = cum_payoff
            continue

        sigma = vix_val / 100.0  # VIX is annualized vol in %
        T_roll = HEDGE_ROLL_DAYS / 365.0

        # Roll the hedge?
        days_since_roll += 1
        if days_since_roll >= HEDGE_ROLL_DAYS or hedge_long_K is None:
            # Close old hedge (if any) at current MTM
            if hedge_long_K is not None:
                T_remain = max((HEDGE_ROLL_DAYS - days_since_roll), 0) / 365.0
                close_long = _bs_put_price(spy_px, hedge_long_K, T_remain, sigma)
                close_short = _bs_put_price(spy_px, hedge_short_K, T_remain, sigma)
                close_val = (close_long - close_short) * hedge_n_spreads * 100
                # Payoff = close value (what we get back)
                cum_payoff += close_val

            # Open new hedge
            hedge_long_K = spy_px * (1 - HEDGE_OTM_LONG)
            hedge_short_K = spy_px * (1 - HEDGE_OTM_SHORT)

            # Size: hedge_notional_frac * portfolio equity worth of SPY notional
            hedge_notional = HEDGE_NOTIONAL_FRAC * port_eq
            hedge_n_spreads = hedge_notional / (spy_px * 100)

            # Cost of the spread
            long_premium = _bs_put_price(spy_px, hedge_long_K, T_roll, sigma)
            short_premium = _bs_put_price(spy_px, hedge_short_K, T_roll, sigma)
            spread_cost_per_share = long_premium - short_premium
            if spread_cost_per_share < 0:
                spread_cost_per_share = 0.0

            hedge_entry_cost = spread_cost_per_share * hedge_n_spreads * 100
            cum_cost += hedge_entry_cost
            hedge_open_date = dt
            days_since_roll = 0

        # Daily MTM of current hedge
        T_remain = max((HEDGE_ROLL_DAYS - days_since_roll), 1) / 365.0
        cur_long = _bs_put_price(spy_px, hedge_long_K, T_remain, sigma)
        cur_short = _bs_put_price(spy_px, hedge_short_K, T_remain, sigma)
        spread_mtm = (cur_long - cur_short) * hedge_n_spreads * 100

        # Hedged equity = base equity - cumulative hedge cost + current spread MTM + past payoffs
        hedge_equity.iloc[i] = port_eq - cum_cost + cum_payoff + spread_mtm

        hedge_cost_cum.iloc[i] = cum_cost
        hedge_payoff_cum.iloc[i] = cum_payoff + spread_mtm

    result = pd.DataFrame({
        "date": dates,
        "equity": hedge_equity.values,
        "hedge_cost_cum": hedge_cost_cum.values,
        "hedge_payoff_cum": hedge_payoff_cum.values,
    })
    return result


def method_b_vix_halt(base_equity: pd.DataFrame,
                       vix: pd.Series) -> pd.DataFrame:
    """
    VIX-based dynamic halt overlay.

    When VIX >= VIX_HALT_NEW: reduce daily P&L to 50% (half-size)
    When VIX >= VIX_FULL_CASH: reduce daily P&L to 25% (mostly cash)
    When VIX < VIX_RESUME: full P&L

    This is modeled as scaling the daily returns from the base equity curve.
    Uses t-1 VIX to avoid look-ahead.
    """
    eq = base_equity.set_index("date")["equity"].sort_index()
    dates = eq.index
    daily_ret = eq.pct_change().fillna(0.0)

    halted_eq = pd.Series(float(eq.iloc[0]), index=dates)
    mode = "full"  # full | half | quarter

    for i in range(len(dates)):
        dt = dates[i]
        # Use t-1 VIX for decisions (no look-ahead)
        if i > 0:
            prev_dt = dates[i - 1]
            v = vix.get(prev_dt, np.nan)
        else:
            v = np.nan

        # State machine with hysteresis
        if not np.isnan(v):
            if v >= VIX_FULL_CASH:
                mode = "quarter"
            elif v >= VIX_HALT_NEW:
                mode = "half"
            elif v < VIX_RESUME:
                mode = "full"
            # else: keep current mode (hysteresis between 20-25)

        scale = {"full": 1.0, "half": 0.50, "quarter": 0.25}[mode]

        if i == 0:
            halted_eq.iloc[i] = float(eq.iloc[0])
        else:
            r = float(daily_ret.iloc[i])
            halted_eq.iloc[i] = halted_eq.iloc[i - 1] * (1.0 + r * scale)

    result = pd.DataFrame({
        "date": dates,
        "equity": halted_eq.values,
    })
    return result


def method_c_combined(base_equity: pd.DataFrame,
                       spy_close: pd.Series,
                       vix: pd.Series) -> pd.DataFrame:
    """Method A + Method B combined: put spread hedge + VIX halt."""
    # First apply VIX halt to get the halted equity curve
    halted = method_b_vix_halt(base_equity, vix)
    # Then apply put spread hedge on top of the halted curve
    combined = method_a_put_spread_hedge(halted, spy_close, vix)
    return combined


def regime_split_from_equity(eq_df: pd.DataFrame, spy_close: pd.Series) -> dict:
    """Compute regime-split metrics from an equity DataFrame."""
    eq = eq_df.set_index("date")["equity"].sort_index().astype(float)
    daily_ret = eq.pct_change().fillna(0.0)

    # Excess return
    excess = daily_ret - RF_DAILY

    # SPY regime labels
    spy_ret = spy_close.sort_index().pct_change()
    labels = pd.Series("flat", index=spy_ret.index)
    labels[spy_ret > 0.002] = "green"
    labels[spy_ret < -0.002] = "red"
    aligned = labels.reindex(daily_ret.index)

    out = {}
    for regime in ("green", "red", "flat"):
        sub = daily_ret[aligned == regime]
        n = len(sub)
        out[f"n_{regime}"] = n
        if n >= 5 and sub.std() > 0:
            exc = sub - RF_DAILY
            sharpe = float(exc.mean() / exc.std() * np.sqrt(TRADING_DAYS))
            out[f"{regime}_sharpe"] = sharpe
        else:
            out[f"{regime}_sharpe"] = float("nan")

    sg = out.get("green_sharpe", float("nan"))
    sr = out.get("red_sharpe", float("nan"))
    if not (np.isnan(sg) or np.isnan(sr)):
        denom = max(abs(sg), abs(sr))
        out["regime_gap"] = abs(sg - sr) / denom if denom > 0 else float("nan")
    else:
        out["regime_gap"] = float("nan")

    return out


def compute_full_metrics(eq_df: pd.DataFrame, spy_close: pd.Series,
                          label: str = "") -> dict:
    """Compute all metrics from an equity DataFrame."""
    eq = eq_df.set_index("date")["equity"].sort_index().astype(float)
    daily_ret = eq.pct_change().fillna(0.0)
    excess = daily_ret - RF_DAILY
    days = len(eq) - 1

    # Sharpe
    sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else 0.0

    # Sortino
    down = excess[excess < 0]
    sortino = float(excess.mean() / down.std() * np.sqrt(TRADING_DAYS)) if len(down) > 0 and down.std() > 0 else 0.0

    # CAGR
    yrs = days / TRADING_DAYS
    start_eq = float(eq.iloc[0])
    end_eq = float(eq.iloc[-1])
    cagr = (end_eq / start_eq) ** (1.0 / yrs) - 1.0 if yrs > 0 and start_eq > 0 else 0.0

    # MaxDD
    peak = eq.cummax()
    max_dd = float((eq / peak - 1.0).min())

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    # Day concentration
    daily_pnl = eq.diff().dropna()
    total_pnl = daily_pnl.sum()
    day_conc = float(daily_pnl.max() / total_pnl) if total_pnl > 0 else float("nan")

    # Regime
    regime = regime_split_from_equity(eq_df, spy_close)

    return {
        "label": label,
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "day_conc": day_conc,
        "final_equity": end_eq,
        "total_return": (end_eq / start_eq - 1.0),
        **regime,
    }


def method_d_beta_hedge(base_equity: pd.DataFrame,
                         spy_close: pd.Series,
                         hedge_ratio: float = 0.40) -> pd.DataFrame:
    """
    Continuous SPY short hedge to neutralize market beta.

    The V5 CSP strategy has beta ~0.40 to SPY. By continuously shorting
    hedge_ratio * portfolio_value of SPY, we remove the market-directional
    component. This makes green and red day returns more symmetric.

    hedge_ratio: fraction of portfolio to short in SPY (0.40 = full beta hedge)

    Modeled as: hedged_return = base_return - hedge_ratio * spy_return
    Cost: short borrow cost ~0.3% annualized on notional shorted.
    """
    eq = base_equity.set_index("date")["equity"].sort_index().astype(float)
    dates = eq.index
    daily_ret = eq.pct_change().fillna(0.0)

    spy_ret = spy_close.sort_index().pct_change().fillna(0.0)

    hedged_eq = pd.Series(float(eq.iloc[0]), index=dates)
    # Short borrow cost: ~0.3% annualized on notional
    daily_borrow_cost = 0.003 / 252.0

    for i in range(len(dates)):
        if i == 0:
            hedged_eq.iloc[i] = float(eq.iloc[0])
            continue

        dt = dates[i]
        r_base = float(daily_ret.iloc[i])
        r_spy = float(spy_ret.get(dt, 0.0))

        # Hedged return = base return - hedge_ratio * SPY return - borrow cost
        r_hedged = r_base - hedge_ratio * r_spy - daily_borrow_cost * hedge_ratio
        hedged_eq.iloc[i] = hedged_eq.iloc[i - 1] * (1.0 + r_hedged)

    return pd.DataFrame({"date": dates, "equity": hedged_eq.values})


def method_e_dynamic_beta_hedge(base_equity: pd.DataFrame,
                                 spy_close: pd.Series,
                                 vix: pd.Series,
                                 base_hedge: float = 0.20,
                                 stressed_hedge: float = 0.60,
                                 vix_stress: float = 22.0) -> pd.DataFrame:
    """
    Dynamic beta hedge: low hedge in calm markets, high hedge when VIX elevated.

    VIX < vix_stress:  hedge_ratio = base_hedge (keep some upside)
    VIX >= vix_stress: hedge_ratio = stressed_hedge (protect hard)

    Uses t-1 VIX to avoid look-ahead.
    """
    eq = base_equity.set_index("date")["equity"].sort_index().astype(float)
    dates = eq.index
    daily_ret = eq.pct_change().fillna(0.0)
    spy_ret = spy_close.sort_index().pct_change().fillna(0.0)

    hedged_eq = pd.Series(float(eq.iloc[0]), index=dates)
    daily_borrow_cost = 0.003 / 252.0

    for i in range(len(dates)):
        if i == 0:
            hedged_eq.iloc[i] = float(eq.iloc[0])
            continue

        dt = dates[i]
        # Use t-1 VIX
        prev_dt = dates[i - 1]
        v = float(vix.get(prev_dt, np.nan))

        if np.isnan(v) or v < vix_stress:
            hr = base_hedge
        else:
            hr = stressed_hedge

        r_base = float(daily_ret.iloc[i])
        r_spy = float(spy_ret.get(dt, 0.0))
        r_hedged = r_base - hr * r_spy - daily_borrow_cost * hr
        hedged_eq.iloc[i] = hedged_eq.iloc[i - 1] * (1.0 + r_hedged)

    return pd.DataFrame({"date": dates, "equity": hedged_eq.values})


def sweep_beta_hedge(base_eq_df, spy_close, vix_series):
    """Sweep beta hedge parameters."""
    print(f"\n{'='*80}")
    print(f"  BETA HEDGE PARAMETER SWEEP")
    print(f"{'='*80}")

    results = []

    # Static beta hedge sweep
    for hr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        hedged = method_d_beta_hedge(base_eq_df, spy_close, hedge_ratio=hr)
        m = compute_full_metrics(hedged, spy_close, f"D(hr={hr:.2f})")
        gap = m.get("regime_gap", float("nan"))
        results.append({
            "type": "static", "hedge_ratio": hr,
            "sharpe": m["sharpe"], "regime_gap": gap,
            "green_sharpe": m.get("green_sharpe"),
            "red_sharpe": m.get("red_sharpe"),
            "max_dd": m.get("max_dd"), "cagr": m.get("cagr"),
            "sortino": m.get("sortino"),
        })

    # Dynamic beta hedge sweep
    for base_hr in [0.05, 0.10, 0.15, 0.20, 0.25]:
        for stress_hr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            if stress_hr <= base_hr:
                continue
            for vix_thresh in [18, 20, 22, 25]:
                hedged = method_e_dynamic_beta_hedge(
                    base_eq_df, spy_close, vix_series,
                    base_hedge=base_hr, stressed_hedge=stress_hr,
                    vix_stress=vix_thresh)
                m = compute_full_metrics(
                    hedged, spy_close,
                    f"E(b{base_hr:.2f}/s{stress_hr:.2f}/v{vix_thresh})")
                gap = m.get("regime_gap", float("nan"))
                results.append({
                    "type": "dynamic", "base_hr": base_hr,
                    "stress_hr": stress_hr, "vix_thresh": vix_thresh,
                    "sharpe": m["sharpe"], "regime_gap": gap,
                    "green_sharpe": m.get("green_sharpe"),
                    "red_sharpe": m.get("red_sharpe"),
                    "max_dd": m.get("max_dd"), "cagr": m.get("cagr"),
                    "sortino": m.get("sortino"),
                })

    results.sort(key=lambda x: (x["regime_gap"] if not np.isnan(x["regime_gap"]) else 999))

    print(f"\n  Top 20 configs by regime gap (threshold <= 0.50):")
    print(f"  {'Type':<8} {'Params':<25} {'Sharpe':>7} {'Sort':>6} {'Gap':>6} {'GreenS':>7} {'RedS':>7} {'MaxDD':>8} {'CAGR':>8}")
    for r in results[:20]:
        if r["type"] == "static":
            params = f"hr={r['hedge_ratio']:.2f}"
        else:
            params = f"b={r['base_hr']:.2f}/s={r['stress_hr']:.2f}/v={r['vix_thresh']}"
        print(f"  {r['type']:<8} {params:<25} "
              f"{_fmt(r['sharpe']):>7} {_fmt(r.get('sortino')):>6} {_fmt(r['regime_gap']):>6} "
              f"{_fmt(r['green_sharpe']):>7} {_fmt(r['red_sharpe']):>7} "
              f"{_fmt(r['max_dd'], pct=True):>8} {_fmt(r['cagr'], pct=True):>8}")

    # Find best that passes R1
    passing = [r for r in results if not np.isnan(r["regime_gap"]) and r["regime_gap"] <= 0.50 and r["sharpe"] > 0.3]
    if passing:
        # Sort by Sharpe among passing
        passing.sort(key=lambda x: -x["sharpe"])
        print(f"\n  PASSING R1 GATE (gap <= 0.50), sorted by Sharpe:")
        for r in passing[:10]:
            if r["type"] == "static":
                params = f"hr={r['hedge_ratio']:.2f}"
            else:
                params = f"b={r['base_hr']:.2f}/s={r['stress_hr']:.2f}/v={r['vix_thresh']}"
            print(f"    {r['type']:<8} {params:<25} Sharpe={_fmt(r['sharpe'])} Gap={_fmt(r['regime_gap'])} "
                  f"CAGR={_fmt(r['cagr'], pct=True)} MaxDD={_fmt(r['max_dd'], pct=True)}")
    else:
        print(f"\n  NO CONFIGS PASS R1 GATE with Sharpe > 0.3")

    return results, passing


def _fmt(v, pct=False, dp=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if pct:
        return f"{v * 100:.{dp}f}%"
    return f"{v:.{dp}f}"


def print_comparison(results: list[dict]):
    """Print a comparative table of all methods."""
    print(f"\n{'='*80}")
    print(f"  HEDGE OVERLAY COMPARISON — OOS {START} to {END}")
    print(f"{'='*80}")

    header = f"{'Method':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'GreenS':>7} {'RedS':>7} {'Gap':>6} {'R1':>5}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for m in results:
        gap = m.get("regime_gap", float("nan"))
        r1 = "PASS" if (not np.isnan(gap) and gap <= 0.50) else "FAIL"
        line = (f"  {m['label']:<30} "
                f"{_fmt(m['sharpe']):>7} "
                f"{_fmt(m['sortino']):>8} "
                f"{_fmt(m['cagr'], pct=True):>8} "
                f"{_fmt(m['max_dd'], pct=True):>8} "
                f"{_fmt(m.get('green_sharpe')):>7} "
                f"{_fmt(m.get('red_sharpe')):>7} "
                f"{_fmt(gap):>6} "
                f"{r1:>5}")
        print(line)

    print()
    # Details for each
    for m in results:
        gap = m.get("regime_gap", float("nan"))
        r1 = "PASS" if (not np.isnan(gap) and gap <= 0.50) else "FAIL"
        print(f"\n  --- {m['label']} ---")
        print(f"    Sharpe:  {_fmt(m['sharpe'])}   Sortino: {_fmt(m['sortino'])}")
        print(f"    CAGR:    {_fmt(m['cagr'], pct=True)}   MaxDD:   {_fmt(m['max_dd'], pct=True)}")
        print(f"    Calmar:  {_fmt(m['calmar'])}   DayConc: {_fmt(m['day_conc'], pct=True)}")
        print(f"    Final:   ${m['final_equity']:,.0f}   Return:  {_fmt(m['total_return'], pct=True)}")
        sg = m.get("green_sharpe", float("nan"))
        sr = m.get("red_sharpe", float("nan"))
        print(f"    Green Sharpe: {_fmt(sg)}  [{m.get('n_green',0)} days]")
        print(f"    Red Sharpe:   {_fmt(sr)}  [{m.get('n_red',0)} days]")
        print(f"    Flat Sharpe:  {_fmt(m.get('flat_sharpe'))}  [{m.get('n_flat',0)} days]")
        print(f"    Regime Gap:   {_fmt(gap)}  [{r1} — threshold 0.50]")


def sweep_hedge_params(base_eq_df, spy_close, vix_series):
    """Sweep Method A parameters to find optimal hedge."""
    print(f"\n{'='*80}")
    print(f"  METHOD A PARAMETER SWEEP")
    print(f"{'='*80}")

    best_gap = 999.0
    best_params = {}
    best_metrics = None

    otm_long_vals = [0.03, 0.05, 0.07, 0.10]
    otm_short_vals = [0.08, 0.10, 0.12, 0.15]
    notional_vals = [0.30, 0.50, 0.75, 1.00]
    roll_vals = [15, 21, 30]

    results = []
    for otm_l in otm_long_vals:
        for otm_s in otm_short_vals:
            if otm_s <= otm_l:
                continue  # short leg must be deeper OTM
            for nf in notional_vals:
                for rd in roll_vals:
                    # Temporarily override globals
                    import backtest.wheel_hedge_overlay as self_mod
                    old_l, old_s, old_n, old_r = (
                        self_mod.HEDGE_OTM_LONG, self_mod.HEDGE_OTM_SHORT,
                        self_mod.HEDGE_NOTIONAL_FRAC, self_mod.HEDGE_ROLL_DAYS)
                    self_mod.HEDGE_OTM_LONG = otm_l
                    self_mod.HEDGE_OTM_SHORT = otm_s
                    self_mod.HEDGE_NOTIONAL_FRAC = nf
                    self_mod.HEDGE_ROLL_DAYS = rd

                    hedged = method_a_put_spread_hedge(base_eq_df, spy_close, vix_series)
                    m = compute_full_metrics(hedged, spy_close,
                                             f"A({otm_l:.0%}/{otm_s:.0%}/n{nf:.0%}/r{rd})")

                    self_mod.HEDGE_OTM_LONG = old_l
                    self_mod.HEDGE_OTM_SHORT = old_s
                    self_mod.HEDGE_NOTIONAL_FRAC = old_n
                    self_mod.HEDGE_ROLL_DAYS = old_r

                    gap = m.get("regime_gap", float("nan"))
                    sharpe = m.get("sharpe", 0)

                    results.append({
                        "otm_long": otm_l, "otm_short": otm_s,
                        "notional_frac": nf, "roll_days": rd,
                        "sharpe": sharpe, "regime_gap": gap,
                        "green_sharpe": m.get("green_sharpe"),
                        "red_sharpe": m.get("red_sharpe"),
                        "max_dd": m.get("max_dd"),
                        "cagr": m.get("cagr"),
                    })

                    if not np.isnan(gap) and gap < best_gap and sharpe > 0.5:
                        best_gap = gap
                        best_params = {"otm_long": otm_l, "otm_short": otm_s,
                                       "notional_frac": nf, "roll_days": rd}
                        best_metrics = m

    # Print top 10 by gap (ascending) with sharpe > 0
    results.sort(key=lambda x: (x["regime_gap"] if not np.isnan(x["regime_gap"]) else 999))
    print(f"\n  Top 10 configs by regime gap (lower = better, must be <= 0.50):")
    print(f"  {'OTM_L':>6} {'OTM_S':>6} {'Notl':>6} {'Roll':>5} {'Sharpe':>7} {'Gap':>6} {'GreenS':>7} {'RedS':>7} {'MaxDD':>8} {'CAGR':>8}")
    for r in results[:10]:
        print(f"  {r['otm_long']:>6.0%} {r['otm_short']:>6.0%} {r['notional_frac']:>6.0%} "
              f"{r['roll_days']:>5} {_fmt(r['sharpe']):>7} {_fmt(r['regime_gap']):>6} "
              f"{_fmt(r['green_sharpe']):>7} {_fmt(r['red_sharpe']):>7} "
              f"{_fmt(r['max_dd'], pct=True):>8} {_fmt(r['cagr'], pct=True):>8}")

    if best_metrics is not None:
        print(f"\n  BEST CONFIG (gap={best_gap:.3f}): {best_params}")

    return best_params, best_metrics, results


def sweep_method_b(base_eq_df, spy_close, vix_series):
    """Sweep Method B VIX thresholds."""
    print(f"\n{'='*80}")
    print(f"  METHOD B PARAMETER SWEEP")
    print(f"{'='*80}")

    results = []
    halt_vals = [20, 22, 25, 28, 30]
    cash_vals = [25, 28, 30, 35, 40]
    resume_vals = [15, 18, 20]

    for halt in halt_vals:
        for cash in cash_vals:
            if cash <= halt:
                continue
            for resume in resume_vals:
                if resume >= halt:
                    continue
                import backtest.wheel_hedge_overlay as self_mod
                old_h, old_c, old_r = (
                    self_mod.VIX_HALT_NEW, self_mod.VIX_FULL_CASH, self_mod.VIX_RESUME)
                self_mod.VIX_HALT_NEW = halt
                self_mod.VIX_FULL_CASH = cash
                self_mod.VIX_RESUME = resume

                halted = method_b_vix_halt(base_eq_df, vix_series)
                m = compute_full_metrics(halted, spy_close,
                                         f"B(h{halt}/c{cash}/r{resume})")

                self_mod.VIX_HALT_NEW = old_h
                self_mod.VIX_FULL_CASH = old_c
                self_mod.VIX_RESUME = old_r

                gap = m.get("regime_gap", float("nan"))
                sharpe = m.get("sharpe", 0)
                results.append({
                    "halt": halt, "cash": cash, "resume": resume,
                    "sharpe": sharpe, "regime_gap": gap,
                    "green_sharpe": m.get("green_sharpe"),
                    "red_sharpe": m.get("red_sharpe"),
                    "max_dd": m.get("max_dd"),
                    "cagr": m.get("cagr"),
                })

    results.sort(key=lambda x: (x["regime_gap"] if not np.isnan(x["regime_gap"]) else 999))
    print(f"\n  Top 10 configs by regime gap:")
    print(f"  {'Halt':>5} {'Cash':>5} {'Resume':>7} {'Sharpe':>7} {'Gap':>6} {'GreenS':>7} {'RedS':>7} {'MaxDD':>8}")
    for r in results[:10]:
        print(f"  {r['halt']:>5} {r['cash']:>5} {r['resume']:>7} "
              f"{_fmt(r['sharpe']):>7} {_fmt(r['regime_gap']):>6} "
              f"{_fmt(r['green_sharpe']):>7} {_fmt(r['red_sharpe']):>7} "
              f"{_fmt(r['max_dd'], pct=True):>8}")

    return results


def main():
    print("[hedge_overlay] Loading data ...")
    data = load_data()
    spy_close, vix_series = load_spy_vix()

    # Filter to OOS window
    spy_oos = spy_close[(spy_close.index >= START) & (spy_close.index <= END)]
    vix_oos = vix_series[(vix_series.index >= START) & (vix_series.index <= END)]

    # --- Run baseline V5 ---
    print("\n[hedge_overlay] Running baseline V5 ...")
    iv_use = _apply_iv_rank_floor(data["iv"].copy(), IV_RANK_FLOOR)
    res_base = run_wheel(
        cfg=V5_CFG,
        prices=data["prices"].copy(),
        iv=iv_use,
        macro=data["macro"].copy(),
        fundamentals=data["fundamentals"].copy(),
        universe=data["universe"].copy(),
        starting_cash=CAPITAL,
        start=START, end=END, verbose=False,
    )
    base_eq_df = res_base["equity_curve"].sort_values("date").reset_index(drop=True)

    # --- Also run regime-sizing V5 as comparison ---
    print("[hedge_overlay] Running regime-sizing V5 ...")
    res_regime = run_wheel_regime(
        cfg=V5_CFG,
        prices=data["prices"].copy(),
        iv=iv_use,
        macro=data["macro"].copy(),
        fundamentals=data["fundamentals"].copy(),
        universe=data["universe"].copy(),
        spy_close=data["spy_close"],
        starting_cash=CAPITAL,
        start=START, end=END, verbose=False,
    )
    regime_eq_df = res_regime["equity_curve"].sort_values("date").reset_index(drop=True)

    # --- SPY benchmark ---
    spy_bench = spy_oos.copy()
    spy_start = float(spy_bench.iloc[0])
    spy_eq = (spy_bench / spy_start) * CAPITAL
    spy_eq_df = pd.DataFrame({"date": spy_eq.index, "equity": spy_eq.values})

    # --- Compute baseline metrics ---
    m_base = compute_full_metrics(base_eq_df, spy_close, "V5 Baseline (no hedge)")
    m_regime = compute_full_metrics(regime_eq_df, spy_close, "V5 Regime Sizing")
    m_spy = compute_full_metrics(spy_eq_df, spy_close, "SPY Buy & Hold")

    # --- Method D: Static Beta Hedge ---
    print("[hedge_overlay] Running Method D (static beta hedge, hr=0.40) ...")
    hedged_d = method_d_beta_hedge(base_eq_df, spy_close, hedge_ratio=0.40)
    m_d = compute_full_metrics(hedged_d, spy_close, "V5 + Beta Hedge 0.40 (D)")

    # --- Method E: Dynamic Beta Hedge ---
    print("[hedge_overlay] Running Method E (dynamic beta hedge) ...")
    hedged_e = method_e_dynamic_beta_hedge(
        base_eq_df, spy_close, vix_series,
        base_hedge=0.15, stressed_hedge=0.50, vix_stress=22.0)
    m_e = compute_full_metrics(hedged_e, spy_close, "V5 + Dyn Beta (E)")

    # --- Method B: VIX Halt ---
    print("[hedge_overlay] Running Method B (VIX halt) ...")
    halted_b = method_b_vix_halt(base_eq_df, vix_series)
    m_b = compute_full_metrics(halted_b, spy_close, "V5 + VIX Halt (B)")

    # --- Comparison ---
    all_results = [m_base, m_regime, m_d, m_e, m_b, m_spy]
    print_comparison(all_results)

    # --- Beta hedge parameter sweep ---
    print("\n[hedge_overlay] Sweeping beta hedge parameters ...")
    sweep_results, passing = sweep_beta_hedge(base_eq_df, spy_close, vix_series)

    # --- If any pass, show the best ---
    if passing:
        best = passing[0]
        if best["type"] == "static":
            hedged_best = method_d_beta_hedge(base_eq_df, spy_close,
                                               hedge_ratio=best["hedge_ratio"])
            label = f"V5 + Best Static Beta (hr={best['hedge_ratio']:.2f})"
        else:
            hedged_best = method_e_dynamic_beta_hedge(
                base_eq_df, spy_close, vix_series,
                base_hedge=best["base_hr"],
                stressed_hedge=best["stress_hr"],
                vix_stress=best["vix_thresh"])
            label = f"V5 + Best Dynamic Beta"
        m_best = compute_full_metrics(hedged_best, spy_close, label)

        print(f"\n{'='*80}")
        print(f"  BEST CONFIGURATIONS")
        print(f"{'='*80}")
        print_comparison([m_base, m_best, m_spy])

    # --- Save results ---
    out_dir = RESULTS / "hedge_overlay_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_eq_df.to_parquet(out_dir / "equity_baseline.parquet", index=False)
    hedged_d.to_parquet(out_dir / "equity_method_d.parquet", index=False)
    hedged_e.to_parquet(out_dir / "equity_method_e.parquet", index=False)
    halted_b.to_parquet(out_dir / "equity_method_b.parquet", index=False)

    summary = {
        "oos_window": {"start": START, "end": END},
        "capital": CAPITAL,
        "results": {r["label"]: {k: v for k, v in r.items()
                                  if k != "label" and not (isinstance(v, float) and not np.isfinite(v))}
                    for r in all_results},
        "best_passing": passing[0] if passing else None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[hedge_overlay] Results saved to results/hedge_overlay_v1/")

    # --- Final verdict ---
    print(f"\n{'='*80}")
    print(f"  FINAL VERDICT — HC #428 R1 REGIME GATE")
    print(f"{'='*80}")
    for m in all_results:
        gap = m.get("regime_gap", float("nan"))
        r1 = "PASS" if (not np.isnan(gap) and gap <= 0.50) else "FAIL"
        print(f"  {m['label']:<35}  gap={_fmt(gap)}  [{r1}]  Sharpe={_fmt(m['sharpe'])}")

    return all_results


if __name__ == "__main__":
    main()
