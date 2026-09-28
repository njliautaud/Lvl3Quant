"""
ETF Rotation Drawdown Protection Overlay v1

Tests whether adding drawdown protection overlays improves the ETF rotation
strategy's risk-adjusted returns.

Base strategy: etf_rotation_v3 quality config
  Sharpe 2.50, CAGR 28.2%, MaxDD -6.7%, 30-day rebalancing

Overlays tested:
  1. Trailing stop: 5% portfolio DD -> 100% cash until recovery
  2. VIX regime: VIX>30 -> 50% size, VIX>40 -> cash
  3. Momentum death cross: SPY 20-SMA < 50-SMA -> 50% allocation
  4. Combined: best of the above

Adversarial gates:
  - Permutation test (p < 0.05)
  - Regime symmetry (|Sharpe_green - Sharpe_red|/max < 0.50)
  - Sub-period: positive in both halves
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = ROOT / "output/etf_rotation_drawdown_overlay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
TXN_COST_BPS = 5.0

# ============================================================================
# METRICS (self-contained, no dependency on walk_forward.py)
# ============================================================================

def sharpe(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def sortino(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 2:
        return float("nan")
    down = r[r < 0]
    if len(down) < 1 or down.std() == 0:
        return float("nan")
    return float(r.mean() / down.std() * np.sqrt(TRADING_DAYS))


def cagr(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 2:
        return float("nan")
    eq = (1.0 + r).cumprod()
    n_years = len(r) / TRADING_DAYS
    if n_years <= 0:
        return float("nan")
    return float(eq.iloc[-1] ** (1.0 / n_years) - 1.0)


def max_dd(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 2:
        return float("nan")
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())


def calmar(rets: pd.Series) -> float:
    c = cagr(rets)
    m = max_dd(rets)
    if not np.isfinite(c) or not np.isfinite(m) or m == 0:
        return float("nan")
    return float(c / abs(m))


def profit_factor(rets: pd.Series) -> float:
    r = rets.dropna()
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return float("nan")
    return float(pos / neg)


def win_rate(rets: pd.Series) -> float:
    r = rets.dropna()
    if len(r) < 1:
        return float("nan")
    return float((r > 0).mean())


def compute_metrics(rets: pd.Series) -> dict:
    return {
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "cagr": cagr(rets),
        "max_dd": max_dd(rets),
        "calmar": calmar(rets),
        "pf": profit_factor(rets),
        "wr": win_rate(rets),
        "n_days": int(rets.dropna().shape[0]),
    }


# ============================================================================
# DATA LOADING (yfinance)
# ============================================================================

def load_data(start: str = "2010-01-01", end: str = "2026-07-23") -> dict:
    """Load SPY, VIX, SHY, and sector ETFs via yfinance."""
    import yfinance as yf

    sector_etfs = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
    all_tickers = ["SPY", "SHY", "^VIX"] + sector_etfs

    print(f"Downloading data for {len(all_tickers)} tickers from {start} to {end}...")
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns: (Price, Ticker)
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["Adj Close"]
    close = close.dropna(how="all")

    # Rename ^VIX -> VIX
    if "^VIX" in close.columns:
        close = close.rename(columns={"^VIX": "VIX"})

    print(f"Data loaded: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return {
        "close": close,
        "spy": close["SPY"].dropna(),
        "vix": close["VIX"].dropna() if "VIX" in close.columns else None,
        "shy": close["SHY"].dropna() if "SHY" in close.columns else None,
        "sector_etfs": sector_etfs,
    }


# ============================================================================
# BASE STRATEGY: ETF rotation (simplified version of quality config)
# ============================================================================

def run_base_rotation(data: dict, hold_days: int = 30, n_long: int = 3,
                      train_months: int = 12) -> pd.Series:
    """
    Simplified ETF rotation: rank sectors by momentum, pick top N.
    Uses walk-forward to avoid lookahead.
    Returns daily portfolio returns.
    """
    close = data["close"]
    spy = data["spy"]
    sector_etfs = data["sector_etfs"]

    # Get sector returns
    sector_close = close[sector_etfs].dropna(how="all")
    sector_rets = sector_close.pct_change()

    # SPY regime filter (60-day MA)
    spy_ma60 = spy.rolling(60, min_periods=30).mean()
    is_bull = spy > spy_ma60

    # Features for ranking (no lookahead — all backward-looking)
    ret_20d = sector_close.pct_change(20)
    ret_60d = sector_close.pct_change(60)
    # Relative strength vs SPY
    spy_ret_20d = spy.pct_change(20)
    rel_str = ret_20d.sub(spy_ret_20d, axis=0)

    # Combined momentum score (z-scored cross-sectionally each day)
    def xs_zscore(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)

    score = (xs_zscore(ret_20d).fillna(0) * 0.4 +
             xs_zscore(ret_60d).fillna(0) * 0.3 +
             xs_zscore(rel_str).fillna(0) * 0.3)

    # Walk-forward: rebalance every hold_days
    all_dates = sector_rets.index
    # Need enough warmup for 60-day lookback
    warmup = max(60, 20 * train_months)
    tradeable_dates = all_dates[warmup:]

    portfolio_rets = pd.Series(0.0, index=tradeable_dates, dtype=float)
    rebal_dates = tradeable_dates[::hold_days]

    current_holdings = []
    current_weight = 0.0
    n_rebals = 0

    for i, date in enumerate(tradeable_dates):
        if date in rebal_dates:
            # Check regime filter
            bull = is_bull.get(date, False)
            if not bull:
                current_holdings = []
                current_weight = 0.0
                portfolio_rets.loc[date] = 0.0
                continue

            # Get scores for this date (backward-looking only)
            day_scores = score.loc[date].dropna()
            if len(day_scores) < n_long:
                current_holdings = []
                current_weight = 0.0
                portfolio_rets.loc[date] = 0.0
                continue

            # Pick top N
            top_n = day_scores.nlargest(n_long).index.tolist()

            # Transaction cost on turnover
            old_set = set(current_holdings)
            new_set = set(top_n)
            turnover = len(old_set.symmetric_difference(new_set))
            tc = (turnover / max(len(new_set), 1)) * TXN_COST_BPS / 10000.0

            current_holdings = top_n
            current_weight = 1.0 / n_long
            n_rebals += 1

            # Today's return = equal-weight of holdings minus tc
            day_ret = sector_rets.loc[date, current_holdings].mean()
            portfolio_rets.loc[date] = (day_ret if pd.notna(day_ret) else 0.0) - tc
        else:
            # Hold period
            if current_holdings:
                # Check intra-hold regime
                bull = is_bull.get(date, True)
                if not bull:
                    portfolio_rets.loc[date] = 0.0
                    continue
                day_ret = sector_rets.loc[date, current_holdings].mean()
                portfolio_rets.loc[date] = day_ret if pd.notna(day_ret) else 0.0
            else:
                portfolio_rets.loc[date] = 0.0

    print(f"Base rotation: {n_rebals} rebalances, {len(tradeable_dates)} days")
    return portfolio_rets.dropna()


# ============================================================================
# OVERLAY 1: TRAILING STOP
# ============================================================================

def overlay_trailing_stop(base_rets: pd.Series, dd_threshold: float = 0.05,
                          recovery_mode: str = "hwm_or_rally",
                          rally_days: int = 10) -> tuple[pd.Series, dict]:
    """
    If portfolio drawdown exceeds dd_threshold, go to cash until:
    - New high watermark, OR
    - rally_days consecutive positive days

    Returns (modified_rets, stats).
    """
    eq = (1.0 + base_rets).cumprod()
    hwm = eq.cummax()
    dd = (eq - hwm) / hwm

    modified = base_rets.copy()
    in_cash = False
    cash_entry_date = None
    n_triggers = 0
    n_recovery = 0
    consecutive_pos = 0
    trigger_dates = []

    for i, (date, ret) in enumerate(base_rets.items()):
        if in_cash:
            # Check recovery conditions
            if recovery_mode == "hwm_or_rally":
                # Track underlying equity (what WOULD have happened)
                if ret > 0:
                    consecutive_pos += 1
                else:
                    consecutive_pos = 0

                # Recovery: new HWM in underlying or sustained rally
                if eq.loc[date] >= hwm.loc[date] or consecutive_pos >= rally_days:
                    in_cash = False
                    n_recovery += 1
                    consecutive_pos = 0

            if in_cash:
                modified.iloc[i] = 0.0  # Cash return (SHY ~ 0 for simplicity)
        else:
            # Check if we should enter cash
            if dd.loc[date] < -dd_threshold:
                in_cash = True
                cash_entry_date = date
                n_triggers += 1
                trigger_dates.append(str(date.date()))
                consecutive_pos = 0
                modified.iloc[i] = 0.0

    stats = {
        "n_triggers": n_triggers,
        "n_recoveries": n_recovery,
        "trigger_dates": trigger_dates[:20],  # Cap for readability
        "days_in_cash": int((modified == 0.0).sum()),
        "pct_in_cash": float((modified == 0.0).mean() * 100),
    }
    return modified, stats


# ============================================================================
# OVERLAY 2: VIX REGIME
# ============================================================================

def overlay_vix_regime(base_rets: pd.Series, vix: pd.Series,
                       vix_half: float = 30.0, vix_full: float = 40.0) -> tuple[pd.Series, dict]:
    """
    VIX > vix_half: reduce to 50% allocation.
    VIX > vix_full: go 100% cash.
    """
    # Align VIX to returns index
    vix_aligned = vix.reindex(base_rets.index).ffill()

    modified = base_rets.copy()
    n_half = 0
    n_full = 0

    for i, (date, ret) in enumerate(base_rets.items()):
        v = vix_aligned.get(date)
        if v is None or pd.isna(v):
            continue
        if v > vix_full:
            modified.iloc[i] = 0.0
            n_full += 1
        elif v > vix_half:
            modified.iloc[i] = ret * 0.5
            n_half += 1

    stats = {
        "n_half_size": n_half,
        "n_full_cash": n_full,
        "n_total_affected": n_half + n_full,
        "pct_affected": float((n_half + n_full) / len(base_rets) * 100),
    }
    return modified, stats


# ============================================================================
# OVERLAY 3: MOMENTUM DEATH CROSS
# ============================================================================

def overlay_death_cross(base_rets: pd.Series, spy: pd.Series,
                        fast_ma: int = 20, slow_ma: int = 50,
                        reduction: float = 0.5) -> tuple[pd.Series, dict]:
    """
    When SPY fast_ma < slow_ma (death cross), reduce allocation to `reduction`.
    """
    spy_fast = spy.rolling(fast_ma, min_periods=fast_ma).mean()
    spy_slow = spy.rolling(slow_ma, min_periods=slow_ma).mean()
    is_death_cross = spy_fast < spy_slow

    # Align to returns
    dc_aligned = is_death_cross.reindex(base_rets.index).ffill().fillna(False)

    modified = base_rets.copy()
    n_affected = 0

    for i, (date, ret) in enumerate(base_rets.items()):
        if dc_aligned.get(date, False):
            modified.iloc[i] = ret * reduction
            n_affected += 1

    stats = {
        "n_days_reduced": n_affected,
        "pct_days_reduced": float(n_affected / len(base_rets) * 100),
    }
    return modified, stats


# ============================================================================
# OVERLAY 4: COMBINED (best of above)
# ============================================================================

def overlay_combined(base_rets: pd.Series, vix: pd.Series, spy: pd.Series,
                     dd_threshold: float = 0.05, rally_days: int = 10,
                     vix_half: float = 30.0, vix_full: float = 40.0,
                     fast_ma: int = 20, slow_ma: int = 50) -> tuple[pd.Series, dict]:
    """
    Apply all three overlays. Most conservative (smallest allocation) wins.
    """
    eq = (1.0 + base_rets).cumprod()
    hwm = eq.cummax()
    dd = (eq - hwm) / hwm

    vix_aligned = vix.reindex(base_rets.index).ffill()
    spy_fast = spy.rolling(fast_ma, min_periods=fast_ma).mean()
    spy_slow = spy.rolling(slow_ma, min_periods=slow_ma).mean()
    is_dc = (spy_fast < spy_slow).reindex(base_rets.index).ffill().fillna(False)

    modified = base_rets.copy()
    in_cash_ts = False
    consecutive_pos = 0
    n_ts_triggers = 0
    n_vix_half = 0
    n_vix_full = 0
    n_dc = 0

    for i, (date, ret) in enumerate(base_rets.items()):
        alloc = 1.0
        reasons = []

        # Trailing stop check
        if in_cash_ts:
            if ret > 0:
                consecutive_pos += 1
            else:
                consecutive_pos = 0
            if eq.loc[date] >= hwm.loc[date] or consecutive_pos >= rally_days:
                in_cash_ts = False
                consecutive_pos = 0
            else:
                alloc = 0.0
                reasons.append("trailing_stop")
        else:
            if dd.loc[date] < -dd_threshold:
                in_cash_ts = True
                n_ts_triggers += 1
                consecutive_pos = 0
                alloc = 0.0
                reasons.append("trailing_stop")

        # VIX check (only if not already in cash from TS)
        v = vix_aligned.get(date)
        if v is not None and pd.notna(v):
            if v > vix_full:
                alloc = min(alloc, 0.0)
                n_vix_full += 1
                reasons.append("vix_cash")
            elif v > vix_half:
                alloc = min(alloc, 0.5)
                n_vix_half += 1
                reasons.append("vix_half")

        # Death cross check
        if is_dc.get(date, False):
            alloc = min(alloc, 0.5)
            n_dc += 1
            reasons.append("death_cross")

        modified.iloc[i] = ret * alloc

    stats = {
        "n_ts_triggers": n_ts_triggers,
        "n_vix_half": n_vix_half,
        "n_vix_full": n_vix_full,
        "n_dc_days": n_dc,
    }
    return modified, stats


# ============================================================================
# ADVERSARIAL GATES
# ============================================================================

def regime_test(rets: pd.Series, spy: pd.Series) -> dict:
    """
    HC #428 R1: |Sharpe_green - Sharpe_red| / max < 0.50
    Green = SPY daily return > +25bps, Red = < -25bps
    """
    spy_ret = spy.pct_change().reindex(rets.index)
    thr = 25 / 1e4

    green_mask = spy_ret > thr
    red_mask = spy_ret < -thr
    flat_mask = ~green_mask & ~red_mask

    sg = sharpe(rets[green_mask])
    sr = sharpe(rets[red_mask])
    sf = sharpe(rets[flat_mask])

    if np.isfinite(sg) and np.isfinite(sr):
        denom = max(abs(sg), abs(sr), 1e-9)
        skew = abs(sg - sr) / denom
    else:
        skew = float("nan")

    return {
        "sharpe_green": sg,
        "sharpe_red": sr,
        "sharpe_flat": sf,
        "n_green": int(green_mask.sum()),
        "n_red": int(red_mask.sum()),
        "n_flat": int(flat_mask.sum()),
        "regime_skew": skew,
        "PASS": bool(np.isfinite(skew) and skew <= 0.50),
    }


def sub_period_test(rets: pd.Series) -> dict:
    """Positive Sharpe in both halves of the sample."""
    mid = len(rets) // 2
    first_half = rets.iloc[:mid]
    second_half = rets.iloc[mid:]

    s1 = sharpe(first_half)
    s2 = sharpe(second_half)

    return {
        "sharpe_first_half": s1,
        "sharpe_second_half": s2,
        "first_half_dates": f"{first_half.index[0].date()} to {first_half.index[-1].date()}",
        "second_half_dates": f"{second_half.index[0].date()} to {second_half.index[-1].date()}",
        "PASS": bool(np.isfinite(s1) and s1 > 0 and np.isfinite(s2) and s2 > 0),
    }


def permutation_test(base_rets: pd.Series, overlay_rets: pd.Series,
                     n_perms: int = 1000, seed: int = 42) -> dict:
    """
    Shuffle the overlay mask (which days were affected) and compare.
    If the overlay improvement is > 95th percentile of random shuffles, p < 0.05.
    """
    rng = np.random.RandomState(seed)

    # Identify which days the overlay changed the return
    affected = (base_rets != overlay_rets)
    n_affected = int(affected.sum())

    if n_affected == 0:
        return {"p_value": 1.0, "PASS": False, "n_affected_days": 0, "note": "no overlay activity"}

    # Actual improvement in Sharpe
    actual_sharpe_diff = sharpe(overlay_rets) - sharpe(base_rets)

    # Permutation: randomly select same number of days to apply the overlay effect
    overlay_effect = overlay_rets.values - base_rets.values  # per-day delta
    actual_effect = overlay_effect[affected.values]  # only affected days

    perm_sharpe_diffs = []
    base_vals = base_rets.values.copy()
    n = len(base_vals)

    for _ in range(n_perms):
        # Randomly pick n_affected days and apply the overlay effects
        perm_idx = rng.choice(n, size=n_affected, replace=False)
        perm_rets = base_vals.copy()
        # Apply shuffled effects (sample from actual effects with replacement)
        shuffled_effects = rng.choice(actual_effect, size=n_affected, replace=True)
        perm_rets[perm_idx] += shuffled_effects
        perm_s = pd.Series(perm_rets)
        perm_sharpe = sharpe(perm_s)
        perm_sharpe_diffs.append(perm_sharpe - sharpe(base_rets))

    perm_arr = np.array(perm_sharpe_diffs)
    p_value = float(np.mean(perm_arr >= actual_sharpe_diff))

    return {
        "actual_sharpe_diff": actual_sharpe_diff,
        "perm_mean_diff": float(np.mean(perm_arr)),
        "perm_p95_diff": float(np.percentile(perm_arr, 95)),
        "p_value": p_value,
        "n_affected_days": n_affected,
        "n_perms": n_perms,
        "PASS": bool(p_value < 0.05),
    }


# ============================================================================
# FULL EVALUATION
# ============================================================================

def evaluate_overlay(name: str, base_rets: pd.Series, overlay_rets: pd.Series,
                     overlay_stats: dict, spy: pd.Series) -> dict:
    """Full evaluation of an overlay."""
    base_m = compute_metrics(base_rets)
    overlay_m = compute_metrics(overlay_rets)

    regime = regime_test(overlay_rets, spy)
    subperiod = sub_period_test(overlay_rets)
    perm = permutation_test(base_rets, overlay_rets)

    # Improvement deltas
    delta = {}
    for k in ["sharpe", "sortino", "cagr", "max_dd", "calmar"]:
        bv = base_m.get(k, float("nan"))
        ov = overlay_m.get(k, float("nan"))
        if np.isfinite(bv) and np.isfinite(ov):
            delta[k] = ov - bv
        else:
            delta[k] = float("nan")

    all_gates_pass = regime["PASS"] and subperiod["PASS"] and perm["PASS"]

    result = {
        "name": name,
        "base_metrics": base_m,
        "overlay_metrics": overlay_m,
        "improvement": delta,
        "overlay_stats": overlay_stats,
        "adversarial_gates": {
            "regime_test": regime,
            "sub_period_test": subperiod,
            "permutation_test": perm,
            "ALL_PASS": all_gates_pass,
        },
        "verdict": _verdict(overlay_m, delta, all_gates_pass),
    }
    return result


def _verdict(overlay_m: dict, delta: dict, gates_pass: bool) -> str:
    """Determine if the overlay is worth adding."""
    sharpe_improved = delta.get("sharpe", 0) > 0
    dd_improved = delta.get("max_dd", 0) > 0  # Less negative = better
    calmar_improved = delta.get("calmar", 0) > 0

    if not gates_pass:
        return "REJECT (fails adversarial gates)"

    if sharpe_improved and dd_improved and calmar_improved:
        return "ACCEPT (improves risk-adjusted returns on all axes)"
    elif dd_improved and calmar_improved:
        return "MARGINAL (reduces drawdown but may hurt Sharpe)"
    elif sharpe_improved:
        return "MARGINAL (improves Sharpe but not drawdown profile)"
    else:
        return "REJECT (no improvement on key metrics)"


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ETF ROTATION DRAWDOWN PROTECTION OVERLAY RESEARCH v1")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    data = load_data()
    spy = data["spy"]
    vix = data["vix"]

    if vix is None:
        print("ERROR: VIX data not available. Cannot run VIX overlay.")
        return

    # Run base strategy
    print("\n" + "=" * 70)
    print("RUNNING BASE ROTATION STRATEGY")
    print("=" * 70)
    base_rets = run_base_rotation(data, hold_days=30, n_long=3)

    base_m = compute_metrics(base_rets)
    print(f"\nBase strategy metrics:")
    print(f"  Sharpe:  {base_m['sharpe']:.3f}")
    print(f"  Sortino: {base_m['sortino']:.3f}")
    print(f"  CAGR:    {base_m['cagr']*100:.1f}%")
    print(f"  MaxDD:   {base_m['max_dd']*100:.1f}%")
    print(f"  Calmar:  {base_m['calmar']:.3f}")
    print(f"  PF:      {base_m['pf']:.3f}")
    print(f"  WR:      {base_m['wr']*100:.1f}%")
    print(f"  N days:  {base_m['n_days']}")

    results = {}

    # -----------------------------------------------------------------------
    # OVERLAY 1: Trailing Stop
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OVERLAY 1: TRAILING STOP (5% DD threshold)")
    print("=" * 70)

    ts_rets, ts_stats = overlay_trailing_stop(base_rets, dd_threshold=0.05, rally_days=10)
    ts_eval = evaluate_overlay("trailing_stop_5pct", base_rets, ts_rets, ts_stats, spy)
    results["trailing_stop"] = ts_eval

    _print_eval(ts_eval)

    # -----------------------------------------------------------------------
    # OVERLAY 2: VIX Regime
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OVERLAY 2: VIX REGIME (>30 half, >40 cash)")
    print("=" * 70)

    vix_rets, vix_stats = overlay_vix_regime(base_rets, vix, vix_half=30.0, vix_full=40.0)
    vix_eval = evaluate_overlay("vix_regime_30_40", base_rets, vix_rets, vix_stats, spy)
    results["vix_regime"] = vix_eval

    _print_eval(vix_eval)

    # -----------------------------------------------------------------------
    # OVERLAY 3: Momentum Death Cross
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OVERLAY 3: MOMENTUM DEATH CROSS (SPY 20/50 SMA)")
    print("=" * 70)

    dc_rets, dc_stats = overlay_death_cross(base_rets, spy, fast_ma=20, slow_ma=50, reduction=0.5)
    dc_eval = evaluate_overlay("death_cross_20_50", base_rets, dc_rets, dc_stats, spy)
    results["death_cross"] = dc_eval

    _print_eval(dc_eval)

    # -----------------------------------------------------------------------
    # OVERLAY 4: Combined
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OVERLAY 4: COMBINED (all three)")
    print("=" * 70)

    comb_rets, comb_stats = overlay_combined(
        base_rets, vix, spy,
        dd_threshold=0.05, rally_days=10,
        vix_half=30.0, vix_full=40.0,
        fast_ma=20, slow_ma=50,
    )
    comb_eval = evaluate_overlay("combined", base_rets, comb_rets, comb_stats, spy)
    results["combined"] = comb_eval

    _print_eval(comb_eval)

    # -----------------------------------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    header = f"{'Overlay':<20} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'Gates':>6} {'Verdict'}"
    print(header)
    print("-" * len(header))

    # Base
    print(f"{'Base (no overlay)':<20} {base_m['sharpe']:>8.3f} {base_m['sortino']:>8.3f} "
          f"{base_m['cagr']*100:>7.1f}% {base_m['max_dd']*100:>7.1f}% {base_m['calmar']:>8.3f} "
          f"{'---':>6} BASE")

    for name, eval_r in results.items():
        om = eval_r["overlay_metrics"]
        gates = eval_r["adversarial_gates"]["ALL_PASS"]
        v = eval_r["verdict"][:30]
        print(f"{name:<20} {om['sharpe']:>8.3f} {om['sortino']:>8.3f} "
              f"{om['cagr']*100:>7.1f}% {om['max_dd']*100:>7.1f}% {om['calmar']:>8.3f} "
              f"{'PASS' if gates else 'FAIL':>6} {v}")

    # -----------------------------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Convert to JSON-safe format
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    output = {
        "timestamp": datetime.now().isoformat(),
        "base_strategy": {
            "description": "ETF rotation v3 quality config (simplified)",
            "hold_days": 30,
            "n_long": 3,
            "regime_filter": "SPY > 60-day MA",
            "metrics": sanitize(base_m),
        },
        "overlays": sanitize(results),
        "recommendation": _final_recommendation(results),
    }

    outpath = OUTPUT_DIR / "results.json"
    outpath.write_text(json.dumps(output, indent=2, default=str))
    print(f"Results saved to {outpath}")

    # Final recommendation
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    print(output["recommendation"])


def _final_recommendation(results: dict) -> str:
    """Generate final recommendation based on all overlay results."""
    accepted = []
    rejected = []

    for name, eval_r in results.items():
        v = eval_r["verdict"]
        if "ACCEPT" in v:
            accepted.append(name)
        else:
            rejected.append(name)

    if not accepted:
        # Check if any improved drawdown even if gates failed
        dd_improvers = []
        for name, eval_r in results.items():
            delta_dd = eval_r["improvement"].get("max_dd", 0)
            if delta_dd is not None and np.isfinite(delta_dd) and delta_dd > 0:
                dd_improvers.append((name, delta_dd))

        if dd_improvers:
            best = max(dd_improvers, key=lambda x: x[1])
            return (f"No overlay passes all adversarial gates. "
                    f"However, '{best[0]}' improves MaxDD by {best[1]*100:.1f}pp. "
                    f"The overlays may still be useful as risk management but lack "
                    f"statistical robustness to prove they add alpha. "
                    f"The base strategy's existing SPY-MA60 regime filter already "
                    f"provides strong drawdown protection (MaxDD only ~6-7%). "
                    f"Adding more overlays risks over-fitting to historical crises "
                    f"that may not repeat in the same form.")
        else:
            return ("No overlay improves the strategy. The base strategy's existing "
                    "SPY-MA60 regime filter already provides effective drawdown protection. "
                    "Additional overlays hurt returns through whipsaw without meaningfully "
                    "reducing an already-small drawdown.")
    else:
        return (f"Accepted overlays: {', '.join(accepted)}. "
                f"These pass all adversarial gates (permutation p<0.05, regime symmetry, "
                f"sub-period consistency) and improve risk-adjusted returns.")


def _print_eval(eval_r: dict):
    """Print evaluation results."""
    om = eval_r["overlay_metrics"]
    delta = eval_r["improvement"]
    stats = eval_r["overlay_stats"]
    gates = eval_r["adversarial_gates"]

    print(f"\nOverlay metrics:")
    print(f"  Sharpe:  {om['sharpe']:.3f} (delta: {delta['sharpe']:+.3f})")
    print(f"  Sortino: {om['sortino']:.3f} (delta: {delta['sortino']:+.3f})")
    print(f"  CAGR:    {om['cagr']*100:.1f}% (delta: {delta['cagr']*100:+.1f}pp)")
    print(f"  MaxDD:   {om['max_dd']*100:.1f}% (delta: {delta['max_dd']*100:+.1f}pp)")
    print(f"  Calmar:  {om['calmar']:.3f} (delta: {delta['calmar']:+.3f})")

    print(f"\nOverlay activity:")
    for k, v in stats.items():
        if isinstance(v, list):
            continue
        print(f"  {k}: {v}")

    print(f"\nAdversarial gates:")
    rt = gates["regime_test"]
    print(f"  Regime test: Sharpe_green={rt['sharpe_green']:.3f}, "
          f"Sharpe_red={rt['sharpe_red']:.3f}, "
          f"skew={rt['regime_skew']:.3f} -> {'PASS' if rt['PASS'] else 'FAIL'}")

    sp = gates["sub_period_test"]
    print(f"  Sub-period: S1={sp['sharpe_first_half']:.3f}, "
          f"S2={sp['sharpe_second_half']:.3f} -> {'PASS' if sp['PASS'] else 'FAIL'}")

    pt = gates["permutation_test"]
    print(f"  Permutation: p={pt['p_value']:.4f}, "
          f"n_affected={pt['n_affected_days']} -> {'PASS' if pt['PASS'] else 'FAIL'}")

    print(f"\nVerdict: {eval_r['verdict']}")


if __name__ == "__main__":
    main()
