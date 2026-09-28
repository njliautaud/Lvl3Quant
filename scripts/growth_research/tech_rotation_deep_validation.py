#!/usr/bin/env python3
"""
Deep Validation of Tech Sub-Industry Rotation Strategy
=======================================================
Original claim: 41.3% CAGR, Sharpe 1.70, Sortino 1.83, Calmar 3.57, MaxDD -11.6%
Universe: {XLK, SMH, SOXX, XSW, IGV, CIBR, SKYY, HACK}

This script performs:
1. Reproduction of original results
2. Survivorship bias check (ETF inception dates)
3. Permutation test (100 trials) - does rotation add value vs random selection?
4. R1 regime test (green/red/flat Sharpe gap)
5. Transaction cost sensitivity (0, 5bps, 10bps slippage)
6. Leveraged version (SOXL/TECL substitution)
7. Comparison to TQQQ+200MA baseline
8. Rotation value-add test vs buy-and-hold XLK

Walk-forward: sliding window (HC #0)
Commission: $0 (Robinhood, HC #694)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import json
import warnings
import time
from datetime import datetime

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/tech_rotation_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Universe ───
TECH_UNIVERSE = ["XLK", "SMH", "SOXX", "XSW", "IGV", "CIBR", "SKYY", "HACK"]
LEVERAGED_MAP = {"SOXX": "SOXL", "SMH": "SOXL", "XLK": "TECL"}
BENCHMARKS = ["SPY", "TQQQ", "QQQ"]

# ─── Known ETF inception dates (for survivorship bias check) ───
ETF_INCEPTION = {
    "XLK":  "1998-12-22",
    "SMH":  "2000-05-05",
    "SOXX": "2001-07-13",
    "IGV":  "2001-07-13",
    "XSW":  "2006-09-28",
    "HACK": "2014-11-11",
    "CIBR": "2015-06-29",
    "SKYY": "2011-07-05",
    "SOXL": "2010-03-11",
    "TECL": "2008-12-17",
    "TQQQ": "2010-02-09",
}

# ─── Strategy parameters (matching original) ───
REGIME_MA = 60
HOLD_DAYS = 21
N_LONG = 2
TARGET_VOL = 0.15
LEV_MIN, LEV_MAX = 0.25, 2.0
MOM_LOOKBACKS = [20, 60]  # ret_20d, ret_60d used in original

np.random.seed(42)


def download_data(start="2005-01-01", end=None):
    """Download all needed tickers."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    all_tickers = sorted(set(TECH_UNIVERSE + list(LEVERAGED_MAP.values()) + BENCHMARKS))
    print(f"Downloading {len(all_tickers)} tickers...")

    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.ffill()

    # Report availability with inception check
    print(f"\nData range: {prices.index[0].date()} to {prices.index[-1].date()}")
    availability = {}
    for t in all_tickers:
        if t in prices.columns:
            first_valid = prices[t].first_valid_index()
            n_valid = prices[t].notna().sum()
            availability[t] = {
                "first_date": str(first_valid.date()) if first_valid else "N/A",
                "n_days": int(n_valid),
                "known_inception": ETF_INCEPTION.get(t, "unknown"),
            }
            print(f"  {t}: from {availability[t]['first_date']} ({n_valid} days) | inception: {ETF_INCEPTION.get(t, 'unknown')}")
        else:
            availability[t] = {"first_date": "N/A", "n_days": 0}
            print(f"  {t}: NOT AVAILABLE")

    return prices, availability


def compute_metrics(daily_returns, label=""):
    """Comprehensive risk-adjusted metrics."""
    dr = daily_returns.dropna()
    if len(dr) < 60 or dr.std() == 0:
        return {"label": label, "valid": False, "n_days": len(dr)}

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    down = dr[dr < 0]
    down_vol = down.std() * np.sqrt(252) if len(down) > 10 else ann_vol
    sortino = ann_ret / down_vol if down_vol > 0 else 0

    years = len(dr) / 252
    total_ret = (1 + dr).prod()
    cagr = total_ret ** (1 / years) - 1 if years > 0 and total_ret > 0 else 0

    cum = (1 + dr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wr = (dr > 0).mean()
    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    return {
        "label": label,
        "valid": True,
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_dd": round(max_dd * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "wr": round(wr * 100, 1),
        "pf": round(pf, 3),
        "years": round(years, 2),
        "n_days": len(dr),
        "total_return": round((total_ret - 1) * 100, 1),
    }


def spy_regime_classify(spy_prices, thresh=0.0005):
    """Classify days as green/red/flat by SPY close-to-close."""
    spy_ret = spy_prices.pct_change()
    regime = pd.Series("flat", index=spy_prices.index)
    regime[spy_ret > thresh] = "green"
    regime[spy_ret < -thresh] = "red"
    return regime


def regime_stratification(strat_returns, spy_returns, thresh=0.0005):
    """R1 regime test: compute Sharpe per regime and gap."""
    aligned = pd.DataFrame({"strat": strat_returns, "spy": spy_returns}).dropna()
    if len(aligned) < 100:
        return {"regime_gap": 999, "r1_pass": False, "regimes": {}}

    results = {}
    for name, mask in [
        ("green", aligned["spy"] > thresh),
        ("red", aligned["spy"] < -thresh),
        ("flat", ~(aligned["spy"] > thresh) & ~(aligned["spy"] < -thresh)),
    ]:
        r = aligned.loc[mask, "strat"]
        if len(r) > 20 and r.std() > 0:
            s = r.mean() / r.std() * np.sqrt(252)
        else:
            s = 0
        results[name] = {
            "sharpe": round(s, 3),
            "n_days": int(len(r)),
            "mean_ret": round(float(r.mean()) * 100, 4) if len(r) > 0 else 0,
        }

    g = results["green"]["sharpe"]
    r = results["red"]["sharpe"]
    max_s = max(abs(g), abs(r))
    gap = abs(g - r) / max_s if max_s > 0 else 999

    return {
        "green_sharpe": g,
        "red_sharpe": r,
        "flat_sharpe": results["flat"]["sharpe"],
        "regime_gap": round(gap, 4),
        "r1_pass": gap <= 0.50,
        "regimes": results,
    }


def run_momentum_rotation(prices, universe, spy_prices,
                          n_long=N_LONG, hold_days=HOLD_DAYS,
                          mom_lookbacks=MOM_LOOKBACKS,
                          regime_ma=REGIME_MA, target_vol=TARGET_VOL,
                          slippage_bps=0, use_regime=True,
                          shuffle_rankings=False):
    """
    Core rotation strategy.
    If shuffle_rankings=True, randomly shuffle the momentum rankings
    (for permutation test).
    """
    avail = [t for t in universe if t in prices.columns and prices[t].notna().sum() > 252]
    if len(avail) < n_long + 1:
        return pd.Series(dtype=float), []

    px = prices[avail].dropna(how="all")
    returns = px.pct_change()

    # Composite momentum score
    mom_scores = None
    for lb in mom_lookbacks:
        m = (px / px.shift(lb) - 1).rank(axis=1, pct=True)
        mom_scores = m if mom_scores is None else mom_scores + m
    mom_scores /= len(mom_lookbacks)

    # Regime filter
    if use_regime:
        spy_ma = spy_prices.rolling(regime_ma, min_periods=30).mean()
        risk_on = spy_prices > spy_ma
    else:
        risk_on = pd.Series(True, index=spy_prices.index)

    warmup = max(mom_lookbacks) + 10
    dates = px.index[warmup:]

    daily_rets = []
    rebal_dates = []
    positions = {}
    last_rebal = None
    prev_holdings = set()

    for date in dates:
        do_rebal = False
        if last_rebal is None:
            do_rebal = True
        elif (date - last_rebal).days >= hold_days:
            do_rebal = True

        if do_rebal:
            last_rebal = date
            rebal_dates.append(date)

            if date not in mom_scores.index:
                positions = {}
            else:
                scores = mom_scores.loc[date].dropna()

                # Shuffle for permutation test
                if shuffle_rankings:
                    scores = scores.sample(frac=1.0)

                ro = risk_on.get(date, False) if use_regime else True
                if len(scores) < n_long or not ro:
                    positions = {}
                else:
                    top = scores.nlargest(n_long)

                    # Vol-targeting
                    if target_vol > 0:
                        book_rets = returns[list(top.index)].loc[:date].tail(60).mean(axis=1).dropna()
                        if len(book_rets) > 20:
                            book_vol = book_rets.std() * np.sqrt(252)
                            if book_vol > 0:
                                lev = np.clip(target_vol / book_vol, LEV_MIN, LEV_MAX)
                            else:
                                lev = 1.0
                        else:
                            lev = 1.0
                    else:
                        lev = 1.0

                    positions = {t: lev / n_long for t in top.index}

                    # Slippage on turnover
                    new_holdings = set(top.index)
                    turnover = len(prev_holdings.symmetric_difference(new_holdings))
                    if turnover > 0 and slippage_bps > 0:
                        slip_cost = slippage_bps / 10000 * lev * (turnover / (2 * n_long))
                        daily_rets.append(-slip_cost)
                    prev_holdings = new_holdings

        day_ret = 0.0
        for ticker, weight in positions.items():
            if ticker in returns.columns and date in returns.index:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    day_ret += weight * r
        daily_rets.append(day_ret)

    # Pad to align with dates
    # Daily rets may be longer than dates due to slippage entries
    if len(daily_rets) > len(dates):
        # Slippage entries are extra; merge them into rebal day returns
        # Rebuild properly
        pass

    # Simpler approach: track properly
    result_rets = pd.Series(dtype=float)
    pos = {}
    prev_h = set()
    last_rb = None
    ret_list = []
    date_list = []

    for date in dates:
        do_rb = False
        if last_rb is None:
            do_rb = True
        elif (date - last_rb).days >= hold_days:
            do_rb = True

        slip_cost = 0.0
        if do_rb:
            last_rb = date
            if date not in mom_scores.index:
                pos = {}
            else:
                scores = mom_scores.loc[date].dropna()
                if shuffle_rankings:
                    scores = scores.sample(frac=1.0)

                ro = risk_on.get(date, False) if use_regime else True
                if len(scores) < n_long or not ro:
                    pos = {}
                else:
                    top = scores.nlargest(n_long)
                    book_rets = returns[list(top.index)].loc[:date].tail(60).mean(axis=1).dropna()
                    if target_vol > 0 and len(book_rets) > 20:
                        bv = book_rets.std() * np.sqrt(252)
                        lev = np.clip(target_vol / bv, LEV_MIN, LEV_MAX) if bv > 0 else 1.0
                    else:
                        lev = 1.0
                    pos = {t: lev / n_long for t in top.index}

                    new_h = set(top.index)
                    to = len(prev_h.symmetric_difference(new_h))
                    if to > 0 and slippage_bps > 0:
                        slip_cost = slippage_bps / 10000 * lev * (to / (2 * n_long))
                    prev_h = new_h

        day_ret = -slip_cost
        for ticker, weight in pos.items():
            if ticker in returns.columns and date in returns.index:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    day_ret += weight * r

        ret_list.append(day_ret)
        date_list.append(date)

    return pd.Series(ret_list, index=date_list), rebal_dates


def run_leveraged_rotation(prices, spy_prices, slippage_bps=0):
    """Run the rotation but substitute leveraged ETFs where available."""
    # Build execution universe: use leveraged where available
    exec_map = {}
    for t in TECH_UNIVERSE:
        if t in LEVERAGED_MAP and LEVERAGED_MAP[t] in prices.columns:
            exec_map[t] = LEVERAGED_MAP[t]
        else:
            exec_map[t] = t

    # Signal universe = unleveraged (rank on original momentum)
    signal_tickers = [t for t in TECH_UNIVERSE if t in prices.columns]
    exec_tickers = [exec_map.get(t, t) for t in signal_tickers]

    if len(signal_tickers) < N_LONG + 1:
        return pd.Series(dtype=float), []

    signal_px = prices[signal_tickers].dropna(how="all")
    exec_returns = prices[list(set(exec_tickers))].pct_change()

    # Momentum on signal (unleveraged)
    mom_scores = None
    for lb in MOM_LOOKBACKS:
        m = (signal_px / signal_px.shift(lb) - 1).rank(axis=1, pct=True)
        mom_scores = m if mom_scores is None else mom_scores + m
    mom_scores /= len(MOM_LOOKBACKS)

    spy_ma = spy_prices.rolling(REGIME_MA, min_periods=30).mean()
    risk_on = spy_prices > spy_ma

    warmup = max(MOM_LOOKBACKS) + 10
    dates = signal_px.index[warmup:]

    ret_list, date_list = [], []
    pos = {}
    last_rb = None
    prev_h = set()

    for date in dates:
        do_rb = last_rb is None or (date - last_rb).days >= HOLD_DAYS
        slip_cost = 0.0

        if do_rb:
            last_rb = date
            if date in mom_scores.index:
                scores = mom_scores.loc[date].dropna()
                ro = risk_on.get(date, False)
                if len(scores) >= N_LONG and ro:
                    top = scores.nlargest(N_LONG)
                    # Map to leveraged for execution
                    lev_holdings = {exec_map.get(t, t): 1.0 / N_LONG for t in top.index}

                    new_h = set(lev_holdings.keys())
                    to = len(prev_h.symmetric_difference(new_h))
                    if to > 0 and slippage_bps > 0:
                        slip_cost = slippage_bps / 10000 * (to / (2 * N_LONG))
                    prev_h = new_h
                    pos = lev_holdings
                else:
                    pos = {}
            else:
                pos = {}

        day_ret = -slip_cost
        for ticker, weight in pos.items():
            if ticker in exec_returns.columns and date in exec_returns.index:
                r = exec_returns.loc[date, ticker]
                if not np.isnan(r):
                    day_ret += weight * r

        ret_list.append(day_ret)
        date_list.append(date)

    return pd.Series(ret_list, index=date_list), []


def run_leveraged_rotation_with_ma(prices, spy_prices, regime_ma=200, slippage_bps=0):
    """Run leveraged rotation with configurable MA filter."""
    exec_map = {}
    for t in TECH_UNIVERSE:
        if t in LEVERAGED_MAP and LEVERAGED_MAP[t] in prices.columns:
            exec_map[t] = LEVERAGED_MAP[t]
        else:
            exec_map[t] = t

    signal_tickers = [t for t in TECH_UNIVERSE if t in prices.columns]
    exec_tickers = [exec_map.get(t, t) for t in signal_tickers]

    if len(signal_tickers) < N_LONG + 1:
        return pd.Series(dtype=float), []

    signal_px = prices[signal_tickers].dropna(how="all")
    exec_returns = prices[list(set(exec_tickers))].pct_change()

    mom_scores = None
    for lb in MOM_LOOKBACKS:
        m = (signal_px / signal_px.shift(lb) - 1).rank(axis=1, pct=True)
        mom_scores = m if mom_scores is None else mom_scores + m
    mom_scores /= len(MOM_LOOKBACKS)

    spy_ma = spy_prices.rolling(regime_ma, min_periods=int(regime_ma * 0.5)).mean()
    risk_on = spy_prices > spy_ma

    warmup = max(MOM_LOOKBACKS) + 10
    dates = signal_px.index[warmup:]

    ret_list, date_list = [], []
    pos = {}
    last_rb = None
    prev_h = set()

    for date in dates:
        do_rb = last_rb is None or (date - last_rb).days >= HOLD_DAYS
        slip_cost = 0.0

        if do_rb:
            last_rb = date
            if date in mom_scores.index:
                scores = mom_scores.loc[date].dropna()
                ro = risk_on.get(date, False)
                if len(scores) >= N_LONG and ro:
                    top = scores.nlargest(N_LONG)
                    lev_holdings = {exec_map.get(t, t): 1.0 / N_LONG for t in top.index}
                    new_h = set(lev_holdings.keys())
                    to = len(prev_h.symmetric_difference(new_h))
                    if to > 0 and slippage_bps > 0:
                        slip_cost = slippage_bps / 10000 * (to / (2 * N_LONG))
                    prev_h = new_h
                    pos = lev_holdings
                else:
                    pos = {}
            else:
                pos = {}

        day_ret = -slip_cost
        for ticker, weight in pos.items():
            if ticker in exec_returns.columns and date in exec_returns.index:
                r = exec_returns.loc[date, ticker]
                if not np.isnan(r):
                    day_ret += weight * r

        ret_list.append(day_ret)
        date_list.append(date)

    return pd.Series(ret_list, index=date_list), []


def run_tqqq_200ma(prices, spy_prices, vol_target=None, slippage_bps=0):
    """TQQQ + 200MA filter benchmark."""
    if "TQQQ" not in prices.columns:
        return pd.Series(dtype=float)

    tqqq_ret = prices["TQQQ"].pct_change()
    spy_ma200 = spy_prices.rolling(200, min_periods=100).mean()
    risk_on = spy_prices > spy_ma200

    dr = tqqq_ret.copy()
    dr[~risk_on] = 0

    if vol_target is not None and vol_target > 0:
        tqqq_vol = tqqq_ret.rolling(21).std() * np.sqrt(252)
        scalar = (vol_target / tqqq_vol).clip(0.1, 2.0)
        dr = dr * scalar

    return dr.dropna().iloc[252:]


def survivorship_bias_check(availability):
    """Check if any ETFs didn't exist for the full backtest period."""
    print("\n" + "=" * 80)
    print("SURVIVORSHIP BIAS CHECK")
    print("=" * 80)

    issues = []
    # The original backtest used data from 2021-06-07 to 2025-12-05
    # But for a REAL long-term test, we need much more history

    for t in TECH_UNIVERSE:
        inception = ETF_INCEPTION.get(t, "unknown")
        print(f"  {t}: inception {inception}")
        if inception != "unknown":
            inc_date = pd.Timestamp(inception)
            if inc_date > pd.Timestamp("2010-01-01"):
                issues.append(f"{t} (inception {inception}) — not available before {inception}")

    # Critical check: HACK (2014), CIBR (2015), SKYY (2011)
    # These are the newest ETFs in the universe
    # If we ran a backtest starting before their inception, results would be inflated
    # because we'd only be selecting from ETFs that SURVIVED

    result = {
        "issues": issues,
        "newest_etf": "CIBR (2015-06-29)",
        "max_clean_backtest_start": "2015-07-01",  # All ETFs existed after this date
        "original_backtest_start": "2021-06-07",
        "survivorship_risk": "LOW" if not issues else "MEDIUM",
        "note": ("Original backtest starts 2021-06-07, all ETFs existed by then. "
                 "However, the universe was SELECTED knowing these ETFs survived/grew. "
                 "The bigger issue is that 2021-2025 is only 2.5 years of data — "
                 "entirely within a tech bull run (post-COVID)."),
    }

    for i in issues:
        print(f"  WARNING: {i}")

    print(f"\n  Survivorship risk: {result['survivorship_risk']}")
    print(f"  Note: {result['note']}")

    return result


def permutation_test(prices, spy_prices, n_trials=100):
    """
    Randomly shuffle momentum rankings each month.
    If random rotation also gets high Sharpe, the 'edge' is just being long tech.
    """
    print("\n" + "=" * 80)
    print(f"PERMUTATION TEST ({n_trials} trials)")
    print("=" * 80)

    # Get real strategy returns
    real_rets, _ = run_momentum_rotation(prices, TECH_UNIVERSE, spy_prices)
    real_metrics = compute_metrics(real_rets, "real")
    real_sharpe = real_metrics.get("sharpe", 0)
    real_cagr = real_metrics.get("cagr", 0)

    print(f"  Real strategy: Sharpe={real_sharpe:.3f}, CAGR={real_cagr:.1f}%")

    shuffled_sharpes = []
    shuffled_cagrs = []
    beat_count = 0

    for i in range(n_trials):
        shuf_rets, _ = run_momentum_rotation(
            prices, TECH_UNIVERSE, spy_prices, shuffle_rankings=True
        )
        sm = compute_metrics(shuf_rets, f"perm_{i}")
        s = sm.get("sharpe", 0)
        c = sm.get("cagr", 0)
        shuffled_sharpes.append(s)
        shuffled_cagrs.append(c)
        if s >= real_sharpe:
            beat_count += 1
        if (i + 1) % 20 == 0:
            print(f"  ... completed {i+1}/{n_trials} trials")

    p_value = beat_count / n_trials
    mean_shuffled_sharpe = np.mean(shuffled_sharpes)
    mean_shuffled_cagr = np.mean(shuffled_cagrs)

    print(f"\n  Permutation results:")
    print(f"    Real Sharpe:      {real_sharpe:.3f}")
    print(f"    Mean shuffled:    {mean_shuffled_sharpe:.3f}")
    print(f"    Shuffled p10/p50/p90: {np.percentile(shuffled_sharpes, 10):.3f} / "
          f"{np.percentile(shuffled_sharpes, 50):.3f} / {np.percentile(shuffled_sharpes, 90):.3f}")
    print(f"    p-value:          {p_value:.3f}")
    print(f"    VERDICT:          {'PASS — rotation adds value' if p_value <= 0.05 else 'FAIL — rotation does NOT add significant value'}")

    print(f"\n  CAGR check (is this just long tech?):")
    print(f"    Real CAGR:        {real_cagr:.1f}%")
    print(f"    Mean random CAGR: {mean_shuffled_cagr:.1f}%")
    print(f"    If random is also high, the 'edge' is BEING LONG TECH, not rotation.")

    return {
        "real_sharpe": real_sharpe,
        "real_cagr": real_cagr,
        "mean_shuffled_sharpe": round(mean_shuffled_sharpe, 3),
        "mean_shuffled_cagr": round(mean_shuffled_cagr, 1),
        "p_value": round(p_value, 4),
        "shuffled_sharpe_p10": round(np.percentile(shuffled_sharpes, 10), 3),
        "shuffled_sharpe_p50": round(np.percentile(shuffled_sharpes, 50), 3),
        "shuffled_sharpe_p90": round(np.percentile(shuffled_sharpes, 90), 3),
        "shuffled_cagr_p10": round(np.percentile(shuffled_cagrs, 10), 1),
        "shuffled_cagr_p50": round(np.percentile(shuffled_cagrs, 50), 1),
        "shuffled_cagr_p90": round(np.percentile(shuffled_cagrs, 90), 1),
        "verdict": "PASS" if p_value <= 0.05 else "FAIL",
        "rotation_adds_value": p_value <= 0.05,
    }


def rotation_value_add(prices, spy_prices):
    """
    HC #697: Does ROTATION add value vs buy-and-hold XLK?
    Compare:
    - Tech rotation strategy
    - Equal-weight buy-and-hold all 8 ETFs
    - Buy-and-hold XLK only
    - Buy-and-hold SMH only
    """
    print("\n" + "=" * 80)
    print("ROTATION VALUE-ADD TEST (HC #697)")
    print("=" * 80)

    results = {}

    # 1. Rotation strategy
    rot_rets, _ = run_momentum_rotation(prices, TECH_UNIVERSE, spy_prices)
    results["rotation"] = compute_metrics(rot_rets, "rotation")

    # 2. Equal-weight buy-and-hold
    avail = [t for t in TECH_UNIVERSE if t in prices.columns and prices[t].notna().sum() > 252]
    if avail:
        ew_rets = prices[avail].pct_change().mean(axis=1).dropna()
        # Apply same regime filter for fair comparison
        spy_ma = spy_prices.rolling(REGIME_MA, min_periods=30).mean()
        risk_on = spy_prices > spy_ma
        ew_rets[~risk_on.reindex(ew_rets.index, fill_value=False)] = 0
        ew_rets = ew_rets.iloc[max(MOM_LOOKBACKS) + 10:]
        results["equal_weight_bh"] = compute_metrics(ew_rets, "equal_weight_bh")

    # 3. XLK buy-and-hold (with regime)
    if "XLK" in prices.columns:
        xlk_ret = prices["XLK"].pct_change().dropna()
        spy_ma = spy_prices.rolling(REGIME_MA, min_periods=30).mean()
        risk_on = spy_prices > spy_ma
        xlk_ret[~risk_on.reindex(xlk_ret.index, fill_value=False)] = 0
        xlk_ret = xlk_ret.iloc[max(MOM_LOOKBACKS) + 10:]
        results["xlk_bh_regime"] = compute_metrics(xlk_ret, "xlk_bh_regime")

    # 4. SMH buy-and-hold (with regime)
    if "SMH" in prices.columns:
        smh_ret = prices["SMH"].pct_change().dropna()
        smh_ret[~risk_on.reindex(smh_ret.index, fill_value=False)] = 0
        smh_ret = smh_ret.iloc[max(MOM_LOOKBACKS) + 10:]
        results["smh_bh_regime"] = compute_metrics(smh_ret, "smh_bh_regime")

    # Print comparison
    print(f"\n  {'Strategy':<25} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7}")
    print("  " + "-" * 65)
    for key, m in results.items():
        if m.get("valid"):
            print(f"  {key:<25} {m['cagr']:>6.1f}% {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
                  f"{m['max_dd']:>6.1f}% {m['calmar']:>7.3f}")

    # Verdict
    rot_sharpe = results.get("rotation", {}).get("sharpe", 0)
    ew_sharpe = results.get("equal_weight_bh", {}).get("sharpe", 0)
    xlk_sharpe = results.get("xlk_bh_regime", {}).get("sharpe", 0)

    rotation_beats_ew = rot_sharpe > ew_sharpe * 1.10  # needs 10% improvement
    rotation_beats_xlk = rot_sharpe > xlk_sharpe * 1.10

    print(f"\n  Rotation Sharpe vs EW B&H: {rot_sharpe:.3f} vs {ew_sharpe:.3f} — "
          f"{'ROTATION ADDS VALUE' if rotation_beats_ew else 'NO SIGNIFICANT VALUE-ADD'}")
    print(f"  Rotation Sharpe vs XLK B&H: {rot_sharpe:.3f} vs {xlk_sharpe:.3f} — "
          f"{'ROTATION ADDS VALUE' if rotation_beats_xlk else 'NO SIGNIFICANT VALUE-ADD'}")

    return {
        "strategies": {k: v for k, v in results.items()},
        "rotation_beats_ew": rotation_beats_ew,
        "rotation_beats_xlk": rotation_beats_xlk,
    }


def per_year_breakdown(daily_returns, label=""):
    """Compute per-year metrics."""
    dr = daily_returns.dropna()
    yearly = {}
    for year in sorted(dr.index.year.unique()):
        yr = dr[dr.index.year == year]
        if len(yr) > 20:
            m = compute_metrics(yr, f"{label}_{year}")
            if m.get("valid"):
                yearly[str(year)] = {
                    "return": m["cagr"],
                    "sharpe": m["sharpe"],
                    "max_dd": m["max_dd"],
                    "wr": m["wr"],
                    "n_days": m["n_days"],
                }
    return yearly


def main():
    t0 = time.time()
    print("=" * 80)
    print("TECH SUB-INDUSTRY ROTATION — DEEP VALIDATION")
    print(f"Started: {pd.Timestamp.now()}")
    print("=" * 80)

    # ── Download data ──
    prices, availability = download_data(start="2005-01-01")
    spy_prices = prices["SPY"].dropna()
    spy_returns = spy_prices.pct_change()

    # ══════════════════════════════════════════════════════════
    # TEST 1: Survivorship Bias Check
    # ══════════════════════════════════════════════════════════
    surv = survivorship_bias_check(availability)

    # ══════════════════════════════════════════════════════════
    # TEST 2: Reproduce Original Results (full available period)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("REPRODUCTION: Original Strategy (full period)")
    print("=" * 80)

    base_rets, base_rebal = run_momentum_rotation(
        prices, TECH_UNIVERSE, spy_prices, slippage_bps=0
    )
    base_metrics = compute_metrics(base_rets, "tech_rotation_base")
    base_regime = regime_stratification(base_rets, spy_returns)
    base_yearly = per_year_breakdown(base_rets, "base")

    print(f"\n  Base strategy results:")
    print(f"    Period: {base_rets.index[0].date()} to {base_rets.index[-1].date()}")
    print(f"    CAGR:    {base_metrics.get('cagr', 0):.1f}%")
    print(f"    Sharpe:  {base_metrics.get('sharpe', 0):.3f}")
    print(f"    Sortino: {base_metrics.get('sortino', 0):.3f}")
    print(f"    Calmar:  {base_metrics.get('calmar', 0):.3f}")
    print(f"    MaxDD:   {base_metrics.get('max_dd', 0):.1f}%")
    print(f"    WR:      {base_metrics.get('wr', 0):.1f}%")
    print(f"    PF:      {base_metrics.get('pf', 0):.3f}")
    print(f"    N days:  {base_metrics.get('n_days', 0)}")

    print(f"\n  R1 Regime Test:")
    print(f"    Green Sharpe: {base_regime['green_sharpe']:.3f}")
    print(f"    Red Sharpe:   {base_regime['red_sharpe']:.3f}")
    print(f"    Regime Gap:   {base_regime['regime_gap']:.4f}")
    print(f"    R1 PASS:      {base_regime['r1_pass']}")

    print(f"\n  Per-year breakdown:")
    for yr, m in base_yearly.items():
        print(f"    {yr}: return={m['return']:.1f}% sharpe={m['sharpe']:.3f} maxDD={m['max_dd']:.1f}%")

    # ══════════════════════════════════════════════════════════
    # TEST 3: Permutation Test
    # ══════════════════════════════════════════════════════════
    perm = permutation_test(prices, spy_prices, n_trials=100)

    # ══════════════════════════════════════════════════════════
    # TEST 4: Transaction Cost Sensitivity
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TRANSACTION COST SENSITIVITY")
    print("=" * 80)

    cost_results = {}
    for slip in [0, 5, 10]:
        rets, _ = run_momentum_rotation(
            prices, TECH_UNIVERSE, spy_prices, slippage_bps=slip
        )
        m = compute_metrics(rets, f"slip_{slip}bps")
        cost_results[f"{slip}bps"] = m
        print(f"  Slippage {slip:>3}bps: CAGR={m.get('cagr',0):>6.1f}%  "
              f"Sharpe={m.get('sharpe',0):.3f}  MaxDD={m.get('max_dd',0):.1f}%")

    # ══════════════════════════════════════════════════════════
    # TEST 5: Rotation Value-Add (HC #697)
    # ══════════════════════════════════════════════════════════
    value_add = rotation_value_add(prices, spy_prices)

    # ══════════════════════════════════════════════════════════
    # TEST 6: Leveraged Version
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("LEVERAGED VERSION (SOXL/TECL substitution)")
    print("=" * 80)

    lev_rets, _ = run_leveraged_rotation(prices, spy_prices, slippage_bps=0)
    lev_metrics = compute_metrics(lev_rets, "leveraged_rotation")
    lev_regime = regime_stratification(lev_rets, spy_returns)
    lev_yearly = per_year_breakdown(lev_rets, "leveraged")

    print(f"  Leveraged results:")
    print(f"    Period: {lev_rets.index[0].date() if len(lev_rets) > 0 else 'N/A'} to "
          f"{lev_rets.index[-1].date() if len(lev_rets) > 0 else 'N/A'}")
    print(f"    CAGR:    {lev_metrics.get('cagr', 0):.1f}%")
    print(f"    Sharpe:  {lev_metrics.get('sharpe', 0):.3f}")
    print(f"    Sortino: {lev_metrics.get('sortino', 0):.3f}")
    print(f"    MaxDD:   {lev_metrics.get('max_dd', 0):.1f}%")
    print(f"    Calmar:  {lev_metrics.get('calmar', 0):.3f}")
    print(f"    R1 gap:  {lev_regime['regime_gap']:.4f} ({'PASS' if lev_regime['r1_pass'] else 'FAIL'})")

    lev_cost_results = {}
    for slip in [0, 5, 10]:
        lr, _ = run_leveraged_rotation(prices, spy_prices, slippage_bps=slip)
        lm = compute_metrics(lr, f"lev_slip_{slip}bps")
        lev_cost_results[f"{slip}bps"] = lm
        print(f"    With {slip}bps slip: CAGR={lm.get('cagr',0):.1f}% Sharpe={lm.get('sharpe',0):.3f}")

    # ══════════════════════════════════════════════════════════
    # TEST 6b: 200MA Crash Filter Variant
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("200MA CRASH FILTER VARIANT (go to cash when SPY < 200MA)")
    print("=" * 80)

    # Run rotation with 200MA instead of 60MA regime filter
    ma200_rets, _ = run_momentum_rotation(
        prices, TECH_UNIVERSE, spy_prices, regime_ma=200, slippage_bps=0
    )
    ma200_metrics = compute_metrics(ma200_rets, "rotation_200MA")
    ma200_regime = regime_stratification(ma200_rets, spy_returns)
    ma200_yearly = per_year_breakdown(ma200_rets, "ma200")

    print(f"  200MA filter results:")
    print(f"    Period: {ma200_rets.index[0].date()} to {ma200_rets.index[-1].date()}")
    print(f"    CAGR:    {ma200_metrics.get('cagr', 0):.1f}%")
    print(f"    Sharpe:  {ma200_metrics.get('sharpe', 0):.3f}")
    print(f"    Sortino: {ma200_metrics.get('sortino', 0):.3f}")
    print(f"    Calmar:  {ma200_metrics.get('calmar', 0):.3f}")
    print(f"    MaxDD:   {ma200_metrics.get('max_dd', 0):.1f}%")
    print(f"    R1 gap:  {ma200_regime['regime_gap']:.4f} ({'PASS' if ma200_regime['r1_pass'] else 'FAIL'})")

    print(f"\n  Per-year:")
    for yr, m in ma200_yearly.items():
        print(f"    {yr}: return={m['return']:.1f}% sharpe={m['sharpe']:.3f} maxDD={m['max_dd']:.1f}%")

    # Also run leveraged + 200MA
    lev_200ma_rets, _ = run_leveraged_rotation_with_ma(prices, spy_prices, regime_ma=200)
    lev_200ma_metrics = compute_metrics(lev_200ma_rets, "leveraged_200MA")
    lev_200ma_regime = regime_stratification(lev_200ma_rets, spy_returns)

    print(f"\n  Leveraged + 200MA:")
    print(f"    CAGR:    {lev_200ma_metrics.get('cagr', 0):.1f}%")
    print(f"    Sharpe:  {lev_200ma_metrics.get('sharpe', 0):.3f}")
    print(f"    MaxDD:   {lev_200ma_metrics.get('max_dd', 0):.1f}%")
    print(f"    R1 gap:  {lev_200ma_regime['regime_gap']:.4f} ({'PASS' if lev_200ma_regime['r1_pass'] else 'FAIL'})")

    # No regime filter at all
    no_regime_rets, _ = run_momentum_rotation(
        prices, TECH_UNIVERSE, spy_prices, use_regime=False, slippage_bps=0
    )
    no_regime_metrics = compute_metrics(no_regime_rets, "rotation_no_filter")
    no_regime_regime = regime_stratification(no_regime_rets, spy_returns)

    print(f"\n  No regime filter (always invested):")
    print(f"    CAGR:    {no_regime_metrics.get('cagr', 0):.1f}%")
    print(f"    Sharpe:  {no_regime_metrics.get('sharpe', 0):.3f}")
    print(f"    MaxDD:   {no_regime_metrics.get('max_dd', 0):.1f}%")
    print(f"    R1 gap:  {no_regime_regime['regime_gap']:.4f} ({'PASS' if no_regime_regime['r1_pass'] else 'FAIL'})")

    # ══════════════════════════════════════════════════════════
    # TEST 7: Comparison to TQQQ+200MA
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON: TQQQ + 200MA")
    print("=" * 80)

    benchmarks = {}

    # TQQQ + 200MA (no vol target)
    tqqq_rets = run_tqqq_200ma(prices, spy_prices)
    tqqq_m = compute_metrics(tqqq_rets, "TQQQ+200MA")
    tqqq_regime = regime_stratification(tqqq_rets, spy_returns)
    benchmarks["TQQQ_200MA"] = {**tqqq_m, "regime": tqqq_regime}

    # TQQQ + 200MA + vol target
    tqqq_vt_rets = run_tqqq_200ma(prices, spy_prices, vol_target=0.30)
    tqqq_vt_m = compute_metrics(tqqq_vt_rets, "TQQQ+200MA+VT30")
    tqqq_vt_regime = regime_stratification(tqqq_vt_rets, spy_returns)
    benchmarks["TQQQ_200MA_VT30"] = {**tqqq_vt_m, "regime": tqqq_vt_regime}

    # SPY buy-and-hold
    spy_bh = spy_returns.dropna().iloc[252:]
    spy_m = compute_metrics(spy_bh, "SPY_BH")
    benchmarks["SPY_BH"] = spy_m

    # XLK buy-and-hold (no regime)
    if "XLK" in prices.columns:
        xlk_bh = prices["XLK"].pct_change().dropna().iloc[252:]
        xlk_m = compute_metrics(xlk_bh, "XLK_BH")
        benchmarks["XLK_BH"] = xlk_m

    print(f"\n  {'Strategy':<25} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7} {'R1 Gap':>7}")
    print("  " + "-" * 75)

    # Print rotation variants
    all_strats = [
        ("TECH ROTATION (60MA)", base_metrics, base_regime),
        ("ROTATION + 200MA", ma200_metrics, ma200_regime),
        ("ROTATION NO FILTER", no_regime_metrics, no_regime_regime),
        ("LEVERAGED (60MA)", lev_metrics, lev_regime),
        ("LEVERAGED + 200MA", lev_200ma_metrics, lev_200ma_regime),
    ]
    for name, m, rg in all_strats:
        if m.get("valid"):
            print(f"  {name:<25} {m.get('cagr',0):>6.1f}% {m.get('sharpe',0):>7.3f} "
                  f"{m.get('sortino',0):>8.3f} {m.get('max_dd',0):>6.1f}% "
                  f"{m.get('calmar',0):>7.3f} {rg['regime_gap']:>7.4f}")

    for name, bm in benchmarks.items():
        if bm.get("valid"):
            rg = bm.get("regime", {}).get("regime_gap", "-")
            rg_str = f"{rg:.4f}" if isinstance(rg, float) else "-"
            print(f"  {name:<25} {bm['cagr']:>6.1f}% {bm['sharpe']:>7.3f} "
                  f"{bm['sortino']:>8.3f} {bm['max_dd']:>6.1f}% "
                  f"{bm['calmar']:>7.3f} {rg_str:>7}")

    # ══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    issues = []
    passes = []

    # R1 regime check
    if not base_regime["r1_pass"]:
        issues.append(f"FAILS R1: regime gap {base_regime['regime_gap']:.4f} > 0.50 threshold")
    else:
        passes.append(f"R1 regime gap {base_regime['regime_gap']:.4f} <= 0.50")

    # Permutation test
    if not perm["rotation_adds_value"]:
        issues.append(f"FAILS permutation: p={perm['p_value']:.4f} > 0.05 — rotation doesn't beat random")
    else:
        passes.append(f"Permutation p={perm['p_value']:.4f} — rotation adds value")

    # Beats TQQQ+200MA?
    tqqq_sharpe = tqqq_m.get("sharpe", 0)
    rot_sharpe = base_metrics.get("sharpe", 0)
    if rot_sharpe > tqqq_sharpe:
        passes.append(f"Beats TQQQ+200MA on Sharpe: {rot_sharpe:.3f} vs {tqqq_sharpe:.3f}")
    else:
        issues.append(f"Does NOT beat TQQQ+200MA on Sharpe: {rot_sharpe:.3f} vs {tqqq_sharpe:.3f}")

    # Short backtest period
    if base_metrics.get("years", 0) < 5:
        issues.append(f"Short backtest: only {base_metrics.get('years',0):.1f} years — mostly tech bull market")

    # Value-add from rotation
    if not value_add.get("rotation_beats_ew"):
        issues.append("Rotation does NOT beat equal-weight buy-and-hold by >10%")
    else:
        passes.append("Rotation beats equal-weight B&H by >10% Sharpe improvement")

    overall = "FAIL" if len(issues) > 0 else "PASS"

    for p in passes:
        print(f"  PASS: {p}")
    for i in issues:
        print(f"  FAIL: {i}")

    print(f"\n  OVERALL: {overall}")
    if overall == "FAIL":
        print(f"  RECOMMENDATION: The tech rotation strategy has {len(issues)} critical issue(s).")
        print(f"  The 41% CAGR is likely driven by tech bull market (2021-2025) + regime selection,")
        print(f"  NOT by predictive momentum rotation edge.")

    # ── Save results ──
    elapsed = time.time() - t0

    report = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "overall_verdict": overall,
        "n_issues": len(issues),
        "issues": issues,
        "passes": passes,

        "base_strategy": {
            "metrics": base_metrics,
            "regime": base_regime,
            "per_year": base_yearly,
        },

        "survivorship_bias": surv,
        "permutation_test": perm,

        "cost_sensitivity": cost_results,

        "rotation_value_add": {
            "rotation_beats_ew": value_add.get("rotation_beats_ew"),
            "rotation_beats_xlk": value_add.get("rotation_beats_xlk"),
            "strategies": {k: v for k, v in value_add.get("strategies", {}).items()},
        },

        "leveraged_version": {
            "metrics": lev_metrics,
            "regime": lev_regime,
            "per_year": lev_yearly,
            "cost_sensitivity": lev_cost_results,
        },

        "ma200_filter": {
            "metrics": ma200_metrics,
            "regime": ma200_regime,
            "per_year": ma200_yearly,
        },

        "leveraged_200ma": {
            "metrics": lev_200ma_metrics,
            "regime": lev_200ma_regime,
        },

        "no_regime_filter": {
            "metrics": no_regime_metrics,
            "regime": no_regime_regime,
        },

        "benchmarks": {k: {kk: vv for kk, vv in v.items() if kk != "regime"} for k, v in benchmarks.items()},

        "tqqq_200ma_comparison": {
            "rotation_sharpe": rot_sharpe,
            "tqqq_sharpe": tqqq_sharpe,
            "rotation_wins": rot_sharpe > tqqq_sharpe,
            "tqqq_cagr": tqqq_m.get("cagr", 0),
            "rotation_cagr": base_metrics.get("cagr", 0),
        },
    }

    outfile = OUT_DIR / "deep_validation_results.json"
    with open(outfile, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Results saved to {outfile}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    return report


if __name__ == "__main__":
    main()
