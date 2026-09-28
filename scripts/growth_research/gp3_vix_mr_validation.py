#!/usr/bin/env python3
"""
GP3 + VIX Mean-Reversion Entry — Full Adversarial Validation
=============================================================
Tests whether adding a VIX mean-reversion ENTRY signal to Gameplan v3
genuinely improves the system or just adds lucky recovery exposure.

Enhancement: When VIX > 20 but dropped 15%+ from 20d peak AND below 10d SMA
(declining), enter UPRO even if confluence < 2.5.  Only if vol < 25%.

Validation suite:
  1. Full-period metrics (GP3 standard vs GP3+MR)
  2. Walk-forward: 3yr train / 1yr OOS, 10+ sliding windows
  3. Permutation test: 200 shuffles of MR timing signal, p<0.05
  4. Sub-period consistency: 3 blocks, all positive Sharpe, CV<0.50
  5. Outlier robustness: remove 10 best days, Sharpe degradation <30%
  6. R1 regime test: |Sharpe_green - Sharpe_red| / max < 0.50
  7. Year-by-year comparison
  8. MR entry statistics
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/gp3_vix_mr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Data Download ───────────────────────────────────────────────────────────

def download_data(start="2012-01-01"):
    """Download SPY, UPRO, GLD, TLT, ^VIX from yfinance."""
    tickers = ["SPY", "UPRO", "GLD", "TLT", "^VIX"]
    cache_path = OUTPUT_DIR / "data_cache.parquet"

    print("Downloading data from yfinance...")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw

    # Flatten column names if needed
    close.columns = [str(c).strip() for c in close.columns]
    close = close.dropna()
    close.to_parquet(cache_path)
    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


# ─── Signal Construction ─────────────────────────────────────────────────────

def compute_confluence_score(df):
    """
    3-timeframe confluence score (0-3):
      SHORT:  5d momentum > 0 AND 10d RSI > 50  → +1
      MEDIUM: 20d MA > 50d MA AND vol < 15%     → +1
      LONG:   200d MA slope > 0 AND vol trend declining → +1
    """
    spy = df["SPY"]
    vix = df["^VIX"]

    # SHORT: 5d momentum + 10d RSI
    mom_5d = spy.pct_change(5)
    delta = spy.diff()
    gain = delta.clip(lower=0).rolling(10).mean()
    loss = (-delta.clip(upper=0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_10d = 100 - (100 / (1 + rs))
    short_score = ((mom_5d > 0) & (rsi_10d > 50)).astype(float)

    # MEDIUM: 20/50 MA cross + vol < 15%
    ma20 = spy.rolling(20).mean()
    ma50 = spy.rolling(50).mean()
    vol_pct = vix / 100  # VIX is already annualized vol %
    medium_score = ((ma20 > ma50) & (vix < 15)).astype(float)

    # LONG: 200d MA slope > 0 + vol trend declining
    ma200 = spy.rolling(200).mean()
    ma200_slope = ma200.diff(20)  # 20d slope of 200d MA
    vol_trend = vix.rolling(20).mean().diff(10)  # vol trend
    long_score = ((ma200_slope > 0) & (vol_trend < 0)).astype(float)

    confluence = short_score + medium_score + long_score
    return confluence


def compute_vix_mr_signal(df):
    """
    VIX Mean-Reversion entry signal:
      - VIX > 20
      - VIX dropped 15%+ from 20d peak
      - VIX < 10d SMA (declining)
      - VIX < 25 (don't override crisis)
    """
    vix = df["^VIX"]
    vix_20d_peak = vix.rolling(20).max()
    vix_drop_pct = (vix - vix_20d_peak) / vix_20d_peak
    vix_10d_sma = vix.rolling(10).mean()

    mr_signal = (
        (vix > 20) &
        (vix_drop_pct <= -0.15) &  # dropped 15%+ from peak
        (vix < vix_10d_sma) &       # below 10d SMA (declining)
        (vix < 25)                   # don't override crisis protection
    ).astype(float)

    return mr_signal


def gp3_standard_allocation(df):
    """
    GP3 standard allocation:
      - September: SPY regardless
      - VIX > 30: GLD (crisis)
      - VIX > 15: SPY
      - VIX < 15 AND confluence >= 2.5: UPRO (with hysteresis exit at < 2.0)
    """
    vix = df["^VIX"]
    confluence = compute_confluence_score(df)

    n = len(df)
    alloc = pd.Series("SPY", index=df.index)

    # Apply hysteresis for UPRO entry/exit
    in_upro = False
    for i in range(n):
        month = df.index[i].month
        v = vix.iloc[i]
        c = confluence.iloc[i]

        if month == 9:
            alloc.iloc[i] = "SPY"
            in_upro = False
        elif v > 30:
            alloc.iloc[i] = "GLD"
            in_upro = False
        elif v > 15:
            alloc.iloc[i] = "SPY"
            in_upro = False
        else:
            # vol < 15%
            if in_upro:
                if c < 2.0:
                    alloc.iloc[i] = "SPY"
                    in_upro = False
                else:
                    alloc.iloc[i] = "UPRO"
            else:
                if c >= 2.5:
                    alloc.iloc[i] = "UPRO"
                    in_upro = True
                else:
                    alloc.iloc[i] = "SPY"

    return alloc


def gp3_mr_allocation(df):
    """
    GP3 + VIX Mean-Reversion allocation:
      Same as standard, PLUS: MR signal triggers UPRO entry even if confluence < 2.5
      (but only if VIX < 25).
    """
    vix = df["^VIX"]
    confluence = compute_confluence_score(df)
    mr_signal = compute_vix_mr_signal(df)

    n = len(df)
    alloc = pd.Series("SPY", index=df.index)
    mr_active = pd.Series(False, index=df.index)

    in_upro = False
    mr_entry = False  # track if current UPRO position was MR-triggered

    for i in range(n):
        month = df.index[i].month
        v = vix.iloc[i]
        c = confluence.iloc[i]
        mr = mr_signal.iloc[i]

        if month == 9:
            alloc.iloc[i] = "SPY"
            in_upro = False
            mr_entry = False
        elif v > 30:
            alloc.iloc[i] = "GLD"
            in_upro = False
            mr_entry = False
        elif v > 25:
            # Between 25-30: SPY (MR doesn't fire here, standard doesn't either since vol>15)
            alloc.iloc[i] = "SPY"
            in_upro = False
            mr_entry = False
        elif v > 15:
            # Vol 15-25: MR can trigger UPRO here
            if mr > 0.5:
                alloc.iloc[i] = "UPRO"
                in_upro = True
                mr_entry = True
                mr_active.iloc[i] = True
            elif in_upro and mr_entry:
                # Stay in UPRO if MR-triggered, exit when VIX drops below 20 or confluence takes over
                if v < 20 and c >= 2.0:
                    # Confluence takes over management
                    alloc.iloc[i] = "UPRO"
                    mr_entry = False
                elif v < 20:
                    alloc.iloc[i] = "SPY"
                    in_upro = False
                    mr_entry = False
                else:
                    # Still in VIX>20 zone, check if MR still valid
                    if mr > 0.5:
                        alloc.iloc[i] = "UPRO"
                        mr_active.iloc[i] = True
                    else:
                        alloc.iloc[i] = "SPY"
                        in_upro = False
                        mr_entry = False
            else:
                alloc.iloc[i] = "SPY"
                in_upro = False
                mr_entry = False
        else:
            # vol < 15: standard confluence logic
            if in_upro:
                if c < 2.0:
                    alloc.iloc[i] = "SPY"
                    in_upro = False
                    mr_entry = False
                else:
                    alloc.iloc[i] = "UPRO"
            else:
                if c >= 2.5:
                    alloc.iloc[i] = "UPRO"
                    in_upro = True
                    mr_entry = False
                else:
                    alloc.iloc[i] = "SPY"

    return alloc, mr_active


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(returns, ann_factor=252):
    """Compute risk-adjusted metrics from a return series."""
    if len(returns) < 20 or returns.std() == 0:
        return {
            "sharpe": 0.0, "sortino": 0.0, "cagr": 0.0,
            "max_dd": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
            "vol_ann": 0.0, "total_return": 0.0, "n_days": len(returns)
        }

    mean_r = returns.mean()
    std_r = returns.std()
    sharpe = mean_r / std_r * np.sqrt(ann_factor) if std_r > 0 else 0

    downside = returns[returns < 0].std()
    sortino = mean_r / downside * np.sqrt(ann_factor) if downside > 0 else 0

    cum = (1 + returns).cumprod()
    total_years = len(returns) / ann_factor
    cagr = (cum.iloc[-1] ** (1 / total_years) - 1) * 100 if total_years > 0 else 0

    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    win_rate = (returns > 0).mean() * 100

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")

    vol_ann = std_r * np.sqrt(ann_factor) * 100
    total_return = (cum.iloc[-1] - 1) * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 2),
        "max_dd": round(max_dd, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "vol_ann": round(vol_ann, 2),
        "total_return": round(total_return, 2),
        "n_days": len(returns)
    }


def allocation_to_returns(alloc, df):
    """Convert allocation series to daily returns."""
    # Compute daily returns for each asset
    rets = pd.DataFrame({
        "SPY": df["SPY"].pct_change(),
        "UPRO": df["UPRO"].pct_change(),
        "GLD": df["GLD"].pct_change(),
    })

    # Map allocation to returns (1-day forward, signal on close → next day return)
    shifted_alloc = alloc.shift(1)  # trade on next day's open→close

    strategy_rets = pd.Series(0.0, index=df.index)
    for asset in ["SPY", "UPRO", "GLD"]:
        mask = shifted_alloc == asset
        strategy_rets[mask] = rets[asset][mask]

    return strategy_rets.dropna()


# ─── Test 1: Full-Period Metrics ─────────────────────────────────────────────

def test_full_period(df):
    """Full-period comparison: GP3 standard vs GP3+MR."""
    print("\n" + "="*70)
    print("TEST 1: FULL-PERIOD METRICS")
    print("="*70)

    alloc_std = gp3_standard_allocation(df)
    alloc_mr, mr_active = gp3_mr_allocation(df)

    rets_std = allocation_to_returns(alloc_std, df)
    rets_mr = allocation_to_returns(alloc_mr, df)

    # Also compute buy-and-hold SPY for reference
    rets_spy = df["SPY"].pct_change().dropna()

    m_std = compute_metrics(rets_std)
    m_mr = compute_metrics(rets_mr)
    m_spy = compute_metrics(rets_spy)

    print(f"\n{'Metric':<20} {'SPY B&H':>12} {'GP3 Std':>12} {'GP3+MR':>12} {'Delta':>12}")
    print("-"*70)
    for key in ["sharpe", "sortino", "cagr", "max_dd", "win_rate", "profit_factor", "vol_ann", "total_return"]:
        delta = m_mr[key] - m_std[key]
        print(f"  {key:<18} {m_spy[key]:>12} {m_std[key]:>12} {m_mr[key]:>12} {delta:>+12.3f}")

    # Allocation breakdown
    print(f"\nAllocation Breakdown (GP3 Standard):")
    for asset in ["SPY", "UPRO", "GLD"]:
        pct = (alloc_std == asset).mean() * 100
        print(f"  {asset}: {pct:.1f}% of days")

    print(f"\nAllocation Breakdown (GP3+MR):")
    for asset in ["SPY", "UPRO", "GLD"]:
        pct = (alloc_mr == asset).mean() * 100
        print(f"  {asset}: {pct:.1f}% of days")

    # MR-specific stats
    mr_days = mr_active.sum()
    mr_years = len(df) / 252
    print(f"\nMR Entry Statistics:")
    print(f"  Total MR-active days: {int(mr_days)}")
    print(f"  MR days/year: {mr_days/mr_years:.1f}")
    print(f"  Additional UPRO days vs standard: {(alloc_mr == 'UPRO').sum() - (alloc_std == 'UPRO').sum()}")

    # MR entry returns
    mr_mask = mr_active.shift(1).fillna(False)
    if mr_mask.sum() > 0:
        mr_rets = df["UPRO"].pct_change()[mr_mask].dropna()
        print(f"  MR-triggered day returns: mean={mr_rets.mean()*100:.3f}%, "
              f"WR={((mr_rets>0).mean()*100):.1f}%, n={len(mr_rets)}")

    return {
        "gp3_standard": m_std,
        "gp3_mr": m_mr,
        "spy_bh": m_spy,
        "mr_active_days": int(mr_days),
        "mr_days_per_year": round(mr_days / mr_years, 1),
        "additional_upro_days": int((alloc_mr == "UPRO").sum() - (alloc_std == "UPRO").sum()),
    }, rets_std, rets_mr, alloc_std, alloc_mr, mr_active


# ─── Test 2: Walk-Forward Validation ────────────────────────────────────────

def test_walkforward(df, train_years=3, test_years=1):
    """Walk-forward: 3yr train, 1yr OOS, sliding windows."""
    print("\n" + "="*70)
    print("TEST 2: WALK-FORWARD VALIDATION (3yr train / 1yr OOS)")
    print("="*70)

    dates = df.index
    total_days = len(dates)
    train_days = train_years * 252
    test_days = test_years * 252
    step_days = test_days  # slide by 1 year

    windows = []
    start = 0
    while start + train_days + test_days <= total_days:
        train_end = start + train_days
        test_end = min(train_end + test_days, total_days)

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        # Compute allocations on test period using signals computed on full data up to test period
        # (signals use lookback windows, so we need enough history)
        full_up_to_test = df.iloc[:test_end]

        alloc_std = gp3_standard_allocation(full_up_to_test)
        alloc_mr, mr_active = gp3_mr_allocation(full_up_to_test)

        # Only evaluate on test period
        test_alloc_std = alloc_std.iloc[train_end:test_end]
        test_alloc_mr = alloc_mr.iloc[train_end:test_end]

        rets_std = allocation_to_returns(test_alloc_std, test_df)
        rets_mr = allocation_to_returns(test_alloc_mr, test_df)

        m_std = compute_metrics(rets_std)
        m_mr = compute_metrics(rets_mr)

        window_info = {
            "window": len(windows) + 1,
            "train_start": str(train_df.index[0].date()),
            "train_end": str(train_df.index[-1].date()),
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
            "std_sharpe": m_std["sharpe"],
            "mr_sharpe": m_mr["sharpe"],
            "sharpe_delta": round(m_mr["sharpe"] - m_std["sharpe"], 3),
            "std_cagr": m_std["cagr"],
            "mr_cagr": m_mr["cagr"],
        }
        windows.append(window_info)

        start += step_days

    print(f"\n  {'Win':>4} {'Test Period':<25} {'Std Sharpe':>12} {'MR Sharpe':>12} {'Delta':>10}")
    print("  " + "-"*65)

    mr_wins = 0
    for w in windows:
        flag = "+" if w["sharpe_delta"] > 0 else "-"
        if w["sharpe_delta"] > 0:
            mr_wins += 1
        print(f"  {w['window']:>4} {w['test_start']} to {w['test_end']}  "
              f"{w['std_sharpe']:>10.3f}  {w['mr_sharpe']:>10.3f}  {w['sharpe_delta']:>+8.3f} {flag}")

    avg_delta = np.mean([w["sharpe_delta"] for w in windows])
    print(f"\n  MR wins: {mr_wins}/{len(windows)} windows")
    print(f"  Avg Sharpe delta: {avg_delta:+.3f}")

    wf_pass = mr_wins > len(windows) / 2 and avg_delta > 0
    print(f"  WF VERDICT: {'PASS' if wf_pass else 'FAIL'}")

    return {
        "windows": windows,
        "mr_win_count": mr_wins,
        "total_windows": len(windows),
        "avg_sharpe_delta": round(avg_delta, 4),
        "pass": wf_pass
    }


# ─── Test 3: Permutation Test ───────────────────────────────────────────────

def test_permutation(df, n_perms=200):
    """
    Permutation test: shuffle WHEN the MR signal fires (not returns).
    Null hypothesis: MR timing doesn't matter, any random VIX>20 entry is equally good.
    """
    print("\n" + "="*70)
    print(f"TEST 3: PERMUTATION TEST ({n_perms} shuffles of MR timing)")
    print("="*70)

    # Get actual MR enhancement value
    alloc_std = gp3_standard_allocation(df)
    alloc_mr, mr_active = gp3_mr_allocation(df)

    rets_std = allocation_to_returns(alloc_std, df)
    rets_mr = allocation_to_returns(alloc_mr, df)

    actual_sharpe_delta = compute_metrics(rets_mr)["sharpe"] - compute_metrics(rets_std)["sharpe"]

    # Identify days where MR COULD have fired (VIX>20, VIX<25)
    vix = df["^VIX"]
    eligible_days = ((vix > 20) & (vix < 25))
    eligible_indices = eligible_days[eligible_days].index

    actual_mr_days = int(mr_active.sum())

    print(f"  Actual MR signal fires: {actual_mr_days} days")
    print(f"  Eligible days (VIX 20-25): {len(eligible_indices)} days")
    print(f"  Actual Sharpe delta: {actual_sharpe_delta:+.3f}")

    if actual_mr_days == 0 or len(eligible_indices) == 0:
        print("  SKIP: No MR signals to permute")
        return {"pass": False, "reason": "no_mr_signals", "p_value": 1.0}

    # Permutation: randomly choose same number of eligible days as MR entries
    np.random.seed(42)
    perm_deltas = []

    for p in range(n_perms):
        # Create shuffled MR signal: same number of days, random eligible dates
        n_mr = min(actual_mr_days, len(eligible_indices))
        shuffled_mr = pd.Series(False, index=df.index)
        chosen = np.random.choice(len(eligible_indices), size=n_mr, replace=False)
        shuffled_mr.loc[eligible_indices[chosen]] = True

        # Build shuffled allocation: standard GP3 + shuffled MR override
        alloc_shuf = alloc_std.copy()
        # On shuffled MR days, override to UPRO
        alloc_shuf[shuffled_mr] = "UPRO"

        rets_shuf = allocation_to_returns(alloc_shuf, df)
        m_shuf = compute_metrics(rets_shuf)
        perm_delta = m_shuf["sharpe"] - compute_metrics(rets_std)["sharpe"]
        perm_deltas.append(perm_delta)

    perm_deltas = np.array(perm_deltas)
    p_value = (perm_deltas >= actual_sharpe_delta).mean()

    print(f"\n  Permutation distribution: mean={perm_deltas.mean():+.4f}, "
          f"std={perm_deltas.std():.4f}")
    print(f"  Actual delta: {actual_sharpe_delta:+.4f}")
    print(f"  p-value: {p_value:.4f}")

    perm_pass = p_value < 0.05
    print(f"  PERMUTATION VERDICT: {'PASS (p<0.05)' if perm_pass else 'FAIL (p>=0.05)'}")

    return {
        "actual_sharpe_delta": round(actual_sharpe_delta, 4),
        "p_value": round(p_value, 4),
        "perm_mean_delta": round(float(perm_deltas.mean()), 4),
        "perm_std_delta": round(float(perm_deltas.std()), 4),
        "n_permutations": n_perms,
        "pass": perm_pass
    }


# ─── Test 4: Sub-Period Consistency ──────────────────────────────────────────

def test_subperiod(rets_std, rets_mr):
    """Split into 3 blocks. All must have positive Sharpe. CV < 0.50."""
    print("\n" + "="*70)
    print("TEST 4: SUB-PERIOD CONSISTENCY (3 blocks)")
    print("="*70)

    n = len(rets_mr)
    block_size = n // 3

    blocks_std = []
    blocks_mr = []

    for i in range(3):
        start = i * block_size
        end = (i + 1) * block_size if i < 2 else n

        block_rets_std = rets_std.iloc[start:end]
        block_rets_mr = rets_mr.iloc[start:end]

        m_std = compute_metrics(block_rets_std)
        m_mr = compute_metrics(block_rets_mr)

        blocks_std.append(m_std)
        blocks_mr.append(m_mr)

        print(f"\n  Block {i+1}: {block_rets_mr.index[0].date()} to {block_rets_mr.index[-1].date()}")
        print(f"    GP3 Std  — Sharpe: {m_std['sharpe']:.3f}, CAGR: {m_std['cagr']:.2f}%")
        print(f"    GP3+MR   — Sharpe: {m_mr['sharpe']:.3f}, CAGR: {m_mr['cagr']:.2f}%")

    sharpes_std = [b["sharpe"] for b in blocks_std]
    sharpes_mr = [b["sharpe"] for b in blocks_mr]

    all_pos_std = all(s > 0 for s in sharpes_std)
    all_pos_mr = all(s > 0 for s in sharpes_mr)

    cv_std = np.std(sharpes_std) / np.mean(sharpes_std) if np.mean(sharpes_std) != 0 else float("inf")
    cv_mr = np.std(sharpes_mr) / np.mean(sharpes_mr) if np.mean(sharpes_mr) != 0 else float("inf")

    print(f"\n  GP3 Std  — All positive: {all_pos_std}, CV: {cv_std:.3f}")
    print(f"  GP3+MR   — All positive: {all_pos_mr}, CV: {cv_mr:.3f}")

    # MR enhancement must maintain consistency
    mr_pass = all_pos_mr and cv_mr < 0.50
    std_pass = all_pos_std and cv_std < 0.50

    print(f"  GP3 Std  VERDICT: {'PASS' if std_pass else 'FAIL'}")
    print(f"  GP3+MR   VERDICT: {'PASS' if mr_pass else 'FAIL'}")

    return {
        "blocks_std": blocks_std,
        "blocks_mr": blocks_mr,
        "all_positive_std": all_pos_std,
        "all_positive_mr": all_pos_mr,
        "cv_std": round(cv_std, 4),
        "cv_mr": round(cv_mr, 4),
        "pass_std": std_pass,
        "pass_mr": mr_pass
    }


# ─── Test 5: Outlier Robustness ──────────────────────────────────────────────

def test_outlier_robustness(rets_std, rets_mr):
    """Remove 10 best days. Sharpe degradation < 30%."""
    print("\n" + "="*70)
    print("TEST 5: OUTLIER ROBUSTNESS (remove 10 best days)")
    print("="*70)

    for label, rets in [("GP3 Std", rets_std), ("GP3+MR", rets_mr)]:
        m_full = compute_metrics(rets)

        # Remove 10 best days
        sorted_rets = rets.sort_values(ascending=False)
        top10_dates = sorted_rets.index[:10]
        rets_trimmed = rets.drop(top10_dates)

        m_trimmed = compute_metrics(rets_trimmed)

        if m_full["sharpe"] != 0:
            degradation = (m_full["sharpe"] - m_trimmed["sharpe"]) / abs(m_full["sharpe"]) * 100
        else:
            degradation = 0

        passed = degradation < 30

        print(f"\n  {label}:")
        print(f"    Full Sharpe:    {m_full['sharpe']:.3f}")
        print(f"    Trimmed Sharpe: {m_trimmed['sharpe']:.3f}")
        print(f"    Degradation:    {degradation:.1f}%")
        print(f"    VERDICT: {'PASS (<30%)' if passed else 'FAIL (>=30%)'}")

    # Return MR-specific results
    m_full_mr = compute_metrics(rets_mr)
    sorted_mr = rets_mr.sort_values(ascending=False)
    rets_mr_trim = rets_mr.drop(sorted_mr.index[:10])
    m_trim_mr = compute_metrics(rets_mr_trim)

    m_full_std = compute_metrics(rets_std)
    sorted_std = rets_std.sort_values(ascending=False)
    rets_std_trim = rets_std.drop(sorted_std.index[:10])
    m_trim_std = compute_metrics(rets_std_trim)

    deg_std = (m_full_std["sharpe"] - m_trim_std["sharpe"]) / abs(m_full_std["sharpe"]) * 100 if m_full_std["sharpe"] != 0 else 0
    deg_mr = (m_full_mr["sharpe"] - m_trim_mr["sharpe"]) / abs(m_full_mr["sharpe"]) * 100 if m_full_mr["sharpe"] != 0 else 0

    return {
        "std_full_sharpe": m_full_std["sharpe"],
        "std_trimmed_sharpe": m_trim_std["sharpe"],
        "std_degradation_pct": round(deg_std, 2),
        "std_pass": deg_std < 30,
        "mr_full_sharpe": m_full_mr["sharpe"],
        "mr_trimmed_sharpe": m_trim_mr["sharpe"],
        "mr_degradation_pct": round(deg_mr, 2),
        "mr_pass": deg_mr < 30,
    }


# ─── Test 6: R1 Regime Test ─────────────────────────────────────────────────

def test_regime(df, rets_std, rets_mr):
    """
    R1 regime test: |Sharpe_green - Sharpe_red| / max < 0.50
    Green day: SPY close > prev close. Red day: SPY close < prev close.
    """
    print("\n" + "="*70)
    print("TEST 6: R1 REGIME TEST (green vs red days)")
    print("="*70)

    spy_rets = df["SPY"].pct_change()
    green_days = spy_rets > 0
    red_days = spy_rets < 0

    # Align with strategy returns
    common_idx = rets_mr.index.intersection(green_days.index)

    for label, rets in [("GP3 Std", rets_std), ("GP3+MR", rets_mr)]:
        idx = rets.index.intersection(common_idx)

        green_rets = rets.loc[idx][green_days.loc[idx]]
        red_rets = rets.loc[idx][red_days.loc[idx]]

        m_green = compute_metrics(green_rets)
        m_red = compute_metrics(red_rets)

        s_g = m_green["sharpe"]
        s_r = m_red["sharpe"]
        max_s = max(abs(s_g), abs(s_r))
        ratio = abs(s_g - s_r) / max_s if max_s > 0 else 0

        passed = ratio < 0.50

        print(f"\n  {label}:")
        print(f"    Green-day Sharpe: {s_g:.3f} ({len(green_rets)} days)")
        print(f"    Red-day Sharpe:   {s_r:.3f} ({len(red_rets)} days)")
        print(f"    Regime ratio:     {ratio:.3f}")
        print(f"    VERDICT: {'PASS (<0.50)' if passed else 'FAIL (>=0.50) — EXPECTED for growth strategy per HC #709'}")

    # MR results for output
    idx = rets_mr.index.intersection(common_idx)
    green_rets = rets_mr.loc[idx][green_days.loc[idx]]
    red_rets = rets_mr.loc[idx][red_days.loc[idx]]
    m_green = compute_metrics(green_rets)
    m_red = compute_metrics(red_rets)
    s_g, s_r = m_green["sharpe"], m_red["sharpe"]
    max_s = max(abs(s_g), abs(s_r))
    ratio = abs(s_g - s_r) / max_s if max_s > 0 else 0

    return {
        "green_sharpe": s_g,
        "red_sharpe": s_r,
        "regime_ratio": round(ratio, 4),
        "pass": ratio < 0.50,
        "note": "Expected to FAIL for growth strategy per HC #709"
    }


# ─── Test 7: Year-by-Year ───────────────────────────────────────────────────

def test_yearly(rets_std, rets_mr, mr_active):
    """Year-by-year comparison."""
    print("\n" + "="*70)
    print("TEST 7: YEAR-BY-YEAR COMPARISON")
    print("="*70)

    years = sorted(set(rets_mr.index.year))
    yearly = []

    print(f"\n  {'Year':>6} {'Std Sharpe':>12} {'MR Sharpe':>12} {'Delta':>10} "
          f"{'Std CAGR':>10} {'MR CAGR':>10} {'MR Days':>8}")
    print("  " + "-"*70)

    for yr in years:
        mask_std = rets_std.index.year == yr
        mask_mr = rets_mr.index.year == yr
        mask_active = mr_active.index.year == yr

        if mask_std.sum() < 20 or mask_mr.sum() < 20:
            continue

        m_std = compute_metrics(rets_std[mask_std])
        m_mr = compute_metrics(rets_mr[mask_mr])
        mr_days = int(mr_active[mask_active].sum())

        delta = m_mr["sharpe"] - m_std["sharpe"]

        print(f"  {yr:>6} {m_std['sharpe']:>12.3f} {m_mr['sharpe']:>12.3f} {delta:>+10.3f} "
              f"{m_std['cagr']:>9.2f}% {m_mr['cagr']:>9.2f}% {mr_days:>8}")

        yearly.append({
            "year": yr,
            "std_sharpe": m_std["sharpe"],
            "mr_sharpe": m_mr["sharpe"],
            "sharpe_delta": round(delta, 3),
            "std_cagr": m_std["cagr"],
            "mr_cagr": m_mr["cagr"],
            "mr_entry_days": mr_days,
        })

    mr_better_years = sum(1 for y in yearly if y["sharpe_delta"] > 0)
    print(f"\n  MR better in {mr_better_years}/{len(yearly)} years")

    return {"years": yearly, "mr_better_count": mr_better_years, "total_years": len(yearly)}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("GP3 + VIX Mean-Reversion Entry — Adversarial Validation")
    print("=" * 70)

    # Download data
    df = download_data()

    # Warmup: skip first 200 days for indicator calculation
    df = df.iloc[200:].copy()
    print(f"  After warmup: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} days")

    results = {}

    # Test 1: Full period
    t1, rets_std, rets_mr, alloc_std, alloc_mr, mr_active = test_full_period(df)
    results["full_period"] = t1

    # Test 2: Walk-forward
    results["walkforward"] = test_walkforward(df)

    # Test 3: Permutation test
    results["permutation"] = test_permutation(df, n_perms=200)

    # Test 4: Sub-period consistency
    results["subperiod"] = test_subperiod(rets_std, rets_mr)

    # Test 5: Outlier robustness
    results["outlier_robustness"] = test_outlier_robustness(rets_std, rets_mr)

    # Test 6: Regime test
    results["regime"] = test_regime(df, rets_std, rets_mr)

    # Test 7: Year-by-year
    results["yearly"] = test_yearly(rets_std, rets_mr, mr_active)

    # ─── Final Verdict ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    tests = {
        "Full-period improvement": t1["gp3_mr"]["sharpe"] > t1["gp3_standard"]["sharpe"],
        "Walk-forward": results["walkforward"]["pass"],
        "Permutation (p<0.05)": results["permutation"]["pass"],
        "Sub-period consistency": results["subperiod"]["pass_mr"],
        "Outlier robustness": results["outlier_robustness"]["mr_pass"],
        "Regime (expected FAIL)": results["regime"]["pass"],
    }

    critical_tests = ["Walk-forward", "Permutation (p<0.05)", "Sub-period consistency", "Outlier robustness"]
    critical_pass = all(tests[t] for t in critical_tests)

    for name, passed in tests.items():
        status = "PASS" if passed else "FAIL"
        critical = " [CRITICAL]" if name in critical_tests else ""
        expected = " [EXPECTED]" if "expected" in name.lower() else ""
        print(f"  {status:>4} — {name}{critical}{expected}")

    # Determine overall verdict
    sharpe_delta = t1["gp3_mr"]["sharpe"] - t1["gp3_standard"]["sharpe"]

    if critical_pass and sharpe_delta > 0:
        verdict = "ACCEPT"
        reason = (f"GP3+MR passes all critical tests. Sharpe improvement: {sharpe_delta:+.3f}. "
                  f"Permutation p={results['permutation']['p_value']:.4f}. "
                  f"MR adds {t1['additional_upro_days']} UPRO days with genuine timing edge.")
    elif not results["permutation"]["pass"]:
        verdict = "REJECT"
        reason = (f"FAILED permutation test (p={results['permutation']['p_value']:.4f}). "
                  f"MR timing is not statistically better than random VIX>20 entries. "
                  f"The enhancement likely captures generic recovery exposure, not a genuine signal.")
    elif not results["walkforward"]["pass"]:
        verdict = "REJECT"
        reason = (f"FAILED walk-forward validation. MR wins only "
                  f"{results['walkforward']['mr_win_count']}/{results['walkforward']['total_windows']} windows. "
                  f"Enhancement is period-dependent.")
    else:
        verdict = "REJECT"
        reason = (f"Failed critical tests. Sharpe delta: {sharpe_delta:+.3f}. "
                  f"Enhancement does not reliably improve GP3.")

    print(f"\n  VERDICT: {verdict}")
    print(f"  REASON: {reason}")

    results["verdict"] = verdict
    results["reason"] = reason
    results["critical_pass"] = critical_pass

    # Save results
    output_path = OUTPUT_DIR / "validation_results.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        return convert(obj)

    results = deep_convert(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
