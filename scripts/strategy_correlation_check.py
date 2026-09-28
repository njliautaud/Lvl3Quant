#!/usr/bin/env python3
"""
Strategy Correlation Check
==========================
Compares daily return series of the 3 validated strategies:
  1. Signal Aggregator A (composite score >= 3 -> long QQQ)
  2. Strategy Rotation v2 F (dynamic VIX-adjusted contrarian)
  3. Adaptive Leveraged Growth v2 (VIX-overlay UPRO/QQQ/SHY vol-timing)

Key question: Does #3 provide diversification benefit vs #1 and #2?
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── COMMON CONFIG ─────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
SLIPPAGE = 0.0002

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/strategy_correlation_check.json")


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 1: Signal Aggregator A (score >= 3 -> long QQQ)
# ═══════════════════════════════════════════════════════════════════════

def run_signal_aggregator_a(close, volume):
    """Reproduce Signal Aggregator variant A."""
    SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLE", "XLF", "XLV"]

    # Compute 5 signals
    spy_sma200 = close["SPY"].rolling(200).mean()
    regime = (close["SPY"] > spy_sma200).astype(float)

    vix = close["^VIX"]
    vix_below_20 = vix < 20
    vix_was_high = vix.rolling(10).max() > 25
    vix_declining = vix < vix.shift(5)
    vix_fade = vix_was_high & vix_declining
    vix_calm = (vix_below_20 | vix_fade).astype(float)

    momentum = (close["QQQ"].pct_change(20) > 0).astype(float)

    vol_20d_avg = volume[SECTOR_ETFS].rolling(20).mean()
    above_avg = volume[SECTOR_ETFS] > vol_20d_avg
    any_streak = pd.DataFrame(index=close.index, columns=SECTOR_ETFS, dtype=int)
    for etf in SECTOR_ETFS:
        arr = above_avg[etf].values.astype(int)
        result = np.zeros(len(arr), dtype=int)
        for i in range(len(arr)):
            if arr[i]:
                result[i] = result[i - 1] + 1 if i > 0 else 1
        any_streak[etf] = result
    volume_surge = (any_streak.max(axis=1) >= 5).astype(float)

    spy_ret_20d = close["SPY"].pct_change(20)
    rsp_ret_20d = close["RSP"].pct_change(20)
    breadth = ((spy_ret_20d > 0) & (rsp_ret_20d > spy_ret_20d * 0.5)).astype(float)

    score = regime + vix_calm + momentum + volume_surge + breadth

    # Variant A: score >= 3 -> QQQ, else cash
    oot_idx = close.loc[OOT_START:].index
    qqq_ret = close["QQQ"].pct_change().loc[oot_idx]
    score_oot = score.loc[oot_idx]

    invested = (score_oot >= 3).shift(1).fillna(False)
    trades_mask = invested != invested.shift(1)

    daily_ret = pd.Series(0.0, index=oot_idx)
    daily_ret[invested] = qqq_ret[invested]
    daily_ret[trades_mask & invested] -= SLIPPAGE

    # Position: 1 = long QQQ, 0 = cash
    position = invested.astype(int)

    return daily_ret, position


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 2: Strategy Rotation v2 F (dynamic VIX-adjusted contrarian)
# ═══════════════════════════════════════════════════════════════════════

def run_strategy_rotation_v2f(close):
    """Reproduce Strategy Rotation v2 variant F."""
    spy = close["SPY"]
    qqq = close["QQQ"]
    vix = close["^VIX"]

    sma200 = spy.rolling(200).mean()
    bull = (spy > sma200).astype(int)
    spy_ret5 = spy.pct_change(5)
    vix_chg5 = vix.pct_change(5)

    spy_ret = spy.pct_change().fillna(0)
    qqq_ret = qqq.pct_change().fillna(0)

    oot_idx = close.loc[OOT_START:].index

    daily_ret = pd.Series(0.0, index=oot_idx)
    position = pd.Series(0, index=oot_idx)  # 0=cash, 1=long
    hold_remaining = 0
    prev_label = None

    for i, date in enumerate(oot_idx):
        if date not in close.index:
            continue

        v = vix.loc[date]
        b = bull.loc[date]

        # Determine label
        if v > 25:
            label = "vix_fade"
        elif b:
            label = "earnings_momentum"
        else:
            label = "dynamic_contrarian"

        # Apply slippage on regime switch
        if label != prev_label and prev_label is not None:
            daily_ret.loc[date] -= SLIPPAGE
        prev_label = label

        if label == "earnings_momentum":
            daily_ret.loc[date] += qqq_ret.loc[date]
            position.loc[date] = 1

        elif label == "dynamic_contrarian":
            threshold = -1.5 * v / 100.0
            hold_days = 8 if v > 25 else 5
            if hold_remaining > 0:
                daily_ret.loc[date] += spy_ret.loc[date]
                position.loc[date] = 1
                hold_remaining -= 1
            elif spy_ret5.loc[date] <= threshold:
                daily_ret.loc[date] += spy_ret.loc[date]
                position.loc[date] = 1
                hold_remaining = hold_days - 1

        elif label == "vix_fade":
            if v > 25 and vix_chg5.loc[date] < 0:
                daily_ret.loc[date] += spy_ret.loc[date]
                position.loc[date] = 1

    return daily_ret, position


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 3: Adaptive Leveraged Growth v2 (VIX-overlay UPRO)
# ═══════════════════════════════════════════════════════════════════════

def run_adaptive_leveraged_v2(close):
    """Reproduce Adaptive Leveraged Growth v2."""
    spy = close["SPY"]
    vix = close["^VIX"]

    sma200 = spy.rolling(200).mean()
    above_sma = (spy > sma200).astype(float)

    vix_level = pd.Series(1, index=close.index)
    vix_level[vix < 20] = 2
    vix_level[vix >= 30] = 0

    if "^VIX3M" in close.columns:
        vix3m = close["^VIX3M"]
        vix_contango = (vix < vix3m).astype(float)
    else:
        vix_contango = pd.Series(1.0, index=close.index)

    # T-1 shift
    above_sma = above_sma.shift(1)
    vix_level = vix_level.shift(1)
    vix_contango = vix_contango.shift(1)

    # Asset returns
    asset_rets = {}
    for asset in ["UPRO", "QQQ", "SHY"]:
        if asset in close.columns:
            asset_rets[asset] = close[asset].pct_change()

    oot_idx = close.loc[OOT_START:].index
    daily_ret = pd.Series(0.0, index=oot_idx)
    position = pd.Series(0, index=oot_idx)  # 0=cash/SHY, 1=long risk
    asset_used = pd.Series("SHY", index=oot_idx)

    current_asset = "SHY"
    equity = 1.0
    equity_history = []
    circuit_breaker = False
    cb_trigger_day = 0

    for day_idx, date in enumerate(oot_idx):
        if date not in close.index:
            continue

        a_sma = above_sma.get(date, np.nan)
        v_lvl = vix_level.get(date, np.nan)
        v_cnt = vix_contango.get(date, np.nan)

        # Target allocation
        if np.isnan(a_sma) or np.isnan(v_lvl):
            target = "SHY"
        elif a_sma < 0.5:
            target = "SHY"
        elif v_lvl == 2:
            target = "UPRO" if v_cnt >= 0.5 else "QQQ"
        elif v_lvl == 1:
            target = "QQQ"
        else:
            target = "SHY"

        # Rolling DD check
        lookback = 40
        recent_eq = equity_history[-lookback:] if len(equity_history) >= lookback else equity_history
        if len(recent_eq) > 0:
            rolling_hwm = max(recent_eq)
            rolling_dd = (rolling_hwm - equity) / rolling_hwm if rolling_hwm > 0 else 0
        else:
            rolling_dd = 0

        if not circuit_breaker:
            if rolling_dd > 0.20:
                circuit_breaker = True
                cb_trigger_day = day_idx
        else:
            if day_idx - cb_trigger_day >= 15:
                if target != "SHY":
                    circuit_breaker = False

        if circuit_breaker:
            target = "SHY"

        # Weekly rebalance only
        is_friday = pd.Timestamp(date).day_name() == "Friday"

        cost = 0
        if target != current_asset and is_friday:
            cost_bps = max(
                {"UPRO": 20, "QQQ": 10, "SHY": 5}.get(current_asset, 10),
                {"UPRO": 20, "QQQ": 10, "SHY": 5}.get(target, 10),
            )
            cost = cost_bps / 10000
            current_asset = target

        day_ret = asset_rets.get(current_asset, pd.Series(0.0, index=close.index)).get(date, 0)
        if np.isnan(day_ret):
            day_ret = 0
        day_ret -= cost

        equity *= 1 + day_ret
        equity_history.append(equity)

        daily_ret.loc[date] = day_ret
        asset_used.loc[date] = current_asset
        position.loc[date] = 1 if current_asset in ["UPRO", "QQQ"] else 0

    return daily_ret, position, asset_used


# ═══════════════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("STRATEGY CORRELATION CHECK")
    print("=" * 70)

    # Download data
    tickers = ["SPY", "QQQ", "TQQQ", "UPRO", "SHY", "RSP", "TLT",
               "XLK", "XLC", "XLY", "XLE", "XLF", "XLV", "^VIX", "^VIX3M"]

    start = "2021-01-01"
    print(f"\nDownloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=OOT_END, auto_adjust=True, progress=False)

    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
    close = close.dropna(subset=["SPY"])
    volume = volume.loc[close.index]

    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")

    # Run strategies
    print("\n--- Running Signal Aggregator A ---")
    ret_agg, pos_agg = run_signal_aggregator_a(close, volume)

    print("--- Running Strategy Rotation v2 F ---")
    ret_rot, pos_rot = run_strategy_rotation_v2f(close)

    print("--- Running Adaptive Leveraged Growth v2 ---")
    ret_lev, pos_lev, asset_lev = run_adaptive_leveraged_v2(close)

    # Align indices
    common_idx = ret_agg.index.intersection(ret_rot.index).intersection(ret_lev.index)
    ret_agg = ret_agg.loc[common_idx]
    ret_rot = ret_rot.loc[common_idx]
    ret_lev = ret_lev.loc[common_idx]
    pos_agg = pos_agg.loc[common_idx]
    pos_rot = pos_rot.loc[common_idx]
    pos_lev = pos_lev.loc[common_idx]
    asset_lev = asset_lev.loc[common_idx]

    print(f"\nCommon OOT period: {common_idx[0].date()} to {common_idx[-1].date()}, {len(common_idx)} days")

    # ── Individual Strategy Metrics ───────────────────────────────────
    def calc_strat_metrics(rets, name):
        dr = rets.dropna()
        ann_ret = dr.mean() * 252
        ann_vol = dr.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        downside = dr[dr < 0].std() * np.sqrt(252)
        sortino = ann_ret / downside if downside > 0 else 0
        eq = (1 + dr).cumprod()
        mdd = ((eq - eq.cummax()) / eq.cummax()).min()
        total_ret = (eq.iloc[-1] - 1) * 100
        gains = dr[dr > 0].sum()
        losses = abs(dr[dr < 0].sum())
        pf = gains / losses if losses > 0 else float("inf")
        wr = (dr[dr != 0] > 0).mean() if (dr != 0).any() else 0
        return {
            "name": name,
            "total_return_pct": round(total_ret, 2),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown_pct": round(mdd * 100, 2),
            "profit_factor": round(pf, 3),
            "win_rate": round(wr, 4),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "pct_invested": round((rets != 0).mean(), 3),
        }

    m_agg = calc_strat_metrics(ret_agg, "Signal Aggregator A")
    m_rot = calc_strat_metrics(ret_rot, "Strategy Rotation v2 F")
    m_lev = calc_strat_metrics(ret_lev, "Adaptive Leveraged v2")

    print(f"\n{'='*70}")
    print(f"INDIVIDUAL STRATEGY METRICS")
    print(f"{'='*70}")
    print(f"{'Strategy':<30} {'Sharpe':>8} {'Sortino':>8} {'Return%':>9} {'MDD%':>8} {'PF':>7} {'WR':>7} {'%Inv':>7}")
    print("-" * 85)
    for m in [m_agg, m_rot, m_lev]:
        print(f"{m['name']:<30} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['total_return_pct']:>+8.1f}% "
              f"{m['max_drawdown_pct']:>7.1f}% {m['profit_factor']:>6.2f} {m['win_rate']:>6.1%} {m['pct_invested']:>6.1%}")

    # ── 1. Pearson Correlation of Daily Returns ───────────────────────
    print(f"\n{'='*70}")
    print(f"1. PEARSON CORRELATION — DAILY RETURNS")
    print(f"{'='*70}")

    returns_df = pd.DataFrame({
        "SigAgg_A": ret_agg,
        "StratRot_v2F": ret_rot,
        "AdaptLev_v2": ret_lev,
    })

    corr_matrix = returns_df.corr()
    print(corr_matrix.round(4).to_string())

    corr_agg_rot = corr_matrix.loc["SigAgg_A", "StratRot_v2F"]
    corr_agg_lev = corr_matrix.loc["SigAgg_A", "AdaptLev_v2"]
    corr_rot_lev = corr_matrix.loc["StratRot_v2F", "AdaptLev_v2"]

    print(f"\nSigAgg_A vs StratRot_v2F:  {corr_agg_rot:.4f}")
    print(f"SigAgg_A vs AdaptLev_v2:  {corr_agg_lev:.4f}")
    print(f"StratRot_v2F vs AdaptLev_v2: {corr_rot_lev:.4f}")

    # ── 1b. Correlation on ACTIVE DAYS ONLY ──────────────────────────
    print(f"\n{'='*70}")
    print(f"1b. CORRELATION — ACTIVE DAYS ONLY (both strategies invested)")
    print(f"{'='*70}")

    def active_corr(r1, r2, p1, p2, n1, n2):
        both_active = (p1 == 1) & (p2 == 1)
        if both_active.sum() < 10:
            return np.nan, 0
        return r1[both_active].corr(r2[both_active]), int(both_active.sum())

    ac_agg_rot, n_agg_rot = active_corr(ret_agg, ret_rot, pos_agg, pos_rot, "A", "F_rot")
    ac_agg_lev, n_agg_lev = active_corr(ret_agg, ret_lev, pos_agg, pos_lev, "A", "F_lev")
    ac_rot_lev, n_rot_lev = active_corr(ret_rot, ret_lev, pos_rot, pos_lev, "F_rot", "F_lev")

    print(f"SigAgg_A vs StratRot_v2F:    {ac_agg_rot:.4f}  ({n_agg_rot} shared active days)")
    print(f"SigAgg_A vs AdaptLev_v2:     {ac_agg_lev:.4f}  ({n_agg_lev} shared active days)")
    print(f"StratRot_v2F vs AdaptLev_v2: {ac_rot_lev:.4f}  ({n_rot_lev} shared active days)")

    # ── 2. Rolling 60-Day Correlation ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"2. ROLLING 60-DAY CORRELATION (summary stats)")
    print(f"{'='*70}")

    rolling_window = 60
    roll_agg_rot = ret_agg.rolling(rolling_window).corr(ret_rot)
    roll_agg_lev = ret_agg.rolling(rolling_window).corr(ret_lev)
    roll_rot_lev = ret_rot.rolling(rolling_window).corr(ret_lev)

    for name, roll in [("SigAgg_A vs StratRot_v2F", roll_agg_rot),
                        ("SigAgg_A vs AdaptLev_v2", roll_agg_lev),
                        ("StratRot_v2F vs AdaptLev_v2", roll_rot_lev)]:
        r = roll.dropna()
        print(f"\n  {name}:")
        print(f"    Mean: {r.mean():.4f}")
        print(f"    Std:  {r.std():.4f}")
        print(f"    Min:  {r.min():.4f}")
        print(f"    Max:  {r.max():.4f}")
        print(f"    Pct > 0.7: {(r > 0.7).mean()*100:.1f}%")
        print(f"    Pct > 0.5: {(r > 0.5).mean()*100:.1f}%")
        print(f"    Pct < 0.3: {(r < 0.3).mean()*100:.1f}%")

    # ── 3. Regime-Specific Correlation ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"3. REGIME-SPECIFIC CORRELATION (SPY vs 200-SMA)")
    print(f"{'='*70}")

    spy_sma200 = close["SPY"].rolling(200).mean()
    bull_mask = (close["SPY"] > spy_sma200).loc[common_idx]
    bear_mask = ~bull_mask

    bull_days = bull_mask.sum()
    bear_days = bear_mask.sum()
    print(f"  Bull days: {bull_days} ({bull_days/len(common_idx)*100:.1f}%)")
    print(f"  Bear days: {bear_days} ({bear_days/len(common_idx)*100:.1f}%)")

    for regime_name, mask in [("BULL", bull_mask), ("BEAR", bear_mask)]:
        if mask.sum() < 30:
            print(f"\n  {regime_name}: too few days ({mask.sum()})")
            continue
        print(f"\n  {regime_name} regime ({mask.sum()} days):")
        r_agg = ret_agg[mask]
        r_rot = ret_rot[mask]
        r_lev = ret_lev[mask]

        c1 = r_agg.corr(r_rot)
        c2 = r_agg.corr(r_lev)
        c3 = r_rot.corr(r_lev)
        print(f"    SigAgg_A vs StratRot_v2F:    {c1:.4f}")
        print(f"    SigAgg_A vs AdaptLev_v2:     {c2:.4f}")
        print(f"    StratRot_v2F vs AdaptLev_v2: {c3:.4f}")

    # ── 4. Signal Agreement Rate ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"4. SIMULTANEOUS SIGNAL AGREEMENT RATE")
    print(f"{'='*70}")

    # Agreement = both in same state (both invested or both cash)
    agree_agg_rot = (pos_agg == pos_rot).mean()
    agree_agg_lev = (pos_agg == pos_lev).mean()
    agree_rot_lev = (pos_rot == pos_lev).mean()

    # Both invested at same time
    both_in_agg_rot = ((pos_agg == 1) & (pos_rot == 1)).mean()
    both_in_agg_lev = ((pos_agg == 1) & (pos_lev == 1)).mean()
    both_in_rot_lev = ((pos_rot == 1) & (pos_lev == 1)).mean()

    # Both cash at same time
    both_out_agg_rot = ((pos_agg == 0) & (pos_rot == 0)).mean()
    both_out_agg_lev = ((pos_agg == 0) & (pos_lev == 0)).mean()
    both_out_rot_lev = ((pos_rot == 0) & (pos_lev == 0)).mean()

    print(f"{'Pair':<35} {'Agreement':>10} {'Both In':>10} {'Both Out':>10}")
    print("-" * 65)
    print(f"{'SigAgg_A vs StratRot_v2F':<35} {agree_agg_rot:>9.1%} {both_in_agg_rot:>9.1%} {both_out_agg_rot:>9.1%}")
    print(f"{'SigAgg_A vs AdaptLev_v2':<35} {agree_agg_lev:>9.1%} {both_in_agg_lev:>9.1%} {both_out_agg_lev:>9.1%}")
    print(f"{'StratRot_v2F vs AdaptLev_v2':<35} {agree_rot_lev:>9.1%} {both_in_rot_lev:>9.1%} {both_out_rot_lev:>9.1%}")

    # Individual investment rates
    print(f"\n  Signal Aggregator A invested: {(pos_agg == 1).mean():.1%}")
    print(f"  Strategy Rotation v2 F invested: {(pos_rot == 1).mean():.1%}")
    print(f"  Adaptive Leveraged v2 invested: {(pos_lev == 1).mean():.1%}")

    # Adaptive Lev asset breakdown
    upro_pct = (asset_lev == "UPRO").mean()
    qqq_pct = (asset_lev == "QQQ").mean()
    shy_pct = (asset_lev == "SHY").mean()
    print(f"\n  Adaptive Leveraged v2 asset breakdown:")
    print(f"    UPRO: {upro_pct:.1%}")
    print(f"    QQQ:  {qqq_pct:.1%}")
    print(f"    SHY:  {shy_pct:.1%}")

    # ── 5. Portfolio Combination Test ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"5. PORTFOLIO COMBINATION TEST")
    print(f"{'='*70}")

    # Equal-weight 3-strategy portfolio
    ew3 = (ret_agg + ret_rot + ret_lev) / 3
    m_ew3 = calc_strat_metrics(ew3, "Equal-Weight 3-Strategy")

    # 2-strategy portfolios
    ew_agg_rot = (ret_agg + ret_rot) / 2
    m_agg_rot = calc_strat_metrics(ew_agg_rot, "SigAgg_A + StratRot_v2F")

    ew_agg_lev = (ret_agg + ret_lev) / 2
    m_agg_lev = calc_strat_metrics(ew_agg_lev, "SigAgg_A + AdaptLev_v2")

    ew_rot_lev = (ret_rot + ret_lev) / 2
    m_rot_lev = calc_strat_metrics(ew_rot_lev, "StratRot_v2F + AdaptLev_v2")

    print(f"{'Portfolio':<30} {'Sharpe':>8} {'Sortino':>8} {'Return%':>9} {'MDD%':>8}")
    print("-" * 65)
    for m in [m_agg, m_rot, m_lev, m_agg_rot, m_agg_lev, m_rot_lev, m_ew3]:
        print(f"{m['name']:<30} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['total_return_pct']:>+8.1f}% {m['max_drawdown_pct']:>7.1f}%")

    # ── 6. KEY VERDICT ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"6. DIVERSIFICATION VERDICT")
    print(f"{'='*70}")

    def verdict(corr, name):
        if corr < 0.3:
            return f"{name}: corr={corr:.3f} -- STRONG DIVERSIFIER (< 0.3)"
        elif corr < 0.5:
            return f"{name}: corr={corr:.3f} -- GOOD DIVERSIFIER (< 0.5)"
        elif corr < 0.7:
            return f"{name}: corr={corr:.3f} -- MODERATE OVERLAP (0.5-0.7)"
        else:
            return f"{name}: corr={corr:.3f} -- REDUNDANT (> 0.7)"

    print(f"  {verdict(corr_agg_rot, 'SigAgg_A vs StratRot_v2F')}")
    print(f"  {verdict(corr_agg_lev, 'SigAgg_A vs AdaptLev_v2')}")
    print(f"  {verdict(corr_rot_lev, 'StratRot_v2F vs AdaptLev_v2')}")

    is_diversifier = corr_agg_lev < 0.5 and corr_rot_lev < 0.5
    print(f"\n  ANSWER: Adaptive Leveraged v2 {'IS' if is_diversifier else 'IS NOT'} a genuine portfolio addition.")

    if is_diversifier:
        sharpe_boost = m_ew3["sharpe"] - max(m_agg["sharpe"], m_rot["sharpe"])
        print(f"  3-strategy portfolio Sharpe: {m_ew3['sharpe']:.3f} vs best individual: {max(m_agg['sharpe'], m_rot['sharpe']):.3f}")
        print(f"  Sharpe improvement from adding AdaptLev: {sharpe_boost:+.3f}")
        mdd_improvement = m_ew3["max_drawdown_pct"] - min(m_agg["max_drawdown_pct"], m_rot["max_drawdown_pct"])
        print(f"  MDD improvement: {mdd_improvement:+.1f}pp")
    else:
        print(f"  The strategies share too much 'long tech in calm markets' exposure.")
        print(f"  Adding AdaptLev to the portfolio adds little diversification.")

    # ── Save Results ──────────────────────────────────────────────────
    # Collect rolling correlation quantiles
    def roll_stats(r):
        r = r.dropna()
        return {
            "mean": round(float(r.mean()), 4),
            "std": round(float(r.std()), 4),
            "min": round(float(r.min()), 4),
            "max": round(float(r.max()), 4),
            "pct_above_0.7": round(float((r > 0.7).mean()), 4),
            "pct_above_0.5": round(float((r > 0.5).mean()), 4),
            "pct_below_0.3": round(float((r < 0.3).mean()), 4),
        }

    results = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "n_common_days": len(common_idx),
        },
        "individual_metrics": {
            "signal_aggregator_a": m_agg,
            "strategy_rotation_v2f": m_rot,
            "adaptive_leveraged_v2": m_lev,
        },
        "pearson_correlation": {
            "sigagg_a_vs_stratrot_v2f": round(corr_agg_rot, 4),
            "sigagg_a_vs_adaptlev_v2": round(corr_agg_lev, 4),
            "stratrot_v2f_vs_adaptlev_v2": round(corr_rot_lev, 4),
        },
        "active_days_correlation": {
            "sigagg_a_vs_stratrot_v2f": {"corr": round(ac_agg_rot, 4) if not np.isnan(ac_agg_rot) else None, "n_days": n_agg_rot},
            "sigagg_a_vs_adaptlev_v2": {"corr": round(ac_agg_lev, 4) if not np.isnan(ac_agg_lev) else None, "n_days": n_agg_lev},
            "stratrot_v2f_vs_adaptlev_v2": {"corr": round(ac_rot_lev, 4) if not np.isnan(ac_rot_lev) else None, "n_days": n_rot_lev},
        },
        "rolling_60d_correlation": {
            "sigagg_a_vs_stratrot_v2f": roll_stats(roll_agg_rot),
            "sigagg_a_vs_adaptlev_v2": roll_stats(roll_agg_lev),
            "stratrot_v2f_vs_adaptlev_v2": roll_stats(roll_rot_lev),
        },
        "regime_correlation": {},
        "signal_agreement": {
            "sigagg_a_vs_stratrot_v2f": {"agreement": round(agree_agg_rot, 4), "both_invested": round(both_in_agg_rot, 4), "both_cash": round(both_out_agg_rot, 4)},
            "sigagg_a_vs_adaptlev_v2": {"agreement": round(agree_agg_lev, 4), "both_invested": round(both_in_agg_lev, 4), "both_cash": round(both_out_agg_lev, 4)},
            "stratrot_v2f_vs_adaptlev_v2": {"agreement": round(agree_rot_lev, 4), "both_invested": round(both_in_rot_lev, 4), "both_cash": round(both_out_rot_lev, 4)},
        },
        "adaptive_lev_asset_breakdown": {
            "UPRO_pct": round(upro_pct, 4),
            "QQQ_pct": round(qqq_pct, 4),
            "SHY_pct": round(shy_pct, 4),
        },
        "portfolio_combinations": {
            "sigagg_a_plus_stratrot_v2f": m_agg_rot,
            "sigagg_a_plus_adaptlev_v2": m_agg_lev,
            "stratrot_v2f_plus_adaptlev_v2": m_rot_lev,
            "equal_weight_3_strategy": m_ew3,
        },
        "verdict": {
            "is_diversifier": is_diversifier,
            "corr_threshold": 0.5,
            "answer": (
                "Adaptive Leveraged v2 IS a genuine portfolio addition"
                if is_diversifier
                else "Adaptive Leveraged v2 is NOT a genuine portfolio addition — too correlated with existing strategies"
            ),
        },
    }

    # Regime correlations
    for regime_name, mask in [("bull", bull_mask), ("bear", bear_mask)]:
        if mask.sum() >= 30:
            r_agg = ret_agg[mask]
            r_rot = ret_rot[mask]
            r_lev = ret_lev[mask]
            results["regime_correlation"][regime_name] = {
                "n_days": int(mask.sum()),
                "sigagg_a_vs_stratrot_v2f": round(float(r_agg.corr(r_rot)), 4),
                "sigagg_a_vs_adaptlev_v2": round(float(r_agg.corr(r_lev)), 4),
                "stratrot_v2f_vs_adaptlev_v2": round(float(r_rot.corr(r_lev)), 4),
            }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    results = main()
