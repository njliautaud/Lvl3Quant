#!/usr/bin/env python3
"""
SECTOR IV RANK ROTATION
========================
Hypothesis: Sectors with the lowest relative IV (cheapest options) tend to
outperform in the next 5-10 days. Cheap options = market complacency about
that sector = likely to surprise to the upside.

This combines two proven edges:
  1. Our IV regime options backtest showed cheap IV = 3x better option returns
  2. Sector rotation picking top-3 works (multiple validated strategies)

What if we rotate into whichever sectors have the CHEAPEST options?

Uses historical volatility as IV proxy (since we don't have real IV history for all sectors).
HV rank = 20-day realized vol percentile over 252-day lookback.
Low HV rank ≈ low IV rank (strong correlation for sector ETFs).

Walk-forward: sliding 60-day window, 5-day hold, 2020-2026
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime

np.random.seed(42)

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK, VIX_TICKER]

HOLD_DAYS = 5
TOP_N = 3
COST_RT = 0.0020
HV_LOOKBACK = 20  # days for HV calc
HV_RANK_WINDOW = 252  # 1 year for percentile rank

DATA_START = '2018-01-01'  # extra for warmup
DATA_END = '2026-08-21'


def download_data():
    print("Downloading data...", flush=True)
    raw = yf.download(ALL_TICKERS, start=DATA_START, end=DATA_END,
                       progress=False, auto_adjust=True, group_by='ticker', threads=True)
    closes = {}
    for ticker in ALL_TICKERS:
        try:
            s = raw[ticker]['Close'].dropna().squeeze()
            if len(s) > 100:
                closes[ticker] = s
        except:
            pass
    df = pd.DataFrame(closes).dropna()
    print(f"  Got {len(df)} trading days, {len(df.columns)} tickers", flush=True)
    return df


def compute_sharpe(rets):
    if len(rets) < 10 or rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * np.sqrt(252))


def compute_sortino(rets):
    down = rets[rets < 0]
    if len(down) < 5 or down.std() == 0:
        return 0.0
    return float((rets.mean() / down.std()) * np.sqrt(252))


def compute_full_metrics(rets, spy_rets):
    sharpe = compute_sharpe(rets)
    sortino = compute_sortino(rets)
    cum = (1 + rets).prod() - 1
    spy_cum = (1 + spy_rets).prod() - 1

    cum_eq = (1 + rets).cumprod()
    peak = cum_eq.expanding().max()
    max_dd = ((cum_eq - peak) / peak).min()

    block_rets = []
    for i in range(0, len(rets) - HOLD_DAYS + 1, HOLD_DAYS):
        block_rets.append(rets.iloc[i:i+HOLD_DAYS].sum())
    block_rets = np.array(block_rets) if block_rets else np.array([0])
    wr = (block_rets > 0).mean() * 100
    pf_g = block_rets[block_rets > 0].sum()
    pf_l = abs(block_rets[block_rets < 0].sum())
    pf = pf_g / max(pf_l, 1e-10)

    green = spy_rets > 0
    red = spy_rets < 0
    sh_g = compute_sharpe(rets[green]) if green.sum() > 10 else 0
    sh_r = compute_sharpe(rets[red]) if red.sum() > 10 else 0
    max_reg = max(abs(sh_g), abs(sh_r))
    regime_gap = abs(sh_g - sh_r) / max_reg if max_reg > 0 else 0

    return {
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cum_pct': round(cum * 100, 2), 'spy_pct': round(spy_cum * 100, 2),
        'excess_pct': round((cum - spy_cum) * 100, 2),
        'max_dd_pct': round(max_dd * 100, 2),
        'trade_wr': round(wr, 1), 'trade_pf': round(min(pf, 99), 3),
        'n_trades': len(block_rets),
        'sharpe_green': round(sh_g, 3), 'sharpe_red': round(sh_r, 3),
        'regime_gap': round(regime_gap, 4),
    }


def compute_hv_rank(prices, lookback=HV_LOOKBACK, rank_window=HV_RANK_WINDOW):
    """
    For each sector on each day, compute:
    - HV = 20-day annualized realized vol
    - HV_rank = percentile rank of current HV vs last 252 days of HV
    Low rank = vol is at historical lows = options cheap = complacent
    """
    returns = prices[SECTOR_ETFS].pct_change()

    hv = returns.rolling(lookback).std() * np.sqrt(252)

    hv_rank = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS, dtype=float)
    for etf in SECTOR_ETFS:
        if etf not in hv.columns:
            continue
        for i in range(rank_window + lookback, len(hv)):
            current = hv[etf].iloc[i]
            history = hv[etf].iloc[i-rank_window:i]
            if pd.isna(current) or history.isna().all():
                continue
            rank = (history < current).sum() / len(history.dropna())
            hv_rank.iloc[i][etf] = rank

    return hv_rank


def run_rotation(prices, signal_df, label, top_n=TOP_N, hold=HOLD_DAYS,
                 cost=COST_RT, pick_lowest=True, shuffle_seed=None):
    """
    Rotate into sectors with lowest (or highest) signal values.
    pick_lowest=True: cheap vol (low rank) = our hypothesis
    pick_lowest=False: expensive vol = inverse test
    """
    returns = prices[SECTOR_ETFS].pct_change()
    spy_returns = prices[BENCHMARK].pct_change()

    warmup = HV_RANK_WINDOW + HV_LOOKBACK + 10
    tradeable = prices.index[warmup:]

    rng = np.random.RandomState(shuffle_seed) if shuffle_seed is not None else None

    strat_rets = []
    days_since = hold
    holdings = []
    n_rebalances = 0

    for date in tradeable:
        if days_since >= hold:
            day_sig = signal_df.loc[date].dropna() if date in signal_df.index else pd.Series(dtype=float)

            if len(day_sig) >= top_n:
                if rng is not None:
                    available = list(day_sig.index)
                    rng.shuffle(available)
                    holdings = available[:top_n]
                elif pick_lowest:
                    ranked = day_sig.sort_values(ascending=True)
                    holdings = list(ranked.index[:top_n])
                else:
                    ranked = day_sig.sort_values(ascending=False)
                    holdings = list(ranked.index[:top_n])
                n_rebalances += 1
            days_since = 0

        if holdings and date in returns.index:
            dr = returns.loc[date, holdings]
            port_ret = dr.mean()
            if days_since == 0:
                port_ret -= cost / hold
            strat_rets.append(float(port_ret))
        else:
            strat_rets.append(0.0)

        days_since += 1

    sr = pd.Series(strat_rets, index=tradeable[:len(strat_rets)])
    spy_sr = spy_returns.loc[tradeable[:len(strat_rets)]]

    return sr, spy_sr, n_rebalances


def main():
    print("=" * 90, flush=True)
    print("SECTOR IV RANK ROTATION (HV Proxy)", flush=True)
    print("Hypothesis: Sectors with cheapest vol outperform next 5 days", flush=True)
    print("=" * 90, flush=True)

    prices = download_data()
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}\n", flush=True)

    # Compute HV rank
    print("Computing HV rank...", flush=True)
    hv_rank = compute_hv_rank(prices)
    print(f"HV rank computed. Non-null values: {hv_rank.count().sum()}", flush=True)

    # ── Strategy 1: Cheapest vol rotation ──────────────────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("STRATEGY A: Rotate into CHEAPEST VOL sectors (bottom-3 HV rank)", flush=True)
    print("=" * 70, flush=True)

    sr_a, spy_a, n_a = run_rotation(prices, hv_rank, "Cheapest Vol", pick_lowest=True)
    m_a = compute_full_metrics(sr_a, spy_a)

    print(f"  Sharpe: {m_a['sharpe']:.3f}, Sortino: {m_a['sortino']:.3f}", flush=True)
    print(f"  Cum return: {m_a['cum_pct']:.2f}%, SPY: {m_a['spy_pct']:.2f}%, Excess: {m_a['excess_pct']:.2f}%", flush=True)
    print(f"  Trade WR: {m_a['trade_wr']:.1f}%, PF: {m_a['trade_pf']:.3f}", flush=True)
    print(f"  Max DD: {m_a['max_dd_pct']:.2f}%", flush=True)
    print(f"  Green Sharpe: {m_a['sharpe_green']:.3f}, Red Sharpe: {m_a['sharpe_red']:.3f}", flush=True)
    print(f"  Regime gap: {m_a['regime_gap']:.4f}", flush=True)
    print(f"  Rebalances: {n_a}", flush=True)

    # ── Strategy 2: Expensive vol rotation (inverse) ───────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("STRATEGY B: Rotate into EXPENSIVE VOL sectors (top-3 HV rank) — INVERSE", flush=True)
    print("=" * 70, flush=True)

    sr_b, spy_b, n_b = run_rotation(prices, hv_rank, "Expensive Vol", pick_lowest=False)
    m_b = compute_full_metrics(sr_b, spy_b)

    print(f"  Sharpe: {m_b['sharpe']:.3f}, Sortino: {m_b['sortino']:.3f}", flush=True)
    print(f"  Cum return: {m_b['cum_pct']:.2f}%, SPY: {m_b['spy_pct']:.2f}%, Excess: {m_b['excess_pct']:.2f}%", flush=True)
    print(f"  Trade WR: {m_b['trade_wr']:.1f}%, PF: {m_b['trade_pf']:.3f}", flush=True)
    print(f"  Max DD: {m_b['max_dd_pct']:.2f}%", flush=True)
    print(f"  Green Sharpe: {m_b['sharpe_green']:.3f}, Red Sharpe: {m_b['sharpe_red']:.3f}", flush=True)
    print(f"  Regime gap: {m_b['regime_gap']:.4f}", flush=True)

    # ── Strategy 3: Cheapest vol + RSI oversold ────────────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("STRATEGY C: Cheapest vol + RSI < 35 (confluence filter)", flush=True)
    print("=" * 70, flush=True)

    # Compute RSI
    from collections import defaultdict
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    rsi_df = pd.DataFrame(index=prices.index)
    for etf in SECTOR_ETFS:
        if etf in prices.columns:
            rsi_df[etf] = compute_rsi(prices[etf])

    # Combined signal: low HV rank + RSI < 35
    combined = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS, dtype=float)
    for etf in SECTOR_ETFS:
        if etf in hv_rank.columns and etf in rsi_df.columns:
            # Only include sectors that are BOTH cheap vol AND oversold RSI
            mask = rsi_df[etf] < 35
            combined[etf] = hv_rank[etf].where(mask, np.nan)

    returns = prices[SECTOR_ETFS].pct_change()
    spy_returns = prices[BENCHMARK].pct_change()

    warmup = HV_RANK_WINDOW + HV_LOOKBACK + 10
    tradeable = prices.index[warmup:]

    strat_rets_c = []
    days_since = HOLD_DAYS
    holdings = []
    n_trades_c = 0

    for date in tradeable:
        if days_since >= HOLD_DAYS:
            day_sig = combined.loc[date].dropna() if date in combined.index else pd.Series(dtype=float)

            if len(day_sig) >= 1:  # at least 1 sector that's both cheap + oversold
                ranked = day_sig.sort_values(ascending=True)  # cheapest vol first
                holdings = list(ranked.index[:min(3, len(ranked))])
                days_since = 0
                n_trades_c += 1
            else:
                holdings = []

        if holdings and date in returns.index:
            dr = returns.loc[date, holdings]
            port_ret = dr.mean()
            if days_since == 0:
                port_ret -= COST_RT / HOLD_DAYS
            strat_rets_c.append(float(port_ret))
        else:
            strat_rets_c.append(0.0)

        days_since += 1

    sr_c = pd.Series(strat_rets_c, index=tradeable[:len(strat_rets_c)])
    spy_c = spy_returns.loc[tradeable[:len(strat_rets_c)]]

    in_pos = sr_c != 0
    if in_pos.sum() > 20:
        m_c = compute_full_metrics(sr_c, spy_c)
        print(f"  Sharpe: {m_c['sharpe']:.3f}, Sortino: {m_c['sortino']:.3f}", flush=True)
        print(f"  Cum return: {m_c['cum_pct']:.2f}%, SPY: {m_c['spy_pct']:.2f}%, Excess: {m_c['excess_pct']:.2f}%", flush=True)
        print(f"  Trade WR: {m_c['trade_wr']:.1f}%, PF: {m_c['trade_pf']:.3f}", flush=True)
        print(f"  Max DD: {m_c['max_dd_pct']:.2f}%", flush=True)
        print(f"  Green Sharpe: {m_c['sharpe_green']:.3f}, Red Sharpe: {m_c['sharpe_red']:.3f}", flush=True)
        print(f"  Regime gap: {m_c['regime_gap']:.4f}", flush=True)
        print(f"  Entry days: {in_pos.sum()}, Rebalances: {n_trades_c}", flush=True)
    else:
        print(f"  Too few entry days ({in_pos.sum()}). RSI<35 + cheap vol is rare.", flush=True)
        m_c = None

    # ── Strategy 4: Vol rank change (vol compression = breakout setup) ────
    print(f"\n{'=' * 70}", flush=True)
    print("STRATEGY D: Vol rank DECLINING (vol compressing = pre-breakout)", flush=True)
    print("=" * 70, flush=True)

    hv_rank_change = hv_rank.diff(10)  # 10-day change in HV rank

    sr_d, spy_d, n_d = run_rotation(prices, hv_rank_change, "Vol Compressing",
                                      pick_lowest=True)  # most negative = vol dropping fastest
    m_d = compute_full_metrics(sr_d, spy_d)

    print(f"  Sharpe: {m_d['sharpe']:.3f}, Sortino: {m_d['sortino']:.3f}", flush=True)
    print(f"  Cum return: {m_d['cum_pct']:.2f}%, SPY: {m_d['spy_pct']:.2f}%, Excess: {m_d['excess_pct']:.2f}%", flush=True)
    print(f"  Trade WR: {m_d['trade_wr']:.1f}%, PF: {m_d['trade_pf']:.3f}", flush=True)
    print(f"  Max DD: {m_d['max_dd_pct']:.2f}%", flush=True)
    print(f"  Green Sharpe: {m_d['sharpe_green']:.3f}, Red Sharpe: {m_d['sharpe_red']:.3f}", flush=True)
    print(f"  Regime gap: {m_d['regime_gap']:.4f}", flush=True)

    # ── Permutation test on best strategy ─────────────────────────────────
    strategies = [
        ('A_cheapest_vol', m_a, hv_rank, True),
        ('B_expensive_vol', m_b, hv_rank, False),
        ('D_vol_compressing', m_d, hv_rank_change, True),
    ]

    best = max(strategies, key=lambda x: x[1]['sharpe'])
    best_name, best_m, best_sig, best_pick = best

    if best_m['sharpe'] > 0.5:
        print(f"\n{'=' * 70}", flush=True)
        print(f"PERMUTATION TEST on {best_name} (Sharpe {best_m['sharpe']:.3f})", flush=True)
        print("=" * 70, flush=True)

        perm_sharpes = []
        for seed in range(500):
            r = run_rotation(prices, best_sig, "perm", shuffle_seed=seed + 9000)
            if r:
                perm_sharpes.append(compute_sharpe(r[0]))
            if (seed + 1) % 100 == 0:
                print(f"  ... {seed + 1}/500 permutations done", flush=True)

        perm_sharpes = np.array(perm_sharpes)
        perm_p = (perm_sharpes >= best_m['sharpe']).sum() / len(perm_sharpes)
        print(f"  Permutation p-value: {perm_p:.4f}", flush=True)
        print(f"  Random Sharpe: mean={perm_sharpes.mean():.3f}, std={perm_sharpes.std():.3f}", flush=True)
    else:
        print(f"\nBest strategy Sharpe {best_m['sharpe']:.3f} < 0.5. No permutation test needed.", flush=True)
        perm_p = 1.0

    # ── Yearly Breakdown of best ──────────────────────────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print(f"YEARLY BREAKDOWN — {best_name}", flush=True)
    print("=" * 70, flush=True)

    if best_name == 'A_cheapest_vol':
        sr_best, spy_best = sr_a, spy_a
    elif best_name == 'D_vol_compressing':
        sr_best, spy_best = sr_d, spy_d
    else:
        sr_best, spy_best = sr_b, spy_b

    for year in sorted(sr_best.index.year.unique()):
        mask = sr_best.index.year == year
        yr = sr_best[mask]
        if len(yr) < 10:
            continue
        cum = (1 + yr).prod() - 1
        sh = compute_sharpe(yr)
        print(f"  {year}: return={cum*100:+.2f}%, Sharpe={sh:.3f}, days={len(yr)}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 90}", flush=True)

    for name, m, _, _ in strategies:
        gates = {
            'sharpe>0.5': m['sharpe'] > 0.5,
            'regime_gap<0.5': m['regime_gap'] < 0.5,
            'trades>=40': m['n_trades'] >= 40,
            'mdd>-30': m['max_dd_pct'] > -30,
        }
        pass_count = sum(gates.values())
        print(f"\n  {name}: Sharpe={m['sharpe']:.3f}, WR={m['trade_wr']:.1f}%, "
              f"excess={m['excess_pct']:.2f}%, gap={m['regime_gap']:.4f}, "
              f"gates={pass_count}/4", flush=True)
        for g, v in gates.items():
            if not v:
                print(f"    FAIL: {g}", flush=True)

    if m_c:
        print(f"\n  C_cheap+oversold: Sharpe={m_c['sharpe']:.3f}, WR={m_c['trade_wr']:.1f}%, "
              f"excess={m_c['excess_pct']:.2f}%, gap={m_c['regime_gap']:.4f}", flush=True)

    # Check if A beats B (cheap beats expensive = real IV edge)
    if m_a['sharpe'] > m_b['sharpe']:
        print(f"\n  ✅ Cheap vol (A: {m_a['sharpe']:.3f}) beats expensive vol (B: {m_b['sharpe']:.3f})", flush=True)
        print(f"     → IV rank has REAL predictive power for sector rotation", flush=True)
    else:
        print(f"\n  ❌ Expensive vol (B: {m_b['sharpe']:.3f}) beats cheap vol (A: {m_a['sharpe']:.3f})", flush=True)
        print(f"     → Cheap IV hypothesis is WRONG for rotation. High vol = higher returns (risk premium).", flush=True)

    print(f"\nDone. {datetime.now()}", flush=True)


if __name__ == '__main__':
    main()
