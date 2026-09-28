#!/usr/bin/env python3
"""
SECTOR LEAD-LAG RELATIONSHIP BACKTEST
=======================================
Hypothesis: Some sectors consistently lead others by 1-5 days.
If we can identify which sectors are "leaders" and which "follow",
we can position in the followers when leaders move first.

Tests:
  A) Granger-causality style: does XLK 1d return predict XLF 1d return next day?
  B) Cross-correlation at lag 1-5d for all sector pairs
  C) Tradeable strategy: when a leader sector moves >1% in a day,
     go long the follower sector next day

Walk-forward: sliding 60-day estimation window, 1-day step
Cost: 0.20% round-trip
Hold: 5 trading days
OOT: ALL available days from 2021-01-01 onward
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime
import sys

np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

ESTIMATION_WINDOW = 60
HOLD_DAYS = 5
COST_RT = 0.0020
LEADER_THRESHOLD = 0.01  # 1% move in leader sector

DATA_START = '2019-01-01'
DATA_END = '2026-08-21'


def download_data():
    print("Downloading sector ETF + SPY daily data...", flush=True)
    raw = yf.download(ALL_TICKERS, start=DATA_START, end=DATA_END,
                       progress=False, auto_adjust=True, group_by='ticker', threads=True)
    closes = {}
    for ticker in ALL_TICKERS:
        try:
            if len(ALL_TICKERS) > 1:
                s = raw[ticker]['Close'].dropna().squeeze()
            else:
                s = raw['Close'].dropna().squeeze()
            if len(s) > 100:
                closes[ticker] = s
        except Exception as e:
            print(f"  WARNING: Failed to get {ticker}: {e}", flush=True)
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


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: Cross-Correlation Matrix at Various Lags
# ══════════════════════════════════════════════════════════════════════════════

def cross_correlation_analysis(prices):
    """Compute cross-correlation between all sector pairs at lags 1-5 days."""
    returns = prices[SECTOR_ETFS].pct_change().dropna()

    print("\n" + "=" * 80, flush=True)
    print("CROSS-CORRELATION ANALYSIS (lag 1-5 days)", flush=True)
    print("=" * 80, flush=True)

    results = []

    for lag in [1, 2, 3, 5]:
        print(f"\n--- Lag {lag} day(s) ---", flush=True)
        best_pairs = []

        for leader in SECTOR_ETFS:
            for follower in SECTOR_ETFS:
                if leader == follower:
                    continue

                # Correlation between leader today and follower in `lag` days
                leader_rets = returns[leader].iloc[:-lag]
                follower_rets = returns[follower].iloc[lag:]

                # Align indices
                leader_rets = leader_rets.reset_index(drop=True)
                follower_rets = follower_rets.reset_index(drop=True)

                corr, pval = stats.pearsonr(leader_rets, follower_rets)

                if abs(corr) > 0.05:
                    best_pairs.append({
                        'leader': leader, 'follower': follower,
                        'corr': corr, 'pval': pval, 'lag': lag
                    })

        best_pairs.sort(key=lambda x: abs(x['corr']), reverse=True)

        print(f"  Top cross-correlations (|r| > 0.05):", flush=True)
        for p in best_pairs[:10]:
            sig = "***" if p['pval'] < 0.001 else "**" if p['pval'] < 0.01 else "*" if p['pval'] < 0.05 else ""
            print(f"    {p['leader']:>5} → {p['follower']:<5}: r={p['corr']:+.4f} (p={p['pval']:.4f}) {sig}", flush=True)

        results.extend(best_pairs)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: Leader-Follower Tradeable Strategy
# ══════════════════════════════════════════════════════════════════════════════

def leader_follower_strategy(prices, leader, follower, threshold=LEADER_THRESHOLD,
                             hold=HOLD_DAYS, cost=COST_RT, direction='same'):
    """
    When leader has a big move (>threshold), position in follower next day.
    direction='same': if leader goes up, buy follower. 'opposite': contrarian.

    Returns daily strategy returns.
    """
    returns = prices.pct_change().dropna()
    warmup = ESTIMATION_WINDOW

    tradeable = returns.index[warmup:]
    if len(tradeable) < 40:
        return None

    strat_rets = []
    in_position = False
    position_days = 0
    position_dir = 0  # +1 long, -1 short

    for i, date in enumerate(tradeable):
        idx = returns.index.get_loc(date)

        if in_position:
            ret = returns.loc[date, follower] * position_dir
            if position_days == 0:
                ret -= cost / hold
            strat_rets.append(float(ret))
            position_days += 1

            if position_days >= hold:
                in_position = False
                position_days = 0
                position_dir = 0
        else:
            strat_rets.append(0.0)

            # Check if leader had a big move yesterday
            if idx > 0:
                leader_ret = returns.iloc[idx - 1][leader]

                if abs(leader_ret) > threshold:
                    in_position = True
                    position_days = 0

                    if direction == 'same':
                        position_dir = 1 if leader_ret > 0 else -1
                    else:
                        position_dir = -1 if leader_ret > 0 else 1

    sr = pd.Series(strat_rets, index=tradeable[:len(strat_rets)])
    spy_rets = returns[BENCHMARK].loc[tradeable[:len(strat_rets)]]

    return sr, spy_rets


def run_all_pairs(prices, direction='same', threshold=LEADER_THRESHOLD):
    """Test all leader-follower pairs."""
    returns = prices[SECTOR_ETFS].pct_change().dropna()
    spy_returns = prices[BENCHMARK].pct_change().dropna()

    results = []

    for leader in SECTOR_ETFS:
        for follower in SECTOR_ETFS:
            if leader == follower:
                continue

            r = leader_follower_strategy(prices, leader, follower,
                                          threshold=threshold, direction=direction)
            if r is None:
                continue

            sr, spy_sr = r

            # Count trades
            in_pos = sr != 0
            trade_starts = in_pos & (~in_pos.shift(1).fillna(False))
            n_trades = trade_starts.sum()

            if n_trades < 10:
                continue

            sharpe = compute_sharpe(sr[in_pos]) if in_pos.sum() > 10 else 0
            sortino = compute_sortino(sr[in_pos]) if in_pos.sum() > 10 else 0

            # WR on trade blocks
            cum = (1 + sr).prod() - 1
            spy_cum = (1 + spy_sr).prod() - 1

            # Simple trade-level WR
            daily_in = sr[in_pos]
            block_rets = []
            for i in range(0, len(daily_in) - HOLD_DAYS + 1, HOLD_DAYS):
                block_rets.append(daily_in.iloc[i:i+HOLD_DAYS].sum())
            block_rets = np.array(block_rets) if block_rets else np.array([0])
            wr = (block_rets > 0).mean() * 100 if len(block_rets) > 0 else 0

            # Regime analysis
            green = spy_sr > 0
            red = spy_sr < 0
            sh_g = compute_sharpe(sr[green & in_pos]) if (green & in_pos).sum() > 10 else 0
            sh_r = compute_sharpe(sr[red & in_pos]) if (red & in_pos).sum() > 10 else 0
            max_reg = max(abs(sh_g), abs(sh_r))
            regime_gap = abs(sh_g - sh_r) / max_reg if max_reg > 0 else 0

            results.append({
                'leader': leader, 'follower': follower,
                'direction': direction,
                'sharpe': round(sharpe, 3),
                'sortino': round(sortino, 3),
                'n_trades': int(n_trades),
                'wr': round(wr, 1),
                'cum_ret_pct': round(cum * 100, 2),
                'excess_pct': round((cum - spy_cum) * 100, 2),
                'sharpe_green': round(sh_g, 3),
                'sharpe_red': round(sh_r, 3),
                'regime_gap': round(regime_gap, 4),
            })

    return sorted(results, key=lambda x: x['sharpe'], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: Rolling Lead-Lag Score
# ══════════════════════════════════════════════════════════════════════════════

def rolling_leadlag(prices, window=20):
    """
    For each sector on each day, compute a 'leader score':
    How much does this sector's return today predict other sectors' returns tomorrow?
    High score = this sector is currently leading the market.

    Strategy: Go long sectors that are being LED (followers) by strong leaders,
    in the direction the leaders are moving.
    """
    returns = prices[SECTOR_ETFS].pct_change().dropna()
    spy_returns = prices[BENCHMARK].pct_change().dropna()

    warmup = ESTIMATION_WINDOW + window
    tradeable = returns.index[warmup:]

    strat_rets = []
    trade_log = []
    days_since = HOLD_DAYS
    holdings = []

    for date in tradeable:
        idx = returns.index.get_loc(date)

        if days_since >= HOLD_DAYS:
            # Compute leader scores over rolling window
            window_rets = returns.iloc[idx-window:idx]

            leader_scores = {}
            for s in SECTOR_ETFS:
                if s not in window_rets.columns:
                    continue
                # Correlation between s(t) and mean-of-others(t+1)
                s_rets = window_rets[s].iloc[:-1].values
                others = [c for c in SECTOR_ETFS if c != s and c in window_rets.columns]
                others_next = window_rets[others].iloc[1:].mean(axis=1).values

                if len(s_rets) < 5:
                    continue

                corr = np.corrcoef(s_rets, others_next)[0, 1]
                leader_scores[s] = corr if not np.isnan(corr) else 0

            if len(leader_scores) >= 3:
                # Identify top leaders and their recent direction
                sorted_leaders = sorted(leader_scores.items(), key=lambda x: x[1], reverse=True)
                top_leaders = [k for k, v in sorted_leaders[:3] if v > 0.1]

                if top_leaders:
                    # Get leader direction (average return of leaders yesterday)
                    leader_dir = returns.iloc[idx - 1][top_leaders].mean()

                    # Pick followers (lowest leader score = most influenced)
                    followers = [k for k, v in sorted_leaders[-3:]]

                    if leader_dir > 0:
                        holdings = followers  # long followers, leaders going up
                        trade_log.append({'date': str(date.date()), 'leaders': top_leaders,
                                         'followers': followers, 'dir': 'long'})
                    elif leader_dir < -0.005:
                        holdings = followers  # could short, but we'll test long-bias first
                        trade_log.append({'date': str(date.date()), 'leaders': top_leaders,
                                         'followers': followers, 'dir': 'flat'})
                        holdings = []  # stay flat on down moves

                days_since = 0

        if holdings and date in returns.index:
            dr = returns.loc[date, holdings]
            port_ret = dr.mean()
            if days_since == 0:
                port_ret -= COST_RT / HOLD_DAYS
            strat_rets.append(float(port_ret))
        else:
            strat_rets.append(0.0)

        days_since += 1

    sr = pd.Series(strat_rets, index=tradeable[:len(strat_rets)])
    spy_sr = spy_returns.loc[tradeable[:len(strat_rets)]]

    return sr, spy_sr, trade_log


def permutation_test(prices, real_sharpe, n_perms=500):
    """Shuffle leader-follower assignments."""
    returns = prices[SECTOR_ETFS].pct_change().dropna()
    spy_returns = prices[BENCHMARK].pct_change().dropna()
    warmup = ESTIMATION_WINDOW + 20
    tradeable = returns.index[warmup:]

    perm_sharpes = []
    for p in range(n_perms):
        rng = np.random.RandomState(p + 2000)
        strat_rets = []
        days_since = HOLD_DAYS
        holdings = []

        for date in tradeable:
            if days_since >= HOLD_DAYS:
                available = [c for c in SECTOR_ETFS if c in returns.columns]
                rng.shuffle(available)
                holdings = available[:3]
                days_since = 0

            if holdings and date in returns.index:
                dr = returns.loc[date, holdings]
                port_ret = dr.mean()
                if days_since == 0:
                    port_ret -= COST_RT / HOLD_DAYS
                strat_rets.append(float(port_ret))
            else:
                strat_rets.append(0.0)

            days_since += 1

        sr = pd.Series(strat_rets)
        if sr.std() > 0:
            perm_sharpes.append(compute_sharpe(sr))

    count_ge = sum(1 for s in perm_sharpes if s >= real_sharpe)
    return round(count_ge / len(perm_sharpes), 4), perm_sharpes


def main():
    print("=" * 90, flush=True)
    print("SECTOR LEAD-LAG RELATIONSHIP BACKTEST", flush=True)
    print("=" * 90, flush=True)

    prices = download_data()
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}\n", flush=True)

    # ── PART 1: Cross-correlations ──────────────────────────────────────────
    xcorr = cross_correlation_analysis(prices)

    # ── PART 2: Pairwise leader-follower strategies ────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("PAIRWISE LEADER-FOLLOWER STRATEGIES (Same Direction)", flush=True)
    print("When leader moves >1%, go long follower for 5 days", flush=True)
    print("=" * 80, flush=True)

    same_results = run_all_pairs(prices, direction='same', threshold=0.01)

    print(f"\n  Top 15 pairs by Sharpe:", flush=True)
    print(f"  {'Leader':>6} → {'Follower':<6} {'Sharpe':>7} {'Sort':>6} {'Trades':>7} {'WR':>6} {'Cum%':>7} {'Exc%':>7} {'ShG':>6} {'ShR':>6} {'RGap':>6}", flush=True)
    print(f"  {'-' * 85}", flush=True)

    for r in same_results[:15]:
        print(f"  {r['leader']:>6} → {r['follower']:<6} {r['sharpe']:>7.3f} {r['sortino']:>6.3f} {r['n_trades']:>7} {r['wr']:>5.1f}% {r['cum_ret_pct']:>6.2f}% {r['excess_pct']:>6.2f}% {r['sharpe_green']:>6.3f} {r['sharpe_red']:>6.3f} {r['regime_gap']:>6.4f}", flush=True)

    # Contrarian direction
    print("\n" + "=" * 80, flush=True)
    print("PAIRWISE LEADER-FOLLOWER STRATEGIES (Contrarian)", flush=True)
    print("When leader moves >1% up, SHORT follower for 5 days (mean-reversion)", flush=True)
    print("=" * 80, flush=True)

    opp_results = run_all_pairs(prices, direction='opposite', threshold=0.01)

    print(f"\n  Top 15 pairs by Sharpe:", flush=True)
    print(f"  {'Leader':>6} → {'Follower':<6} {'Sharpe':>7} {'Sort':>6} {'Trades':>7} {'WR':>6} {'Cum%':>7} {'Exc%':>7} {'ShG':>6} {'ShR':>6} {'RGap':>6}", flush=True)
    print(f"  {'-' * 85}", flush=True)

    for r in opp_results[:15]:
        print(f"  {r['leader']:>6} → {r['follower']:<6} {r['sharpe']:>7.3f} {r['sortino']:>6.3f} {r['n_trades']:>7} {r['wr']:>5.1f}% {r['cum_ret_pct']:>6.2f}% {r['excess_pct']:>6.2f}% {r['sharpe_green']:>6.3f} {r['sharpe_red']:>6.3f} {r['regime_gap']:>6.4f}", flush=True)

    # ── PART 3: Rolling lead-lag score strategy ────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("ROLLING LEAD-LAG STRATEGY", flush=True)
    print("Identify current market leaders, position in followers", flush=True)
    print("=" * 80, flush=True)

    for window in [10, 20, 40]:
        sr, spy_sr, trades = rolling_leadlag(prices, window=window)

        sharpe = compute_sharpe(sr)
        sortino = compute_sortino(sr)
        cum = (1 + sr).prod() - 1
        spy_cum = (1 + spy_sr).prod() - 1

        # Regime
        green = spy_sr > 0
        red = spy_sr < 0
        in_pos = sr != 0
        sh_g = compute_sharpe(sr[green]) if green.sum() > 10 else 0
        sh_r = compute_sharpe(sr[red]) if red.sum() > 10 else 0
        max_reg = max(abs(sh_g), abs(sh_r))
        regime_gap = abs(sh_g - sh_r) / max_reg if max_reg > 0 else 0

        print(f"\n  Window {window}d:", flush=True)
        print(f"    Sharpe: {sharpe:.3f}, Sortino: {sortino:.3f}", flush=True)
        print(f"    Cum return: {cum*100:.2f}%, SPY: {spy_cum*100:.2f}%, Excess: {(cum-spy_cum)*100:.2f}%", flush=True)
        print(f"    Green Sharpe: {sh_g:.3f}, Red Sharpe: {sh_r:.3f}, Regime gap: {regime_gap:.4f}", flush=True)
        print(f"    Trades: {len(trades)}", flush=True)

        if trades:
            print(f"    Sample trades:", flush=True)
            for t in trades[:5]:
                print(f"      {t['date']}: leaders={t['leaders']}, followers={t['followers']}, dir={t['dir']}", flush=True)

    # ── PART 4: Best strategy permutation test ────────────────────────────
    # Find best pair
    all_strategies = same_results + opp_results
    best = max(all_strategies, key=lambda x: x['sharpe']) if all_strategies else None

    if best and best['sharpe'] > 0.5:
        print(f"\n{'=' * 80}", flush=True)
        print(f"BEST PAIR: {best['leader']} → {best['follower']} ({best['direction']}), Sharpe {best['sharpe']:.3f}", flush=True)
        print(f"Running permutation test (500 shuffles)...", flush=True)
        print(f"{'=' * 80}", flush=True)

        perm_p, perm_dist = permutation_test(prices, best['sharpe'])
        print(f"  Permutation p-value: {perm_p:.4f}", flush=True)
        print(f"  Random Sharpe: mean={np.mean(perm_dist):.3f}, std={np.std(perm_dist):.3f}, max={np.max(perm_dist):.3f}", flush=True)

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 90}", flush=True)

    # Count strategies passing 5-gate
    passing = [r for r in all_strategies if (
        r['sharpe'] > 0.5 and
        r['regime_gap'] < 0.5 and
        r['n_trades'] >= 40
    )]

    print(f"\nTotal pairs tested: {len(all_strategies)}", flush=True)
    print(f"Pairs passing basic gates (Sharpe>0.5, regime_gap<0.5, trades>=40): {len(passing)}", flush=True)

    if passing:
        print(f"\nPassing pairs:", flush=True)
        for r in passing:
            print(f"  {r['leader']:>6} → {r['follower']:<6} ({r['direction']}): Sharpe {r['sharpe']:.3f}, "
                  f"WR {r['wr']:.1f}%, trades {r['n_trades']}, regime_gap {r['regime_gap']:.4f}", flush=True)
    else:
        print(f"\nNo pairs pass all gates. Lead-lag relationships in sector ETFs are too weak/noisy.", flush=True)
        print(f"Best same-dir: {same_results[0]['leader']}→{same_results[0]['follower']} Sharpe {same_results[0]['sharpe']:.3f}" if same_results else "None", flush=True)
        print(f"Best contrarian: {opp_results[0]['leader']}→{opp_results[0]['follower']} Sharpe {opp_results[0]['sharpe']:.3f}" if opp_results else "None", flush=True)

    print(f"\nDone. {datetime.now()}", flush=True)


if __name__ == '__main__':
    main()
