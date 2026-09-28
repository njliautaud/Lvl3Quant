#!/usr/bin/env python3
"""
Pairs Trading Backtest within Quality Universe
Variants A-F with 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]

SECTORS = {
    'Tech': ['AAPL', 'MSFT', 'AVGO', 'GOOGL', 'META', 'AMZN'],
    'Finance': ['JPM', 'V', 'MA'],
    'Healthcare': ['UNH', 'LLY', 'ABBV', 'MRK', 'JNJ'],
    'Consumer': ['PG', 'KO', 'PEP', 'HD', 'COST', 'WMT'],
}

CAPITAL = 669.0
PAIR_SIZE = 200.0  # per pair trade ($100 long / $100 short)
SLIPPAGE_BPS = 2   # per side
LOOKBACK = 60
MAX_HOLD = 10
Z_ENTRY = 2.0
Z_EXIT = 0.0
START = '2022-01-01'
END = '2026-07-31'

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading price data...")
data = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=False)
prices = data['Close'].dropna(how='all').ffill().dropna()
returns = prices.pct_change().dropna()
print(f"Data: {prices.shape[0]} days, {prices.shape[1]} tickers, {prices.index[0].date()} to {prices.index[-1].date()}")

# SPY for regime classification
_spy_raw = yf.download('SPY', start=START, end=END, auto_adjust=True, progress=False)['Close'].ffill()
# Flatten to Series if multi-level columns
if isinstance(_spy_raw, pd.DataFrame):
    spy = _spy_raw.squeeze()
else:
    spy = _spy_raw

# Pre-compute regime for every date
_regime_cache = {}
for idx_pos in range(len(spy)):
    dt = spy.index[idx_pos]
    if idx_pos < 60:
        _regime_cache[dt] = 'bull'
    else:
        ret = float(spy.iloc[idx_pos]) / float(spy.iloc[idx_pos - 60]) - 1
        _regime_cache[dt] = 'bull' if ret > 0 else 'bear'

def classify_regime(date):
    """Bull if SPY 60d return > 0, else bear."""
    if date in _regime_cache:
        return _regime_cache[date]
    # Find nearest date
    nearest = spy.index[spy.index.searchsorted(date, side='right') - 1]
    return _regime_cache.get(nearest, 'bull')

# ── Slippage cost ───────────────────────────────────────────────────────────
def pair_slippage_cost(notional_per_leg=100.0):
    """Total slippage for opening+closing a pair (2 legs open + 2 legs close) = 4 sides."""
    return notional_per_leg * 2 * (SLIPPAGE_BPS / 10000) * 4


# ── Variant A: Same-Sector Pairs ────────────────────────────────────────────
def variant_a(prices, returns):
    """For each sector, find highest 60d rolling corr pair. Trade spread z-score."""
    trades = []
    for sector_name, tickers in SECTORS.items():
        sector_tickers = [t for t in tickers if t in prices.columns]
        if len(sector_tickers) < 2:
            continue
        pairs = list(combinations(sector_tickers, 2))

        # Track open positions to avoid overlapping trades on same pair
        open_until = {}  # pair -> exit_idx

        for i in range(LOOKBACK, len(prices)):
            window = prices.iloc[i-LOOKBACK:i]
            # Find best corr pair in this sector
            best_corr = -1
            best_pair = None
            for p in pairs:
                c = window[p[0]].corr(window[p[1]])
                if c > best_corr:
                    best_corr = c
                    best_pair = p

            if best_pair is None or best_corr < 0.5:
                continue

            t1, t2 = best_pair
            pair_key = (t1, t2)

            # Skip if we have an open position on this pair
            if pair_key in open_until and i < open_until[pair_key]:
                continue

            ratio = window[t1] / window[t2]
            mu = ratio.mean()
            sigma = ratio.std()
            if sigma < 1e-8:
                continue

            current_ratio = prices[t1].iloc[i] / prices[t2].iloc[i]
            z = (current_ratio - mu) / sigma

            if abs(z) >= Z_ENTRY:
                long_tk = t2 if z > 0 else t1
                short_tk = t1 if z > 0 else t2

                entry_date = prices.index[i]
                exit_idx = None
                for j in range(i+1, min(i+MAX_HOLD+1, len(prices))):
                    r = prices[t1].iloc[j] / prices[t2].iloc[j]
                    z_j = (r - mu) / sigma
                    if abs(z_j) <= Z_EXIT:
                        exit_idx = j
                        break
                if exit_idx is None:
                    exit_idx = min(i + MAX_HOLD, len(prices) - 1)

                open_until[pair_key] = exit_idx

                long_ret = prices[long_tk].iloc[exit_idx] / prices[long_tk].iloc[i] - 1
                short_ret = -(prices[short_tk].iloc[exit_idx] / prices[short_tk].iloc[i] - 1)
                spread_ret = (long_ret + short_ret) / 2
                slip = pair_slippage_cost() / PAIR_SIZE
                net_ret = spread_ret - slip

                trades.append({
                    'entry': entry_date,
                    'exit': prices.index[exit_idx],
                    'long': long_tk,
                    'short': short_tk,
                    'spread_ret': net_ret,
                    'sector': sector_name,
                })

    return trades


# ── Variant B: Distance Method ──────────────────────────────────────────────
def variant_b(prices, returns):
    """Rank all pairs by SSD. Trade top 5 when distance exceeds 2 stdev."""
    trades = []
    all_pairs = list(combinations(prices.columns, 2))
    open_until = {}

    # Check every 5 days to keep runtime manageable
    for i in range(LOOKBACK, len(prices), 5):
        window = prices.iloc[i-LOOKBACK:i]
        norm = (window - window.mean()) / window.std()

        ssd = []
        for p in all_pairs:
            d = ((norm[p[0]] - norm[p[1]]) ** 2).sum()
            ssd.append((p, d))
        ssd.sort(key=lambda x: x[1])
        top5 = ssd[:5]

        for (t1, t2), _ in top5:
            pair_key = (t1, t2)
            if pair_key in open_until and i < open_until[pair_key]:
                continue

            norm_diff = norm[t1] - norm[t2]
            mu_d = norm_diff.mean()
            sigma_d = norm_diff.std()
            if sigma_d < 1e-8:
                continue

            cur_norm_t1 = (prices[t1].iloc[i] - window[t1].mean()) / window[t1].std()
            cur_norm_t2 = (prices[t2].iloc[i] - window[t2].mean()) / window[t2].std()
            cur_diff = cur_norm_t1 - cur_norm_t2
            z = (cur_diff - mu_d) / sigma_d

            if abs(z) >= Z_ENTRY:
                long_tk = t2 if z > 0 else t1
                short_tk = t1 if z > 0 else t2

                entry_date = prices.index[i]
                exit_idx = None
                for j in range(i+1, min(i+MAX_HOLD+1, len(prices))):
                    w2 = prices.iloc[max(0,j-LOOKBACK):j]
                    if len(w2) < 20:
                        continue
                    s1 = w2[t1].std()
                    s2 = w2[t2].std()
                    if s1 < 1e-8 or s2 < 1e-8:
                        continue
                    n1 = (prices[t1].iloc[j] - w2[t1].mean()) / s1
                    n2 = (prices[t2].iloc[j] - w2[t2].mean()) / s2
                    nd = n1 - n2
                    z_j = (nd - mu_d) / sigma_d
                    if abs(z_j) <= Z_EXIT:
                        exit_idx = j
                        break
                if exit_idx is None:
                    exit_idx = min(i + MAX_HOLD, len(prices) - 1)

                open_until[pair_key] = exit_idx

                long_ret = prices[long_tk].iloc[exit_idx] / prices[long_tk].iloc[i] - 1
                short_ret = -(prices[short_tk].iloc[exit_idx] / prices[short_tk].iloc[i] - 1)
                spread_ret = (long_ret + short_ret) / 2
                slip = pair_slippage_cost() / PAIR_SIZE

                trades.append({
                    'entry': entry_date,
                    'exit': prices.index[exit_idx],
                    'long': long_tk,
                    'short': short_tk,
                    'spread_ret': spread_ret - slip,
                })

    return trades


# ── Variant C: Ratio Mean Reversion ─────────────────────────────────────────
def variant_c(prices, returns):
    """Price ratio z-score. Trade when |z|>2, exit at z=0 or 10 days.
    Uses top 30 correlated pairs and checks every 5 days for tractability."""
    trades = []

    # Pre-select top 30 most correlated pairs to limit combinatorial explosion
    corr_matrix = prices.iloc[LOOKBACK:LOOKBACK*2].corr()
    all_pairs = list(combinations(prices.columns, 2))
    pair_corrs = [(p, abs(corr_matrix.loc[p[0], p[1]])) for p in all_pairs]
    pair_corrs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = [p[0] for p in pair_corrs[:30]]

    open_until = {}

    for i in range(LOOKBACK, len(prices), 3):
        for t1, t2 in top_pairs:
            pair_key = (t1, t2)
            if pair_key in open_until and i < open_until[pair_key]:
                continue

            window = prices.iloc[i-LOOKBACK:i]
            ratio = window[t1] / window[t2]
            mu = ratio.mean()
            sigma = ratio.std()
            if sigma < 1e-8:
                continue

            cur_ratio = prices[t1].iloc[i] / prices[t2].iloc[i]
            z = (cur_ratio - mu) / sigma

            if abs(z) < Z_ENTRY:
                continue

            if z < -Z_ENTRY:
                long_tk, short_tk = t1, t2
                target_direction = 1  # expect z to rise toward 0
            else:
                long_tk, short_tk = t2, t1
                target_direction = -1

            entry_date = prices.index[i]
            exit_idx = None
            for j in range(i+1, min(i+MAX_HOLD+1, len(prices))):
                r = prices[t1].iloc[j] / prices[t2].iloc[j]
                z_j = (r - mu) / sigma
                if target_direction == 1 and z_j >= Z_EXIT:
                    exit_idx = j
                    break
                elif target_direction == -1 and z_j <= Z_EXIT:
                    exit_idx = j
                    break
            if exit_idx is None:
                exit_idx = min(i + MAX_HOLD, len(prices) - 1)

            open_until[pair_key] = exit_idx

            long_ret = prices[long_tk].iloc[exit_idx] / prices[long_tk].iloc[i] - 1
            short_ret = -(prices[short_tk].iloc[exit_idx] / prices[short_tk].iloc[i] - 1)
            spread_ret = (long_ret + short_ret) / 2
            slip = pair_slippage_cost() / PAIR_SIZE

            trades.append({
                'entry': entry_date,
                'exit': prices.index[exit_idx],
                'long': long_tk,
                'short': short_tk,
                'spread_ret': spread_ret - slip,
            })

    return trades


# ── Variant D: Sector-Neutral ───────────────────────────────────────────────
def variant_d(prices, returns):
    """Same as A but require one long + one short in same sector simultaneously.
    Since A already pairs within sector, D is structurally identical."""
    return variant_a(prices, returns)


# ── Variant E: Correlation Breakdown ────────────────────────────────────────
def variant_e(prices, returns):
    """When 20d corr drops below 0.3 for normally-correlated pair (60d corr>0.7),
    buy both equally expecting correlation to re-establish. Hold 10 days."""
    trades = []
    all_pairs = list(combinations(prices.columns, 2))
    open_until = {}

    for i in range(LOOKBACK, len(prices)):
        window_60 = returns.iloc[max(0,i-LOOKBACK):i]
        window_20 = returns.iloc[max(0,i-20):i]
        if len(window_20) < 15 or len(window_60) < 40:
            continue

        for t1, t2 in all_pairs:
            pair_key = (t1, t2)
            if pair_key in open_until and i < open_until[pair_key]:
                continue

            corr_60 = window_60[t1].corr(window_60[t2])
            corr_20 = window_20[t1].corr(window_20[t2])

            if corr_60 > 0.7 and corr_20 < 0.3:
                entry_date = prices.index[i]
                exit_idx = min(i + MAX_HOLD, len(prices) - 1)
                open_until[pair_key] = exit_idx

                ret_t1 = prices[t1].iloc[exit_idx] / prices[t1].iloc[i] - 1
                ret_t2 = prices[t2].iloc[exit_idx] / prices[t2].iloc[i] - 1
                avg_ret = (ret_t1 + ret_t2) / 2

                slip = pair_slippage_cost() / PAIR_SIZE

                trades.append({
                    'entry': entry_date,
                    'exit': prices.index[exit_idx],
                    'long': t1,
                    'short': t2,
                    'spread_ret': avg_ret - slip,
                })

    return trades


# ── Variant F: Momentum Spread ──────────────────────────────────────────────
def variant_f(prices, returns):
    """Within each sector, long highest 20d return, short lowest. Monthly rebalance."""
    trades = []
    monthly_idx = []
    prev_month = None
    for idx_pos, dt in enumerate(prices.index):
        ym = (dt.year, dt.month)
        if ym != prev_month:
            monthly_idx.append(idx_pos)
            prev_month = ym

    for mi in range(len(monthly_idx)):
        entry_idx = monthly_idx[mi]
        if entry_idx < 20 or entry_idx >= len(prices) - 1:
            continue
        if mi + 1 < len(monthly_idx):
            exit_idx = min(monthly_idx[mi+1], len(prices) - 1)
        else:
            exit_idx = len(prices) - 1

        for sector_name, tickers in SECTORS.items():
            sector_tickers = [t for t in tickers if t in prices.columns]
            if len(sector_tickers) < 2:
                continue

            mom = {}
            for t in sector_tickers:
                mom[t] = prices[t].iloc[entry_idx] / prices[t].iloc[entry_idx-20] - 1

            sorted_mom = sorted(mom.items(), key=lambda x: x[1])
            short_tk = sorted_mom[0][0]
            long_tk = sorted_mom[-1][0]

            long_ret = prices[long_tk].iloc[exit_idx] / prices[long_tk].iloc[entry_idx] - 1
            short_ret = -(prices[short_tk].iloc[exit_idx] / prices[short_tk].iloc[entry_idx] - 1)
            spread_ret = (long_ret + short_ret) / 2
            slip = pair_slippage_cost() / PAIR_SIZE

            trades.append({
                'entry': prices.index[entry_idx],
                'exit': prices.index[exit_idx],
                'long': long_tk,
                'short': short_tk,
                'spread_ret': spread_ret - slip,
                'sector': sector_name,
            })

    return trades


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(trades, capital=CAPITAL):
    """Compute strategy metrics from trade list."""
    if not trades:
        return {
            'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
            'max_drawdown': 0, 'total_return': 0, 'num_trades': 0,
            'bull_sharpe': 0, 'bear_sharpe': 0, 'regime_gap': 0,
            'permutation_p': 1.0, 'five_gate': 'FAIL',
        }

    df = pd.DataFrame(trades)
    df['entry'] = pd.to_datetime(df['entry'])
    df['exit'] = pd.to_datetime(df['exit'])
    rets = df['spread_ret'].values
    n = len(rets)

    win_rate = np.mean(rets > 0)
    gross_wins = rets[rets > 0].sum() if (rets > 0).any() else 0
    gross_losses = abs(rets[rets < 0].sum()) if (rets < 0).any() else 1e-8
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0

    trading_days = (df['exit'].max() - df['entry'].min()).days
    if trading_days <= 0:
        trading_days = 1
    years = trading_days / 365.25
    trades_per_year = n / years if years > 0 else n

    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1e-8
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-8

    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Equity curve & drawdown
    equity = capital
    peak = capital
    max_dd = 0
    for r in rets:
        pnl = PAIR_SIZE * r
        equity += pnl
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)
    total_return = (equity - capital) / capital

    # Regime analysis
    df['regime'] = df['entry'].apply(classify_regime)
    bull_rets = df[df['regime'] == 'bull']['spread_ret'].values
    bear_rets = df[df['regime'] == 'bear']['spread_ret'].values

    def _sharpe(r, tpy):
        if len(r) < 2:
            return 0
        s = np.std(r, ddof=1)
        return (np.mean(r) / s) * np.sqrt(tpy) if s > 0 else 0

    bull_sharpe = _sharpe(bull_rets, trades_per_year) if len(bull_rets) > 1 else 0
    bear_sharpe = _sharpe(bear_rets, trades_per_year) if len(bear_rets) > 1 else 0

    denom = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom if denom > 0 else 0

    # Permutation test: shuffle trade returns 1000x, see how often shuffled Sharpe >= observed
    observed_sharpe = sharpe
    np.random.seed(42)
    perm_count = 0
    for _ in range(1000):
        perm_rets = np.random.permutation(rets)
        pm = np.mean(perm_rets)
        ps = np.std(perm_rets, ddof=1) if n > 1 else 1e-8
        psharpe = (pm / ps) * np.sqrt(trades_per_year) if ps > 0 else 0
        if psharpe >= observed_sharpe:
            perm_count += 1
    permutation_p = perm_count / 1000

    # 5-Gate validation
    g1 = sharpe > 0.5
    g2 = permutation_p < 0.05
    g3 = regime_gap < 0.5
    g4 = max_dd > -0.50
    g5 = n >= 20
    five_gate = 'PASS' if all([g1, g2, g3, g4, g5]) else 'FAIL'

    return {
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 4),
        'max_drawdown': round(max_dd, 4),
        'total_return': round(total_return, 4),
        'num_trades': n,
        'bull_sharpe': round(bull_sharpe, 4),
        'bear_sharpe': round(bear_sharpe, 4),
        'regime_gap': round(regime_gap, 4),
        'permutation_p': round(permutation_p, 4),
        'five_gate': five_gate,
        'gate_details': {
            'g1_sharpe_gt_0.5': g1,
            'g2_perm_p_lt_0.05': g2,
            'g3_regime_gap_lt_0.5': g3,
            'g4_max_dd_gt_neg50pct': g4,
            'g5_min_20_trades': g5,
        },
    }


# ── Run All Variants ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*70)
    print("PAIRS TRADING BACKTEST — QUALITY UNIVERSE")
    print("="*70)

    variants = {
        'A_same_sector': variant_a,
        'B_distance': variant_b,
        'C_ratio_mr': variant_c,
        'D_sector_neutral': variant_d,
        'E_corr_breakdown': variant_e,
        'F_momentum_spread': variant_f,
    }

    results = {}
    for name, func in variants.items():
        print(f"\nRunning Variant {name}...")
        try:
            trades = func(prices, returns)
            metrics = compute_metrics(trades)
            results[name] = metrics

            print(f"  Trades: {metrics['num_trades']}")
            print(f"  Sharpe: {metrics['sharpe']:.4f}  Sortino: {metrics['sortino']:.4f}")
            print(f"  WR: {metrics['win_rate']:.2%}  PF: {metrics['profit_factor']:.2f}")
            print(f"  MaxDD: {metrics['max_drawdown']:.2%}  Return: {metrics['total_return']:.2%}")
            print(f"  Bull Sharpe: {metrics['bull_sharpe']:.4f}  Bear Sharpe: {metrics['bear_sharpe']:.4f}")
            print(f"  Regime Gap: {metrics['regime_gap']:.4f}  Perm p: {metrics['permutation_p']:.4f}")
            print(f"  5-Gate: {metrics['five_gate']}")
            for g, v in metrics['gate_details'].items():
                status = "PASS" if v else "FAIL"
                print(f"    {g}: {status}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e), 'five_gate': 'FAIL'}

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    valid = {k: v for k, v in results.items() if 'sharpe' in v}
    if valid:
        best = max(valid, key=lambda k: valid[k]['sharpe'])
        print(f"\nBest variant by Sharpe: {best} (Sharpe={valid[best]['sharpe']:.4f})")
        passing = [k for k, v in valid.items() if v.get('five_gate') == 'PASS']
        if passing:
            print(f"Variants passing 5-gate: {', '.join(passing)}")
        else:
            print("No variants pass all 5 gates.")

    # Save
    output = {
        'strategy': 'pairs_trading_quality_universe',
        'period': f'{START} to {END}',
        'capital': CAPITAL,
        'pair_size': PAIR_SIZE,
        'slippage_bps_per_side': SLIPPAGE_BPS,
        'universe': UNIVERSE,
        'run_date': datetime.now().isoformat(),
        'variants': results,
    }

    out_path = '/home/jupiter/Lvl3Quant/data/pairs_trading_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
