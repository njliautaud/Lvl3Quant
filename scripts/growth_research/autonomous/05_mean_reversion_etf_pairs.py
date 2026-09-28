#!/usr/bin/env python3
"""
ETF Mean Reversion Pairs
=========================
Trade mean-reversion on highly correlated ETF pairs:
- SPY/QQQ ratio
- GLD/GDX ratio (gold vs gold miners)
- XLF/XLU ratio (financials vs utilities = risk barometer)

When pairs deviate from their normal ratio, trade the reversion.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *

def main():
    print("=" * 70)
    print("ETF MEAN REVERSION PAIRS")
    print("=" * 70)

    tickers = ['SPY', 'QQQ', 'GLD', 'GDX', 'XLF', 'XLU', 'SHY', 'TLT']
    prices = download_etfs(tickers, start='2010-01-01')
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows")

    rets = {t: prices[t].pct_change() for t in tickers}
    start = 252

    def trade_pair(prices, asset_a, asset_b, lookback=63, entry_z=1.5, exit_z=0.5):
        """Trade mean reversion on ratio A/B"""
        ratio = prices[asset_a] / prices[asset_b]
        ratio_ma = ratio.rolling(lookback).mean()
        ratio_std = ratio.rolling(lookback).std()
        z_score = (ratio - ratio_ma) / ratio_std.clip(lower=0.001)

        equity = pd.Series(INITIAL_CAPITAL, index=prices.index)
        position = 0  # 1 = long A/short B, -1 = short A/long B, 0 = flat

        for i in range(start, len(prices)):
            z = z_score.iloc[i]
            ret_a = rets[asset_a].iloc[i]
            ret_b = rets[asset_b].iloc[i]

            if np.isnan(z):
                equity.iloc[i] = equity.iloc[i-1]
                continue

            # Entry signals
            if position == 0:
                if z > entry_z:  # Ratio too high, short A / long B
                    position = -1
                elif z < -entry_z:  # Ratio too low, long A / short B
                    position = 1

            # Exit signals
            if position == 1 and z > -exit_z:
                position = 0
            elif position == -1 and z < exit_z:
                position = 0

            # P&L
            if position == 1:
                pair_ret = ret_a - ret_b  # Long A, short B
            elif position == -1:
                pair_ret = ret_b - ret_a  # Long B, short A
            else:
                pair_ret = rets['SHY'].iloc[i]  # Flat = cash

            # Scale position size (50% to keep it conservative)
            port_ret = 0.5 * pair_ret + 0.5 * rets['SHY'].iloc[i]
            equity.iloc[i] = equity.iloc[i-1] * (1 + port_ret) if np.isfinite(port_ret) else equity.iloc[i-1]

        return equity.iloc[start:]

    configs = {}

    # Pair 1: SPY/QQQ
    configs['SPY/QQQ MR'] = trade_pair(prices, 'SPY', 'QQQ', lookback=63)

    # Pair 2: GLD/GDX
    configs['GLD/GDX MR'] = trade_pair(prices, 'GLD', 'GDX', lookback=63)

    # Pair 3: XLF/XLU
    configs['XLF/XLU MR'] = trade_pair(prices, 'XLF', 'XLU', lookback=63)

    # Combined: equal-weight all 3 pairs
    combined_keys = ['SPY/QQQ MR', 'GLD/GDX MR', 'XLF/XLU MR']
    min_len = min(len(configs[k]) for k in combined_keys)
    combined_rets = pd.DataFrame()
    for k in combined_keys:
        combined_rets[k] = configs[k].iloc[-min_len:].pct_change()

    combined_eq = (1 + combined_rets.mean(axis=1)).cumprod() * INITIAL_CAPITAL
    combined_eq.iloc[0] = INITIAL_CAPITAL
    configs['Combined 3-Pair'] = combined_eq

    # Longer lookback variants
    configs['SPY/QQQ MR 126d'] = trade_pair(prices, 'SPY', 'QQQ', lookback=126)
    configs['GLD/GDX MR 126d'] = trade_pair(prices, 'GLD', 'GDX', lookback=126)

    # SPY benchmark
    spy_eq = prices['SPY'].iloc[start:] / prices['SPY'].iloc[start] * INITIAL_CAPITAL
    configs['SPY B&H'] = spy_eq

    # Results
    print(f"\n{'Strategy':<25s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s}")
    print(f"{'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

    best_name, best_sharpe = None, -999
    for name, eq in configs.items():
        m = compute_metrics(eq, name)
        print(f"{m['name']:<25s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%}")
        if name != 'SPY B&H' and m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_name = name

    print(f"\nBest: {best_name}")
    metrics = compute_metrics(configs[best_name], best_name)
    adv = full_adversarial(configs[best_name], prices['SPY'])

    print(f"Adversarial: {adv['gates_passed']}/3")
    for test in ['permutation', 'subperiod', 'regime']:
        t = adv[test]
        key = 'p_value' if test == 'permutation' else 'cv' if test == 'subperiod' else 'gap'
        print(f"  {test}: {t.get(key, '?')} {'PASS' if t['pass'] else 'FAIL'}")

    emit_result(
        name=f"ETF Mean Reversion ({best_name})",
        description="Pairs mean reversion on correlated ETFs",
        metrics=metrics,
        adversarial=adv
    )

if __name__ == '__main__':
    main()
