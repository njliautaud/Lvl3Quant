#!/usr/bin/env python3
"""
Dispersion Trading v1 — Index vs Constituent Implied Vol
=========================================================
Dispersion trades exploit the gap between index implied vol (VIX/index options)
and weighted-average constituent implied vol. When index vol is expensive relative
to constituent vol (correlation risk premium), sell index vol + buy constituent vol.

Proxy approach (no real options data needed):
- Use realized vol as proxy for implied vol
- SPY realized vol vs. weighted-average sector ETF realized vol
- When SPY vol > sector-vol-portfolio → "sell dispersion" (short vol trades)
- When SPY vol < sector-vol-portfolio → "buy dispersion" (long vol trades)

We also use VIX as additional signal for implied vol richness.

Universe: SPY (index) vs sector ETFs (constituents)
Capital: $645, OOT: 2022-01-01 to 2026-07-25
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json, os, warnings, traceback
from datetime import datetime
warnings.filterwarnings('ignore')

SECTORS = ['XLK','XLF','XLE','XLV','XLC','XLI','XLY','XLP','XLU','XLRE','XLB']
INDEX = 'SPY'
ALL_TICKERS = [INDEX] + SECTORS
VIX_TICKER = '^VIX'

OOT_START = '2022-01-01'
OOT_END = '2026-07-25'
CAPITAL = 645.0
TRAIN_WINDOW = 252

# Approximate sector weights in SPY (as of ~2026)
SECTOR_WEIGHTS = {
    'XLK': 0.30, 'XLF': 0.13, 'XLV': 0.12, 'XLI': 0.09,
    'XLY': 0.10, 'XLC': 0.09, 'XLP': 0.06, 'XLE': 0.04,
    'XLU': 0.03, 'XLRE': 0.02, 'XLB': 0.02
}

SLIPPAGE_BPS = 10
N_PERMS = 100
RESULTS_DIR = '/home/nick/Lvl3Quant/research/findings'
os.makedirs(RESULTS_DIR, exist_ok=True)


def download_data():
    print(f"Downloading {len(ALL_TICKERS)+1} tickers...")
    trade_data = yf.download(ALL_TICKERS, start='2018-01-01', end=OOT_END,
                              auto_adjust=True, progress=False)
    trade_prices = trade_data['Close'] if isinstance(trade_data.columns, pd.MultiIndex) else trade_data

    vix_data = yf.download([VIX_TICKER], start='2018-01-01', end=OOT_END,
                            auto_adjust=True, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix = vix_data['Close'].iloc[:, 0]
    else:
        vix = vix_data['Close'] if 'Close' in str(vix_data.columns) else vix_data.iloc[:, 0]
    vix = pd.Series(vix.values.flatten(), index=vix.index, name='VIX')

    trade_prices = trade_prices.dropna(how='all')
    print(f"Got {len(trade_prices)} days, {trade_prices.shape[1]} tickers, VIX {len(vix)} days")
    return trade_prices, vix


def compute_dispersion_signal(prices, vix, lookback=21):
    """
    Compute dispersion signal: difference between index realized vol
    and weighted-average sector realized vol.

    Positive dispersion = index vol > constituent vol → correlation premium → sell dispersion
    Negative dispersion = index vol < constituent vol → sell constituents, buy index
    """
    spy = prices[INDEX]
    spy_ret = spy.pct_change()
    spy_vol = spy_ret.rolling(lookback).std() * np.sqrt(252)

    # Weighted average sector vol
    sector_vols = pd.DataFrame()
    for s in SECTORS:
        if s in prices.columns:
            ret = prices[s].pct_change()
            sector_vols[s] = ret.rolling(lookback).std() * np.sqrt(252)

    # Weighted average
    weights = pd.Series({s: SECTOR_WEIGHTS.get(s, 0) for s in sector_vols.columns})
    weights = weights / weights.sum()
    weighted_vol = (sector_vols * weights).sum(axis=1)

    # Implied correlation proxy
    # If all sectors had correlation=1, index vol = weighted vol
    # Actual index vol < weighted vol due to diversification
    # implied_corr ≈ (index_vol / weighted_vol)^2
    implied_corr = (spy_vol / weighted_vol).clip(0, 2) ** 2

    # Dispersion = weighted_vol - index_vol (positive = diversification benefit)
    dispersion = weighted_vol - spy_vol

    # VIX premium over realized (richness of implied)
    vix_premium = vix / 100 - spy_vol  # VIX in % vs annualized realized

    features = pd.DataFrame(index=prices.index)
    features['spy_vol'] = spy_vol
    features['weighted_sector_vol'] = weighted_vol
    features['dispersion'] = dispersion
    features['implied_corr'] = implied_corr
    features['vix'] = vix.reindex(prices.index)
    features['vix_premium'] = vix_premium

    # Z-scores
    features['dispersion_z'] = (dispersion - dispersion.rolling(63).mean()) / dispersion.rolling(63).std()
    features['corr_z'] = (implied_corr - implied_corr.rolling(63).mean()) / implied_corr.rolling(63).std()
    features['vix_z'] = (vix.reindex(prices.index) - vix.reindex(prices.index).rolling(63).mean()) / vix.reindex(prices.index).rolling(63).std()

    # Cross-sector correlation (rolling)
    sector_rets = prices[SECTORS].pct_change()
    corr_matrix = sector_rets.rolling(lookback).corr()
    # Average pairwise correlation
    avg_corr_list = []
    for date in prices.index:
        try:
            cm = corr_matrix.loc[date]
            if isinstance(cm, pd.DataFrame) and cm.shape[0] > 1:
                mask = np.triu(np.ones(cm.shape), k=1).astype(bool)
                avg_c = cm.values[mask].mean()
                avg_corr_list.append(avg_c)
            else:
                avg_corr_list.append(np.nan)
        except:
            avg_corr_list.append(np.nan)
    features['avg_sector_corr'] = avg_corr_list

    return features


def run_dispersion_backtest(prices, vix, variant='A', seed=42):
    """
    Variants:
    A: Simple dispersion signal — go long sectors (short SPY) when dispersion widens
    B: Correlation regime — buy when implied corr is extremely high (panic, mean revert)
    C: VIX premium — sell index vol proxy when VIX premium is wide
    D: Sector selection — in low-corr regime, pick lowest-corr sector pair, long/short
    E: Mean-reversion on dispersion z-score
    F: Combined (dispersion + corr + VIX premium)
    """
    features = compute_dispersion_signal(prices, vix)

    oot_dates = features.index[features.index >= OOT_START]
    if len(oot_dates) < 40: return None

    equity = CAPITAL
    equity_curve = []
    trades = []
    position = None

    for i, date in enumerate(oot_dates):
        date_idx = features.index.get_loc(date)
        if date_idx < TRAIN_WINDOW:
            equity_curve.append(equity)
            continue

        today = features.iloc[date_idx]
        today_prices = prices.iloc[date_idx]

        disp_z = today.get('dispersion_z', 0)
        corr_z = today.get('corr_z', 0)
        vix_z = today.get('vix_z', 0)
        avg_corr = today.get('avg_sector_corr', 0.5)
        disp = today.get('dispersion', 0)

        # ── Close existing position ──
        if position is not None:
            days_held = (date - position['entry_date']).days

            # Calculate current P&L
            pnl = 0
            for leg in position['legs']:
                ticker = leg['ticker']
                if ticker in today_prices.index and not pd.isna(today_prices[ticker]):
                    current = today_prices[ticker]
                    leg_pnl = leg['shares'] * (current - leg['entry_price'])
                    pnl += leg_pnl

            pnl_pct = pnl / max(position['capital_used'], 1)

            should_close = False
            close_reason = ''

            # Exit rules depend on variant
            max_hold = 10 if variant in ('A', 'E', 'F') else 5
            tp = 0.03  # 3% take profit
            sl = -0.02  # 2% stop loss

            if variant == 'B':
                # Close when correlation normalizes
                if corr_z < 0.5 or days_held >= 5:
                    should_close = True
                    close_reason = 'corr_normalize' if corr_z < 0.5 else 'time'
            elif variant in ('A', 'E', 'F'):
                # Close on reversion or time
                if disp_z < 0.5 or days_held >= max_hold:
                    should_close = True
                    close_reason = 'signal_revert' if disp_z < 0.5 else 'time'
            else:
                if days_held >= max_hold:
                    should_close = True
                    close_reason = 'time'

            # Universal stops
            if pnl_pct >= tp:
                should_close = True
                close_reason = 'tp'
            elif pnl_pct <= sl:
                should_close = True
                close_reason = 'sl'

            if should_close:
                # Add slippage
                total_notional = sum(abs(l['shares'] * today_prices.get(l['ticker'], l['entry_price']))
                                    for l in position['legs'] if l['ticker'] in today_prices.index)
                slippage = total_notional * SLIPPAGE_BPS / 10000 * 2  # RT
                pnl -= slippage
                equity += pnl

                trades.append({
                    'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'type': position.get('trade_type', 'dispersion'),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct * 100, 2),
                    'days_held': days_held,
                    'close_reason': close_reason,
                    'entry_disp_z': position.get('entry_disp_z', 0)
                })
                position = None

        # ── Open new position ──
        if position is None and equity > 50:
            signal = None
            legs = []
            trade_type = 'dispersion'

            if pd.isna(disp_z) or pd.isna(corr_z):
                equity_curve.append(equity)
                continue

            if variant == 'A':
                # Long dispersion when it widens (high z-score)
                # = Long sectors, short index
                if disp_z > 1.5:
                    signal = 'long_dispersion'
                    alloc = min(equity * 0.40, 250)
                    # Short SPY
                    spy_price = today_prices.get(INDEX, np.nan)
                    if not pd.isna(spy_price) and spy_price > 0:
                        spy_shares = -alloc / spy_price
                        legs.append({'ticker': INDEX, 'shares': spy_shares, 'entry_price': spy_price})
                        # Long top 3 sectors by weight
                        sector_alloc = alloc / 3
                        for s in ['XLK', 'XLF', 'XLV']:
                            if s in today_prices.index and not pd.isna(today_prices[s]):
                                legs.append({'ticker': s, 'shares': sector_alloc / today_prices[s],
                                           'entry_price': today_prices[s]})

            elif variant == 'B':
                # Fade extreme correlation (>1.5z = panic)
                if corr_z > 1.5:
                    signal = 'fade_corr'
                    alloc = min(equity * 0.40, 250)
                    # Long lowest-corr sectors (most diversification benefit when corr mean-reverts)
                    sector_rets = prices[SECTORS].iloc[max(0,date_idx-21):date_idx].pct_change()
                    corr_to_spy = {}
                    for s in SECTORS:
                        if s in sector_rets.columns and INDEX in prices.columns:
                            spy_ret = prices[INDEX].iloc[max(0,date_idx-21):date_idx].pct_change()
                            common = sector_rets[s].dropna().index.intersection(spy_ret.dropna().index)
                            if len(common) > 10:
                                c = sector_rets[s].loc[common].corr(spy_ret.loc[common])
                                corr_to_spy[s] = c
                    if corr_to_spy:
                        # Long lowest-corr sectors
                        sorted_sectors = sorted(corr_to_spy, key=corr_to_spy.get)[:3]
                        per_sector = alloc / len(sorted_sectors)
                        for s in sorted_sectors:
                            if s in today_prices.index and not pd.isna(today_prices[s]):
                                legs.append({'ticker': s, 'shares': per_sector / today_prices[s],
                                           'entry_price': today_prices[s]})
                        # Short SPY as hedge
                        spy_price = today_prices.get(INDEX, np.nan)
                        if not pd.isna(spy_price) and spy_price > 0:
                            legs.append({'ticker': INDEX, 'shares': -alloc / spy_price,
                                       'entry_price': spy_price})

            elif variant == 'C':
                # VIX premium trade — when VIX >> realized, sell vol proxy
                # Proxy: short straddle via short both directions
                # We'll use: short high-vol sector, long low-vol sector
                if vix_z > 1.5:
                    signal = 'sell_vix_premium'
                    alloc = min(equity * 0.40, 250)
                    # Short highest vol, long lowest vol
                    sector_vols = {}
                    for s in SECTORS:
                        if s in prices.columns:
                            vol = prices[s].iloc[max(0,date_idx-21):date_idx].pct_change().std() * np.sqrt(252)
                            sector_vols[s] = vol
                    if len(sector_vols) >= 4:
                        sorted_secs = sorted(sector_vols, key=sector_vols.get)
                        # Long lowest 2 vol
                        for s in sorted_secs[:2]:
                            if s in today_prices.index and not pd.isna(today_prices[s]):
                                legs.append({'ticker': s, 'shares': alloc / 4 / today_prices[s],
                                           'entry_price': today_prices[s]})
                        # Short highest 2 vol
                        for s in sorted_secs[-2:]:
                            if s in today_prices.index and not pd.isna(today_prices[s]):
                                legs.append({'ticker': s, 'shares': -alloc / 4 / today_prices[s],
                                           'entry_price': today_prices[s]})

            elif variant == 'D':
                # In low-corr regimes, pick most divergent pair
                if avg_corr < 0.4:
                    signal = 'low_corr_pair'
                    alloc = min(equity * 0.40, 250)
                    # Find most divergent pair in last 5 days
                    rets_5d = {}
                    for s in SECTORS:
                        if s in prices.columns and date_idx >= 5:
                            r = (prices[s].iloc[date_idx] / prices[s].iloc[date_idx-5]) - 1
                            rets_5d[s] = r
                    if len(rets_5d) >= 4:
                        sorted_by_ret = sorted(rets_5d, key=rets_5d.get)
                        worst = sorted_by_ret[0]
                        best = sorted_by_ret[-1]
                        # Mean revert: long worst, short best
                        for s, d in [(worst, 1), (best, -1)]:
                            if s in today_prices.index and not pd.isna(today_prices[s]):
                                legs.append({'ticker': s, 'shares': d * alloc / 2 / today_prices[s],
                                           'entry_price': today_prices[s]})

            elif variant == 'E':
                # Pure dispersion z-score mean reversion
                if disp_z > 2.0:  # Dispersion unusually wide → will compress
                    signal = 'disp_compress'
                    alloc = min(equity * 0.40, 250)
                    spy_price = today_prices.get(INDEX, np.nan)
                    if not pd.isna(spy_price) and spy_price > 0:
                        # Long SPY (bet index outperforms = compression)
                        legs.append({'ticker': INDEX, 'shares': alloc / spy_price,
                                   'entry_price': spy_price})
                elif disp_z < -2.0:  # Dispersion unusually narrow → will widen
                    signal = 'disp_widen'
                    alloc = min(equity * 0.40, 250)
                    spy_price = today_prices.get(INDEX, np.nan)
                    if not pd.isna(spy_price) and spy_price > 0:
                        legs.append({'ticker': INDEX, 'shares': -alloc / spy_price,
                                   'entry_price': spy_price})
                        # Long sectors
                        for s in ['XLK', 'XLF', 'XLV']:
                            if s in today_prices.index and not pd.isna(today_prices[s]):
                                legs.append({'ticker': s, 'shares': alloc / 3 / today_prices[s],
                                           'entry_price': today_prices[s]})

            elif variant == 'F':
                # Combined: all signals vote
                votes = 0
                if disp_z > 1.0: votes += 1
                if corr_z > 1.0: votes += 1
                if vix_z > 1.0: votes += 1

                if votes >= 2:
                    signal = 'combined_long_disp'
                    alloc = min(equity * 0.40, 250)
                    spy_price = today_prices.get(INDEX, np.nan)
                    if not pd.isna(spy_price) and spy_price > 0:
                        legs.append({'ticker': INDEX, 'shares': -alloc / spy_price,
                                   'entry_price': spy_price})
                        for s in ['XLK', 'XLF', 'XLV']:
                            if s in today_prices.index and not pd.isna(today_prices[s]):
                                legs.append({'ticker': s, 'shares': alloc / 3 / today_prices[s],
                                           'entry_price': today_prices[s]})

            if signal and legs:
                cap_used = sum(abs(l['shares'] * l['entry_price']) for l in legs)
                position = {
                    'legs': legs, 'entry_date': date,
                    'capital_used': cap_used,
                    'trade_type': trade_type,
                    'entry_disp_z': round(disp_z, 2) if not pd.isna(disp_z) else 0
                }

        equity_curve.append(equity)

    if not equity_curve: return None
    ea = np.array(equity_curve)
    rets = np.diff(ea) / ea[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) < 40 or np.std(rets) == 0: return None

    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252)
    neg = rets[rets < 0]
    sortino = np.mean(rets) / (np.std(neg) * np.sqrt(252)) if len(neg) > 0 and np.std(neg) > 0 else 0
    cagr = (ea[-1] / CAPITAL) ** (252 / len(rets)) - 1
    peak = np.maximum.accumulate(ea)
    mdd = np.min((ea - peak) / peak) * 100

    if trades:
        pnls = [t['pnl'] for t in trades]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        pf = sum(wins) / sum(losses) if losses else 999
    else:
        wr, pf = 0, 0

    return {'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
            'cagr': round(cagr * 100, 2), 'mdd': round(mdd, 2),
            'wr': round(wr, 2), 'pf': round(pf, 3),
            'n_trades': len(trades), 'n_days': len(rets),
            'final_equity': round(ea[-1], 2),
            'equity_curve': ea.tolist(),
            'oot_dates': [d.strftime('%Y-%m-%d') for d in oot_dates[:len(ea)]],
            'trades': trades}


def regime_stratify(metrics, prices):
    if INDEX not in prices.columns: return {'regime_balance_ok': False}
    spy = prices[INDEX].dropna()
    oot_spy = spy[spy.index >= OOT_START]
    spy_rets = oot_spy.pct_change().dropna()
    ea = np.array(metrics['equity_curve'])
    oot_dates = pd.to_datetime(metrics['oot_dates'])
    strat_rets = pd.Series(np.diff(ea) / ea[:-1], index=oot_dates[1:len(ea)])

    green = spy_rets[spy_rets > 0.001].index
    red = spy_rets[spy_rets < -0.001].index
    flat = spy_rets[(spy_rets >= -0.001) & (spy_rets <= 0.001)].index

    def ss(rs, ds):
        s = rs[rs.index.isin(ds)]
        if len(s) < 10 or np.std(s) == 0: return 0.0, len(s)
        return round(np.mean(s) / np.std(s) * np.sqrt(252), 3), len(s)

    sg, ng = ss(strat_rets, green)
    sr, nr = ss(strat_rets, red)
    sf, nf = ss(strat_rets, flat)
    mx = max(abs(sg), abs(sr), 0.001)
    gap = abs(sg - sr) / mx
    return {'sharpe_green': sg, 'n_green': ng, 'sharpe_red': sr, 'n_red': nr,
            'sharpe_flat': sf, 'n_flat': nf, 'regime_gap': round(gap, 3),
            'regime_balance_ok': gap <= 0.50}


def permutation_test(prices, vix, variant, n_perms=N_PERMS):
    actual = run_dispersion_backtest(prices, vix, variant=variant)
    if not actual: return None, None
    actual_sharpe = actual['sharpe']
    perm_sharpes = []
    for i in range(n_perms):
        # Circular shift VIX to break signal timing
        rng = np.random.RandomState(i)
        shift = rng.randint(60, len(vix) - 60)
        vix_s = pd.Series(np.roll(vix.values, shift), index=vix.index, name='VIX')
        pr = run_dispersion_backtest(prices, vix_s, variant=variant, seed=i)
        if pr: perm_sharpes.append(pr['sharpe'])
    if not perm_sharpes: return actual, {'p_value': 1.0, 'significant': False}
    pv = np.mean([s >= actual_sharpe for s in perm_sharpes])
    return actual, {'actual_sharpe': actual_sharpe,
                    'perm_mean': round(np.mean(perm_sharpes), 3),
                    'perm_std': round(np.std(perm_sharpes), 3),
                    'p_value': round(pv, 3), 'significant': pv < 0.05}


def main():
    print("=" * 70)
    print("DISPERSION TRADING v1")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    prices, vix = download_data()

    # Quick dispersion analysis
    features = compute_dispersion_signal(prices, vix)
    oot_feat = features[features.index >= OOT_START]
    print(f"\nDispersion stats (OOT):")
    print(f"  Avg dispersion: {oot_feat['dispersion'].mean():.4f}")
    print(f"  Avg implied corr: {oot_feat['implied_corr'].mean():.3f}")
    print(f"  Avg sector corr: {oot_feat['avg_sector_corr'].mean():.3f}")
    print(f"  VIX premium mean: {oot_feat['vix_premium'].mean():.4f}")

    variants = [
        ('A', 'Long Dispersion (disp_z>1.5)'),
        ('B', 'Fade High Correlation (corr_z>1.5)'),
        ('C', 'Sell VIX Premium (vix_z>1.5)'),
        ('D', 'Low-Corr Pair Mean-Revert'),
        ('E', 'Dispersion Z-Score Mean-Revert'),
        ('F', 'Combined (2/3 signals)'),
    ]

    results = {}
    for v, label in variants:
        print(f"\n{'=' * 60}\nVariant {v}: {label}\n{'=' * 60}")
        try:
            r = run_dispersion_backtest(prices, vix, variant=v)
            if r:
                regime = regime_stratify(r, prices)
                print(f"  Sharpe={r['sharpe']}, Sortino={r['sortino']}, CAGR={r['cagr']}%, "
                      f"Trades={r['n_trades']}, WR={r['wr']}%, PF={r['pf']}, MDD={r['mdd']}%")
                print(f"  Regime: green={regime['sharpe_green']}, red={regime['sharpe_red']}, "
                      f"gap={regime['regime_gap']}, pass={regime['regime_balance_ok']}")
                rc = {k: v for k, v in r.items() if k not in ('equity_curve', 'oot_dates', 'trades')}
                results[v] = {'variant': v, 'label': label, 'metrics': rc, 'regime': regime,
                             'trades_sample': r.get('trades', [])[:10]}
            else:
                print(f"  No valid results")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    # Perm test on best positive-Sharpe variant
    positive = {v: r for v, r in results.items() if r['metrics']['sharpe'] > 0}
    if positive:
        best = max(positive, key=lambda v: positive[v]['metrics']['sharpe'])
        print(f"\n--- Perm Test on Variant {best} (Sharpe={positive[best]['metrics']['sharpe']}) ---")
        _, perm = permutation_test(prices, vix, variant=best, n_perms=N_PERMS)
        if perm:
            results[best]['permutation'] = perm
            print(f"  p={perm['p_value']}, sig={perm['significant']}")
    else:
        print("\n--- No positive-Sharpe variants, skipping perm test ---")

    final = {'strategy': 'Dispersion Trading v1', 'run_date': datetime.now().isoformat(),
             'capital': CAPITAL, 'oot_start': OOT_START, 'oot_end': OOT_END,
             'variants': results}

    out = os.path.join(RESULTS_DIR, 'dispersion_trade_v1_results.json')
    with open(out, 'w') as f: json.dump(final, f, indent=2, default=str)

    print(f"\n{'=' * 70}\nRESULTS SAVED: {out}\n{'=' * 70}")
    for v, r in sorted(results.items()):
        m = r['metrics']
        rg = r['regime']
        ok = "PASS" if rg.get('regime_balance_ok') else "FAIL"
        ps = f", perm_p={r['permutation']['p_value']}" if 'permutation' in r else ""
        print(f"  {v} ({r['label']}): Sharpe={m['sharpe']}, CAGR={m['cagr']}%, "
              f"WR={m['wr']}%, Regime={ok}(gap={rg.get('regime_gap', '?')}){ps}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
