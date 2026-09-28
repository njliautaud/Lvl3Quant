#!/usr/bin/env python3
"""
VIX Term Structure Trading v1 — Contango/Backwardation Signal
=============================================================
Academic literature shows VIX term structure slope predicts equity returns:
- Steep contango (VIX << VIX3M/VXV) = complacency → stay long / buy dips
- Backwardation (VIX >> VIX3M) = panic → fade the panic after 1-3 days

We use VIX/VIX3M ratio as the primary signal. Trade sector ETFs (liquid, cheap options)
or SPY itself.

Universe: XLK, XLF, XLE, XLI, XLY, SPY, QQQ
Capital: $645, OOT: 2022-01-01 to 2026-07-25
Sliding walk-forward (HC #0)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json, os, warnings, traceback
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings('ignore')

TRADE_TICKERS = ['XLK', 'XLF', 'XLE', 'XLI', 'XLY', 'SPY', 'QQQ']
SIGNAL_TICKERS = ['^VIX', '^VIX3M']  # VIX and VIX3M (3-month VIX)
ALL_TICKERS = TRADE_TICKERS + SIGNAL_TICKERS

TRAIN_WINDOW = 504   # 2 years for ML model
OOT_START = '2022-01-01'
OOT_END = '2026-07-25'
CAPITAL = 645.0

# Option cost assumptions (BS approx for ATM, DTE=14)
OPTION_COST_PCT = 0.025   # ~2.5% of underlying for ATM 14DTE call
SLIPPAGE_BPS = 10          # 10 bps for options (bid-ask + fill)
COMMISSION = 0.65          # Per contract

N_PERMS = 100
RESULTS_DIR = '/home/nick/Lvl3Quant/research/findings'
os.makedirs(RESULTS_DIR, exist_ok=True)


def download_data():
    print(f"Downloading {len(ALL_TICKERS)} tickers...")
    # Download trade tickers
    trade_data = yf.download(TRADE_TICKERS, start='2018-01-01', end=OOT_END,
                              auto_adjust=True, progress=False)
    trade_prices = trade_data['Close'] if isinstance(trade_data.columns, pd.MultiIndex) else trade_data

    # Download VIX indices separately (yfinance handles ^VIX)
    vix_data = yf.download(['^VIX'], start='2018-01-01', end=OOT_END,
                            auto_adjust=True, progress=False)
    vix3m_data = yf.download(['^VIX3M'], start='2018-01-01', end=OOT_END,
                              auto_adjust=True, progress=False)

    # Handle MultiIndex
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix = vix_data['Close'].iloc[:, 0] if vix_data['Close'].shape[1] == 1 else vix_data['Close']['^VIX']
    else:
        vix = vix_data['Close'] if 'Close' in str(vix_data.columns) else vix_data.iloc[:, 0]

    if isinstance(vix3m_data.columns, pd.MultiIndex):
        vix3m = vix3m_data['Close'].iloc[:, 0] if vix3m_data['Close'].shape[1] == 1 else vix3m_data['Close']['^VIX3M']
    else:
        vix3m = vix3m_data['Close'] if 'Close' in str(vix3m_data.columns) else vix3m_data.iloc[:, 0]

    vix = pd.Series(vix.values.flatten(), index=vix.index, name='VIX')
    vix3m = pd.Series(vix3m.values.flatten(), index=vix3m.index, name='VIX3M')

    print(f"Trade prices: {len(trade_prices)} days, {trade_prices.shape[1]} tickers")
    print(f"VIX: {len(vix)} days, VIX3M: {len(vix3m)} days")

    return trade_prices, vix, vix3m


def build_features(trade_prices, vix, vix3m):
    """Build VIX term structure features."""
    # Align all data
    common = trade_prices.index.intersection(vix.index).intersection(vix3m.index)
    tp = trade_prices.loc[common]
    v = vix.loc[common]
    v3m = vix3m.loc[common]

    features = pd.DataFrame(index=common)

    # Core term structure signals
    features['vix'] = v
    features['vix3m'] = v3m
    features['vts_ratio'] = v / v3m  # <1 = contango, >1 = backwardation
    features['vts_spread'] = v - v3m  # Negative = contango

    # VIX dynamics
    features['vix_5d_chg'] = v.pct_change(5)
    features['vix_10d_chg'] = v.pct_change(10)
    features['vix_z20'] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    features['vix_z60'] = (v - v.rolling(60).mean()) / v.rolling(60).std()
    features['vix_percentile_252'] = v.rolling(252).apply(lambda x: (x < x.iloc[-1]).mean(), raw=False)

    # Term structure dynamics
    features['vts_ratio_5d_chg'] = features['vts_ratio'].pct_change(5)
    features['vts_ratio_z20'] = (features['vts_ratio'] - features['vts_ratio'].rolling(20).mean()) / features['vts_ratio'].rolling(20).std()
    features['vts_slope'] = features['vts_ratio'].rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False)

    # SPY context
    spy = tp['SPY'] if 'SPY' in tp.columns else tp.iloc[:, 0]
    features['spy_ret_5d'] = spy.pct_change(5)
    features['spy_ret_21d'] = spy.pct_change(21)
    features['spy_vol_21d'] = spy.pct_change().rolling(21).std() * np.sqrt(252)
    features['spy_z50'] = (spy - spy.rolling(50).mean()) / spy.rolling(50).std()

    # Interaction features
    features['vix_x_spy_vol'] = features['vix'] * features['spy_vol_21d']
    features['vts_x_spy_ret'] = features['vts_ratio'] * features['spy_ret_5d']

    return features, tp


def run_vts_backtest(trade_prices, vix, vix3m, variant='A', seed=42):
    """
    Variants:
    A: Rules-based — buy when VTS ratio drops below 0.85 (deep contango), sell on revert
    B: Rules-based — fade backwardation: buy when ratio > 1.05 and starts falling
    C: ML-based — GBM classifier predicts 5-day forward SPY return sign
    D: Rules + options — buy calls on contango entry, puts on backwardation
    E: Sector rotation — rotate to highest-momentum sector during contango, defensive during backw.
    F: Combined signals — A + B + momentum filter
    """
    features, tp = build_features(trade_prices, vix, vix3m)

    oot_mask = features.index >= OOT_START
    oot_dates = features.index[oot_mask]

    if len(oot_dates) < 40:
        return None

    equity = CAPITAL
    equity_curve = []
    trades = []
    position = None  # {'ticker': str, 'direction': str, 'entry_price': float, 'entry_date': date, 'type': 'equity'|'option'}

    # For ML variant, train model on sliding window
    feature_cols = [c for c in features.columns if c not in ('vix', 'vix3m')]

    for i, date in enumerate(oot_dates):
        date_idx = features.index.get_loc(date)

        if date_idx < TRAIN_WINDOW:
            equity_curve.append(equity)
            continue

        today_feat = features.iloc[date_idx]
        today_prices = tp.iloc[date_idx]

        vts_ratio = today_feat['vts_ratio']
        vts_ratio_z = today_feat.get('vts_ratio_z20', 0)
        vix_val = today_feat['vix']
        vix_z = today_feat.get('vix_z20', 0)

        # ── Close existing position ──
        if position is not None:
            ticker = position['ticker']
            if ticker not in today_prices.index or pd.isna(today_prices[ticker]):
                equity_curve.append(equity)
                continue

            current_price = today_prices[ticker]
            days_held = (date - position['entry_date']).days

            should_close = False
            close_reason = ''

            if variant in ('A', 'F'):
                # Close when VTS normalizes (ratio > 0.95) or 10 day max hold
                if vts_ratio > 0.95 or days_held >= 10:
                    should_close = True
                    close_reason = 'vts_normalize' if vts_ratio > 0.95 else 'time_stop'

            elif variant == 'B':
                # Close when backwardation subsides or 5 day max hold
                if vts_ratio < 1.0 or days_held >= 5:
                    should_close = True
                    close_reason = 'backw_subside' if vts_ratio < 1.0 else 'time_stop'

            elif variant == 'C':
                # Close after 5 days (holding period matches prediction horizon)
                if days_held >= 5:
                    should_close = True
                    close_reason = 'horizon'

            elif variant == 'D':
                # Options: close after 5 days or 30% profit or 25% loss
                entry_cost = position.get('option_cost', position['entry_price'] * OPTION_COST_PCT)
                option_delta = 0.50  # ATM approximation
                price_change = current_price - position['entry_price']
                if position['direction'] == 'short':
                    price_change = -price_change
                option_pnl_pct = (price_change * option_delta) / entry_cost

                if option_pnl_pct >= 0.30 or option_pnl_pct <= -0.25 or days_held >= 5:
                    should_close = True
                    close_reason = 'tp' if option_pnl_pct >= 0.30 else ('sl' if option_pnl_pct <= -0.25 else 'time')

            elif variant == 'E':
                # Sector rotation: rebalance weekly
                if days_held >= 5:
                    should_close = True
                    close_reason = 'rebalance'

            if should_close:
                if position.get('type') == 'option':
                    # Option P&L
                    entry_cost = position.get('option_cost', position['entry_price'] * OPTION_COST_PCT)
                    option_delta = 0.50
                    price_change = current_price - position['entry_price']
                    if position['direction'] == 'short':
                        price_change = -price_change
                    pnl = position['size'] * (price_change * option_delta - entry_cost * SLIPPAGE_BPS / 10000)
                    # Can't lose more than premium paid
                    pnl = max(pnl, -position['size'] * entry_cost)
                else:
                    # Equity P&L
                    if position['direction'] == 'long':
                        pnl = position['shares'] * (current_price - position['entry_price'])
                    else:
                        pnl = position['shares'] * (position['entry_price'] - current_price)
                    pnl -= abs(position['shares']) * current_price * SLIPPAGE_BPS / 10000 * 2  # RT slippage

                equity += pnl
                trades.append({
                    'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'direction': position['direction'],
                    'type': position.get('type', 'equity'),
                    'pnl': round(pnl, 2),
                    'days_held': days_held,
                    'close_reason': close_reason,
                    'entry_vts': position.get('entry_vts', 0),
                    'exit_vts': round(vts_ratio, 3)
                })
                position = None

        # ── Open new position ──
        if position is None and equity > 50:
            signal = None
            ticker = 'SPY'
            direction = 'long'
            trade_type = 'equity'

            if variant == 'A':
                # Deep contango entry
                if vts_ratio < 0.85 and vix_z > 0.5:
                    signal = 'contango_buy'
                    direction = 'long'

            elif variant == 'B':
                # Fade backwardation (panic → recovery)
                if vts_ratio > 1.05 and today_feat.get('vts_slope', 0) < 0:
                    signal = 'backw_fade'
                    direction = 'long'

            elif variant == 'C':
                # ML prediction
                if date_idx >= TRAIN_WINDOW + 5:
                    train_start = date_idx - TRAIN_WINDOW
                    train_feat = features.iloc[train_start:date_idx][feature_cols].dropna()

                    # Create labels: 5-day forward SPY return > 0
                    spy = tp['SPY'] if 'SPY' in tp.columns else tp.iloc[:, 0]
                    fwd_ret = spy.pct_change(5).shift(-5)
                    train_labels = (fwd_ret.loc[train_feat.index] > 0).astype(int)

                    valid = train_labels.dropna().index.intersection(train_feat.dropna().index)
                    if len(valid) > 100:
                        X_train = train_feat.loc[valid].values
                        y_train = train_labels.loc[valid].values

                        try:
                            model = GradientBoostingClassifier(
                                n_estimators=50, max_depth=3, learning_rate=0.1,
                                random_state=seed)
                            model.fit(X_train, y_train)

                            X_today = features.iloc[date_idx:date_idx+1][feature_cols].values
                            if not np.any(np.isnan(X_today)):
                                prob = model.predict_proba(X_today)[0]
                                if len(prob) > 1 and prob[1] > 0.60:
                                    signal = 'ml_long'
                                    direction = 'long'
                                elif len(prob) > 1 and prob[1] < 0.40:
                                    signal = 'ml_short'
                                    direction = 'short'
                        except Exception:
                            pass

            elif variant == 'D':
                # Options version of A + B
                trade_type = 'option'
                if vts_ratio < 0.85 and vix_z > 0.5:
                    signal = 'contango_call'
                    direction = 'long'
                elif vts_ratio > 1.05 and today_feat.get('vts_slope', 0) < 0:
                    signal = 'backw_call'
                    direction = 'long'

            elif variant == 'E':
                # Sector rotation based on VIX regime
                sectors = [t for t in ['XLK', 'XLF', 'XLE', 'XLI', 'XLY'] if t in today_prices.index]
                if sectors:
                    if vts_ratio < 0.95:  # Contango = risk-on
                        # Pick highest 21d momentum sector
                        mom = {}
                        for s in sectors:
                            if s in tp.columns:
                                ret = tp[s].iloc[max(0,date_idx-21):date_idx].pct_change().sum()
                                mom[s] = ret
                        if mom:
                            ticker = max(mom, key=mom.get)
                            signal = 'sector_riskOn'
                            direction = 'long'
                    elif vts_ratio > 1.0:  # Backwardation = defensive
                        ticker = 'XLU' if 'XLU' in today_prices.index else 'XLP'
                        if ticker not in today_prices.index:
                            ticker = 'SPY'
                        signal = 'sector_defensive'
                        direction = 'long'

            elif variant == 'F':
                # Combined: contango + backwardation fade + momentum filter
                spy_ret_5d = today_feat.get('spy_ret_5d', 0)
                if vts_ratio < 0.85 and vix_z > 0.5:
                    signal = 'combined_contango'
                    direction = 'long'
                elif vts_ratio > 1.05 and today_feat.get('vts_slope', 0) < 0 and spy_ret_5d < -0.02:
                    signal = 'combined_backw_fade'
                    direction = 'long'

            if signal and ticker in today_prices.index and not pd.isna(today_prices[ticker]):
                entry_price = today_prices[ticker]
                alloc = min(equity * 0.50, 300)  # Max $300 per trade

                if trade_type == 'option':
                    option_cost = entry_price * OPTION_COST_PCT
                    n_contracts = max(1, int(alloc / (option_cost * 100)))
                    position = {
                        'ticker': ticker, 'direction': direction, 'type': 'option',
                        'entry_price': entry_price, 'entry_date': date,
                        'option_cost': option_cost, 'size': n_contracts * 100,
                        'entry_vts': round(vts_ratio, 3)
                    }
                else:
                    shares = alloc / entry_price
                    position = {
                        'ticker': ticker, 'direction': direction, 'type': 'equity',
                        'entry_price': entry_price, 'entry_date': date,
                        'shares': shares, 'entry_vts': round(vts_ratio, 3)
                    }

        equity_curve.append(equity)

    if not equity_curve:
        return None

    ea = np.array(equity_curve)
    rets = np.diff(ea) / ea[:-1]
    rets = rets[np.isfinite(rets)]

    if len(rets) < 40 or np.std(rets) == 0:
        return None

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

    return {
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 2), 'mdd': round(mdd, 2),
        'wr': round(wr, 2), 'pf': round(pf, 3),
        'n_trades': len(trades), 'n_days': len(rets),
        'final_equity': round(ea[-1], 2),
        'equity_curve': ea.tolist(),
        'oot_dates': [d.strftime('%Y-%m-%d') for d in oot_dates[:len(ea)]],
        'trades': trades
    }


def regime_stratify(metrics, trade_prices):
    if 'SPY' not in trade_prices.columns:
        return {'regime_balance_ok': False}
    spy = trade_prices['SPY'].dropna()
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


def permutation_test(trade_prices, vix, vix3m, variant, n_perms=N_PERMS):
    actual = run_vts_backtest(trade_prices, vix, vix3m, variant=variant)
    if not actual:
        return None, None
    actual_sharpe = actual['sharpe']
    perm_sharpes = []

    for i in range(n_perms):
        # Shuffle VIX/VIX3M time alignment (break signal-return relationship)
        vix_shuffled = vix.copy()
        vix3m_shuffled = vix3m.copy()
        rng = np.random.RandomState(i)

        # Circular shift VIX data by random offset
        shift = rng.randint(60, len(vix) - 60)
        vix_vals = vix_shuffled.values
        vix_shuffled = pd.Series(np.roll(vix_vals, shift), index=vix_shuffled.index, name='VIX')
        vix3m_vals = vix3m_shuffled.values
        vix3m_shuffled = pd.Series(np.roll(vix3m_vals, shift), index=vix3m_shuffled.index, name='VIX3M')

        pr = run_vts_backtest(trade_prices, vix_shuffled, vix3m_shuffled, variant=variant, seed=i)
        if pr:
            perm_sharpes.append(pr['sharpe'])

    if not perm_sharpes:
        return actual, {'p_value': 1.0, 'significant': False}
    pv = np.mean([s >= actual_sharpe for s in perm_sharpes])
    return actual, {
        'actual_sharpe': actual_sharpe,
        'perm_mean': round(np.mean(perm_sharpes), 3),
        'perm_std': round(np.std(perm_sharpes), 3),
        'p_value': round(pv, 3),
        'significant': pv < 0.05
    }


def main():
    print("=" * 70)
    print("VIX TERM STRUCTURE TRADING v1")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    trade_prices, vix, vix3m = download_data()

    # Quick VTS analysis
    common = vix.index.intersection(vix3m.index)
    ratio = vix.loc[common] / vix3m.loc[common]
    print(f"\nVTS Ratio stats: mean={ratio.mean():.3f}, std={ratio.std():.3f}")
    print(f"  Contango (<0.85): {(ratio < 0.85).sum()} days ({(ratio < 0.85).mean()*100:.1f}%)")
    print(f"  Normal (0.85-1.05): {((ratio >= 0.85) & (ratio <= 1.05)).sum()} days")
    print(f"  Backwardation (>1.05): {(ratio > 1.05).sum()} days ({(ratio > 1.05).mean()*100:.1f}%)")

    variants = [
        ('A', 'Contango Buy (ratio<0.85)'),
        ('B', 'Fade Backwardation (ratio>1.05 falling)'),
        ('C', 'ML GBM Classifier'),
        ('D', 'Options (calls on contango/backw)'),
        ('E', 'Sector Rotation by VIX Regime'),
        ('F', 'Combined Signals + Momentum'),
    ]

    results = {}
    for v, label in variants:
        print(f"\n{'=' * 60}\nVariant {v}: {label}\n{'=' * 60}")
        try:
            r = run_vts_backtest(trade_prices, vix, vix3m, variant=v)
            if r:
                regime = regime_stratify(r, trade_prices)
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
        _, perm = permutation_test(trade_prices, vix, vix3m, variant=best, n_perms=N_PERMS)
        if perm:
            results[best]['permutation'] = perm
            print(f"  p={perm['p_value']}, sig={perm['significant']}")
    else:
        print("\n--- No positive-Sharpe variants, skipping perm test ---")

    final = {
        'strategy': 'VIX Term Structure Trading v1',
        'run_date': datetime.now().isoformat(),
        'capital': CAPITAL,
        'universe': TRADE_TICKERS,
        'oot_start': OOT_START,
        'oot_end': OOT_END,
        'variants': results
    }

    out = os.path.join(RESULTS_DIR, 'vix_term_structure_v1_results.json')
    with open(out, 'w') as f:
        json.dump(final, f, indent=2, default=str)

    print(f"\n{'=' * 70}\nRESULTS SAVED: {out}\n{'=' * 70}")
    for v, r in sorted(results.items()):
        m = r['metrics']
        rg = r['regime']
        ok = "PASS" if rg.get('regime_balance_ok') else "FAIL"
        perm_str = ""
        if 'permutation' in r:
            perm_str = f", perm_p={r['permutation']['p_value']}"
        print(f"  {v} ({r['label']}): Sharpe={m['sharpe']}, CAGR={m['cagr']}%, "
              f"WR={m['wr']}%, Regime={ok}(gap={rg.get('regime_gap', '?')}){perm_str}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
