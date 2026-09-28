#!/usr/bin/env python3
"""
Factor ETF LightGBM Ranker v1
==============================

Survivorship-free approach to the QM Ranker concept.
Instead of ranking individual stocks (survivorship bias from NVDA/TSLA),
rank a broader universe of factor/thematic/sector ETFs.

Universe: ~40 ETFs covering:
- Sectors (XLK, XLF, XLE, etc.)
- Factors (MTUM, VLUE, QUAL, SIZE, USMV)
- Themes (ARKK, KWEB, IBB, XBI, TAN)
- Geography (EFA, EEM, FXI, EWJ)
- Fixed income/alternatives (TLT, HYG, LQD, GLD, SLV, DBC)
- Broad indices (QQQ, IWM, MDY, DIA)

All ETFs → zero survivorship bias.
LightGBM cross-sectional ranking with 20+ features.
Monthly rebalance, top K holdings.
Walk-forward: 252d train, 21d test, sliding.
Transaction costs: 10bps each way (20bps RT).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'factor_etf_ranker_v1_results.json'

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass

# Expanded universe — all major, long-lived ETFs
UNIVERSE = [
    # Sectors
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC',
    # Broad indices
    'QQQ', 'IWM', 'MDY', 'DIA',
    # Factor ETFs
    'MTUM', 'VLUE', 'QUAL', 'USMV',
    # International
    'EFA', 'EEM', 'FXI', 'EWJ', 'EWZ', 'VWO',
    # Thematic
    'IBB', 'XBI', 'IYR', 'VNQ',
    # Fixed income & alternatives
    'TLT', 'IEF', 'HYG', 'LQD', 'GLD', 'SLV', 'DBC',
    # Other
    'IEMG', 'SMH', 'XHB',
]

# Defensive ETFs for regime shift
DEFENSIVE = {'XLP', 'XLU', 'TLT', 'IEF', 'GLD', 'USMV'}
RISK_ON = {'XLK', 'QQQ', 'XLY', 'IWM', 'EEM', 'XBI', 'SMH'}


def download_data():
    """Download all ETF data."""
    import yfinance as yf

    tickers = list(set(UNIVERSE + ['SPY']))
    fprint(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start='2008-01-01', auto_adjust=True, progress=False)

    close = data['Close'].dropna(how='all')

    # Drop ETFs with >30% missing (newer ETFs)
    valid_cols = []
    for col in close.columns:
        if col in UNIVERSE and close[col].isna().mean() < 0.30:
            valid_cols.append(col)
    valid_cols = list(set(valid_cols + ['SPY']))

    close = close[valid_cols].ffill().dropna()

    n_etfs = len([c for c in close.columns if c in UNIVERSE])
    fprint(f"  Loaded {n_etfs} ETFs + SPY ({len(close)} days)")
    return close


def build_features(close_df, date_idx, window=252):
    """Build cross-sectional features for all ETFs at a given date."""
    etf_cols = [c for c in close_df.columns if c in UNIVERSE]

    features = {}
    for etf in etf_cols:
        prices = close_df[etf].iloc[max(0, date_idx - window):date_idx + 1]
        if len(prices) < 60:
            continue

        lr = np.log(prices / prices.shift(1)).dropna()
        if len(lr) < 20:
            continue

        # Momentum features
        ret_5d = prices.iloc[-1] / prices.iloc[-6] - 1 if len(prices) >= 6 else 0
        ret_21d = prices.iloc[-1] / prices.iloc[-22] - 1 if len(prices) >= 22 else 0
        ret_63d = prices.iloc[-1] / prices.iloc[-64] - 1 if len(prices) >= 64 else 0
        ret_126d = prices.iloc[-1] / prices.iloc[-127] - 1 if len(prices) >= 127 else 0
        ret_252d = prices.iloc[-1] / prices.iloc[-min(253, len(prices))] - 1

        # 12-1 momentum (skip last month)
        mom_12_1 = ret_252d - ret_21d

        # Momentum acceleration
        ret_63d_prev = prices.iloc[-64] / prices.iloc[-127] - 1 if len(prices) >= 127 else 0
        mom_accel = ret_63d - ret_63d_prev

        # Volatility features
        vol_20d = float(lr.iloc[-20:].std() * np.sqrt(252)) if len(lr) >= 20 else 0.20
        vol_60d = float(lr.iloc[-60:].std() * np.sqrt(252)) if len(lr) >= 60 else 0.20
        vol_ratio = vol_20d / vol_60d if vol_60d > 0 else 1.0

        # Risk-adjusted momentum
        sharpe_63d = float(lr.iloc[-63:].mean() / lr.iloc[-63:].std()) if len(lr) >= 63 and lr.iloc[-63:].std() > 0 else 0

        # MaxDD (recent)
        rolling_max = prices.iloc[-63:].cummax() if len(prices) >= 63 else prices.cummax()
        dd = (prices.iloc[-63:] / rolling_max - 1) if len(prices) >= 63 else (prices / rolling_max - 1)
        maxdd = float(dd.min())

        # Distance from 52-week high
        high_52w = prices.iloc[-min(252, len(prices)):].max()
        dist_high = float(prices.iloc[-1] / high_52w)

        # Skewness and kurtosis
        skew = float(lr.iloc[-63:].skew()) if len(lr) >= 63 else 0
        kurt = float(lr.iloc[-63:].kurtosis()) if len(lr) >= 63 else 0

        # Volume momentum (use price as proxy since volume data may be inconsistent)
        price_vol = float(lr.rolling(20).std().iloc[-1]) if len(lr) >= 20 else 0

        # Relative strength vs SPY
        if 'SPY' in close_df.columns:
            spy_prices = close_df['SPY'].iloc[max(0, date_idx - window):date_idx + 1]
            spy_ret_63d = spy_prices.iloc[-1] / spy_prices.iloc[-64] - 1 if len(spy_prices) >= 64 else 0
            rel_strength = ret_63d - spy_ret_63d
        else:
            rel_strength = 0

        # Trend strength (price vs SMA)
        sma_50 = prices.iloc[-50:].mean() if len(prices) >= 50 else prices.mean()
        sma_200 = prices.iloc[-200:].mean() if len(prices) >= 200 else prices.mean()
        above_sma50 = float(prices.iloc[-1] / sma_50 - 1)
        above_sma200 = float(prices.iloc[-1] / sma_200 - 1)

        features[etf] = {
            'ret_5d': ret_5d,
            'ret_21d': ret_21d,
            'ret_63d': ret_63d,
            'ret_126d': ret_126d,
            'ret_252d': ret_252d,
            'mom_12_1': mom_12_1,
            'mom_accel': mom_accel,
            'vol_20d': vol_20d,
            'vol_60d': vol_60d,
            'vol_ratio': vol_ratio,
            'sharpe_63d': sharpe_63d,
            'maxdd': maxdd,
            'dist_high': dist_high,
            'skew': skew,
            'kurtosis': kurt,
            'price_vol': price_vol,
            'rel_strength': rel_strength,
            'above_sma50': above_sma50,
            'above_sma200': above_sma200,
        }

    return features


def run_wf_lgbm(close_df, top_k=5, train_days=252, test_days=21,
                defensive_shift=False, shift_strength=1.5,
                name='variant'):
    """
    Run walk-forward LightGBM cross-sectional ranking.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        fprint("ERROR: lightgbm not installed. Install with: pip install lightgbm")
        return None, [], [], []

    etf_cols = [c for c in close_df.columns if c in UNIVERSE]
    spy = close_df['SPY']
    spy_sma200 = spy.rolling(200).mean()

    dates = close_df.index
    start_idx = train_days + 252  # need 252d lookback + train window

    # Build all features at once (faster)
    fprint(f"  [{name}] Building features...")
    all_dates = list(range(start_idx, len(dates), test_days))

    monthly_returns = []
    capital = 10000
    capital_history = [(dates[start_idx], capital)]
    trades = []
    feature_importance = None

    for fold_i, test_start in enumerate(all_dates[:-1]):
        test_end = min(test_start + test_days, len(dates) - 1)

        # Build training data: rank ETFs by forward return
        train_X = []
        train_y = []

        for t in range(max(start_idx - train_days, 252), test_start, test_days):
            feats = build_features(close_df, t)
            if not feats:
                continue

            # Forward return (21d) for each ETF
            fwd_rets = {}
            for etf in feats:
                if t + test_days < len(dates):
                    fwd = float(close_df[etf].iloc[t + test_days] / close_df[etf].iloc[t] - 1)
                    fwd_rets[etf] = fwd

            if not fwd_rets:
                continue

            # Cross-sectional rank as label
            sorted_etfs = sorted(fwd_rets.items(), key=lambda x: x[1], reverse=True)
            n = len(sorted_etfs)
            rank_map = {etf: (n - i) / n for i, (etf, _) in enumerate(sorted_etfs)}

            for etf, feat_dict in feats.items():
                if etf in rank_map:
                    train_X.append(list(feat_dict.values()))
                    train_y.append(rank_map[etf])

        if len(train_X) < 50:
            monthly_returns.append(0)
            capital_history.append((dates[test_end], capital))
            continue

        train_X = np.array(train_X)
        train_y = np.array(train_y)

        # Train LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': -1,
            'n_jobs': 4,
        }

        ds = lgb.Dataset(train_X, label=train_y)
        model = lgb.train(params, ds, num_boost_round=100)

        if feature_importance is None:
            feature_importance = model.feature_importance(importance_type='gain')
        else:
            feature_importance += model.feature_importance(importance_type='gain')

        # Predict on test period
        test_feats = build_features(close_df, test_start)
        if not test_feats:
            monthly_returns.append(0)
            capital_history.append((dates[test_end], capital))
            continue

        predictions = {}
        for etf, feat_dict in test_feats.items():
            pred = model.predict(np.array([list(feat_dict.values())]))[0]
            predictions[etf] = pred

        # Defensive shift in bear markets
        if defensive_shift:
            spy_idx = test_start
            if spy.iloc[spy_idx] < spy_sma200.iloc[spy_idx]:
                for etf in predictions:
                    if etf in DEFENSIVE:
                        predictions[etf] *= shift_strength
                    elif etf in RISK_ON:
                        predictions[etf] /= shift_strength

        # Select top K
        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Equal weight
        per_pos = capital / top_k
        period_pnl = 0

        for etf, score in ranked:
            entry_price = float(close_df[etf].iloc[test_start])
            exit_price = float(close_df[etf].iloc[test_end])
            ret = (exit_price / entry_price - 1)

            # Transaction costs: 10bps each way = 20bps RT
            ret -= 0.002

            pos_pnl = per_pos * ret
            period_pnl += pos_pnl

            trades.append({
                'entry_date': str(dates[test_start].date()),
                'exit_date': str(dates[test_end].date()),
                'etf': etf,
                'score': round(score, 4),
                'return_pct': round(ret * 100, 2),
                'pnl': round(pos_pnl, 2),
            })

        capital += period_pnl
        capital = max(0, capital)

        ret_pct = period_pnl / max(capital - period_pnl, 1)
        monthly_returns.append(ret_pct)
        capital_history.append((dates[test_end], capital))

        if capital <= 0:
            break

    # Metrics
    r = np.array(monthly_returns)
    r_nz = r[r != 0] if np.any(r != 0) else r

    sharpe = np.mean(r_nz) / np.std(r_nz) * np.sqrt(12) if np.std(r_nz) > 0 else 0
    ds_rets = r_nz[r_nz < 0]
    ds_vol = np.std(ds_rets) * np.sqrt(12) if len(ds_rets) > 0 else 1e-6
    sortino = np.mean(r_nz) * 12 / ds_vol if ds_vol > 0 else 0

    if len(capital_history) > 1:
        years = (capital_history[-1][0] - capital_history[0][0]).days / 365.25
        if years > 0 and capital > 0:
            cagr = (capital / 10000) ** (1/years) - 1
        else:
            cagr = -1
    else:
        years = 0
        cagr = 0

    peak = 10000
    maxdd = 0
    for _, val in capital_history:
        peak = max(peak, val)
        dd = (val - peak) / peak if peak > 0 else 0
        maxdd = min(maxdd, dd)

    tpnls = [t['pnl'] for t in trades]
    wr = sum(1 for p in tpnls if p > 0) / len(tpnls) * 100 if tpnls else 0
    gp = sum(p for p in tpnls if p > 0)
    gl = abs(sum(p for p in tpnls if p < 0))
    pf = gp / gl if gl > 0 else float('inf')
    calmar = abs(cagr / maxdd) if maxdd < 0 else 0

    # Feature importance (normalized)
    feat_names = list(build_features(close_df, start_idx).get(etf_cols[0], {}).keys()) if etf_cols else []
    feat_imp = {}
    if feature_importance is not None and len(feat_names) == len(feature_importance):
        total = feature_importance.sum()
        if total > 0:
            for fn, fi in zip(feat_names, feature_importance):
                feat_imp[fn] = round(float(fi / total * 100), 1)

    # Top ETFs by selection frequency
    etf_freq = {}
    for t in trades:
        etf = t['etf']
        etf_freq[etf] = etf_freq.get(etf, 0) + 1
    top_etfs = sorted(etf_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    metrics = {
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr': round(cagr * 100, 1),
        'maxdd': round(maxdd * 100, 1),
        'wr': round(wr, 1),
        'pf': round(pf, 2),
        'calmar': round(calmar, 2),
        'final_capital': round(capital, 2),
        'n_trades': len(trades),
        'n_folds': len(monthly_returns),
        'years': round(years, 1),
        'top_features': dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:8]),
        'top_etfs': {e: c for e, c in top_etfs},
    }

    return metrics, trades, monthly_returns, capital_history


def adversarial_gates(monthly_returns, close_df, name):
    """4-gate adversarial validation."""
    r = np.array(monthly_returns)
    r_nz = r[r != 0] if np.any(r != 0) else r
    gates = {}

    fprint(f"  Adversarial gates for {name}:")

    # 1. Permutation test
    actual_sharpe = np.mean(r_nz) / np.std(r_nz) * np.sqrt(12) if np.std(r_nz) > 0 else 0
    n_perm = 1000
    perm_sharpes = []
    for _ in range(n_perm):
        p = np.random.permutation(r_nz)
        ps = np.mean(p) / np.std(p) * np.sqrt(12) if np.std(p) > 0 else 0
        perm_sharpes.append(ps)
    p_val = np.mean([ps >= actual_sharpe for ps in perm_sharpes])
    gates['permutation'] = {
        'actual_sharpe': round(float(actual_sharpe), 3),
        'p_value': round(float(p_val), 3),
        'pass': p_val < 0.05,
    }
    fprint(f"    Perm: Sharpe={actual_sharpe:.2f}, p={p_val:.3f} {'PASS' if p_val < 0.05 else 'FAIL'}")

    # 2. R1 Regime
    spy = close_df['SPY']
    spy_monthly = spy.resample('ME').last().pct_change().dropna()
    if len(r_nz) > 6:
        bull_mask = np.zeros(len(r_nz), dtype=bool)
        bear_mask = np.zeros(len(r_nz), dtype=bool)
        for i in range(len(r_nz)):
            off = len(r_nz) - i
            if off <= len(spy_monthly):
                sr = float(spy_monthly.iloc[-off])
                if sr >= 0: bull_mask[i] = True
                else: bear_mask[i] = True
            else:
                bull_mask[i] = True

        if bull_mask.sum() >= 3 and bear_mask.sum() >= 3:
            br = r_nz[bull_mask]; ber = r_nz[bear_mask]
            bs = np.mean(br) / np.std(br) * np.sqrt(12) if np.std(br) > 0 else 0
            brs = np.mean(ber) / np.std(ber) * np.sqrt(12) if np.std(ber) > 0 else 0
            ms = max(abs(bs), abs(brs))
            gap = abs(bs - brs) / ms if ms > 0 else 0
            gates['regime_r1'] = {
                'bull_sharpe': round(float(bs), 2),
                'bear_sharpe': round(float(brs), 2),
                'gap': round(float(gap), 3),
                'pass': gap < 0.50,
            }
            fprint(f"    R1: bull={bs:.2f}, bear={brs:.2f}, gap={gap:.3f} {'PASS' if gap < 0.50 else 'FAIL'}")

    # 3. Sub-period
    mid = len(r_nz) // 2
    if mid >= 3:
        h1s = np.mean(r_nz[:mid]) / np.std(r_nz[:mid]) * np.sqrt(12) if np.std(r_nz[:mid]) > 0 else 0
        h2s = np.mean(r_nz[mid:]) / np.std(r_nz[mid:]) * np.sqrt(12) if np.std(r_nz[mid:]) > 0 else 0
        sp = h1s > 0 and h2s > 0
        gates['sub_period'] = {'h1_sharpe': round(float(h1s), 2), 'h2_sharpe': round(float(h2s), 2), 'pass': sp}
        fprint(f"    Sub: H1={h1s:.2f}, H2={h2s:.2f} {'PASS' if sp else 'FAIL'}")

    # 4. Outlier
    if len(r_nz) >= 10:
        nr = max(1, int(len(r_nz) * 0.05))
        si = np.argsort(r_nz)[::-1]
        trimmed = np.delete(r_nz, si[:nr])
        ts = np.mean(trimmed) / np.std(trimmed) * np.sqrt(12) if np.std(trimmed) > 0 else 0
        op = ts > 0
        gates['outlier'] = {'trimmed_sharpe': round(float(ts), 2), 'n_removed': nr, 'pass': op}
        fprint(f"    Outlier: trimmed={ts:.2f} {'PASS' if op else 'FAIL'}")

    n_pass = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                 if gates.get(k, {}).get('pass') is True)
    n_total = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                  if gates.get(k, {}).get('pass') is not None)
    fprint(f"    GATES: {n_pass}/{n_total}")

    return gates


def main():
    fprint("=" * 70)
    fprint("FACTOR ETF LightGBM RANKER v1")
    fprint("=" * 70)
    fprint(f"Universe: {len(UNIVERSE)} ETFs (zero survivorship bias)")
    fprint(f"Approach: LightGBM cross-sectional ranking, monthly rebalance")
    fprint()

    close = download_data()

    variants = [
        {'name': 'A_Top3', 'top_k': 3, 'desc': 'Top 3, no regime shift'},
        {'name': 'B_Top5', 'top_k': 5, 'desc': 'Top 5, no regime shift'},
        {'name': 'C_Top3_DefShift', 'top_k': 3, 'defensive_shift': True, 'desc': 'Top 3 + defensive shift'},
        {'name': 'D_Top5_DefShift', 'top_k': 5, 'defensive_shift': True, 'desc': 'Top 5 + defensive shift'},
        {'name': 'E_Top5_StrongShift', 'top_k': 5, 'defensive_shift': True, 'shift_strength': 2.0,
         'desc': 'Top 5 + strong shift (2x)'},
        {'name': 'F_Top7', 'top_k': 7, 'desc': 'Top 7, more diversified'},
    ]

    results = {}
    gate_results = {}

    for v in variants:
        vname = v['name']
        fprint(f"\n--- {vname} ({v['desc']}) ---")

        kwargs = {
            'top_k': v['top_k'],
            'defensive_shift': v.get('defensive_shift', False),
            'shift_strength': v.get('shift_strength', 1.5),
            'name': vname,
        }

        result = run_wf_lgbm(close, **kwargs)
        if result is None:
            continue

        metrics, trades, monthly_returns, cap_hist = result

        fprint(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
               f"CAGR: {metrics['cagr']}%, MaxDD: {metrics['maxdd']}%, "
               f"WR: {metrics['wr']}%, PF: {metrics['pf']}, Calmar: {metrics['calmar']}")
        fprint(f"  ${10000} → ${metrics['final_capital']} over {metrics['years']}y, "
               f"{metrics['n_trades']} trades, {metrics['n_folds']} folds")
        if metrics.get('top_features'):
            fprint(f"  Top features: {dict(list(metrics['top_features'].items())[:5])}")
        if metrics.get('top_etfs'):
            fprint(f"  Top ETFs: {dict(list(metrics['top_etfs'].items())[:5])}")

        gates = adversarial_gates(monthly_returns, close, vname)

        results[vname] = {'desc': v['desc'], 'metrics': metrics}
        gate_results[vname] = gates

    # Summary
    fprint(f"\n{'='*70}")
    fprint("SUMMARY")
    fprint(f"{'='*70}")

    valid = [(n, r) for n, r in results.items()
             if r['metrics']['sharpe'] > 0 and r['metrics']['n_trades'] > 20]

    if valid:
        def sort_key(item):
            n, r = item
            g = gate_results[n]
            np_ = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                      if g.get(k, {}).get('pass') is True)
            return (np_, r['metrics']['sharpe'])

        valid.sort(key=sort_key, reverse=True)

        for n, r in valid:
            m = r['metrics']
            g = gate_results[n]
            np_ = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                      if g.get(k, {}).get('pass') is True)
            nt = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                     if g.get(k, {}).get('pass') is not None)
            fprint(f"  {n}: Sharpe {m['sharpe']}, CAGR {m['cagr']}%, MaxDD {m['maxdd']}%, "
                   f"WR {m['wr']}%, PF {m['pf']}, Calmar {m['calmar']} | "
                   f"Gates {np_}/{nt}")

        winner_name = valid[0][0]
        winner = results[winner_name]
        wm = winner['metrics']
        fprint(f"\n  WINNER: {winner_name}")
        fprint(f"    {winner['desc']}")
        fprint(f"    Sharpe {wm['sharpe']}, Sortino {wm['sortino']}, CAGR {wm['cagr']}%, "
               f"MaxDD {wm['maxdd']}%, WR {wm['wr']}%, PF {wm['pf']}, Calmar {wm['calmar']}")
        fprint(f"    Top features: {wm.get('top_features', {})}")
        fprint(f"    Top ETFs: {wm.get('top_etfs', {})}")
    else:
        fprint("  NO VALID VARIANTS")
        winner_name = None

    # Save
    output = {
        'strategy': 'Factor ETF LightGBM Ranker v1',
        'timestamp': datetime.now().isoformat(),
        'universe': UNIVERSE,
        'universe_size': len(UNIVERSE),
        'starting_capital': 10000,
        'results': results,
        'gates': gate_results,
        'winner': winner_name,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow
    if MLFLOW_OK and winner_name:
        try:
            exp_name = 'factor_etf_ranker_v1'
            try:
                mlflow.create_experiment(exp_name)
            except:
                pass
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=f'v1_{winner_name}'):
                wm = results[winner_name]['metrics']
                wg = gate_results[winner_name]
                mlflow.log_metrics({
                    'sharpe': wm.get('sharpe', 0),
                    'sortino': wm.get('sortino', 0),
                    'cagr': wm.get('cagr', 0),
                    'maxdd': wm.get('maxdd', 0),
                    'wr': wm.get('wr', 0),
                    'pf': wm.get('pf', 0),
                    'calmar': wm.get('calmar', 0),
                    'perm_p': wg.get('permutation', {}).get('p_value', -1),
                    'r1_gap': wg.get('regime_r1', {}).get('gap', -1),
                    'n_pass': sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                                  if wg.get(k, {}).get('pass') is True),
                })
                mlflow.log_params({
                    'winner': winner_name,
                    'universe_size': len(UNIVERSE),
                    'n_variants': len(variants),
                })
                fprint(f"MLflow logged (exp: {exp_name})")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    fprint(f"\n{'='*70}")
    fprint("DONE")
    fprint(f"{'='*70}")

    return output


if __name__ == '__main__':
    main()
