#!/usr/bin/env python3
"""
Carry/Yield Strategy Research
==============================
Exploits yield differentials and carry across asset classes:
1. Bond carry: yield curve slope → long bonds when steep, short when flat/inverted
2. Dividend carry: high div yield stocks outperform when spreads are tight
3. FX carry proxy: UUP (dollar) vs EEM (EM) based on rate differentials
4. Commodity carry: contango/backwardation in commodity curves (via ETFs)

These are fundamentally different from momentum — they harvest risk premia,
not trend-following alpha.

HC #713: Fixed capital, no DCA
HC #714: Income + growth focus, ML/AI for exploration
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/carry_yield')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000


def download_data():
    """Download ETFs representing different carry sources"""
    tickers = {
        # Core
        'SPY': 'US Equities',
        'TLT': 'Long Bonds (20y+)',
        'IEF': 'Intermediate Bonds (7-10y)',
        'SHY': 'Short Bonds (1-3y)',
        'TIP': 'TIPS (inflation-linked)',
        # Carry sources
        'DVY': 'High Dividend (iShares)',
        'VYM': 'High Dividend (Vanguard)',
        'HYG': 'High Yield Corporate',
        'LQD': 'Investment Grade Corporate',
        'EMB': 'EM Bonds',
        # FX/Commodity carry proxies
        'UUP': 'US Dollar Bull',
        'EEM': 'Emerging Markets',
        'GLD': 'Gold',
        'DBC': 'Commodities Broad',
        # Leverage
        'UPRO': '3x SPY',
    }

    prices = yf.download(list(tickers.keys()), start='2010-01-01', progress=False)['Close']
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    vix = yf.download('^VIX', start='2010-01-01', progress=False)['Close']
    prices['VIX'] = vix

    # Treasury yields as carry signals
    tnx = yf.download('^TNX', start='2010-01-01', progress=False)['Close']  # 10Y yield
    irx = yf.download('^IRX', start='2010-01-01', progress=False)['Close']  # 13-week T-bill
    prices['TNX'] = tnx
    prices['IRX'] = irx

    prices = prices.dropna()
    print(f"[DATA] {len(prices)} rows, {len(prices.columns)} columns")
    print(f"[DATA] Range: {prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}")
    return prices


def compute_carry_features(prices):
    """Build carry-related features"""
    features = pd.DataFrame(index=prices.index)

    # 1. Yield curve slope (10Y - 3M) — steep = more carry in bonds
    if 'TNX' in prices.columns and 'IRX' in prices.columns:
        features['yield_curve_slope'] = prices['TNX'] - prices['IRX']
        features['yield_curve_slope_z'] = (
            (features['yield_curve_slope'] - features['yield_curve_slope'].rolling(252).mean()) /
            features['yield_curve_slope'].rolling(252).std()
        )

    # 2. Credit spread proxy (HYG yield - IEF yield via return differential)
    if 'HYG' in prices.columns and 'IEF' in prices.columns:
        hyg_ret = prices['HYG'].pct_change(21)
        ief_ret = prices['IEF'].pct_change(21)
        features['credit_spread_proxy'] = ief_ret - hyg_ret  # Wider spread when HYG underperforms
        features['credit_spread_z'] = (
            (features['credit_spread_proxy'] - features['credit_spread_proxy'].rolling(252).mean()) /
            features['credit_spread_proxy'].rolling(252).std()
        )

    # 3. Dividend carry: DVY vs SPY relative strength
    if 'DVY' in prices.columns:
        features['div_carry'] = (prices['DVY'] / prices['SPY']).pct_change(63)  # 3-month relative

    # 4. EM carry: EEM vs SPY (FX + equity carry differential)
    if 'EEM' in prices.columns:
        features['em_carry'] = (prices['EEM'] / prices['SPY']).pct_change(63)

    # 5. Real rate proxy: TIP vs IEF
    if 'TIP' in prices.columns and 'IEF' in prices.columns:
        features['real_rate_proxy'] = (prices['TIP'] / prices['IEF']).pct_change(63)

    # 6. Commodity carry proxy: DBC momentum (contango = negative carry)
    if 'DBC' in prices.columns:
        features['commodity_carry'] = prices['DBC'].pct_change(63)

    # 7. Dollar strength (carry flow indicator)
    if 'UUP' in prices.columns:
        features['dollar_momentum'] = prices['UUP'].pct_change(63)

    # 8. VIX level (risk premium)
    features['vix_level'] = prices['VIX']
    features['vix_percentile'] = prices['VIX'].rolling(252).rank(pct=True)

    # 9. Bond momentum signals
    features['tlt_mom_1m'] = prices['TLT'].pct_change(21)
    features['tlt_mom_3m'] = prices['TLT'].pct_change(63)

    # 10. SPY momentum (context)
    features['spy_mom_1m'] = prices['SPY'].pct_change(21)
    features['spy_mom_3m'] = prices['SPY'].pct_change(63)

    # 11. Volatility features
    spy_ret = prices['SPY'].pct_change()
    features['realized_vol_20d'] = spy_ret.rolling(20).std() * np.sqrt(252)
    features['vol_of_vol'] = features['realized_vol_20d'].rolling(63).std()

    return features.dropna()


def strategy_yield_curve_carry(prices, features):
    """
    Strategy 1: Yield Curve Carry
    When yield curve is steep (>1.5 z-score): overweight long bonds (TLT)
    When yield curve is flat/inverted: overweight short bonds + gold
    """
    start_idx = 300
    equity = np.ones(len(prices)) * INITIAL_CAPITAL

    spy_ret = prices['SPY'].pct_change()
    tlt_ret = prices['TLT'].pct_change()
    shy_ret = prices['SHY'].pct_change()
    gld_ret = prices['GLD'].pct_change()

    for i in range(start_idx, len(prices)):
        date = prices.index[i]
        if date not in features.index:
            equity[i] = equity[i-1] * (1 + spy_ret.iloc[i] * 0.6 + shy_ret.iloc[i] * 0.4)
            continue

        slope_z = features.loc[date, 'yield_curve_slope_z']

        if slope_z > 1.0:
            # Steep curve: bonds will rally as curve normalizes
            port_ret = 0.40 * spy_ret.iloc[i] + 0.40 * tlt_ret.iloc[i] + 0.20 * shy_ret.iloc[i]
        elif slope_z < -0.5:
            # Flat/inverted: recession risk, go defensive
            port_ret = 0.20 * spy_ret.iloc[i] + 0.20 * gld_ret.iloc[i] + 0.60 * shy_ret.iloc[i]
        else:
            # Normal: balanced
            port_ret = 0.50 * spy_ret.iloc[i] + 0.30 * tlt_ret.iloc[i] + 0.20 * shy_ret.iloc[i]

        equity[i] = equity[i-1] * (1 + port_ret)

    return pd.Series(equity[start_idx:], index=prices.index[start_idx:])


def strategy_credit_carry(prices, features):
    """
    Strategy 2: Credit Carry
    Harvest credit spread: long HYG when spreads are wide (high carry)
    Reduce when spreads are tight (low carry, high risk)
    """
    start_idx = 300
    equity = np.ones(len(prices)) * INITIAL_CAPITAL

    spy_ret = prices['SPY'].pct_change()
    hyg_ret = prices['HYG'].pct_change()
    lqd_ret = prices['LQD'].pct_change() if 'LQD' in prices.columns else spy_ret * 0
    shy_ret = prices['SHY'].pct_change()

    for i in range(start_idx, len(prices)):
        date = prices.index[i]
        if date not in features.index:
            equity[i] = equity[i-1] * (1 + shy_ret.iloc[i])
            continue

        credit_z = features.loc[date, 'credit_spread_z']

        if credit_z > 1.0:
            # Wide spreads: high carry, buy HY
            port_ret = 0.30 * spy_ret.iloc[i] + 0.50 * hyg_ret.iloc[i] + 0.20 * shy_ret.iloc[i]
        elif credit_z < -0.5:
            # Tight spreads: low carry, risk of blowout
            port_ret = 0.30 * spy_ret.iloc[i] + 0.10 * hyg_ret.iloc[i] + 0.60 * shy_ret.iloc[i]
        else:
            # Normal
            port_ret = 0.40 * spy_ret.iloc[i] + 0.30 * hyg_ret.iloc[i] + 0.30 * shy_ret.iloc[i]

        equity[i] = equity[i-1] * (1 + port_ret)

    return pd.Series(equity[start_idx:], index=prices.index[start_idx:])


def strategy_ml_carry_timing(prices, features):
    """
    Strategy 3: ML Carry Timing
    Use GBM to predict which carry source outperforms next month
    Walk-forward: 12m train, 1m test, sliding
    """
    # Define carry streams and their returns
    carry_assets = ['SPY', 'TLT', 'HYG', 'GLD', 'DVY']
    available = [a for a in carry_assets if a in prices.columns]

    monthly_prices = prices[available].resample('ME').last().dropna()
    monthly_returns = monthly_prices.pct_change().dropna()

    # Monthly features
    monthly_features = features.resample('ME').last().dropna()

    # Align
    common_dates = monthly_returns.index.intersection(monthly_features.index)
    monthly_returns = monthly_returns.loc[common_dates]
    monthly_features = monthly_features.loc[common_dates]

    # Target: which asset has highest return next month
    target = monthly_returns.shift(-1).idxmax(axis=1).dropna()

    # Align again
    common = target.index.intersection(monthly_features.index)
    target = target.loc[common]
    X = monthly_features.loc[common]

    # Walk-forward
    train_months = 12
    predictions = []

    feat_cols = [c for c in X.columns if X[c].notna().sum() > train_months + 5]
    X = X[feat_cols].fillna(0)

    for i in range(train_months, len(X) - 1):
        X_train = X.iloc[:i]
        y_train = target.iloc[:i]

        # Encode labels
        label_map = {a: j for j, a in enumerate(available)}
        y_encoded = y_train.map(label_map)

        # Remove NaN labels
        valid = y_encoded.notna()
        X_train = X_train[valid]
        y_encoded = y_encoded[valid].astype(int)

        if len(X_train) < 6:
            predictions.append({'date': X.index[i], 'pred': 'SPY'})
            continue

        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42
        )
        try:
            model.fit(X_train, y_encoded)
            pred_class = model.predict(X.iloc[i:i+1])[0]
            inv_map = {v: k for k, v in label_map.items()}
            pred_asset = inv_map[pred_class]
        except:
            pred_asset = 'SPY'

        predictions.append({'date': X.index[i], 'pred': pred_asset})

    pred_df = pd.DataFrame(predictions).set_index('date')

    # Build equity curve from predictions
    start_idx = 300
    daily_prices_cut = prices.iloc[start_idx:]
    equity = np.ones(len(daily_prices_cut)) * INITIAL_CAPITAL
    current_asset = 'SPY'

    for i in range(1, len(daily_prices_cut)):
        date = daily_prices_cut.index[i]

        # Check if we have a new monthly prediction
        monthly_preds_before = pred_df[pred_df.index <= date]
        if len(monthly_preds_before) > 0:
            current_asset = monthly_preds_before.iloc[-1]['pred']

        if current_asset in prices.columns:
            asset_ret = prices[current_asset].pct_change()
            if date in asset_ret.index:
                ret = asset_ret.loc[date]
                if np.isfinite(ret):
                    equity[i] = equity[i-1] * (1 + ret)
                    continue

        equity[i] = equity[i-1]

    return pd.Series(equity, index=daily_prices_cut.index)


def strategy_multi_carry_blend(prices, features):
    """
    Strategy 4: Multi-Carry Blend
    Equal-weight across carry signals, dynamically rebalanced monthly.
    Long assets with positive carry, short (or zero) assets with negative carry.
    """
    start_idx = 300
    equity = np.ones(len(prices)) * INITIAL_CAPITAL

    # Monthly rebalance
    carry_signals = {}
    spy_ret = prices['SPY'].pct_change()
    tlt_ret = prices['TLT'].pct_change()
    hyg_ret = prices['HYG'].pct_change()
    gld_ret = prices['GLD'].pct_change()
    shy_ret = prices['SHY'].pct_change()

    last_rebal = None

    weights = {'SPY': 0.25, 'TLT': 0.25, 'HYG': 0.25, 'GLD': 0.25}

    for i in range(start_idx, len(prices)):
        date = prices.index[i]

        # Monthly rebalance
        if last_rebal is None or date.month != last_rebal.month:
            if date in features.index:
                f = features.loc[date]

                scores = {}
                # SPY: buy when vol low, momentum positive
                scores['SPY'] = 1.0 if f.get('spy_mom_3m', 0) > 0 else 0.3

                # TLT: buy when yield curve steep (bonds have carry)
                scores['TLT'] = 1.0 if f.get('yield_curve_slope_z', 0) > 0.5 else 0.3

                # HYG: buy when credit spreads wide (high carry)
                scores['HYG'] = 1.0 if f.get('credit_spread_z', 0) > 0 else 0.3

                # GLD: buy when real rates falling / dollar weak
                scores['GLD'] = 1.0 if f.get('dollar_momentum', 0) < 0 else 0.3

                total = sum(scores.values())
                weights = {k: v/total for k, v in scores.items()}
                last_rebal = date

        ret_map = {
            'SPY': spy_ret.iloc[i],
            'TLT': tlt_ret.iloc[i],
            'HYG': hyg_ret.iloc[i],
            'GLD': gld_ret.iloc[i],
        }

        port_ret = sum(weights.get(k, 0) * v for k, v in ret_map.items() if np.isfinite(v))
        equity[i] = equity[i-1] * (1 + port_ret)

    return pd.Series(equity[start_idx:], index=prices.index[start_idx:])


def compute_metrics(equity_series, name="Strategy"):
    """Standard risk-adjusted metrics"""
    returns = equity_series.pct_change().dropna()
    if len(returns) < 50:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': 0,
                'calmar': 0, 'win_rate': 0, 'final_equity': equity_series.iloc[-1]}

    ann_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0
    cummax = equity_series.cummax()
    max_dd = ((equity_series - cummax) / cummax).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    wr = (returns > 0).mean()

    return {
        'name': name, 'sharpe': sharpe, 'sortino': sortino, 'cagr': ann_ret,
        'max_dd': max_dd, 'calmar': calmar, 'win_rate': wr,
        'final_equity': equity_series.iloc[-1]
    }


def permutation_test(equity_series, prices, n_perms=200):
    """Permutation test: shuffle carry signals"""
    real = compute_metrics(equity_series)
    real_sharpe = real['sharpe']

    # Simple benchmark: buy SPY for same period
    spy_eq = prices['SPY'].reindex(equity_series.index)
    spy_eq = spy_eq / spy_eq.iloc[0] * INITIAL_CAPITAL

    # Shuffle daily returns and rebuild equity
    returns = equity_series.pct_change().dropna()
    perm_sharpes = []

    for _ in range(n_perms):
        # Shuffle returns (breaks temporal structure = no signal)
        shuffled = returns.sample(frac=1.0, replace=False).values
        eq = np.cumprod(1 + shuffled) * INITIAL_CAPITAL
        eq_series = pd.Series(eq, index=returns.index)
        pm = compute_metrics(eq_series, "perm")
        perm_sharpes.append(pm['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    return {
        'real_sharpe': real_sharpe,
        'perm_mean': np.mean(perm_sharpes),
        'perm_std': np.std(perm_sharpes),
        'p_value': p_value,
        'pass': p_value < 0.05
    }


def regime_analysis(equity_series, spy_prices):
    """R1 regime test"""
    returns = equity_series.pct_change().dropna()
    spy_ret = spy_prices.reindex(returns.index).pct_change().dropna()
    common = returns.index.intersection(spy_ret.index)
    returns = returns.loc[common]
    spy_ret = spy_ret.loc[common]

    green = spy_ret > 0
    red = spy_ret <= 0

    g_sharpe = returns[green].mean() / returns[green].std() * np.sqrt(252) if green.sum() > 50 else 0
    r_sharpe = returns[red].mean() / returns[red].std() * np.sqrt(252) if red.sum() > 50 else 0
    gap = abs(g_sharpe - r_sharpe) / max(abs(g_sharpe), abs(r_sharpe), 0.01)

    return {'green_sharpe': g_sharpe, 'red_sharpe': r_sharpe, 'gap': gap, 'pass': gap <= 0.50}


def main():
    print("=" * 80)
    print("CARRY/YIELD STRATEGY RESEARCH")
    print("Harvesting risk premia across asset classes")
    print("=" * 80)

    print("\n[STEP 1] Downloading data...")
    prices = download_data()

    print("\n[STEP 2] Building carry features...")
    features = compute_carry_features(prices)
    print(f"  Features: {len(features.columns)}, rows: {len(features)}")

    # Build SPY benchmark
    start_idx = 300
    spy_bench = prices['SPY'].iloc[start_idx:] / prices['SPY'].iloc[start_idx] * INITIAL_CAPITAL

    strategies = {
        'SPY B&H': spy_bench,
    }

    print("\n[STEP 3] Running strategies...")

    print("\n  [3a] Yield Curve Carry...")
    strategies['Yield Curve Carry'] = strategy_yield_curve_carry(prices, features)

    print("  [3b] Credit Carry...")
    strategies['Credit Carry'] = strategy_credit_carry(prices, features)

    print("  [3c] ML Carry Timing...")
    strategies['ML Carry Timing'] = strategy_ml_carry_timing(prices, features)

    print("  [3d] Multi-Carry Blend...")
    strategies['Multi-Carry Blend'] = strategy_multi_carry_blend(prices, features)

    # Results
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    all_metrics = []
    for name, eq in strategies.items():
        m = compute_metrics(eq, name)
        all_metrics.append(m)

    print(f"\n  {'Strategy':<25s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s} {'Calmar':>7s}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
    for m in all_metrics:
        print(f"  {m['name']:<25s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%} {m['calmar']:>7.3f}")

    # Find best strategy (highest Sharpe that beats SPY)
    spy_metrics = [m for m in all_metrics if m['name'] == 'SPY B&H'][0]
    best = max([m for m in all_metrics if m['name'] != 'SPY B&H'],
               key=lambda x: x['sharpe'])

    print(f"\n  BEST: {best['name']} (Sharpe {best['sharpe']:.3f} vs SPY {spy_metrics['sharpe']:.3f})")

    # Adversarial on best
    print("\n" + "=" * 80)
    print(f"ADVERSARIAL VALIDATION: {best['name']}")
    print("=" * 80)

    print(f"\n  [1/3] Permutation test (200 shuffles)...")
    perm = permutation_test(strategies[best['name']], prices, n_perms=200)
    print(f"    Real Sharpe: {perm['real_sharpe']:.3f}")
    print(f"    Perm mean: {perm['perm_mean']:.3f} ± {perm['perm_std']:.3f}")
    print(f"    p-value: {perm['p_value']:.3f} → {'PASS' if perm['pass'] else 'FAIL'}")

    print(f"\n  [2/3] R1 regime test...")
    regime = regime_analysis(strategies[best['name']], prices['SPY'])
    print(f"    Green Sharpe: {regime['green_sharpe']:.3f}")
    print(f"    Red Sharpe: {regime['red_sharpe']:.3f}")
    print(f"    Gap: {regime['gap']:.3f} → {'PASS' if regime['pass'] else 'FAIL'}")

    # Sub-period
    print(f"\n  [3/3] Sub-period consistency...")
    eq = strategies[best['name']]
    returns = eq.pct_change().dropna()
    block_size = len(returns) // 4
    block_sharpes = []
    for b in range(4):
        s = b * block_size
        e = (b+1) * block_size if b < 3 else len(returns)
        br = returns.iloc[s:e]
        bs = br.mean() / br.std() * np.sqrt(252) if br.std() > 0 else 0
        block_sharpes.append(bs)
        print(f"    Block {b+1}: Sharpe {bs:.3f}")
    cv = np.std(block_sharpes) / max(np.mean(block_sharpes), 0.01)
    print(f"    CV: {cv:.3f} → {'PASS' if cv < 0.50 else 'FAIL'}")

    gates = sum([perm['pass'], regime['pass'], cv < 0.50])
    print(f"\n  ADVERSARIAL SUMMARY: {gates}/3 gates → {'PASS' if gates >= 2 else 'FAIL'}")

    # Save
    pd.DataFrame(all_metrics).to_csv(OUTPUT_DIR / 'carry_results.csv', index=False)

    print("\n" + "=" * 80)
    print(f"COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
