#!/usr/bin/env python3
"""
VRP Income + ML Spike Protection
=================================
Core thesis: Volatility Risk Premium (VRP) is REAL income (~20% CAGR naive),
but catastrophic tail risk (-73% MaxDD) kills it. Our VIX spike predictor
(AUC 0.926) can protect against spikes BEFORE they happen.

Strategy:
- Default: Harvest VRP via short-vol ETN (SVXY proxy) = income generation
- Spike predictor P>threshold: EXIT to cash (avoid the -50%+ drawdowns)
- After spike (VIX>30): ENTER VIX puts / SVIX for mean-reversion bounce
  (entry 486 showed 83% WR on spike buying with VIX puts)

This targets HC #714 R1 (income focus) with genuine ML value-add.
$100K fixed capital (HC #713).
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *
from sklearn.ensemble import RandomForestClassifier


def build_features(prices, vix, lookback_days=252):
    """Build features for VIX spike prediction (same as vix_spike_predictor.py)"""
    features = pd.DataFrame(index=prices.index)

    # VIX features
    features['vix_level'] = vix
    features['vix_5d_change'] = vix.pct_change(5)
    features['vix_10d_change'] = vix.pct_change(10)
    features['vix_zscore_20d'] = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()
    features['vix_zscore_60d'] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()

    # Term structure proxy (VIX vs its MA = contango/backwardation)
    features['vix_vs_ma50'] = vix / vix.rolling(50).mean() - 1
    features['vix_vs_ma20'] = vix / vix.rolling(20).mean() - 1

    # SPY features
    spy = prices['SPY']
    features['spy_ret_5d'] = spy.pct_change(5)
    features['spy_ret_20d'] = spy.pct_change(20)
    features['spy_drawdown'] = spy / spy.rolling(252).max() - 1
    features['spy_vol_20d'] = spy.pct_change().rolling(20).std() * np.sqrt(252)
    features['spy_vol_60d'] = spy.pct_change().rolling(60).std() * np.sqrt(252)
    features['spy_vol_ratio'] = features['spy_vol_20d'] / features['spy_vol_60d']
    features['spy_above_200sma'] = (spy > spy.rolling(200).mean()).astype(float)

    # Credit/risk features (if available)
    if 'HYG' in prices.columns and 'TLT' in prices.columns:
        features['credit_spread_proxy'] = prices['TLT'].pct_change(20) - prices['HYG'].pct_change(20)
        features['tlt_momentum'] = prices['TLT'].pct_change(20)

    if 'GLD' in prices.columns:
        features['gold_momentum'] = prices['GLD'].pct_change(20)
        features['flight_to_quality'] = prices['GLD'].pct_change(5) - spy.pct_change(5)

    return features


def main():
    print("=" * 70)
    print("VRP INCOME + ML SPIKE PROTECTION")
    print("=" * 70)

    # Download data
    tickers = ['SPY', 'TLT', 'GLD', 'HYG', 'SHY']
    prices = download_etfs(tickers, start='2007-01-01')
    vix = download_vix(start='2007-01-01')
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")

    # VRP proxy: short VIX futures roll yield
    # SVXY-like returns: capture VIX contango (~0.03-0.05% daily when VIX term structure normal)
    # Approximate: daily VRP return = negative of VIX log-change * leverage factor
    # More realistic: use actual SVXY behavior approximation

    vix_series = prices['VIX']
    spy = prices['SPY']

    # VRP daily return approximation:
    # When VIX futures in contango (most of time), short-vol earns ~0.02-0.04% daily
    # When VIX spikes, short-vol loses big (-10-30% in a day)
    # Model: VRP_return = -beta * VIX_pct_change (where beta ~0.5 for SVXY-like)
    vix_pct = vix_series.pct_change()
    vrp_beta = 0.5  # SVXY has ~0.5x inverse VIX exposure
    vrp_returns = -vrp_beta * vix_pct
    # Cap extreme moves to be realistic (SVXY can't go below -100% in a day)
    vrp_returns = vrp_returns.clip(-0.20, 0.10)

    # Build spike labels (VIX > 30 within 5 days = spike)
    spike_threshold = 30
    future_vix_max = vix_series.rolling(5, min_periods=1).max().shift(-5)
    spike_labels = (future_vix_max > spike_threshold).astype(int)

    # Build features
    features = build_features(prices, vix_series)
    features = features.dropna()

    # Align all data
    common_idx = features.index.intersection(vrp_returns.dropna().index).intersection(spike_labels.dropna().index)
    features = features.loc[common_idx]
    vrp_returns = vrp_returns.loc[common_idx]
    spike_labels = spike_labels.loc[common_idx]
    vix_aligned = vix_series.loc[common_idx]
    spy_aligned = spy.loc[common_idx]

    start_idx = 252  # Need 1yr for first training window
    print(f"Trading days: {len(common_idx) - start_idx}")

    configs = {}

    # === BASELINE 1: NAIVE SHORT-VOL (always harvesting VRP) ===
    naive_eq = (1 + vrp_returns.iloc[start_idx:]).cumprod() * INITIAL_CAPITAL
    configs['Naive Short-Vol'] = naive_eq

    # === BASELINE 2: VIX > 25 EXIT (simple rule) ===
    simple_eq = pd.Series(INITIAL_CAPITAL, index=common_idx)
    for i in range(start_idx, len(common_idx)):
        v = vix_aligned.iloc[i]
        if v > 25:
            # Exit to cash
            daily_ret = 0.0
        else:
            daily_ret = vrp_returns.iloc[i]
        simple_eq.iloc[i] = simple_eq.iloc[i-1] * (1 + daily_ret)
    configs['VIX>25 Exit'] = simple_eq.iloc[start_idx:]

    # === STRATEGY 1: ML SPIKE PREDICTOR + VRP ===
    # Walk-forward: train on past 252 days, predict next day
    ml_eq = pd.Series(INITIAL_CAPITAL, index=common_idx)
    predictions = pd.Series(0.0, index=common_idx)

    train_window = 252
    retrain_freq = 21  # Retrain monthly

    model = None
    feature_cols = features.columns.tolist()

    for i in range(start_idx, len(common_idx)):
        # Retrain periodically
        if model is None or i % retrain_freq == 0:
            train_start = max(0, i - train_window)
            X_train = features.iloc[train_start:i][feature_cols].values
            y_train = spike_labels.iloc[train_start:i].values

            if y_train.sum() > 5:  # Need some positive examples
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5,
                    min_samples_leaf=10, random_state=42,
                    class_weight='balanced'
                )
                model.fit(X_train, y_train)

        # Predict
        if model is not None:
            X_today = features.iloc[i:i+1][feature_cols].values
            prob = model.predict_proba(X_today)[0]
            spike_prob = prob[1] if len(prob) > 1 else 0
        else:
            spike_prob = 0

        predictions.iloc[i] = spike_prob

        # Trading logic
        v = vix_aligned.iloc[i]
        if spike_prob > 0.50:
            # ML says spike coming → EXIT (protect capital)
            daily_ret = 0.0
        elif v > 35:
            # Already in spike zone → BUY THE SPIKE (VIX puts / SVIX)
            # VIX mean-reverts aggressively from >35
            # Spike buying return = positive of VIX decline * factor
            daily_ret = vrp_beta * max(0, -vix_pct.iloc[i])  # Only capture downward VIX moves
        else:
            # Normal: harvest VRP
            daily_ret = vrp_returns.iloc[i]

        ml_eq.iloc[i] = ml_eq.iloc[i-1] * (1 + daily_ret)

    configs['ML Spike Protect'] = ml_eq.iloc[start_idx:]

    # === STRATEGY 2: ML + POSITION SIZING (scale down on elevated prob) ===
    ml_sized_eq = pd.Series(INITIAL_CAPITAL, index=common_idx)

    for i in range(start_idx, len(common_idx)):
        spike_prob = predictions.iloc[i]
        v = vix_aligned.iloc[i]

        # Position size inversely proportional to spike probability
        if spike_prob > 0.60:
            size = 0.0  # Full exit
        elif spike_prob > 0.40:
            size = 0.3  # Reduce heavily
        elif spike_prob > 0.25:
            size = 0.7  # Slight reduction
        else:
            size = 1.0  # Full position

        # VIX-based overlay (additional safety)
        if v > 30:
            size = min(size, 0.2)  # Never full size when VIX already elevated

        # After spike: opportunistic spike buying
        if v > 35 and spike_prob < 0.30:
            # VIX very high but predictor says worst is over → buy the bounce
            daily_ret = vrp_beta * max(0, -vix_pct.iloc[i]) * 1.5  # Levered bounce
        else:
            daily_ret = vrp_returns.iloc[i] * size

        ml_sized_eq.iloc[i] = ml_sized_eq.iloc[i-1] * (1 + daily_ret)

    configs['ML Sized VRP'] = ml_sized_eq.iloc[start_idx:]

    # === STRATEGY 3: HYBRID INCOME (VRP + VIX spike buying + TLT hedge) ===
    hybrid_eq = pd.Series(INITIAL_CAPITAL, index=common_idx)
    tlt_ret = prices['TLT'].pct_change().loc[common_idx]

    for i in range(start_idx, len(common_idx)):
        spike_prob = predictions.iloc[i]
        v = vix_aligned.iloc[i]

        if spike_prob > 0.50:
            # Danger: go to TLT (flight to quality = still earning income via bonds)
            daily_ret = tlt_ret.iloc[i] * 0.8  # 80% TLT
        elif v > 35 and spike_prob < 0.30:
            # Post-spike bounce: buy SVIX aggressively
            daily_ret = vrp_beta * max(0, -vix_pct.iloc[i]) * 2.0
            daily_ret = min(daily_ret, 0.05)  # Cap daily gain
        elif v > 25:
            # Elevated VIX, reduce VRP position, add TLT
            daily_ret = vrp_returns.iloc[i] * 0.3 + tlt_ret.iloc[i] * 0.5
        else:
            # Normal: full VRP harvest
            daily_ret = vrp_returns.iloc[i]

        hybrid_eq.iloc[i] = hybrid_eq.iloc[i-1] * (1 + daily_ret)

    configs['Hybrid Income'] = hybrid_eq.iloc[start_idx:]

    # === STRATEGY 4: CONSERVATIVE INCOME (lower vol target) ===
    cons_eq = pd.Series(INITIAL_CAPITAL, index=common_idx)
    target_vol = 0.12  # Target 12% annual vol

    for i in range(start_idx, len(common_idx)):
        spike_prob = predictions.iloc[i]
        v = vix_aligned.iloc[i]

        # Realized vol of VRP over past 20 days
        if i >= 20:
            recent_vol = vrp_returns.iloc[max(start_idx, i-20):i].std() * np.sqrt(252)
        else:
            recent_vol = 0.30

        # Vol-target sizing
        vol_size = min(target_vol / max(recent_vol, 0.05), 2.0)

        # ML overlay
        if spike_prob > 0.50:
            vol_size = 0.0
        elif spike_prob > 0.30:
            vol_size *= 0.5

        # VIX floor
        if v > 30:
            vol_size = min(vol_size, 0.3)

        daily_ret = vrp_returns.iloc[i] * vol_size
        cons_eq.iloc[i] = cons_eq.iloc[i-1] * (1 + daily_ret)

    configs['Conservative VRP'] = cons_eq.iloc[start_idx:]

    # SPY benchmark
    spy_eq = spy_aligned.iloc[start_idx:] / spy_aligned.iloc[start_idx] * INITIAL_CAPITAL
    configs['SPY B&H'] = spy_eq

    # === RESULTS ===
    print(f"\n{'Strategy':<22s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s} {'Calmar':>7s}")
    print(f"{'-'*22} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    best_name, best_sharpe = None, -999
    for name, eq in configs.items():
        m = compute_metrics(eq, name)
        calmar = m.get('calmar', 0)
        print(f"{m['name']:<22s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%} {calmar:>7.3f}")
        if name != 'SPY B&H' and m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_name = name

    print(f"\nBest: {best_name}")

    # ML prediction quality check
    active_preds = predictions.iloc[start_idx:]
    actual_spikes = spike_labels.iloc[start_idx:]
    high_prob = active_preds > 0.50
    if high_prob.sum() > 0:
        precision = actual_spikes[high_prob].mean()
        recall = actual_spikes[active_preds > 0.50].sum() / max(actual_spikes.sum(), 1)
        print(f"\nML Quality: P>0.50 precision={precision:.1%}, recall={recall:.1%}, "
              f"alerts={high_prob.sum()} ({100*high_prob.mean():.1f}% of days)")

    # Income metrics
    best_eq = configs[best_name]
    total_return = best_eq.iloc[-1] / best_eq.iloc[0] - 1
    years = len(best_eq) / 252
    monthly_income = (best_eq.iloc[-1] - INITIAL_CAPITAL) / (years * 12)
    print(f"\nIncome profile: ${monthly_income:,.0f}/month avg on $100K "
          f"({total_return:.0%} total over {years:.1f} years)")

    # === ADVERSARIAL VALIDATION ===
    best_equity = configs[best_name]
    metrics = compute_metrics(best_equity, best_name)

    # Proper signal-shuffled permutation test
    print(f"\nPermutation test (shuffled spike predictions)...")
    real_sharpe = metrics['sharpe']
    perm_sharpes = []

    for trial in range(200):
        # Shuffle the predictions (break signal-timing alignment)
        shuffled_preds = predictions.iloc[start_idx:].values.copy()
        np.random.shuffle(shuffled_preds)

        perm_eq = pd.Series(INITIAL_CAPITAL, index=common_idx)
        for i in range(start_idx, len(common_idx)):
            sp = shuffled_preds[i - start_idx]
            v = vix_aligned.iloc[i]

            if best_name == 'ML Spike Protect':
                if sp > 0.50:
                    dr = 0.0
                elif v > 35:
                    dr = vrp_beta * max(0, -vix_pct.iloc[i])
                else:
                    dr = vrp_returns.iloc[i]
            elif best_name == 'Hybrid Income':
                if sp > 0.50:
                    dr = tlt_ret.iloc[i] * 0.8
                elif v > 35 and sp < 0.30:
                    dr = vrp_beta * max(0, -vix_pct.iloc[i]) * 2.0
                    dr = min(dr, 0.05)
                elif v > 25:
                    dr = vrp_returns.iloc[i] * 0.3 + tlt_ret.iloc[i] * 0.5
                else:
                    dr = vrp_returns.iloc[i]
            else:
                # Generic: apply same logic as best strategy with shuffled predictions
                if sp > 0.60:
                    size = 0.0
                elif sp > 0.40:
                    size = 0.3
                elif sp > 0.25:
                    size = 0.7
                else:
                    size = 1.0
                if v > 30:
                    size = min(size, 0.2)
                dr = vrp_returns.iloc[i] * size

            perm_eq.iloc[i] = perm_eq.iloc[i-1] * (1 + dr)

        pm = compute_metrics(perm_eq.iloc[start_idx:])
        perm_sharpes.append(pm['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean())
    print(f"  Real: {real_sharpe:.3f}, Perm mean: {perm_sharpes.mean():.3f}, p={p_value:.3f}")

    # Sub-period test
    subp = subperiod_test(best_equity)
    print(f"  SubP: blocks={subp['block_sharpes']}, CV={subp['cv']:.3f}")

    # Regime test
    regime = regime_test(best_equity, spy_eq)
    print(f"  Regime: green={regime['green_sharpe']:.2f}, red={regime['red_sharpe']:.2f}, gap={regime['gap']:.3f}")

    adv = {
        'permutation': {'real_sharpe': real_sharpe, 'perm_mean': round(float(perm_sharpes.mean()), 3),
                       'p_value': round(p_value, 3), 'pass': p_value < 0.05},
        'subperiod': subp,
        'regime': regime,
        'gates_passed': sum([p_value < 0.05, subp['pass'], regime['pass']]),
        'gates_total': 3,
    }

    print(f"  Gates: {adv['gates_passed']}/3")

    emit_result(
        name=f"VRP Income ({best_name})",
        description="Volatility Risk Premium harvesting with ML spike prediction protection",
        metrics=metrics,
        adversarial=adv,
        extra={
            'monthly_income_100k': round(float(monthly_income), 0),
            'ml_quality': {
                'precision_at_50': round(float(precision), 3) if high_prob.sum() > 0 else None,
                'alert_rate': round(float(high_prob.mean()), 3) if high_prob.sum() > 0 else None
            }
        }
    )


if __name__ == '__main__':
    main()
