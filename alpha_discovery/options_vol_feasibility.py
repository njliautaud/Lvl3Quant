#!/usr/bin/env python3
"""
Options Vol Trading Feasibility Analysis

Tests whether we can monetize our vol prediction (IC=0.67 at 1s) via options:

1. Compute daily realized vol from our ES MBO data (100 days)
2. Download VIX data as IV proxy (free from Yahoo Finance)
3. Measure variance risk premium (VRP = IV - RV)
4. Test whether our intraday features predict next-day realized vol
5. Estimate edge from a vol trading strategy

Key question: Can we predict daily/intraday realized vol BETTER than
the options market (i.e., VIX), and is the edge tradable?

Usage:
  python options_vol_feasibility.py
"""

import gc
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    format='%(asctime)s [vol_feas] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger('vol_feas')

MBO_DIR = ROOT_DIR / 'data' / 'processed' / 'mbo_features_cache'
RESULTS_DIR = ROOT_DIR / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ES constants
TICK_SIZE = 0.25
BAR_INTERVAL_S = 0.01  # 10ms bars in MBO data


def load_mid_prices(fpath):
    """Load mid prices from MBO cache."""
    data = np.load(str(fpath))
    raw = data['mbo_features']
    mid = raw[:, 0].copy()  # column 0 = mid price
    # Forward-fill NaN
    mask = np.isnan(mid)
    if mask.any():
        first_valid = np.argmax(~mask)
        if first_valid > 0:
            mid[:first_valid] = mid[first_valid]
        for i in range(1, len(mid)):
            if np.isnan(mid[i]):
                mid[i] = mid[i-1]
    return mid


def load_features_subsampled(fpath, subsample=1000):
    """Load features subsampled at 10s intervals."""
    data = np.load(str(fpath))
    raw = data['mbo_features']
    mid = raw[:, 0].copy()
    # Exclude price levels (0=mid, 3=microprice, 8=best_bid, 9=best_ask)
    exclude = [0, 3, 8, 9]
    keep = [i for i in range(raw.shape[1]) if i not in exclude]
    features = raw[:, keep]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    # Subsample
    indices = np.arange(0, len(features), subsample)
    return features[indices].astype(np.float32)


def compute_daily_rvol(mid):
    """Compute daily realized vol from 10ms mid prices.
    Returns: annualized realized vol (as percentage)
    """
    # Returns at 1-second intervals (every 100 bars)
    step = 100  # 1 second
    prices = mid[::step]
    prices = prices[prices > 0]
    returns = np.diff(np.log(prices))
    returns = returns[np.isfinite(returns)]
    if len(returns) < 100:
        return np.nan
    # Daily vol = std of 1s returns × sqrt(seconds_per_day)
    # Trading day ~6.5 hours = 23400 seconds
    daily_vol = np.std(returns) * np.sqrt(23400)
    # Annualize: × sqrt(252)
    annual_vol = daily_vol * np.sqrt(252)
    return annual_vol * 100  # as percentage


def compute_intraday_vol_windows(mid, window_seconds=3600):
    """Compute realized vol in rolling windows (for intraday analysis).
    Returns array of rvol values, one per window.
    """
    step = 100  # 1s
    prices = mid[::step]
    prices = prices[prices > 0]
    returns = np.diff(np.log(prices))
    returns = returns[np.isfinite(returns)]

    window_bars = window_seconds
    n_windows = len(returns) // window_bars
    vols = []
    for i in range(n_windows):
        chunk = returns[i*window_bars:(i+1)*window_bars]
        if len(chunk) > 10:
            vol = np.std(chunk) * np.sqrt(23400) * np.sqrt(252) * 100
            vols.append(vol)
    return np.array(vols)


def compute_end_of_day_features(mid, features_sub):
    """Compute end-of-day summary features for next-day vol prediction."""
    # 1. Today's realized vol
    rvol = compute_daily_rvol(mid)

    # 2. Intraday vol pattern (first half vs second half)
    half = len(mid) // 2
    rvol_first = compute_daily_rvol(mid[:half])
    rvol_second = compute_daily_rvol(mid[half:])
    vol_ratio = rvol_second / rvol_first if rvol_first > 0 else 1.0

    # 3. Daily return
    prices_1s = mid[::100]
    prices_1s = prices_1s[prices_1s > 0]
    daily_return = (prices_1s[-1] - prices_1s[0]) / prices_1s[0] if len(prices_1s) > 1 else 0

    # 4. Mean of last-hour features (proxy for end-of-day state)
    n_last_hour = min(360, len(features_sub))  # ~1 hour at 10s intervals
    eod_feats = np.mean(features_sub[-n_last_hour:], axis=0)

    # 5. Feature volatility (std of features over the day)
    feat_vol = np.std(features_sub, axis=0)

    # Combine
    summary = np.concatenate([
        [rvol, vol_ratio, abs(daily_return) * 100],
        eod_feats[:20],  # Top 20 mean features
        feat_vol[:20],   # Top 20 feature volatilities
    ])

    return summary, rvol


def download_vix_data(start_date, end_date):
    """Download VIX data from Yahoo Finance."""
    try:
        import requests
    except ImportError:
        log.error("requests not installed")
        return None

    # Use Yahoo Finance API
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()) + 86400

    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
           f"?period1={start_ts}&period2={end_ts}&interval=1d")

    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            log.warning(f"Yahoo Finance returned {r.status_code}")
            return None

        data = r.json()
        chart = data.get('chart', {}).get('result', [{}])[0]
        timestamps = chart.get('timestamp', [])
        closes = chart.get('indicators', {}).get('quote', [{}])[0].get('close', [])

        if not timestamps or not closes:
            log.warning("No VIX data returned")
            return None

        # Build date -> VIX close dict
        vix_data = {}
        for ts, close in zip(timestamps, closes):
            if close is not None:
                date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                vix_data[date] = close

        log.info(f"Downloaded VIX data: {len(vix_data)} days ({min(vix_data.keys())} to {max(vix_data.keys())})")
        return vix_data

    except Exception as e:
        log.warning(f"Failed to download VIX: {e}")
        return None


def main():
    log.info("=" * 70)
    log.info("OPTIONS VOL TRADING FEASIBILITY ANALYSIS")
    log.info("=" * 70)

    # 1. Load all days and compute daily realized vol
    files = sorted(MBO_DIR.glob('*_mbo_features.npz'))
    if not files:
        log.error(f"No MBO files in {MBO_DIR}")
        return

    log.info(f"Found {len(files)} MBO days")

    daily_data = []
    feature_summaries = []

    for i, fpath in enumerate(files):
        date_str = fpath.stem.replace('_mbo_features', '')
        try:
            mid = load_mid_prices(fpath)
            rvol = compute_daily_rvol(mid)

            # Also get intraday windows
            hourly_vols = compute_intraday_vol_windows(mid, window_seconds=3600)

            # Get features for prediction
            features_sub = load_features_subsampled(fpath, subsample=1000)
            summary, _ = compute_end_of_day_features(mid, features_sub)
            feature_summaries.append(summary)

            daily_data.append({
                'date': date_str,
                'rvol_annual': rvol,
                'n_bars': len(mid),
                'hourly_vols': hourly_vols,
            })

            if (i + 1) % 10 == 0 or i == 0:
                log.info(f"  [{i+1}/{len(files)}] {date_str}: rvol={rvol:.1f}%")

            del mid, features_sub
            gc.collect()

        except Exception as e:
            log.warning(f"  Error {date_str}: {e}")
            continue

    log.info(f"\nLoaded {len(daily_data)} days with valid rvol")

    # Extract arrays
    dates = [d['date'] for d in daily_data]
    rvols = np.array([d['rvol_annual'] for d in daily_data])

    log.info(f"\n{'='*70}")
    log.info(f"PART 1: DAILY REALIZED VOL STATISTICS")
    log.info(f"{'='*70}")
    log.info(f"  Mean daily rvol (annualized): {np.mean(rvols):.1f}%")
    log.info(f"  Std of daily rvol: {np.std(rvols):.1f}%")
    log.info(f"  Min: {np.min(rvols):.1f}%  Max: {np.max(rvols):.1f}%")
    log.info(f"  Median: {np.median(rvols):.1f}%")
    log.info(f"  Rvol autocorrelation (lag-1): {np.corrcoef(rvols[:-1], rvols[1:])[0,1]:.3f}")

    # 2. Download VIX
    log.info(f"\n{'='*70}")
    log.info(f"PART 2: VIX vs REALIZED VOL (Variance Risk Premium)")
    log.info(f"{'='*70}")

    vix_data = download_vix_data(dates[0], dates[-1])

    if vix_data:
        # Match VIX to our dates
        matched_vix = []
        matched_rvol = []
        matched_dates = []
        for i, d in enumerate(dates):
            if d in vix_data:
                matched_vix.append(vix_data[d])
                matched_rvol.append(rvols[i])
                matched_dates.append(d)

        matched_vix = np.array(matched_vix)
        matched_rvol = np.array(matched_rvol)

        log.info(f"  Matched {len(matched_vix)} days with both VIX and rvol")
        log.info(f"  VIX mean: {np.mean(matched_vix):.1f}")
        log.info(f"  Rvol mean: {np.mean(matched_rvol):.1f}%")

        # Variance Risk Premium
        vrp = matched_vix - matched_rvol
        log.info(f"\n  VARIANCE RISK PREMIUM (VIX - Realized Vol):")
        log.info(f"    Mean VRP: {np.mean(vrp):.1f} vol points")
        log.info(f"    Std VRP: {np.std(vrp):.1f}")
        log.info(f"    VRP > 0 (seller wins): {np.mean(vrp > 0)*100:.0f}% of days")
        log.info(f"    Mean absolute VRP: {np.mean(np.abs(vrp)):.1f}")

        # VIX as predictor of next-day rvol
        if len(matched_vix) > 2:
            # VIX today vs rvol tomorrow
            vix_today = matched_vix[:-1]
            rvol_tomorrow = matched_rvol[1:]
            corr, pval = pearsonr(vix_today, rvol_tomorrow)
            ic, _ = spearmanr(vix_today, rvol_tomorrow)
            log.info(f"\n  VIX PREDICTIVE POWER:")
            log.info(f"    VIX → next-day rvol: Pearson r={corr:.3f}, Spearman IC={ic:.3f}")

            # Yesterday's rvol as predictor
            rvol_today = matched_rvol[:-1]
            corr2, _ = pearsonr(rvol_today, rvol_tomorrow)
            ic2, _ = spearmanr(rvol_today, rvol_tomorrow)
            log.info(f"    Yesterday rvol → today rvol: Pearson r={corr2:.3f}, Spearman IC={ic2:.3f}")

            # Can we beat VIX?
            log.info(f"\n  KEY QUESTION: Can our model beat VIX as vol predictor?")
            log.info(f"    VIX → next-day: IC={ic:.3f}")
            log.info(f"    Yesterday rvol → today: IC={ic2:.3f}")
            log.info(f"    Our 1s rvol prediction: IC=0.674 (but this is INTRA-day)")
            log.info(f"    We need to test inter-day prediction...")
    else:
        log.warning("  Could not download VIX data — using rvol-only analysis")

    # 3. Test if end-of-day features predict next-day vol
    log.info(f"\n{'='*70}")
    log.info(f"PART 3: CAN MBO FEATURES PREDICT NEXT-DAY REALIZED VOL?")
    log.info(f"{'='*70}")

    if len(feature_summaries) >= 20:
        X_all = np.array(feature_summaries)
        y_all = rvols

        # Remove any NaN/inf
        valid = np.all(np.isfinite(X_all), axis=1) & np.isfinite(y_all)
        X_all = X_all[valid]
        y_all = y_all[valid]

        # Next-day prediction: features today → rvol tomorrow
        X_today = X_all[:-1]
        y_tomorrow = y_all[1:]

        log.info(f"  Feature matrix: {X_today.shape}")
        log.info(f"  Target: next-day rvol ({len(y_tomorrow)} samples)")

        # Simple: correlation of each feature with next-day vol
        log.info(f"\n  Individual feature correlations with next-day rvol:")
        feat_labels = (
            ['rvol', 'vol_ratio', 'abs_daily_return'] +
            [f'mean_feat_{i}' for i in range(20)] +
            [f'std_feat_{i}' for i in range(20)]
        )

        best_ic = 0
        best_feat = ""
        for j in range(min(X_today.shape[1], len(feat_labels))):
            if np.std(X_today[:, j]) > 0:
                ic_j, p_j = spearmanr(X_today[:, j], y_tomorrow)
                if abs(ic_j) > 0.15:
                    log.info(f"    {feat_labels[j]:30s}  IC={ic_j:+.3f}  p={p_j:.4f}")
                if abs(ic_j) > abs(best_ic):
                    best_ic = ic_j
                    best_feat = feat_labels[j]

        log.info(f"\n  Best single feature: {best_feat} IC={best_ic:+.3f}")

        # Walk-forward LightGBM for next-day vol prediction
        try:
            import lightgbm as lgb

            log.info(f"\n  Walk-forward LightGBM (next-day rvol prediction):")
            lgbm_params = {
                'objective': 'regression',
                'metric': 'mse',
                'learning_rate': 0.05,
                'num_leaves': 15,  # Small — few samples
                'max_depth': 3,
                'min_child_samples': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.5,
                'reg_alpha': 1.0,
                'reg_lambda': 5.0,
                'verbose': -1,
                'seed': 42,
            }

            min_train = 20
            fold_ics = []
            preds_all = []
            actuals_all = []

            for test_idx in range(min_train, len(X_today)):
                X_tr = X_today[:test_idx]
                y_tr = y_tomorrow[:test_idx]
                X_te = X_today[test_idx:test_idx+1]
                y_te = y_tomorrow[test_idx]

                dtrain = lgb.Dataset(X_tr, label=y_tr)
                model = lgb.train(lgbm_params, dtrain, num_boost_round=50)
                pred = model.predict(X_te)[0]
                preds_all.append(pred)
                actuals_all.append(y_te)

            preds_arr = np.array(preds_all)
            actuals_arr = np.array(actuals_all)

            if len(preds_arr) > 5:
                ic_model, p_model = spearmanr(preds_arr, actuals_arr)
                pearson_model, _ = pearsonr(preds_arr, actuals_arr)
                mae = np.mean(np.abs(preds_arr - actuals_arr))

                log.info(f"    Model predictions: {len(preds_arr)} days")
                log.info(f"    Spearman IC: {ic_model:+.3f} (p={p_model:.4f})")
                log.info(f"    Pearson r: {pearson_model:+.3f}")
                log.info(f"    MAE: {mae:.1f} vol points")
                log.info(f"    Pred mean: {np.mean(preds_arr):.1f}  Actual mean: {np.mean(actuals_arr):.1f}")
                log.info(f"    Pred std: {np.std(preds_arr):.1f}   Actual std: {np.std(actuals_arr):.1f}")

                # Compare to naive (yesterday's vol)
                naive_preds = actuals_arr[:-1]  # yesterday's actual as prediction
                actual_shifted = actuals_arr[1:]  # today's actual
                if len(naive_preds) > 5:
                    ic_naive, _ = spearmanr(naive_preds, actual_shifted)
                    mae_naive = np.mean(np.abs(naive_preds - actual_shifted))
                    log.info(f"\n    NAIVE (yesterday's vol) comparison:")
                    log.info(f"      IC: {ic_naive:+.3f}")
                    log.info(f"      MAE: {mae_naive:.1f} vol points")
                    log.info(f"      Model improvement: IC {ic_model:+.3f} vs naive {ic_naive:+.3f}")

        except ImportError:
            log.warning("  LightGBM not available — skipping model test")

    # 4. Intraday vol prediction aggregation test
    log.info(f"\n{'='*70}")
    log.info(f"PART 4: INTRADAY VOL PREDICTION (Hourly Windows)")
    log.info(f"{'='*70}")

    # For each day, compute hourly windows and test if first-half
    # features predict second-half vol
    intraday_results = []
    for i, fpath in enumerate(files[:len(daily_data)]):
        if i >= len(daily_data):
            break
        try:
            mid = load_mid_prices(fpath)
            hourly = daily_data[i]['hourly_vols']
            if len(hourly) >= 4:
                # First 2 hours vol → last hours vol correlation
                first_half_vol = np.mean(hourly[:len(hourly)//2])
                second_half_vol = np.mean(hourly[len(hourly)//2:])
                intraday_results.append((first_half_vol, second_half_vol))
            del mid
        except:
            continue

    if intraday_results:
        first_h = np.array([r[0] for r in intraday_results])
        second_h = np.array([r[1] for r in intraday_results])
        ic_intra, _ = spearmanr(first_h, second_h)
        log.info(f"  First-half → second-half vol IC: {ic_intra:+.3f}")
        log.info(f"  (This tests intraday vol persistence)")

    # 5. Economic analysis
    log.info(f"\n{'='*70}")
    log.info(f"PART 5: ECONOMIC ANALYSIS — OPTIONS VOL TRADING")
    log.info(f"{'='*70}")

    mean_rvol = np.mean(rvols)
    std_rvol = np.std(rvols)

    log.info(f"  ES options cost assumptions:")
    log.info(f"    ATM straddle bid-ask spread: ~1.0-2.0 vol points")
    log.info(f"    Commission: ~$1.50 per contract (buy+sell)")
    log.info(f"    Straddle notional: ~$250,000 per ES contract")
    log.info(f"    Vega per ES straddle: ~$250-400 per vol point")
    log.info(f"")
    log.info(f"  Mean realized vol: {mean_rvol:.1f}%")
    log.info(f"  Vol of vol (daily): {std_rvol:.1f}%")

    # If VIX data available
    if vix_data and len(matched_vix) > 0:
        mean_vrp = np.mean(vrp)
        std_vrp = np.std(vrp)

        log.info(f"  Mean VRP: {mean_vrp:+.1f} vol points")
        log.info(f"  Std VRP: {std_vrp:.1f} vol points")
        log.info(f"")

        # Strategy: sell straddle when VRP > mean, buy when VRP < mean
        # Edge = our improvement over VIX in predicting rvol
        log.info(f"  BASELINE STRATEGY: Systematic vol selling")
        log.info(f"    Always sell straddle, collect VRP")
        vega = 300  # $ per vol point per contract
        spread_cost = 1.5  # vol points bid-ask

        gross_daily = mean_vrp * vega / 252  # daily carry from VRP
        cost_per_trade = spread_cost * vega  # cost to enter/exit
        # Assume roll weekly (5 trading days)
        cost_per_day = cost_per_trade * 2 / 5  # buy + sell, amortized over 5 days
        net_daily = gross_daily - cost_per_day

        log.info(f"    Gross daily carry: ${gross_daily:.0f}")
        log.info(f"    Spread cost (amortized): ${cost_per_day:.0f}/day")
        log.info(f"    Net daily P&L: ${net_daily:.0f}")
        log.info(f"    Net annual: ${net_daily * 252:.0f}")

        # Risk: vol blow-up
        max_vrp_loss = np.min(vrp) * vega  # worst day
        log.info(f"    Worst-day VRP loss: ${max_vrp_loss:.0f}")

        # Sharpe of systematic vol selling
        vrp_daily_pnl = vrp * vega / 252
        vrp_sharpe = np.mean(vrp_daily_pnl) / np.std(vrp_daily_pnl) * np.sqrt(252)
        log.info(f"    VRP carry Sharpe (pre-cost): {vrp_sharpe:.2f}")

        log.info(f"\n  ENHANCED STRATEGY: Directional vol trading with prediction")
        log.info(f"    If we predict rvol with IC above VIX benchmark:")
        log.info(f"    - Buy straddle when predicted rvol >> VIX (vol expansion)")
        log.info(f"    - Sell straddle when predicted rvol << VIX (vol compression)")
        log.info(f"    Edge = prediction IC × vol_of_vol × vega")

        for pred_ic in [0.1, 0.2, 0.3, 0.5]:
            # IC → linear edge
            edge_vol = pred_ic * std_vrp * 0.5  # conservative: half the theoretical
            edge_dollar = edge_vol * vega
            net_edge = edge_dollar - spread_cost * vega  # minus spread cost
            trades_per_month = 20  # daily trading
            monthly = net_edge * trades_per_month
            annual = monthly * 12

            log.info(f"    IC={pred_ic}: edge={edge_vol:.1f}vp (${edge_dollar:.0f}), "
                    f"net=${net_edge:.0f}/trade, annual=${annual:,.0f}")

    # 6. Conclusion
    log.info(f"\n{'='*70}")
    log.info(f"CONCLUSION")
    log.info(f"{'='*70}")

    log.info(f"  1. Our 1s rvol prediction (IC=0.674) is INTRA-bar, not inter-day")
    log.info(f"  2. Options require predicting vol over hours/days, not seconds")
    log.info(f"  3. The question is: does intraday book state predict multi-hour vol?")
    log.info(f"  4. If our next-day prediction IC > 0.2, options trading is viable")
    log.info(f"  5. If IC < 0.1, we can't beat the market's vol pricing")

    # Save results
    results = {
        'n_days': len(daily_data),
        'mean_rvol': float(mean_rvol),
        'std_rvol': float(std_rvol),
        'rvol_autocorr': float(np.corrcoef(rvols[:-1], rvols[1:])[0,1]),
        'dates': dates,
        'daily_rvols': rvols.tolist(),
    }

    if vix_data and len(matched_vix) > 0:
        results['mean_vix'] = float(np.mean(matched_vix))
        results['mean_vrp'] = float(np.mean(vrp))
        results['std_vrp'] = float(np.std(vrp))
        results['vrp_pct_positive'] = float(np.mean(vrp > 0) * 100)

    out_path = RESULTS_DIR / f'options_vol_feasibility_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
