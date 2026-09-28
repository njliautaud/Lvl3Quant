#!/usr/bin/env python3
"""
IV-RV Timing Backtest: Does the spread between implied and realized volatility
predict better sector ETF dip-buying timing?

Hypothesis: When VIX >> realized vol (fear premium), oversold sectors mean-revert.
When realized > implied (complacent), dips continue.

8 signal variants tested with permutation testing & regime stratification.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLC', 'XLRE', 'XLB']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Trade parameters
HOLD_DAYS = 5
TP_PCT = 0.03      # +3% take profit
SL_PCT = -0.05     # -5% stop loss
COST_RT = 0.001    # 0.10% round-trip cost

# Validation thresholds
REGIME_GAP_THRESHOLD = 0.50
PERM_P_THRESHOLD = 0.05
DAY_CONC_THRESHOLD = 0.70
N_PERMUTATIONS = 1000

RESULTS_DIR = '/home/jupiter/Lvl3Quant/scripts/growth_research/results'


def download_data():
    """Download all required price data."""
    tickers = SECTOR_ETFS + [BENCHMARK, VIX_TICKER]
    print(f"Downloading data for {len(tickers)} tickers from {START_DATE} to {END_DATE}...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    # Drop any fully-NaN columns and forward fill
    close = close.dropna(how='all', axis=1).ffill()
    print(f"  Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


def compute_features(close):
    """Compute RSI, realized vol, vol premium for all tickers."""
    features = {}

    # SPY realized vol (20-day, annualized)
    spy_ret = close[BENCHMARK].pct_change()
    spy_rv20 = spy_ret.rolling(20).std() * np.sqrt(252) * 100  # annualized, in % points
    features['spy_rv20'] = spy_rv20

    # VIX (already in annualized % points)
    features['vix'] = close[VIX_TICKER]

    # Vol premium = VIX - SPY realized vol
    features['vol_premium'] = features['vix'] - spy_rv20

    # Vol premium 5-day change
    features['vol_premium_chg5'] = features['vol_premium'] - features['vol_premium'].shift(5)

    # Vol premium sign change (for regime transition)
    vp = features['vol_premium']
    vp_was_neg = (vp.shift(1) < 0) | (vp.shift(2) < 0) | (vp.shift(3) < 0)
    vp_now_pos = vp > 0
    features['vol_premium_transition'] = vp_was_neg & vp_now_pos

    # Per-sector features
    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            continue
        ret = close[etf].pct_change()

        # RSI(14)
        delta = ret.copy()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        features[f'{etf}_rsi'] = 100 - (100 / (1 + rs))

        # Sector realized vol (20-day, annualized)
        features[f'{etf}_rv20'] = ret.rolling(20).std() * np.sqrt(252) * 100

        # 10-day low & bounce detection
        low10 = close[etf].rolling(10).min()
        features[f'{etf}_near_10d_low'] = close[etf].shift(1) <= low10.shift(1) * 1.005
        features[f'{etf}_bounce'] = ret > 0  # today closed higher

    # SPY daily returns for regime classification
    features['spy_daily_ret'] = spy_ret

    return features


def generate_signals(features, close):
    """Generate buy signals for all 8 variants. Returns dict of {variant: Series of ETF to buy}."""
    signals = {}
    vix = features['vix']
    spy_rv = features['spy_rv20']
    vp = features['vol_premium']
    vp_chg5 = features['vol_premium_chg5']
    vp_trans = features['vol_premium_transition']

    available_etfs = [e for e in SECTOR_ETFS if e in close.columns]

    # Build RSI DataFrame
    rsi_df = pd.DataFrame({etf: features[f'{etf}_rsi'] for etf in available_etfs}, index=close.index)
    rv_df = pd.DataFrame({etf: features[f'{etf}_rv20'] for etf in available_etfs}, index=close.index)
    near_low_df = pd.DataFrame({etf: features[f'{etf}_near_10d_low'] for etf in available_etfs}, index=close.index)
    bounce_df = pd.DataFrame({etf: features[f'{etf}_bounce'] for etf in available_etfs}, index=close.index)

    def pick_lowest_rsi(mask_series, rsi_threshold=35):
        """Given a boolean date mask, pick the lowest-RSI sector on each True date."""
        result = pd.Series(index=close.index, dtype=object)
        for dt in mask_series.index:
            if not mask_series.loc[dt]:
                continue
            day_rsi = rsi_df.loc[dt]
            oversold = day_rsi[day_rsi < rsi_threshold]
            if len(oversold) > 0:
                result.loc[dt] = oversold.idxmin()
        return result

    def pick_lowest_rsi_40(mask_series):
        return pick_lowest_rsi(mask_series, rsi_threshold=40)

    # A: High Vol Premium Dip-Buy (VIX > RV by >5, RSI < 35)
    mask_a = vp > 5
    signals['A_HighVolPremium'] = pick_lowest_rsi(mask_a)

    # B: Extreme Vol Premium (VIX > RV by >10, RSI < 35)
    mask_b = vp > 10
    signals['B_ExtremeVolPremium'] = pick_lowest_rsi(mask_b)

    # C: Vol Premium Expanding (VP increased over 5d, RSI < 35)
    mask_c = (vp > 0) & (vp_chg5 > 0)
    signals['C_VolPremExpanding'] = pick_lowest_rsi(mask_c)

    # D: Vol Compression Entry (RV < 10%, RSI < 35)
    mask_d = spy_rv < 10
    signals['D_VolCompression'] = pick_lowest_rsi(mask_d)

    # E: Mean-Reversion Sweet Spot (VP > 5, RV > 15%, RSI < 35)
    mask_e = (vp > 5) & (spy_rv > 15)
    signals['E_MeanRevSweetSpot'] = pick_lowest_rsi(mask_e)

    # F: Sector-Specific IV-RV (VIX > sector RV by >5, RSI < 35)
    result_f = pd.Series(index=close.index, dtype=object)
    for dt in close.index:
        candidates = []
        for etf in available_etfs:
            etf_rsi = rsi_df.loc[dt, etf] if not pd.isna(rsi_df.loc[dt, etf]) else 100
            etf_rv = rv_df.loc[dt, etf] if not pd.isna(rv_df.loc[dt, etf]) else 0
            vix_val = vix.loc[dt] if not pd.isna(vix.loc[dt]) else 0
            if vix_val - etf_rv > 5 and etf_rsi < 35:
                candidates.append((etf, etf_rsi))
        if candidates:
            result_f.loc[dt] = min(candidates, key=lambda x: x[1])[0]
    signals['F_SectorSpecificIVRV'] = result_f

    # G: Vol Premium + Bounce (Signal A + bounce confirmation)
    result_g = pd.Series(index=close.index, dtype=object)
    for dt in close.index:
        if not (vp.loc[dt] > 5 if not pd.isna(vp.loc[dt]) else False):
            continue
        candidates = []
        for etf in available_etfs:
            etf_rsi = rsi_df.loc[dt, etf] if not pd.isna(rsi_df.loc[dt, etf]) else 100
            is_near_low = near_low_df.loc[dt, etf] if not pd.isna(near_low_df.loc[dt, etf]) else False
            is_bounce = bounce_df.loc[dt, etf] if not pd.isna(bounce_df.loc[dt, etf]) else False
            if etf_rsi < 35 and is_near_low and is_bounce:
                candidates.append((etf, etf_rsi))
        if candidates:
            result_g.loc[dt] = min(candidates, key=lambda x: x[1])[0]
    signals['G_VolPremBounce'] = result_g

    # H: Vol Regime Transition (VP crosses neg->pos within 3d, RSI < 40)
    mask_h = vp_trans.fillna(False)
    signals['H_VolRegimeTransition'] = pick_lowest_rsi_40(mask_h)

    return signals


def simulate_trades(signals_series, close, features):
    """Simulate trades for a signal variant. Returns list of trade dicts."""
    trades = []
    available_etfs = [e for e in SECTOR_ETFS if e in close.columns]
    dates = close.index.tolist()

    in_trade = False
    trade_end_idx = -1

    for i, dt in enumerate(dates):
        if i <= trade_end_idx:
            continue  # still in a trade

        etf = signals_series.get(dt, None) if isinstance(signals_series, dict) else signals_series.loc[dt]
        if pd.isna(etf) or etf is None or etf not in close.columns:
            continue

        entry_price = close.loc[dt, etf]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Simulate hold period with TP/SL
        exit_price = None
        exit_reason = 'hold'
        exit_date = dt

        for j in range(1, HOLD_DAYS + 1):
            if i + j >= len(dates):
                break
            future_dt = dates[i + j]
            future_price = close.loc[future_dt, etf]
            if pd.isna(future_price):
                continue

            ret = (future_price - entry_price) / entry_price

            if ret >= TP_PCT:
                exit_price = entry_price * (1 + TP_PCT)
                exit_reason = 'TP'
                exit_date = future_dt
                trade_end_idx = i + j
                break
            elif ret <= SL_PCT:
                exit_price = entry_price * (1 + SL_PCT)
                exit_reason = 'SL'
                exit_date = future_dt
                trade_end_idx = i + j
                break

            if j == HOLD_DAYS:
                exit_price = future_price
                exit_reason = 'hold'
                exit_date = future_dt
                trade_end_idx = i + j

        if exit_price is None:
            # Not enough future data
            continue

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - COST_RT

        # Regime: SPY return on entry day
        spy_ret = features['spy_daily_ret'].loc[dt] if dt in features['spy_daily_ret'].index else 0
        regime = 'green' if spy_ret >= 0 else 'red'

        trades.append({
            'entry_date': dt,
            'exit_date': exit_date,
            'etf': etf,
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'gross_ret': float(gross_ret),
            'net_ret': float(net_ret),
            'exit_reason': exit_reason,
            'regime': regime,
        })

    return trades


def compute_metrics(trades):
    """Compute all required metrics from trade list."""
    if len(trades) == 0:
        return {
            'trades': 0, 'win_rate': 0, 'avg_return': 0,
            'sharpe': 0, 'sortino': 0, 'profit_factor': 0,
            'max_dd': 0, 'sharpe_green': 0, 'sharpe_red': 0,
            'regime_gap': 999, 'day_concentration': 0,
            'perm_p': 1.0,
            'pass_regime': False, 'pass_perm': False, 'pass_dayconc': False,
            'pass_all': False,
        }

    rets = np.array([t['net_ret'] for t in trades])
    n = len(rets)
    wins = np.sum(rets > 0)

    wr = wins / n
    avg_ret = np.mean(rets)

    # Annualized Sharpe (assume ~50 trades/year as scaling, use per-trade Sharpe * sqrt(n_annual))
    # More robust: use daily-equivalent scaling
    trades_per_year = n / ((pd.Timestamp(trades[-1]['exit_date']) - pd.Timestamp(trades[0]['entry_date'])).days / 365.25) if n > 1 else 50
    trades_per_year = max(trades_per_year, 1)

    if np.std(rets) > 0:
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = rets[rets < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(rets) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    # Profit factor
    gross_wins = np.sum(rets[rets > 0])
    gross_losses = np.abs(np.sum(rets[rets < 0]))
    pf = gross_wins / gross_losses if gross_losses > 0 else (999 if gross_wins > 0 else 0)

    # Max drawdown (on cumulative returns)
    cum = np.cumsum(rets)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    # Regime-stratified Sharpe
    green_rets = np.array([t['net_ret'] for t in trades if t['regime'] == 'green'])
    red_rets = np.array([t['net_ret'] for t in trades if t['regime'] == 'red'])

    def trade_sharpe(r):
        if len(r) < 2 or np.std(r) == 0:
            return 0.0
        tpy = len(r) / ((pd.Timestamp(trades[-1]['exit_date']) - pd.Timestamp(trades[0]['entry_date'])).days / 365.25)
        tpy = max(tpy, 1)
        return (np.mean(r) / np.std(r)) * np.sqrt(tpy)

    sharpe_green = trade_sharpe(green_rets)
    sharpe_red = trade_sharpe(red_rets)

    max_abs = max(abs(sharpe_green), abs(sharpe_red), 0.001)
    regime_gap = abs(sharpe_green - sharpe_red) / max_abs

    # Day concentration: max fraction of trades on any single date
    dates = [str(t['entry_date'])[:10] for t in trades]
    date_counts = pd.Series(dates).value_counts()
    day_conc = float(date_counts.max() / n) if n > 0 else 0

    # Permutation test
    observed_mean = np.mean(rets)
    perm_count = 0
    all_rets_shuffled = rets.copy()
    for _ in range(N_PERMUTATIONS):
        np.random.shuffle(all_rets_shuffled)
        # Randomly flip signs to test if mean return is significant
        random_signs = np.random.choice([-1, 1], size=n)
        perm_mean = np.mean(rets * random_signs)
        if perm_mean >= observed_mean:
            perm_count += 1
    perm_p = perm_count / N_PERMUTATIONS

    pass_regime = regime_gap < REGIME_GAP_THRESHOLD
    pass_perm = perm_p < PERM_P_THRESHOLD
    pass_dayconc = day_conc < DAY_CONC_THRESHOLD
    pass_all = pass_regime and pass_perm and pass_dayconc and sharpe > 0.5

    return {
        'trades': n,
        'win_rate': round(wr, 4),
        'avg_return': round(avg_ret, 6),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(min(pf, 99), 3),
        'max_dd': round(max_dd, 4),
        'sharpe_green': round(sharpe_green, 3),
        'sharpe_red': round(sharpe_red, 3),
        'regime_gap': round(regime_gap, 3),
        'day_concentration': round(day_conc, 4),
        'perm_p': round(perm_p, 4),
        'pass_regime': pass_regime,
        'pass_perm': pass_perm,
        'pass_dayconc': pass_dayconc,
        'pass_all': pass_all,
    }


def main():
    print("=" * 90)
    print("IV-RV TIMING BACKTEST: Implied vs Realized Vol for Sector Dip-Buying")
    print("=" * 90)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Sectors: {', '.join(SECTOR_ETFS)}")
    print(f"Exit: {HOLD_DAYS}d hold / +{TP_PCT*100:.0f}% TP / {SL_PCT*100:.0f}% SL / {COST_RT*100:.2f}% cost")
    print()

    # Download and prepare data
    close = download_data()
    features = compute_features(close)

    print("\nGenerating signals for 8 variants...")
    signals = generate_signals(features, close)

    all_results = {}

    print("\nSimulating trades and computing metrics...")
    print("-" * 90)

    header = f"{'Variant':<25} {'Trades':>6} {'WR':>6} {'AvgRet':>8} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD':>7} {'SG':>6} {'SR':>6} {'RGap':>6} {'DConc':>6} {'Perm-p':>7} {'PASS':>5}"
    print(header)
    print("-" * 90)

    np.random.seed(42)

    for variant_name, sig in signals.items():
        trade_count = sig.dropna().shape[0] if hasattr(sig, 'dropna') else 0
        trades = simulate_trades(sig, close, features)
        metrics = compute_metrics(trades)
        all_results[variant_name] = metrics

        flag = "YES" if metrics['pass_all'] else "NO"
        regime_flag = "+" if metrics['pass_regime'] else "X"
        perm_flag = "+" if metrics['pass_perm'] else "X"
        dayc_flag = "+" if metrics['pass_dayconc'] else "X"

        print(f"{variant_name:<25} {metrics['trades']:>6} {metrics['win_rate']:>6.1%} {metrics['avg_return']:>8.4f} "
              f"{metrics['sharpe']:>7.3f} {metrics['sortino']:>8.3f} {metrics['profit_factor']:>6.2f} {metrics['max_dd']:>7.4f} "
              f"{metrics['sharpe_green']:>6.2f} {metrics['sharpe_red']:>6.2f} "
              f"{metrics['regime_gap']:>6.3f}{regime_flag} {metrics['day_concentration']:>6.3f}{dayc_flag} "
              f"{metrics['perm_p']:>6.4f}{perm_flag} {flag:>5}")

    print("-" * 90)

    # Summary
    print("\n" + "=" * 90)
    print("VALIDATION SUMMARY")
    print("=" * 90)

    passing = [k for k, v in all_results.items() if v['pass_all']]
    failing = [k for k, v in all_results.items() if not v['pass_all']]

    print(f"\nPASSING variants ({len(passing)}):")
    for v in passing:
        m = all_results[v]
        print(f"  {v}: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, WR={m['win_rate']:.1%}, "
              f"PF={m['profit_factor']:.2f}, {m['trades']} trades")

    if not passing:
        print("  (none)")

    print(f"\nFAILING variants ({len(failing)}):")
    for v in failing:
        m = all_results[v]
        reasons = []
        if not m['pass_regime']:
            reasons.append(f"regime_gap={m['regime_gap']:.3f}")
        if not m['pass_perm']:
            reasons.append(f"perm_p={m['perm_p']:.4f}")
        if not m['pass_dayconc']:
            reasons.append(f"day_conc={m['day_concentration']:.3f}")
        if m['sharpe'] <= 0.5:
            reasons.append(f"sharpe={m['sharpe']:.3f}")
        print(f"  {v}: FAIL ({', '.join(reasons)})")

    # Best variant
    best = max(all_results.items(), key=lambda x: x[1]['sharpe'])
    print(f"\nBest variant by Sharpe: {best[0]} (Sharpe={best[1]['sharpe']:.3f})")

    # Save results
    output = {
        'metadata': {
            'backtest': 'IV-RV Timing',
            'hypothesis': 'Vol premium (VIX - realized vol) predicts sector dip-buying opportunities',
            'period': f'{START_DATE} to {END_DATE}',
            'sectors': SECTOR_ETFS,
            'exit_rules': f'{HOLD_DAYS}d hold / +{TP_PCT*100:.0f}% TP / {SL_PCT*100:.0f}% SL',
            'cost_rt': COST_RT,
            'n_permutations': N_PERMUTATIONS,
            'run_timestamp': datetime.now().isoformat(),
        },
        'results': all_results,
        'summary': {
            'passing_variants': passing,
            'failing_variants': failing,
            'best_variant': best[0],
            'best_sharpe': best[1]['sharpe'],
        }
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'iv_rv_timing_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print("=" * 90)


if __name__ == '__main__':
    main()
