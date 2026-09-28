#!/usr/bin/env python3
"""
Sector ETF Vol Compression Breakout — Backtest v1
===================================================
Hypothesis: When a sector ETF's realized volatility compresses to an unusually
low level (bottom 10th percentile of rolling 60-day distribution), it's about
to make a big directional move. Combined with directional bias (RSI + momentum),
we can predict the 5-day direction.

Universe: 11 sector ETFs (XLK, XLF, XLE, XLU, XLP, XLY, XLV, XLI, XLB, XLC, XLRE) + SPY + VIX
Period: 2020-01-01 to 2026-08-20
Hold: 5 trading days
Cost: 0.20% round-trip
Window: Sliding 60-day calibration, NEVER expanding

Outputs: Sharpe, Sortino, WR, PF, MaxDD, regime-stratified Sharpe, magnitude analysis.
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/sector_vol_compression_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Constants ───
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
BENCHMARKS = ['SPY', '^VIX']
ALL_TICKERS = SECTOR_ETFS + ['SPY']  # VIX downloaded separately
START_DATE = '2019-06-01'  # Extra buffer for lookback
END_DATE = '2026-08-20'
HOLD_DAYS = 5
COST_RT = 0.0020  # 0.20% round-trip
VOL_SHORT = 10    # 10-day realized vol
VOL_LONG = 60     # 60-day realized vol
VOL_RATIO_THRESH = 0.50  # vol ratio < 0.5 = compression
BB_PERIOD = 20
BB_STD = 2
RSI_PERIOD = 14
MOM_PERIOD = 20
RSI_LONG_THRESH = 60
RSI_SHORT_THRESH = 40
CALIBRATION_WINDOW = 60  # Sliding window for threshold calibration

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def compute_rsi(series, period=14):
    """RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_bollinger_width(close, period=20, num_std=2):
    """Bollinger Band width = (upper - lower) / middle."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma
    return width

def compute_signals(df):
    """Compute all signals for a single ETF DataFrame with 'Close' column."""
    close = df['Close'].copy()
    ret = close.pct_change()

    # Realized vol (annualized)
    vol_short = ret.rolling(VOL_SHORT).std() * np.sqrt(252)
    vol_long = ret.rolling(VOL_LONG).std() * np.sqrt(252)
    vol_ratio = vol_short / vol_long.replace(0, np.nan)

    # Bollinger width
    bb_width = compute_bollinger_width(close, BB_PERIOD, BB_STD)

    # RSI
    rsi = compute_rsi(close, RSI_PERIOD)

    # Momentum (20-day return)
    momentum = close.pct_change(MOM_PERIOD)

    # Forward 5-day return
    fwd_ret = close.shift(-HOLD_DAYS) / close - 1

    signals = pd.DataFrame({
        'close': close,
        'ret': ret,
        'vol_short': vol_short,
        'vol_long': vol_long,
        'vol_ratio': vol_ratio,
        'bb_width': bb_width,
        'rsi': rsi,
        'momentum': momentum,
        'fwd_5d_ret': fwd_ret,
        'fwd_5d_abs_ret': fwd_ret.abs(),
    }, index=df.index)

    return signals

def classify_regime(spy_ret_5d):
    """Classify market regime based on SPY 5-day return."""
    if spy_ret_5d > 0.01:
        return 'bull'
    elif spy_ret_5d < -0.01:
        return 'bear'
    else:
        return 'flat'

def walk_forward_backtest(all_signals, spy_data):
    """Walk-forward backtest with sliding 60-day calibration window."""

    trades = []

    # Determine OOT start: need at least VOL_LONG + CALIBRATION_WINDOW days of warmup
    warmup = VOL_LONG + CALIBRATION_WINDOW + 10

    # Get common dates across all ETFs (use any ETF that has data)
    all_dates = set()
    for ticker, sig in all_signals.items():
        all_dates.update(sig.index)
    all_dates = sorted(all_dates)

    if len(all_dates) < warmup + HOLD_DAYS:
        log("ERROR: Not enough data for walk-forward")
        return pd.DataFrame()

    # OOT starts after warmup
    oot_start_idx = warmup
    oot_dates = all_dates[oot_start_idx:-HOLD_DAYS]  # Leave room for forward return

    # Filter to 2020-01-01 onwards for OOT
    oot_dates = [d for d in oot_dates if d >= pd.Timestamp('2020-01-01')]

    log(f"  OOT period: {oot_dates[0].strftime('%Y-%m-%d')} to {oot_dates[-1].strftime('%Y-%m-%d')} ({len(oot_dates)} days)")

    for date in oot_dates:
        for ticker, sig in all_signals.items():
            if date not in sig.index:
                continue

            row = sig.loc[date]

            # Skip if key signals are NaN
            if pd.isna(row['vol_ratio']) or pd.isna(row['rsi']) or pd.isna(row['fwd_5d_ret']):
                continue

            # ─── Sliding window calibration ───
            # Use last CALIBRATION_WINDOW days to compute vol_ratio percentile
            mask = (sig.index < date) & (sig.index >= date - pd.Timedelta(days=CALIBRATION_WINDOW * 2))
            hist_data = sig.loc[mask, 'vol_ratio'].dropna()

            if len(hist_data) < 30:
                continue

            # Adaptive threshold: 10th percentile of recent vol_ratio
            vol_ratio_thresh_adaptive = hist_data.quantile(0.10)

            # ─── Entry conditions ───
            vol_compressed = row['vol_ratio'] < min(VOL_RATIO_THRESH, vol_ratio_thresh_adaptive)

            if not vol_compressed:
                continue

            # Directional bias
            rsi_val = row['rsi']
            mom_val = row['momentum']

            if rsi_val > RSI_LONG_THRESH and mom_val > 0:
                direction = 'long'
                predicted_ret = row['fwd_5d_ret']
            elif rsi_val < RSI_SHORT_THRESH and mom_val < 0:
                direction = 'short'
                predicted_ret = -row['fwd_5d_ret']  # Short profits from decline
            else:
                # Also track undirected compression events for magnitude analysis
                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'direction': 'undirected',
                    'vol_ratio': row['vol_ratio'],
                    'rsi': rsi_val,
                    'bb_width': row['bb_width'],
                    'fwd_5d_ret': row['fwd_5d_ret'],
                    'fwd_5d_abs_ret': row['fwd_5d_abs_ret'],
                    'entry_signal': False,
                })
                continue

            # SPY regime classification
            spy_fwd = None
            if date in spy_data.index:
                spy_row = spy_data.loc[date]
                if not pd.isna(spy_row.get('fwd_5d_ret', np.nan)):
                    spy_fwd = spy_row['fwd_5d_ret']
            regime = classify_regime(spy_fwd) if spy_fwd is not None else 'unknown'

            # Net return after costs
            net_ret = predicted_ret - COST_RT

            trades.append({
                'date': date,
                'ticker': ticker,
                'direction': direction,
                'vol_ratio': row['vol_ratio'],
                'rsi': rsi_val,
                'bb_width': row['bb_width'],
                'momentum': mom_val,
                'fwd_5d_ret': row['fwd_5d_ret'],
                'fwd_5d_abs_ret': row['fwd_5d_abs_ret'],
                'predicted_ret': predicted_ret,
                'net_ret': net_ret,
                'regime': regime,
                'entry_signal': True,
            })

    return pd.DataFrame(trades)

def compute_metrics(returns):
    """Compute Sharpe, Sortino, WR, PF, MaxDD from a series of returns."""
    if len(returns) == 0:
        return {}

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    wr = len(wins) / len(returns) if len(returns) > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf

    mean_ret = returns.mean()
    std_ret = returns.std()

    # Annualize (assuming ~50 trades/year for 5-day holds = 250/5)
    ann_factor = np.sqrt(252 / HOLD_DAYS)
    sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0

    downside = returns[returns < 0].std()
    sortino = (mean_ret / downside) * ann_factor if downside > 0 else 0

    # Max drawdown from cumulative returns
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    return {
        'n_trades': len(returns),
        'mean_ret': mean_ret,
        'median_ret': returns.median(),
        'std_ret': std_ret,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'total_return': cum.iloc[-1] - 1 if len(cum) > 0 else 0,
        'avg_win': wins.mean() if len(wins) > 0 else 0,
        'avg_loss': losses.mean() if len(losses) > 0 else 0,
    }

def permutation_test(returns, n_perms=5000):
    """Permutation test: what fraction of random shuffles beat actual Sharpe?"""
    if len(returns) < 10:
        return 1.0

    actual_sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0

    count_better = 0
    for _ in range(n_perms):
        perm = np.random.permutation(returns.values)
        perm_sharpe = perm.mean() / perm.std() if perm.std() > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perms

def run_magnitude_analysis(trades_df):
    """Test whether vol compression predicts MAGNITUDE of subsequent move."""
    log("\n" + "="*70)
    log("MAGNITUDE ANALYSIS: Does vol compression predict move SIZE?")
    log("="*70)

    # All compression events (directed + undirected)
    all_compressed = trades_df[trades_df['vol_ratio'] < VOL_RATIO_THRESH].copy()

    if len(all_compressed) < 20:
        log("  Not enough compression events for magnitude analysis")
        return {}

    # Compare abs returns: compressed vs non-compressed days
    # We need the non-compressed baseline - use directed trades' universe for fair comparison
    directed = trades_df[trades_df['entry_signal'] == True]
    undirected = trades_df[trades_df['entry_signal'] == False]

    log(f"  Directed compression signals: {len(directed)}")
    log(f"  Undirected compression events (no RSI bias): {len(undirected)}")

    # Magnitude after compression vs overall
    all_abs = trades_df['fwd_5d_abs_ret'].dropna()
    compressed_abs = all_compressed['fwd_5d_abs_ret'].dropna()

    if len(all_abs) > 0 and len(compressed_abs) > 0:
        overall_median_abs = all_abs.median()
        compressed_median_abs = compressed_abs.median()
        magnitude_lift = compressed_median_abs / overall_median_abs - 1 if overall_median_abs > 0 else 0

        log(f"\n  Median |5d return| — all events:   {overall_median_abs*100:.3f}%")
        log(f"  Median |5d return| — compressed:    {compressed_median_abs*100:.3f}%")
        log(f"  Magnitude lift from compression:     {magnitude_lift*100:.1f}%")

        # Statistical test: are compressed abs returns larger?
        from scipy import stats
        if len(compressed_abs) >= 10 and len(all_abs) >= 10:
            stat, pval = stats.mannwhitneyu(compressed_abs, all_abs, alternative='greater')
            log(f"  Mann-Whitney U (compressed > all): p = {pval:.4f}")

        # Quintile analysis of vol_ratio vs abs return magnitude
        q_data = trades_df[['vol_ratio', 'fwd_5d_abs_ret']].dropna()
        if len(q_data) > 50:
            q_data['vol_ratio_quintile'] = pd.qcut(q_data['vol_ratio'], 5, labels=['Q1_lowest','Q2','Q3','Q4','Q5_highest'])
            quintile_stats = q_data.groupby('vol_ratio_quintile')['fwd_5d_abs_ret'].agg(['median', 'mean', 'count'])
            log(f"\n  Vol Ratio Quintile → Median |5d Ret|:")
            for idx, row in quintile_stats.iterrows():
                log(f"    {idx}: median={row['median']*100:.3f}%, mean={row['mean']*100:.3f}%, n={int(row['count'])}")

        return {
            'overall_median_abs_ret': overall_median_abs,
            'compressed_median_abs_ret': compressed_median_abs,
            'magnitude_lift_pct': magnitude_lift * 100,
        }

    return {}


def main():
    log("="*70)
    log("SECTOR ETF VOL COMPRESSION BREAKOUT — BACKTEST v1")
    log("Hypothesis: Low vol ratio (<0.5) + directional bias → 5d move")
    log(f"Universe: {', '.join(SECTOR_ETFS)}")
    log(f"Cost: {COST_RT*100:.2f}% round-trip | Hold: {HOLD_DAYS}d | Window: sliding {CALIBRATION_WINDOW}d")
    log("="*70)

    # ─── Step 1: Download data ───
    log("\n[1/6] Downloading sector ETF + SPY + VIX data...")

    cache_file = os.path.join(OUTPUT_DIR, 'sector_data_cache.pkl')

    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 24:
            log(f"  Using cached data ({age_hours:.1f}h old)")
            data = pd.read_pickle(cache_file)
        else:
            data = None
    else:
        data = None

    if data is None:
        data = {}
        # Download sector ETFs + SPY
        tickers_to_dl = SECTOR_ETFS + ['SPY']
        log(f"  Downloading {len(tickers_to_dl)} tickers...")
        raw = yf.download(tickers_to_dl, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True, threads=True)

        if raw is not None and not raw.empty:
            for t in tickers_to_dl:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        # Try both orderings
                        try:
                            td = raw.xs(t, level=1, axis=1)
                        except KeyError:
                            try:
                                td = raw.xs(t, level='Ticker', axis=1)
                            except:
                                td = raw[t] if t in raw.columns.get_level_values(0) else None
                    else:
                        td = raw

                    if td is not None and 'Close' in td.columns:
                        td = td[['Open','High','Low','Close','Volume']].dropna(subset=['Close'])
                        if len(td) > 100:
                            data[t] = td
                            log(f"    {t}: {len(td)} days ({td.index[0].strftime('%Y-%m-%d')} to {td.index[-1].strftime('%Y-%m-%d')})")
                except Exception as e:
                    log(f"    {t}: FAILED ({e})")

        # Download VIX
        try:
            vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if vix is not None and not vix.empty:
                # Handle MultiIndex columns from yfinance
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                data['^VIX'] = vix
                log(f"    VIX: {len(vix)} days")
        except Exception as e:
            log(f"    VIX: FAILED ({e})")

        # Cache
        pd.to_pickle(data, cache_file)
        log(f"  Cached {len(data)} tickers")

    missing = [t for t in SECTOR_ETFS if t not in data]
    if missing:
        log(f"  WARNING: Missing ETFs: {missing}")

    # ─── Step 2: Compute signals ───
    log("\n[2/6] Computing signals for each sector ETF...")

    all_signals = {}
    for ticker in SECTOR_ETFS:
        if ticker not in data:
            continue
        sig = compute_signals(data[ticker])
        all_signals[ticker] = sig
        n_compressed = (sig['vol_ratio'] < VOL_RATIO_THRESH).sum()
        log(f"  {ticker}: {len(sig)} days, {n_compressed} compression events (vol_ratio < {VOL_RATIO_THRESH})")

    # SPY signals for regime classification
    spy_sig = compute_signals(data['SPY']) if 'SPY' in data else pd.DataFrame()

    # VIX for additional context
    vix_data = data.get('^VIX', pd.DataFrame())

    # ─── Step 3: Walk-forward backtest ───
    log("\n[3/6] Running walk-forward backtest...")
    trades_df = walk_forward_backtest(all_signals, spy_sig)

    if trades_df.empty:
        log("ERROR: No trades generated!")
        return

    # Split into directed trades (actual signals) and all events
    directed = trades_df[trades_df['entry_signal'] == True].copy()
    undirected = trades_df[trades_df['entry_signal'] == False].copy()

    log(f"\n  Total events: {len(trades_df)}")
    log(f"  Directed signals (traded): {len(directed)}")
    log(f"  Undirected compression (magnitude only): {len(undirected)}")

    if len(directed) < 20:
        log(f"\n  WARNING: Only {len(directed)} directed trades — may be statistically weak")

    # ─── Step 4: Overall results ───
    log("\n[4/6] Computing overall metrics...")
    log("="*70)

    if len(directed) > 0:
        metrics = compute_metrics(directed['net_ret'])

        log(f"\n  OVERALL DIRECTED STRATEGY RESULTS ({metrics['n_trades']} trades)")
        log(f"  {'─'*50}")
        log(f"  Sharpe Ratio:    {metrics['sharpe']:.3f}")
        log(f"  Sortino Ratio:   {metrics['sortino']:.3f}")
        log(f"  Win Rate:        {metrics['win_rate']*100:.1f}%")
        log(f"  Profit Factor:   {metrics['profit_factor']:.3f}")
        log(f"  Max Drawdown:    {metrics['max_drawdown']*100:.2f}%")
        log(f"  Total Return:    {metrics['total_return']*100:.2f}%")
        log(f"  Mean Trade:      {metrics['mean_ret']*100:.4f}%")
        log(f"  Median Trade:    {metrics['median_ret']*100:.4f}%")
        log(f"  Avg Win:         {metrics['avg_win']*100:.4f}%")
        log(f"  Avg Loss:        {metrics['avg_loss']*100:.4f}%")

        # ─── By direction ───
        log(f"\n  BY DIRECTION:")
        for direction in ['long', 'short']:
            dir_trades = directed[directed['direction'] == direction]
            if len(dir_trades) < 5:
                log(f"    {direction.upper()}: {len(dir_trades)} trades (too few)")
                continue
            dm = compute_metrics(dir_trades['net_ret'])
            log(f"    {direction.upper()}: n={dm['n_trades']}, Sharpe={dm['sharpe']:.3f}, WR={dm['win_rate']*100:.1f}%, PF={dm['profit_factor']:.3f}")

        # ─── By ticker ───
        log(f"\n  BY SECTOR ETF:")
        for ticker in SECTOR_ETFS:
            t_trades = directed[directed['ticker'] == ticker]
            if len(t_trades) < 3:
                log(f"    {ticker}: {len(t_trades)} trades")
                continue
            tm = compute_metrics(t_trades['net_ret'])
            log(f"    {ticker}: n={tm['n_trades']}, Sharpe={tm['sharpe']:.3f}, WR={tm['win_rate']*100:.1f}%, PF={tm['profit_factor']:.3f}, mean={tm['mean_ret']*100:.3f}%")

        # ─── Step 5: Regime stratification ───
        log(f"\n[5/6] Regime stratification...")
        log("="*70)

        regime_results = {}
        for regime in ['bull', 'bear', 'flat', 'unknown']:
            r_trades = directed[directed['regime'] == regime]
            if len(r_trades) < 5:
                log(f"  {regime.upper()}: {len(r_trades)} trades (too few)")
                continue
            rm = compute_metrics(r_trades['net_ret'])
            regime_results[regime] = rm
            log(f"  {regime.upper()}: n={rm['n_trades']}, Sharpe={rm['sharpe']:.3f}, WR={rm['win_rate']*100:.1f}%, PF={rm['profit_factor']:.3f}, MDD={rm['max_drawdown']*100:.1f}%")

        # Regime gap test
        if 'bull' in regime_results and 'bear' in regime_results:
            bull_s = regime_results['bull']['sharpe']
            bear_s = regime_results['bear']['sharpe']
            gap = abs(bull_s - bear_s) / max(abs(bull_s), abs(bear_s), 0.001)
            log(f"\n  Regime gap: |Sharpe_bull - Sharpe_bear| / max = {gap:.3f}")
            if gap > 0.50:
                log(f"  *** FAIL: Regime gap {gap:.3f} > 0.50 — strategy is regime-tailored, NOT genuine edge ***")
            else:
                log(f"  PASS: Regime gap within bounds")

        # ─── Per-year analysis ───
        log(f"\n  PER-YEAR BREAKDOWN:")
        directed['year'] = directed['date'].dt.year
        for year in sorted(directed['year'].unique()):
            y_trades = directed[directed['year'] == year]
            if len(y_trades) < 3:
                continue
            ym = compute_metrics(y_trades['net_ret'])
            log(f"    {year}: n={ym['n_trades']}, Sharpe={ym['sharpe']:.3f}, WR={ym['win_rate']*100:.1f}%, PF={ym['profit_factor']:.3f}, total={ym['total_return']*100:.2f}%")

        # ─── Day-concentration check ───
        if len(directed) > 10:
            day_counts = directed.groupby('date').size()
            max_day = day_counts.max()
            total = len(directed)
            day_conc = max_day / total
            log(f"\n  Day concentration: max {max_day} trades on one day, {day_conc*100:.1f}% of total")
            if day_conc > 0.70:
                log(f"  *** FAIL: Day concentration {day_conc:.2f} > 0.70 ***")

        # ─── Permutation test ───
        log(f"\n  PERMUTATION TEST (5000 shuffles)...")
        p_val = permutation_test(directed['net_ret'])
        log(f"  p-value: {p_val:.4f}")
        if p_val > 0.05:
            log(f"  *** FAIL: Not statistically significant (p={p_val:.4f} > 0.05) ***")
        else:
            log(f"  PASS: Significant at 5% level")

        # ─── Sharpe verdict ───
        log(f"\n{'='*70}")
        if metrics['sharpe'] < 0.5:
            log(f"  VERDICT: DEAD — Sharpe {metrics['sharpe']:.3f} < 0.5 threshold")
        elif metrics['sharpe'] < 1.0:
            log(f"  VERDICT: WEAK — Sharpe {metrics['sharpe']:.3f} (between 0.5 and 1.0)")
        else:
            log(f"  VERDICT: PROMISING — Sharpe {metrics['sharpe']:.3f} >= 1.0")
        log("="*70)
    else:
        log("  No directed trades to analyze!")
        metrics = {}

    # ─── Step 6: Magnitude analysis ───
    mag_results = run_magnitude_analysis(trades_df)

    # ─── Also test relaxed thresholds ───
    log(f"\n{'='*70}")
    log("SENSITIVITY ANALYSIS: Varying vol_ratio threshold and RSI bounds")
    log("="*70)

    sensitivity_results = []
    for vr_thresh in [0.40, 0.50, 0.60, 0.70]:
        for rsi_lo, rsi_hi in [(30, 70), (35, 65), (40, 60), (45, 55)]:
            # Quick re-filter
            sens_trades = []
            for date_idx, row_data in trades_df.iterrows():
                if row_data['vol_ratio'] >= vr_thresh:
                    continue
                rsi_val = row_data['rsi']
                fwd = row_data['fwd_5d_ret']
                if pd.isna(fwd):
                    continue

                if rsi_val > rsi_hi:
                    net = fwd - COST_RT
                elif rsi_val < rsi_lo:
                    net = -fwd - COST_RT
                else:
                    continue
                sens_trades.append(net)

            if len(sens_trades) >= 10:
                arr = np.array(sens_trades)
                sr = (arr.mean() / arr.std()) * np.sqrt(252/HOLD_DAYS) if arr.std() > 0 else 0
                wr = (arr > 0).mean()
                sensitivity_results.append({
                    'vol_thresh': vr_thresh,
                    'rsi_bounds': f"{rsi_lo}/{rsi_hi}",
                    'n': len(arr),
                    'sharpe': sr,
                    'wr': wr,
                    'mean': arr.mean()
                })

    if sensitivity_results:
        sens_df = pd.DataFrame(sensitivity_results).sort_values('sharpe', ascending=False)
        log(f"\n  Top 10 parameter combos by Sharpe:")
        for i, row in sens_df.head(10).iterrows():
            log(f"    VR<{row['vol_thresh']}, RSI {row['rsi_bounds']}: n={row['n']}, Sharpe={row['sharpe']:.3f}, WR={row['wr']*100:.1f}%")

        best = sens_df.iloc[0]
        log(f"\n  Best combo: vol_ratio<{best['vol_thresh']}, RSI {best['rsi_bounds']} → Sharpe={best['sharpe']:.3f}")

    # ─── Save results ───
    results = {
        'overall_metrics': metrics,
        'regime_results': {k: v for k, v in regime_results.items()} if 'regime_results' in dir() else {},
        'magnitude': mag_results,
        'n_directed': len(directed),
        'n_undirected': len(undirected),
        'sensitivity': sensitivity_results,
        'params': {
            'vol_short': VOL_SHORT,
            'vol_long': VOL_LONG,
            'vol_ratio_thresh': VOL_RATIO_THRESH,
            'rsi_long': RSI_LONG_THRESH,
            'rsi_short': RSI_SHORT_THRESH,
            'hold_days': HOLD_DAYS,
            'cost_rt': COST_RT,
            'calibration_window': CALIBRATION_WINDOW,
        }
    }

    results_file = os.path.join(OUTPUT_DIR, 'results.json')

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert)

    if len(directed) > 0:
        trades_file = os.path.join(OUTPUT_DIR, 'trades.csv')
        directed.to_csv(trades_file, index=False)
        log(f"\n  Saved trades to {trades_file}")

    log(f"  Saved results to {results_file}")
    log("\nDONE.")


if __name__ == '__main__':
    main()
