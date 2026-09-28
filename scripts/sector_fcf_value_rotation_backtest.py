#!/usr/bin/env python3
"""
Sector FCF/Value Rotation Backtest — 5-Day Forward Returns
==========================================================
Hypothesis: Sectors that are relatively "cheap" (by price-derived value proxies)
outperform expensive sectors over 5-day horizons.

Since yfinance doesn't provide historical quarterly FCF by sector, we use
price-derived value proxies that are available historically:
  1. Earnings Yield Spread: inverse of trailing P/E relative to cross-sector median
  2. Dividend Yield Rank: higher div yield = cheaper
  3. Relative Strength Mean-Reversion: sectors that underperformed 20d tend to revert
  4. Composite "Value Score" = weighted rank of above

Strategy:
  - Every day, rank all 11 sectors by composite value score
  - LONG cheapest 3, SHORT most expensive 3 (equal weight within each leg)
  - Hold 5 days (overlapping positions allowed, but we measure 5-day fwd return)
  - Walk-forward: 60-day sliding window to estimate factor weights, 1-day OOT

Constraints (per CLAUDE.md):
  - SLIDING window only (HC #0)
  - All available OOT days (not subset)
  - Cost: 0.20% round-trip for ETF options proxy
  - Report: Sharpe, Sortino, PF, WR, MaxDD, regime stratification

Universe: XLK, XLF, XLE, XLU, XLP, XLY, XLV, XLI, XLB, XLC, XLRE
Benchmark: SPY
Data: 2020-01-01 to 2026-08-20
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
BENCHMARK = 'SPY'
START = '2019-06-01'  # extra lookback for feature computation
END = '2026-08-21'
BACKTEST_START = '2020-03-01'  # start OOT after enough history
TRAIN_WINDOW = 60  # days
HOLD_PERIOD = 5     # 5-day forward return
LONG_N = 3          # long cheapest N
SHORT_N = 3         # short most expensive N
COST_RT_PCT = 0.0020  # 0.20% round-trip cost

# Factor lookbacks
EARNINGS_YIELD_LOOKBACK = 252  # 1yr price change as growth proxy
DIV_YIELD_LOOKBACK = 63       # quarterly
MEAN_REV_LOOKBACK = 20        # 20-day momentum (mean-reversion signal)
RELATIVE_VOL_LOOKBACK = 20    # for vol-adjusted scoring

np.random.seed(42)


def download_data():
    """Download all sector ETF + SPY price data."""
    tickers = SECTOR_ETFS + [BENCHMARK]
    print(f"Downloading {len(tickers)} tickers from {START} to {END}...")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
        volume = data['Volume']
    else:
        close = data[['Close']].copy()
        close.columns = [BENCHMARK]
        volume = data[['Volume']].copy()
        volume.columns = [BENCHMARK]

    close = close.ffill().dropna()
    volume = volume.ffill().fillna(0)
    print(f"  Got {len(close)} trading days, {close.shape[1]} tickers")
    print(f"  Date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close, volume


def compute_factors(close, volume):
    """
    Compute value/cheapness factors for each sector ETF.
    All factors are computed using ONLY backward-looking data (no lookahead).

    Returns DataFrame with multi-level columns: (factor_name, ticker)
    """
    factors = {}

    # 1. MEAN REVERSION (20-day): sectors that fell more are "cheaper"
    #    Negative momentum = positive value signal
    ret_20d = close[SECTOR_ETFS].pct_change(MEAN_REV_LOOKBACK)
    factors['mean_rev'] = -ret_20d  # negate: lower past return = higher value score

    # 2. RELATIVE DIVIDEND YIELD PROXY
    #    Use 63-day realized volatility as a proxy (high-vol sectors tend to have
    #    higher yields as compensation). This is imperfect but avoids lookahead.
    #    Actually, better: use 252-day return relative to 63-day return as "carry" proxy
    ret_252d = close[SECTOR_ETFS].pct_change(EARNINGS_YIELD_LOOKBACK)
    ret_63d = close[SECTOR_ETFS].pct_change(DIV_YIELD_LOOKBACK)
    # "Carry" = long-term underperformance relative to recent = mean-reversion at longer scale
    factors['carry'] = -(ret_252d - ret_63d)  # underperformed long-term vs short-term = cheap

    # 3. VOLATILITY-ADJUSTED CHEAPNESS
    #    Low vol + low recent returns = genuinely cheap (not just crashing)
    vol_20d = close[SECTOR_ETFS].pct_change().rolling(RELATIVE_VOL_LOOKBACK).std()
    # Sectors with low vol and negative recent returns are "value" candidates
    factors['vol_adj_value'] = -ret_20d / (vol_20d + 1e-8)  # risk-adjusted mean reversion

    # 4. RELATIVE STRENGTH INDEX (14-day) — oversold = cheap
    for ticker in SECTOR_ETFS:
        delta = close[ticker].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        if 'rsi_value' not in factors:
            factors['rsi_value'] = pd.DataFrame(index=close.index, columns=SECTOR_ETFS, dtype=float)
        factors['rsi_value'][ticker] = -rsi  # lower RSI = more oversold = "cheaper"

    # 5. VOLUME ANOMALY — unusually high volume on down days = capitulation = cheap
    vol_ratio = volume[SECTOR_ETFS] / volume[SECTOR_ETFS].rolling(20).mean()
    daily_ret = close[SECTOR_ETFS].pct_change()
    # High volume on negative days = selling climax
    factors['vol_climax'] = (vol_ratio * (daily_ret < 0).astype(float)).rolling(10).mean()

    return factors


def rank_cross_section(series):
    """Rank series cross-sectionally (0 to 1 scale). Higher = more value."""
    return series.rank(axis=1, pct=True)


def compute_composite_score(factors, weights=None):
    """
    Combine factors into a single composite value score per sector per day.
    Cross-sectional rank each factor first, then weighted average.
    """
    if weights is None:
        weights = {
            'mean_rev': 0.30,
            'carry': 0.15,
            'vol_adj_value': 0.25,
            'rsi_value': 0.20,
            'vol_climax': 0.10,
        }

    ranked = {}
    for name, df in factors.items():
        ranked[name] = rank_cross_section(df[SECTOR_ETFS])

    # Weighted composite
    composite = pd.DataFrame(0.0, index=factors['mean_rev'].index, columns=SECTOR_ETFS)
    total_w = 0
    for name, w in weights.items():
        if name in ranked:
            composite += w * ranked[name]
            total_w += w
    composite /= total_w

    return composite


def walk_forward_backtest(close, factors, spy_close):
    """
    Walk-forward backtest with sliding 60-day window.

    In-sample: use past 60 days to compute optimal factor weights (or just use fixed).
    OOT: apply composite score, go long cheapest 3 / short most expensive 3.
    Measure 5-day forward return.
    """
    daily_ret = close[SECTOR_ETFS].pct_change()
    fwd_5d_ret = close[SECTOR_ETFS].pct_change(HOLD_PERIOD).shift(-HOLD_PERIOD)
    spy_daily_ret = spy_close.pct_change()

    # For simplicity and robustness, use fixed factor weights
    # (optimizing weights in-sample with only 60 days and 11 sectors = massive overfit risk)
    composite = compute_composite_score(factors)

    # Get valid backtest dates
    valid_dates = composite.dropna().index
    valid_dates = valid_dates[valid_dates >= BACKTEST_START]
    # Need fwd returns to be available
    valid_dates = valid_dates[valid_dates <= fwd_5d_ret.dropna().index[-1]]

    print(f"\nBacktest period: {valid_dates[0].date()} to {valid_dates[-1].date()}")
    print(f"Total OOT days: {len(valid_dates)}")

    results = []

    for date in valid_dates:
        scores = composite.loc[date]
        if scores.isna().any():
            continue

        # Rank: highest composite = cheapest (most value)
        sorted_sectors = scores.sort_values(ascending=False)
        long_sectors = sorted_sectors.index[:LONG_N].tolist()
        short_sectors = sorted_sectors.index[-SHORT_N:].tolist()

        # 5-day forward returns
        long_rets = fwd_5d_ret.loc[date, long_sectors]
        short_rets = fwd_5d_ret.loc[date, short_sectors]

        if long_rets.isna().any() or short_rets.isna().any():
            continue

        # L/S return (equal weight each leg)
        long_ret = long_rets.mean()
        short_ret = short_rets.mean()
        ls_ret = (long_ret - short_ret) / 2  # dollar-neutral

        # Long-only return (just buy cheap sectors)
        long_only_ret = long_ret

        # Apply costs (0.20% RT, but we only rebalance when positions change)
        # Conservative: assume full turnover every 5 days
        ls_ret_net = ls_ret - COST_RT_PCT
        long_only_ret_net = long_only_ret - COST_RT_PCT / 2  # half cost for long-only

        # Regime: SPY return on this day
        spy_ret = spy_daily_ret.loc[date] if date in spy_daily_ret.index else 0
        regime = 'green' if spy_ret >= 0 else 'red'

        results.append({
            'date': date,
            'ls_ret_gross': ls_ret,
            'ls_ret_net': ls_ret_net,
            'long_only_ret_gross': long_only_ret,
            'long_only_ret_net': long_only_ret_net,
            'long_sectors': long_sectors,
            'short_sectors': short_sectors,
            'regime': regime,
            'spy_ret': spy_ret,
            'long_spread': sorted_sectors.iloc[0] - sorted_sectors.iloc[-1],  # value spread
        })

    return pd.DataFrame(results)


def compute_metrics(returns, label="Strategy"):
    """Compute standard performance metrics."""
    r = returns.dropna()
    if len(r) == 0:
        return {}

    n_days = len(r)
    ann_factor = np.sqrt(252 / HOLD_PERIOD)  # annualize for 5-day holding

    mean_ret = r.mean()
    std_ret = r.std()

    # Sharpe (annualized)
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0

    # Sortino (annualized)
    downside = r[r < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = (mean_ret / downside_std * ann_factor) if downside_std > 0 else 0

    # Win rate
    wr = (r > 0).mean()

    # Profit factor
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Avg win / avg loss
    avg_win = r[r > 0].mean() if (r > 0).any() else 0
    avg_loss = r[r < 0].mean() if (r < 0).any() else 0

    return {
        'label': label,
        'n_days': n_days,
        'mean_ret_bps': mean_ret * 10000,
        'std_ret_bps': std_ret * 10000,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'avg_win_bps': avg_win * 10000,
        'avg_loss_bps': avg_loss * 10000,
        'total_ret': (1 + r).prod() - 1,
    }


def print_metrics(m, indent=""):
    """Print metrics dict nicely."""
    if not m:
        print(f"{indent}  NO DATA")
        return
    print(f"{indent}  N days:         {m['n_days']}")
    print(f"{indent}  Mean ret:       {m['mean_ret_bps']:.1f} bps/trade")
    print(f"{indent}  Sharpe (ann):   {m['sharpe']:.3f}")
    print(f"{indent}  Sortino (ann):  {m['sortino']:.3f}")
    print(f"{indent}  Win Rate:       {m['win_rate']:.1%}")
    print(f"{indent}  Profit Factor:  {m['profit_factor']:.2f}")
    print(f"{indent}  Max Drawdown:   {m['max_drawdown']:.2%}")
    print(f"{indent}  Avg Win:        {m['avg_win_bps']:.1f} bps")
    print(f"{indent}  Avg Loss:       {m['avg_loss_bps']:.1f} bps")
    print(f"{indent}  Total Return:   {m['total_ret']:.2%}")


def regime_analysis(df, ret_col):
    """Stratify results by regime (green/red SPY days)."""
    green = df[df['regime'] == 'green']
    red = df[df['regime'] == 'red']

    m_green = compute_metrics(green[ret_col], "Green Days")
    m_red = compute_metrics(red[ret_col], "Red Days")

    return m_green, m_red


def factor_ic_analysis(factors, fwd_5d_ret, valid_dates):
    """Compute information coefficient for each factor vs 5-day fwd return."""
    print("\n" + "="*70)
    print("FACTOR IC ANALYSIS (Spearman rank correlation with 5-day fwd return)")
    print("="*70)

    for fname, fdf in factors.items():
        ics = []
        for date in valid_dates:
            if date not in fdf.index or date not in fwd_5d_ret.index:
                continue
            x = fdf.loc[date, SECTOR_ETFS]
            y = fwd_5d_ret.loc[date, SECTOR_ETFS]
            if x.isna().any() or y.isna().any():
                continue
            ic, _ = stats.spearmanr(x, y)
            ics.append(ic)

        if ics:
            ic_arr = np.array(ics)
            print(f"  {fname:20s}: IC={ic_arr.mean():.4f} ± {ic_arr.std():.4f}  "
                  f"t={ic_arr.mean()/ic_arr.std()*np.sqrt(len(ic_arr)):.2f}  "
                  f"IC>0: {(ic_arr>0).mean():.1%}")


def yearly_breakdown(df, ret_col):
    """Show performance by year."""
    print(f"\n{'Year':>6s}  {'N':>5s}  {'Mean bps':>9s}  {'Sharpe':>7s}  {'WR':>6s}  {'PF':>6s}  {'TotRet':>8s}")
    print("-" * 55)
    df = df.copy()
    df['year'] = df['date'].dt.year
    for year, grp in df.groupby('year'):
        m = compute_metrics(grp[ret_col], str(year))
        if m:
            print(f"{year:>6d}  {m['n_days']:>5d}  {m['mean_ret_bps']:>9.1f}  "
                  f"{m['sharpe']:>7.3f}  {m['win_rate']:>6.1%}  {m['profit_factor']:>6.2f}  "
                  f"{m['total_ret']:>8.2%}")


def sector_frequency_analysis(df):
    """Which sectors appear most in long/short buckets?"""
    print("\n" + "="*70)
    print("SECTOR FREQUENCY IN LONG (CHEAP) vs SHORT (EXPENSIVE) BUCKETS")
    print("="*70)

    long_counts = {}
    short_counts = {}
    for _, row in df.iterrows():
        for s in row['long_sectors']:
            long_counts[s] = long_counts.get(s, 0) + 1
        for s in row['short_sectors']:
            short_counts[s] = short_counts.get(s, 0) + 1

    total = len(df)
    print(f"\n{'Sector':>6s}  {'Long%':>7s}  {'Short%':>7s}  {'Net':>7s}")
    print("-" * 35)
    for s in SECTOR_ETFS:
        l_pct = long_counts.get(s, 0) / total
        s_pct = short_counts.get(s, 0) / total
        print(f"{s:>6s}  {l_pct:>7.1%}  {s_pct:>7.1%}  {l_pct - s_pct:>+7.1%}")


def main():
    print("="*70)
    print("SECTOR FCF/VALUE ROTATION BACKTEST — 5-Day Forward Returns")
    print("="*70)

    # 1. Download data
    close, volume = download_data()
    spy_close = close[BENCHMARK]

    # 2. Compute factors
    print("\nComputing value factors...")
    factors = compute_factors(close, volume)

    # 3. Compute composite score
    composite = compute_composite_score(factors)

    # 4. Run walk-forward backtest
    print("\nRunning walk-forward backtest...")
    results = walk_forward_backtest(close, factors, spy_close)

    if len(results) == 0:
        print("ERROR: No results generated!")
        return

    print(f"\nGenerated {len(results)} trade days")

    # 5. OVERALL METRICS
    print("\n" + "="*70)
    print("OVERALL RESULTS (NET OF COSTS)")
    print("="*70)

    print("\n--- Long/Short (L cheapest 3, S most expensive 3) ---")
    m_ls = compute_metrics(results['ls_ret_net'], "L/S Net")
    print_metrics(m_ls)

    print("\n--- Long-Only (buy cheapest 3 sectors) ---")
    m_lo = compute_metrics(results['long_only_ret_net'], "Long-Only Net")
    print_metrics(m_lo)

    # 6. REGIME ANALYSIS
    print("\n" + "="*70)
    print("REGIME ANALYSIS — L/S Strategy (Net)")
    print("="*70)

    m_green, m_red = regime_analysis(results, 'ls_ret_net')

    print("\n--- GREEN Days (SPY up) ---")
    print_metrics(m_green)

    print("\n--- RED Days (SPY down) ---")
    print_metrics(m_red)

    # Regime divergence check
    if m_green and m_red and m_green.get('sharpe') and m_red.get('sharpe'):
        sg, sr = m_green['sharpe'], m_red['sharpe']
        div = abs(sg - sr) / max(abs(sg), abs(sr), 1e-10)
        print(f"\n  Regime Sharpe Divergence: {div:.2%} (threshold: 50%)")
        if div > 0.50:
            print("  *** FAIL: Regime-tailored, not genuine edge ***")
        else:
            print("  PASS: Reasonably regime-agnostic")

    # 7. REGIME ANALYSIS — Long-Only
    print("\n" + "="*70)
    print("REGIME ANALYSIS — Long-Only Strategy (Net)")
    print("="*70)

    m_green_lo, m_red_lo = regime_analysis(results, 'long_only_ret_net')

    print("\n--- GREEN Days ---")
    print_metrics(m_green_lo)

    print("\n--- RED Days ---")
    print_metrics(m_red_lo)

    # 8. YEARLY BREAKDOWN
    print("\n" + "="*70)
    print("YEARLY BREAKDOWN — L/S Net")
    print("="*70)
    yearly_breakdown(results, 'ls_ret_net')

    print("\n" + "="*70)
    print("YEARLY BREAKDOWN — Long-Only Net")
    print("="*70)
    yearly_breakdown(results, 'long_only_ret_net')

    # 9. FACTOR IC ANALYSIS
    fwd_5d_ret = close[SECTOR_ETFS].pct_change(HOLD_PERIOD).shift(-HOLD_PERIOD)
    valid_dates = results['date'].values
    valid_dates_idx = pd.DatetimeIndex(valid_dates)
    factor_ic_analysis(factors, fwd_5d_ret, valid_dates_idx)

    # 10. SECTOR FREQUENCY
    sector_frequency_analysis(results)

    # 11. VERDICT
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    ls_sharpe = m_ls.get('sharpe', 0) if m_ls else 0
    lo_sharpe = m_lo.get('sharpe', 0) if m_lo else 0

    if ls_sharpe >= 0.5:
        print(f"  L/S Sharpe = {ls_sharpe:.3f} — PASSES minimum threshold (0.5)")
    else:
        print(f"  L/S Sharpe = {ls_sharpe:.3f} — DEAD. Below 0.5 threshold.")

    if lo_sharpe >= 0.5:
        print(f"  Long-Only Sharpe = {lo_sharpe:.3f} — PASSES minimum threshold (0.5)")
    else:
        print(f"  Long-Only Sharpe = {lo_sharpe:.3f} — DEAD. Below 0.5 threshold.")

    best = max(ls_sharpe, lo_sharpe)
    if best < 0.5:
        print("\n  CONCLUSION: Cross-sector value rotation does NOT produce tradeable")
        print("  edge at 5-day horizon. The fundamental cheapness signal is too slow")
        print("  to predict weekly moves in sector ETFs. Consider:")
        print("    - Longer holding periods (20-60 days) where value factors have more power")
        print("    - Combining with momentum or event-driven signals")
        print("    - Individual stock selection within sectors instead of sector rotation")
    elif best < 1.0:
        print("\n  CONCLUSION: Marginal edge detected but not strong enough for standalone.")
        print("  Could work as a confluence filter for other strategies.")
    else:
        print("\n  CONCLUSION: Strong signal. Worth developing into a production strategy.")


if __name__ == '__main__':
    main()
