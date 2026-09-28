#!/usr/bin/env python3
"""
Industry-Level Momentum Spillover Backtest v1
==============================================
Hypothesis: Intra-sector dispersion predicts mean reversion.
  H1: When sub-industries diverge (high dispersion), laggards catch up (5-day MR).
  H2: When sub-industries converge (low dispersion, same direction), sector ETF follows.

Walk-forward sliding 60-day window. FIFO cost: 0.20% RT.
Universe: 11 SPDR sector ETFs + their industry-level sub-ETFs.
Period: 2020-01-01 to 2026-08-20.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from scipy import stats

# =============================================================================
# CONFIG
# =============================================================================
START_DATE = '2019-10-01'  # Extra lookback for rolling calcs
END_DATE = '2026-08-20'
SIGNAL_START = '2020-03-01'  # Start generating signals after warmup
TRAIN_WINDOW = 60   # Trading days
HOLDING_PERIOD = 5  # Days
COST_RT_PCT = 0.0020  # 20 bps round trip
DISPERSION_LOOKBACK = 5  # Days for dispersion calc

# Sector ETFs and their sub-industry ETFs
SECTOR_MAP = {
    'XLK': {
        'name': 'Technology',
        'subs': ['SMH', 'IGV', 'HACK', 'SKYY'],
        'sub_names': ['Semiconductors', 'Software', 'Cybersecurity', 'Cloud Computing']
    },
    'XLF': {
        'name': 'Financials',
        'subs': ['KBE', 'KIE', 'KRE', 'IAI'],
        'sub_names': ['Banks', 'Insurance', 'Regional Banks', 'Broker-Dealers']
    },
    'XLE': {
        'name': 'Energy',
        'subs': ['XOP', 'OIH', 'AMLP'],
        'sub_names': ['E&P', 'Oil Services', 'MLPs']
    },
    'XLV': {
        'name': 'Healthcare',
        'subs': ['IBB', 'XBI', 'IHI', 'XHS'],
        'sub_names': ['Biotech Large', 'Biotech Small', 'Medical Devices', 'Health Services']
    },
    'XLI': {
        'name': 'Industrials',
        'subs': ['ITA', 'XAR', 'JETS'],
        'sub_names': ['Defense', 'Aerospace & Defense', 'Airlines']
    },
    'XLY': {
        'name': 'Consumer Disc',
        'subs': ['XRT', 'XHB', 'IBUY', 'PEJ'],
        'sub_names': ['Retail', 'Homebuilders', 'Online Retail', 'Leisure']
    },
    'XLP': {
        'name': 'Consumer Staples',
        'subs': ['PBJ', 'FXG'],
        'sub_names': ['Food & Bev', 'Consumer Staples Growth']
    },
    'XLU': {
        'name': 'Utilities',
        'subs': ['TAN', 'ICLN'],
        'sub_names': ['Solar', 'Clean Energy']
    },
    'XLB': {
        'name': 'Materials',
        'subs': ['GDX', 'XME', 'REMX'],
        'sub_names': ['Gold Miners', 'Metals & Mining', 'Rare Earth']
    },
    'XLC': {
        'name': 'Comm Services',
        'subs': ['SOCL', 'IYZ'],
        'sub_names': ['Social Media', 'Telecom']
    },
    'XLRE': {
        'name': 'Real Estate',
        'subs': ['VNQ', 'MORT', 'REM'],
        'sub_names': ['REITs Broad', 'Mortgage REITs', 'Mortgage REITs Alt']
    },
}

# =============================================================================
# DATA DOWNLOAD
# =============================================================================
def download_all_data():
    """Download sector ETFs + sub-industry ETFs + SPY for regime."""
    all_tickers = set(['SPY'])
    for sector, info in SECTOR_MAP.items():
        all_tickers.add(sector)
        for sub in info['subs']:
            all_tickers.add(sub)

    all_tickers = sorted(all_tickers)
    print(f"Downloading {len(all_tickers)} tickers: {', '.join(all_tickers)}")

    data = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
        volume = data['Volume']
    else:
        close = data
        volume = data

    close = close.ffill().dropna(how='all')
    volume = volume.ffill().fillna(0)

    # Drop tickers with insufficient data (< 200 days)
    valid = close.columns[close.notna().sum() > 200]
    close = close[valid]
    volume = volume[[c for c in valid if c in volume.columns]]

    print(f"Got {len(close)} trading days, {close.shape[1]} valid tickers")
    print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    # Show which subs are available per sector
    for sector, info in SECTOR_MAP.items():
        avail = [s for s in info['subs'] if s in close.columns]
        missing = [s for s in info['subs'] if s not in close.columns]
        print(f"  {sector} ({info['name']}): {len(avail)}/{len(info['subs'])} subs "
              f"({', '.join(avail)})" + (f" [MISSING: {', '.join(missing)}]" if missing else ""))

    return close, volume


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def compute_features(close, volume):
    """
    For each sector, compute:
    - 5d returns for sector ETF and each sub-industry
    - Intra-sector dispersion (std of sub-industry 5d returns)
    - Dispersion z-score (relative to trailing 60d)
    - Laggard identification (worst-performing sub-industry)
    - Convergence signal (all subs same direction)
    - Volume-weighted convergence
    """
    returns_1d = close.pct_change()
    returns_5d = close.pct_change(DISPERSION_LOOKBACK)

    features = {}

    for sector, info in SECTOR_MAP.items():
        if sector not in close.columns:
            continue

        avail_subs = [s for s in info['subs'] if s in close.columns]
        if len(avail_subs) < 2:
            continue

        sector_key = sector

        # Sub-industry returns
        sub_ret_5d = returns_5d[avail_subs]
        sector_ret_5d = returns_5d[sector]

        # 1. DISPERSION: std of sub-industry 5d returns
        dispersion = sub_ret_5d.std(axis=1)

        # 2. DISPERSION Z-SCORE: rolling z-score of dispersion
        disp_mean = dispersion.rolling(TRAIN_WINDOW, min_periods=20).mean()
        disp_std = dispersion.rolling(TRAIN_WINDOW, min_periods=20).std()
        disp_z = (dispersion - disp_mean) / (disp_std + 1e-8)

        # 3. LAGGARD: which sub underperformed the most
        laggard_idx = sub_ret_5d.idxmin(axis=1)
        leader_idx = sub_ret_5d.idxmax(axis=1)

        # Spread: leader - laggard return
        leader_ret = sub_ret_5d.max(axis=1)
        laggard_ret = sub_ret_5d.min(axis=1)
        spread = leader_ret - laggard_ret

        # 4. CONVERGENCE: fraction of subs moving same direction
        signs = np.sign(sub_ret_5d)
        def _convergence(row):
            mode_vals = row.mode()
            if len(mode_vals) == 0:
                return 0.5
            return (row == mode_vals.iloc[0]).mean()
        convergence = signs.apply(_convergence, axis=1)

        # All same direction flag
        all_same_dir = (signs.nunique(axis=1) == 1).astype(float)

        # 5. VOLUME SURGE: are subs seeing unusual volume?
        sub_vol = volume[[s for s in avail_subs if s in volume.columns]]
        if len(sub_vol.columns) > 0:
            vol_ratio = sub_vol / sub_vol.rolling(20, min_periods=10).mean()
            avg_vol_surge = vol_ratio.mean(axis=1)
        else:
            avg_vol_surge = pd.Series(1.0, index=close.index)

        # 6. LAGGARD vs SECTOR: how much did laggard underperform sector?
        laggard_underperf = laggard_ret - sector_ret_5d

        # 7. SECTOR MOMENTUM: 20d momentum of sector ETF
        sector_mom_20d = close[sector].pct_change(20)

        features[sector_key] = pd.DataFrame({
            'dispersion': dispersion,
            'disp_z': disp_z,
            'spread': spread,
            'convergence': convergence,
            'all_same_dir': all_same_dir,
            'vol_surge': avg_vol_surge,
            'laggard_underperf': laggard_underperf,
            'sector_mom_20d': sector_mom_20d,
            'sector_ret_5d': sector_ret_5d,
            'laggard_ret': laggard_ret,
            'leader_ret': leader_ret,
        }, index=close.index)

        # Store laggard ticker for each day
        features[sector_key]['laggard_ticker'] = laggard_idx
        features[sector_key]['leader_ticker'] = leader_idx

    return features, returns_5d


# =============================================================================
# REGIME CLASSIFICATION
# =============================================================================
def classify_regime(close):
    """Classify each day as bull/bear/flat based on SPY."""
    if 'SPY' not in close.columns:
        return pd.Series('flat', index=close.index)

    spy = close['SPY']
    spy_ret = spy.pct_change(5)

    regime = pd.Series('flat', index=close.index)
    regime[spy_ret > 0.01] = 'bull'
    regime[spy_ret < -0.01] = 'bear'

    return regime


# =============================================================================
# STRATEGY 1: HIGH DISPERSION → BUY LAGGARD (MEAN REVERSION)
# =============================================================================
def strategy_laggard_mr(features, close, returns_5d, regime):
    """
    When intra-sector dispersion z-score > 1.0, buy the laggard sub-industry ETF.
    Hold for 5 days. Sliding 60-day calibration window.
    """
    print("\n" + "="*80)
    print("STRATEGY 1: HIGH DISPERSION → BUY LAGGARD (INTRA-SECTOR MEAN REVERSION)")
    print("="*80)

    all_trades = []

    for sector, feat in features.items():
        feat = feat.dropna(subset=['dispersion', 'disp_z'])
        signal_mask = feat.index >= pd.Timestamp(SIGNAL_START)
        feat_signal = feat[signal_mask]

        for i in range(len(feat_signal)):
            row = feat_signal.iloc[i]
            date = feat_signal.index[i]

            # HIGH DISPERSION SIGNAL: z-score > 1.0
            if row['disp_z'] > 1.0 and pd.notna(row['laggard_ticker']):
                laggard = row['laggard_ticker']

                if laggard not in close.columns:
                    continue

                # Find entry and exit dates
                date_loc = close.index.get_loc(date)
                if date_loc + HOLDING_PERIOD >= len(close):
                    continue

                entry_price = close[laggard].iloc[date_loc]
                exit_price = close[laggard].iloc[date_loc + HOLDING_PERIOD]

                if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                    continue

                gross_ret = (exit_price / entry_price) - 1
                net_ret = gross_ret - COST_RT_PCT

                day_regime = regime.iloc[date_loc] if date_loc < len(regime) else 'flat'

                all_trades.append({
                    'date': date,
                    'sector': sector,
                    'ticker': laggard,
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_return': gross_ret,
                    'net_return': net_ret,
                    'disp_z': row['disp_z'],
                    'spread': row['spread'],
                    'regime': day_regime,
                })

    if not all_trades:
        print("  NO TRADES GENERATED")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    print(f"\n  Total trades: {len(trades_df)}")
    print(f"  Date range: {trades_df['date'].min().date()} to {trades_df['date'].max().date()}")
    print(f"  Sectors traded: {trades_df['sector'].nunique()}")

    return trades_df


# =============================================================================
# STRATEGY 2: CONVERGENCE + VOLUME → SECTOR ETF DIRECTION
# =============================================================================
def strategy_convergence(features, close, returns_5d, regime):
    """
    When all sub-industries move same direction on above-avg volume,
    buy/sell the sector ETF in that direction. Hold 5 days.
    """
    print("\n" + "="*80)
    print("STRATEGY 2: CONVERGENCE + VOLUME → SECTOR ETF FOLLOWS")
    print("="*80)

    all_trades = []

    for sector, feat in features.items():
        feat = feat.dropna(subset=['all_same_dir', 'vol_surge'])
        signal_mask = feat.index >= pd.Timestamp(SIGNAL_START)
        feat_signal = feat[signal_mask]

        if sector not in close.columns:
            continue

        for i in range(len(feat_signal)):
            row = feat_signal.iloc[i]
            date = feat_signal.index[i]

            # CONVERGENCE SIGNAL: all subs same dir + volume surge
            if row['all_same_dir'] == 1.0 and row['vol_surge'] > 1.2:
                # Direction: follow the consensus
                direction = 'long' if row['leader_ret'] > 0 else 'short'

                date_loc = close.index.get_loc(date)
                if date_loc + HOLDING_PERIOD >= len(close):
                    continue

                entry_price = close[sector].iloc[date_loc]
                exit_price = close[sector].iloc[date_loc + HOLDING_PERIOD]

                if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                    continue

                if direction == 'long':
                    gross_ret = (exit_price / entry_price) - 1
                else:
                    gross_ret = (entry_price / exit_price) - 1

                net_ret = gross_ret - COST_RT_PCT

                day_regime = regime.iloc[date_loc] if date_loc < len(regime) else 'flat'

                all_trades.append({
                    'date': date,
                    'sector': sector,
                    'ticker': sector,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_return': gross_ret,
                    'net_return': net_ret,
                    'convergence': row['convergence'],
                    'vol_surge': row['vol_surge'],
                    'regime': day_regime,
                })

    if not all_trades:
        print("  NO TRADES GENERATED")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    print(f"\n  Total trades: {len(trades_df)}")
    print(f"  Date range: {trades_df['date'].min().date()} to {trades_df['date'].max().date()}")

    return trades_df


# =============================================================================
# STRATEGY 3: ADAPTIVE THRESHOLD (WALK-FORWARD)
# =============================================================================
def strategy_adaptive_wf(features, close, returns_5d, regime):
    """
    Walk-forward sliding window:
    - Train on 60 days: find optimal dispersion threshold that maximizes
      laggard-buying Sharpe ratio
    - Test on next day: if dispersion > threshold, buy laggard
    - Slide forward
    """
    print("\n" + "="*80)
    print("STRATEGY 3: WALK-FORWARD ADAPTIVE DISPERSION THRESHOLD")
    print("="*80)

    all_trades = []

    for sector, feat in features.items():
        if sector not in close.columns:
            continue

        avail_subs = [s for s in SECTOR_MAP[sector]['subs'] if s in close.columns]
        if len(avail_subs) < 2:
            continue

        feat = feat.dropna(subset=['dispersion', 'laggard_ticker'])
        dates = feat.index[feat.index >= pd.Timestamp(SIGNAL_START)]

        for date in dates:
            date_loc_feat = feat.index.get_loc(date)

            # Need TRAIN_WINDOW days before
            if date_loc_feat < TRAIN_WINDOW:
                continue

            # TRAIN: look at last 60 days
            train_feat = feat.iloc[date_loc_feat - TRAIN_WINDOW:date_loc_feat]

            # In training window: for each day, compute what laggard-buying would have returned
            train_rets = []
            train_disps = []
            for j in range(len(train_feat)):
                t_date = train_feat.index[j]
                t_row = train_feat.iloc[j]
                laggard = t_row['laggard_ticker']

                if laggard not in close.columns:
                    continue

                t_date_loc = close.index.get_loc(t_date)
                if t_date_loc + HOLDING_PERIOD >= len(close):
                    continue

                ep = close[laggard].iloc[t_date_loc]
                xp = close[laggard].iloc[t_date_loc + HOLDING_PERIOD]
                if pd.isna(ep) or pd.isna(xp) or ep == 0:
                    continue

                train_rets.append((xp / ep) - 1 - COST_RT_PCT)
                train_disps.append(t_row['dispersion'])

            if len(train_rets) < 10:
                continue

            train_rets = np.array(train_rets)
            train_disps = np.array(train_disps)

            # Find threshold: dispersion percentile where avg return is positive
            best_thresh = None
            best_sharpe = -999

            for pctile in [50, 60, 70, 80, 90]:
                thresh = np.percentile(train_disps, pctile)
                mask = train_disps > thresh
                if mask.sum() < 3:
                    continue
                subset_rets = train_rets[mask]
                if subset_rets.std() == 0:
                    continue
                sr = subset_rets.mean() / subset_rets.std() * np.sqrt(252 / HOLDING_PERIOD)
                if sr > best_sharpe:
                    best_sharpe = sr
                    best_thresh = thresh

            if best_thresh is None or best_sharpe < 0:
                continue

            # TEST: apply threshold to current day
            current_disp = feat.iloc[date_loc_feat]['dispersion']
            if current_disp <= best_thresh:
                continue

            laggard = feat.iloc[date_loc_feat]['laggard_ticker']
            if laggard not in close.columns:
                continue

            date_loc_close = close.index.get_loc(date)
            if date_loc_close + HOLDING_PERIOD >= len(close):
                continue

            entry_price = close[laggard].iloc[date_loc_close]
            exit_price = close[laggard].iloc[date_loc_close + HOLDING_PERIOD]

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                continue

            gross_ret = (exit_price / entry_price) - 1
            net_ret = gross_ret - COST_RT_PCT

            day_regime = regime.iloc[date_loc_close] if date_loc_close < len(regime) else 'flat'

            all_trades.append({
                'date': date,
                'sector': sector,
                'ticker': laggard,
                'direction': 'long',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_return': gross_ret,
                'net_return': net_ret,
                'train_sharpe': best_sharpe,
                'threshold': best_thresh,
                'regime': day_regime,
            })

    if not all_trades:
        print("  NO TRADES GENERATED")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    print(f"\n  Total trades: {len(trades_df)}")
    print(f"  Date range: {trades_df['date'].min().date()} to {trades_df['date'].max().date()}")

    return trades_df


# =============================================================================
# ANALYSIS & REPORTING
# =============================================================================
def analyze_trades(trades_df, strategy_name):
    """Full performance analysis with regime stratification."""
    if trades_df.empty:
        print(f"\n  {strategy_name}: NO TRADES")
        return None

    rets = trades_df['net_return'].values

    # Core metrics
    n_trades = len(rets)
    win_rate = (rets > 0).mean()
    avg_ret = rets.mean()

    # Annualized Sharpe (5-day holding)
    periods_per_year = 252 / HOLDING_PERIOD
    sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year) if rets.std() > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = downside.std() if len(downside) > 1 else rets.std()
    sortino = (rets.mean() / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0

    # Profit factor
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown (cumulative)
    cum_rets = (1 + pd.Series(rets)).cumprod()
    rolling_max = cum_rets.cummax()
    drawdowns = (cum_rets - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # T-stat
    t_stat = stats.ttest_1samp(rets, 0).statistic if len(rets) > 1 else 0

    print(f"\n{'='*60}")
    print(f"  {strategy_name} RESULTS")
    print(f"{'='*60}")
    print(f"  Trades:         {n_trades}")
    print(f"  Win Rate:       {win_rate:.1%}")
    print(f"  Avg Return:     {avg_ret:.4%} per trade")
    print(f"  Sharpe:         {sharpe:.3f}")
    print(f"  Sortino:        {sortino:.3f}")
    print(f"  Profit Factor:  {profit_factor:.3f}")
    print(f"  Max Drawdown:   {max_dd:.2%}")
    print(f"  T-stat:         {t_stat:.3f}")
    print(f"  Total Return:   {cum_rets.iloc[-1] - 1:.2%}")

    # Regime stratification
    print(f"\n  --- Regime Stratification ---")
    for reg in ['bull', 'bear', 'flat']:
        reg_trades = trades_df[trades_df['regime'] == reg]
        if len(reg_trades) < 5:
            print(f"  {reg:>5}: {len(reg_trades)} trades (too few)")
            continue
        reg_rets = reg_trades['net_return'].values
        reg_sharpe = (reg_rets.mean() / reg_rets.std()) * np.sqrt(periods_per_year) if reg_rets.std() > 0 else 0
        reg_wr = (reg_rets > 0).mean()
        reg_pf = reg_rets[reg_rets > 0].sum() / abs(reg_rets[reg_rets < 0].sum()) if (reg_rets < 0).any() else float('inf')
        print(f"  {reg:>5}: {len(reg_trades):4d} trades | Sharpe={reg_sharpe:+.3f} | WR={reg_wr:.1%} | PF={reg_pf:.2f}")

    # Regime balance check
    sharpes = {}
    for reg in ['bull', 'bear']:
        reg_trades = trades_df[trades_df['regime'] == reg]
        if len(reg_trades) >= 5:
            reg_rets = reg_trades['net_return'].values
            sharpes[reg] = (reg_rets.mean() / reg_rets.std()) * np.sqrt(periods_per_year) if reg_rets.std() > 0 else 0

    if 'bull' in sharpes and 'bear' in sharpes:
        max_s = max(abs(sharpes['bull']), abs(sharpes['bear']))
        if max_s > 0:
            regime_divergence = abs(sharpes['bull'] - sharpes['bear']) / max_s
            regime_ok = "PASS" if regime_divergence <= 0.50 else "FAIL (regime-tailored)"
            print(f"\n  Regime divergence: {regime_divergence:.2f} → {regime_ok}")

    # Per-sector breakdown
    print(f"\n  --- Per-Sector Breakdown ---")
    for sector in sorted(trades_df['sector'].unique()):
        sec_trades = trades_df[trades_df['sector'] == sector]
        sec_rets = sec_trades['net_return'].values
        sec_wr = (sec_rets > 0).mean()
        sec_sharpe = (sec_rets.mean() / sec_rets.std()) * np.sqrt(periods_per_year) if sec_rets.std() > 0 else 0
        print(f"  {sector:>5}: {len(sec_trades):4d} trades | Sharpe={sec_sharpe:+.3f} | WR={sec_wr:.1%} | Avg={sec_rets.mean():.4%}")

    # Year-by-year
    print(f"\n  --- Year-by-Year ---")
    trades_df_copy = trades_df.copy()
    trades_df_copy['year'] = trades_df_copy['date'].dt.year
    for yr in sorted(trades_df_copy['year'].unique()):
        yr_trades = trades_df_copy[trades_df_copy['year'] == yr]
        yr_rets = yr_trades['net_return'].values
        yr_sharpe = (yr_rets.mean() / yr_rets.std()) * np.sqrt(periods_per_year) if yr_rets.std() > 0 else 0
        yr_wr = (yr_rets > 0).mean()
        print(f"  {yr}: {len(yr_trades):4d} trades | Sharpe={yr_sharpe:+.3f} | WR={yr_wr:.1%} | Tot={yr_rets.sum():.2%}")

    verdict = "ALIVE" if sharpe >= 0.5 else "DEAD (Sharpe < 0.5)"
    print(f"\n  >>> VERDICT: {verdict} <<<")

    return {
        'n_trades': n_trades,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        't_stat': t_stat,
        'total_return': float(cum_rets.iloc[-1] - 1),
        'verdict': verdict,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("INDUSTRY-LEVEL MOMENTUM SPILLOVER BACKTEST v1")
    print(f"Period: {SIGNAL_START} to {END_DATE}")
    print(f"Walk-forward: {TRAIN_WINDOW}d sliding window, {HOLDING_PERIOD}d holding period")
    print(f"Cost: {COST_RT_PCT*100:.2f}% round-trip")
    print("="*80)

    # Download data
    close, volume = download_all_data()

    # Compute features
    features, returns_5d = compute_features(close, volume)
    print(f"\nFeatures computed for {len(features)} sectors")

    # Regime classification
    regime = classify_regime(close)
    regime_counts = regime.value_counts()
    print(f"\nRegime distribution:")
    for r, c in regime_counts.items():
        print(f"  {r}: {c} days ({c/len(regime)*100:.1f}%)")

    # Run strategies
    results = {}

    # Strategy 1: Fixed threshold laggard mean reversion
    trades1 = strategy_laggard_mr(features, close, returns_5d, regime)
    r1 = analyze_trades(trades1, "S1: High Dispersion → Buy Laggard")
    results['S1_laggard_mr'] = r1

    # Strategy 2: Convergence momentum
    trades2 = strategy_convergence(features, close, returns_5d, regime)
    r2 = analyze_trades(trades2, "S2: Convergence + Volume → Follow")
    results['S2_convergence'] = r2

    # Strategy 3: Walk-forward adaptive
    trades3 = strategy_adaptive_wf(features, close, returns_5d, regime)
    r3 = analyze_trades(trades3, "S3: Walk-Forward Adaptive Threshold")
    results['S3_adaptive_wf'] = r3

    # Combined: all non-overlapping trades
    all_trades = pd.concat([t for t in [trades1, trades2, trades3] if not t.empty], ignore_index=True)
    if not all_trades.empty:
        all_trades = all_trades.sort_values('date')
        r_all = analyze_trades(all_trades, "COMBINED (All Strategies)")
        results['combined'] = r_all

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    for name, r in results.items():
        if r is not None:
            print(f"  {name:25s}: Sharpe={r['sharpe']:+.3f} | Sortino={r['sortino']:+.3f} | "
                  f"WR={r['win_rate']:.1%} | PF={r['profit_factor']:.2f} | "
                  f"DD={r['max_drawdown']:.1%} | {r['verdict']}")

    # Save results
    output_dir = Path('/home/jupiter/Lvl3Quant/output/growth_research/industry_spillover_v1')
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({k: v for k, v in results.items() if v is not None}, f, indent=2, default=str)

    # Save trade logs
    for name, trades in [('s1', trades1), ('s2', trades2), ('s3', trades3)]:
        if not trades.empty:
            trades.to_csv(output_dir / f'{name}_trades.csv', index=False)

    print(f"\nResults saved to {output_dir}")

    return results


if __name__ == '__main__':
    main()
