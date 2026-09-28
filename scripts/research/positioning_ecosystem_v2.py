#!/usr/bin/env python3
"""
Positioning Ecosystem v2 — Full HC #740 Implementation (Jupiter)

The user mandated: "Breadth and flows of cash vs stock prices CTA positions
fed positioning everything has an effect"

This builds MACRO POSITIONING INDICATORS from available price/volume data and tests
whether they improve our proven stock-picking signals. Unlike flow_enhanced_v1 which
tested stock-level flow (MFI, OBV), this tests MACRO/CROSS-ASSET positioning:

1. CTA PRESSURE: Moving average crossover intensity across major assets
   (when price crosses key MAs, CTAs mechanically buy/sell billions)
2. FED FLOW PROXY: TLT/HYG/LQD relative performance (rate expectations)
3. RISK APPETITE: SPY volume vs SHY/BIL volume ratio (equity vs safety)
4. SECTOR ROTATION: XLK/XLE/XLF/XLU relative strength (where money flows)
5. BREADTH SIGNAL: % advancing stocks (broad vs narrow market)
6. VIX TERM STRUCTURE: VIX level + contango/backwardation proxy (via VIX ETPs)
7. OPTIONS FLOW PROXY: Put/call volume ratio from equity option chains

These are used as FILTERS on our proven signals: only take trades when
macro positioning is favorable.

Uses cached stock data from flow_enhanced_signals_v1.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json, os, sys, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/positioning_ecosystem_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============== DATA LOADING ==============

def load_stock_data():
    """Load cached stock data."""
    cache = '/home/jupiter/Lvl3Quant/output/flow_enhanced_signals_v1/prices_cache.parquet'
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        col_map = {c: c.title() for c in df.columns if c.lower() in ['close','open','high','low','volume']}
        df = df.rename(columns=col_map)
        print(f"  Loaded {df['ticker'].nunique()} stocks, {len(df)} rows")
        return df
    print("ERROR: No cached stock data")
    sys.exit(1)


def load_macro_etfs(start='2013-01-01', end='2026-07-22'):
    """Download macro ETF data for positioning indicators."""
    cache = os.path.join(OUTPUT_DIR, 'macro_cache.parquet')
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        print(f"  Loaded cached macro data: {df['ticker'].nunique()} ETFs")
        return df

    etfs = {
        # Market
        'SPY': 'sp500', 'QQQ': 'nasdaq', 'IWM': 'smallcap', 'DIA': 'dow',
        # Bonds / Fed proxy
        'TLT': 'longbond', 'IEF': 'midbond', 'SHY': 'shortbond',
        'HYG': 'highyield', 'LQD': 'investment_grade', 'BIL': 'tbill',
        # Sectors
        'XLK': 'tech', 'XLE': 'energy', 'XLF': 'financials',
        'XLU': 'utilities', 'XLP': 'staples', 'XLI': 'industrials',
        'XLV': 'healthcare', 'XLRE': 'realestate',
        # Vol
        'VIXY': 'vix_short',  # VIX short-term futures ETF
        # Commodities
        'GLD': 'gold', 'USO': 'oil',
        # International
        'EEM': 'emerging', 'EFA': 'developed',
    }

    all_data = []
    tickers = list(etfs.keys())
    print(f"  Downloading {len(tickers)} macro ETFs...")
    try:
        data = yf.download(tickers, start=start, end=end, group_by='ticker', threads=True, progress=False)
        for t in tickers:
            try:
                td = data[t].dropna(subset=['Close']).copy()
                if len(td) < 100:
                    continue
                td['ticker'] = t
                td['label'] = etfs[t]
                td.index.name = 'date'
                all_data.append(td.reset_index())
            except:
                pass
    except Exception as e:
        print(f"  Download error: {e}")

    if not all_data:
        print("ERROR: No macro data downloaded")
        sys.exit(1)

    df = pd.concat(all_data, ignore_index=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == '' else c[0] for c in df.columns]

    # Normalize column names
    col_map = {c: c.title() for c in df.columns if c.lower() in ['close','open','high','low','volume']}
    df = df.rename(columns=col_map)

    df.to_parquet(cache)
    print(f"  Downloaded {df['ticker'].nunique()} macro ETFs, {len(df)} rows")
    return df


# ============== POSITIONING INDICATORS ==============

def compute_cta_pressure(macro_df):
    """
    CTA Pressure Index: How many major assets are above their 50/100/200 SMA.
    When prices cross these levels, CTA trend-followers mechanically trade.
    High = bullish positioning, Low = bearish positioning.
    """
    cta_tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'USO', 'EEM', 'EFA']
    signals = []

    for ticker in cta_tickers:
        tdf = macro_df[macro_df['ticker'] == ticker].sort_values('date').copy()
        if len(tdf) < 200:
            continue
        for ma in [50, 100, 200]:
            tdf[f'above_sma{ma}'] = (tdf['Close'] > tdf['Close'].rolling(ma).mean()).astype(float)
        signals.append(tdf[['date'] + [c for c in tdf.columns if 'above_sma' in c]])

    if not signals:
        return pd.Series(dtype=float)

    # Merge all on date, compute average
    merged = signals[0][['date']].copy()
    all_cols = []
    for i, s in enumerate(signals):
        s = s.copy()
        for c in [c for c in s.columns if 'above_sma' in c]:
            new_name = f'{c}_{i}'
            s = s.rename(columns={c: new_name})
            all_cols.append(new_name)
        merged = merged.merge(s[['date'] + [c for c in s.columns if c != 'date']], on='date', how='outer')

    # CTA pressure = average of all above_sma signals
    merged['cta_pressure'] = merged[all_cols].mean(axis=1)
    merged['date'] = pd.to_datetime(merged['date'])
    return merged.set_index('date')['cta_pressure']


def compute_fed_flow(macro_df):
    """
    Fed Flow Proxy: TLT relative strength vs SPY.
    Rising TLT/SPY = flight to safety (defensive positioning)
    Falling TLT/SPY = risk-on (aggressive positioning)
    """
    tlt = macro_df[macro_df['ticker'] == 'TLT'].set_index('date')['Close']
    spy = macro_df[macro_df['ticker'] == 'SPY'].set_index('date')['Close']
    hyg = macro_df[macro_df['ticker'] == 'HYG'].set_index('date')['Close']

    ratio = pd.DataFrame(index=tlt.index)
    ratio['tlt_spy'] = tlt / spy.reindex(tlt.index)
    ratio['hyg_spy'] = hyg.reindex(tlt.index) / spy.reindex(tlt.index)

    # Z-score of 20d change
    ratio['tlt_spy_z'] = (ratio['tlt_spy'].pct_change(20) - ratio['tlt_spy'].pct_change(20).rolling(252).mean()) / ratio['tlt_spy'].pct_change(20).rolling(252).std()

    # Fed flow: negative = risk-on (good for long equity), positive = defensive
    ratio.index = pd.to_datetime(ratio.index)
    return -ratio['tlt_spy_z']  # Invert so positive = risk-on


def compute_risk_appetite(macro_df):
    """
    Risk Appetite: SPY volume / (SHY + BIL volume).
    High ratio = money flowing into equities over safety.
    """
    spy = macro_df[macro_df['ticker'] == 'SPY'].set_index('date')[['Volume']].rename(columns={'Volume': 'spy_vol'})
    shy = macro_df[macro_df['ticker'] == 'SHY'].set_index('date')[['Volume']].rename(columns={'Volume': 'shy_vol'})
    bil = macro_df[macro_df['ticker'] == 'BIL'].set_index('date')[['Volume']].rename(columns={'Volume': 'bil_vol'})

    merged = spy.join(shy, how='outer').join(bil, how='outer')
    merged['safety_vol'] = merged['shy_vol'].fillna(0) + merged['bil_vol'].fillna(0)
    merged['risk_ratio'] = merged['spy_vol'] / merged['safety_vol'].replace(0, np.nan)

    # Smooth and z-score
    merged['risk_z'] = (merged['risk_ratio'].rolling(5).mean() - merged['risk_ratio'].rolling(252).mean()) / merged['risk_ratio'].rolling(252).std()

    merged.index = pd.to_datetime(merged.index)
    return merged['risk_z']


def compute_sector_rotation(macro_df):
    """
    Sector Rotation Signal: Defensive vs Cyclical sector relative strength.
    Defensive = XLU + XLP + XLV. Cyclical = XLK + XLF + XLI.
    Rising defensive/cyclical = risk-off. Falling = risk-on.
    """
    sectors = {}
    for t in ['XLK', 'XLE', 'XLF', 'XLU', 'XLP', 'XLI', 'XLV']:
        sdf = macro_df[macro_df['ticker'] == t].set_index('date')['Close']
        if len(sdf) > 100:
            sectors[t] = sdf

    if len(sectors) < 5:
        return pd.Series(dtype=float)

    df = pd.DataFrame(sectors)
    # Normalize to pct change over 20d
    pct = df.pct_change(20)

    defensive = pct[['XLU', 'XLP', 'XLV']].mean(axis=1)
    cyclical = pct[['XLK', 'XLF', 'XLI']].mean(axis=1)

    # Cyclical outperformance = risk-on
    rotation = cyclical - defensive
    rotation.index = pd.to_datetime(rotation.index)
    return rotation


def compute_breadth_indicator(stock_df):
    """
    Market Breadth: % of stocks with positive 20d returns.
    High breadth = broad participation (durable rally).
    Low breadth = narrow (fragile, concentrated).
    """
    stock_df2 = stock_df.copy()
    stock_df2['date'] = pd.to_datetime(stock_df2['date'])
    stock_df2['ret_20d'] = stock_df2.groupby('ticker')['Close'].pct_change(20)

    breadth = stock_df2.groupby('date').apply(lambda x: (x['ret_20d'] > 0).mean())
    return breadth


def compute_vix_regime(macro_df):
    """
    VIX Regime from VIXY ETF.
    VIXY decays in contango (normal), spikes in backwardation (crisis).
    Use 20d rate of change as fear indicator.
    """
    vixy = macro_df[macro_df['ticker'] == 'VIXY'].set_index('date')['Close']
    if len(vixy) < 50:
        return pd.Series(dtype=float)

    # Rate of change (positive = fear rising)
    vixy_roc = vixy.pct_change(5)
    vixy_roc.index = pd.to_datetime(vixy_roc.index)
    return vixy_roc


# ============== STOCK-LEVEL FEATURES ==============

def compute_stock_features(df):
    """Compute signal features per stock."""
    results = []
    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        g['ret_1d'] = g['Close'].pct_change()

        # RSI-14
        delta = g['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g['rsi'] = 100 - (100 / (1 + rs))

        # Vol percentile
        g['vol_20d'] = g['ret_1d'].rolling(20).std()
        g['vol_pctile'] = g['vol_20d'].rolling(252).rank(pct=True)

        # Volume ratio
        g['vol_avg_20d'] = g['Volume'].rolling(20).mean()
        g['vol_ratio'] = g['Volume'] / g['vol_avg_20d'].replace(0, np.nan)

        # MFI
        tp = (g['High'] + g['Low'] + g['Close']) / 3
        rmf = tp * g['Volume']
        pos = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
        g['mfi'] = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))

        # Forward returns
        for h in [5, 10, 21]:
            g[f'fwd_ret_{h}d'] = g['Close'].shift(-h) / g['Close'] - 1

        results.append(g)

    return pd.concat(results, ignore_index=True)


# ============== SIGNAL + POSITIONING FILTER ==============

def detect_base_signals(df):
    """Our proven base signals."""
    signals = {}

    # Oversold bounce (RSI < 20)
    mask = df['rsi'] < 20
    signals['oversold'] = df[mask][['date', 'ticker', 'fwd_ret_10d', 'fwd_ret_21d']].copy()

    # 3% drop
    mask = df['ret_1d'] < -0.03
    signals['drop3pct'] = df[mask][['date', 'ticker', 'fwd_ret_10d', 'fwd_ret_21d']].copy()

    # Confluence: drop + vol compression
    mask = (df['ret_1d'] < -0.03) & (df['vol_pctile'] < 0.10)
    signals['confluence'] = df[mask][['date', 'ticker', 'fwd_ret_10d', 'fwd_ret_21d']].copy()

    # MFI-enhanced
    mask = (df['ret_1d'] < -0.03) & (df['mfi'] < 20)
    signals['drop3_mfi'] = df[mask][['date', 'ticker', 'fwd_ret_10d', 'fwd_ret_21d']].copy()

    for k in signals:
        signals[k]['date'] = pd.to_datetime(signals[k]['date'])

    return signals


def apply_positioning_filter(signal_df, positioning_series, filter_name, condition_fn, hold_col='fwd_ret_10d'):
    """Apply a macro positioning filter to stock-level signals."""
    if positioning_series is None or len(positioning_series) == 0:
        return None

    merged = signal_df.merge(
        positioning_series.rename('pos_val').reset_index().rename(columns={'index': 'date'}),
        on='date', how='inner'
    )

    if len(merged) == 0:
        return None

    filtered = merged[condition_fn(merged['pos_val'])]
    if len(filtered) < 20:
        return None

    return filtered


def backtest(signal_df, hold_col='fwd_ret_10d'):
    """Quick backtest with permutation test and regime analysis."""
    if signal_df is None or len(signal_df) < 20:
        return None

    rets = signal_df[hold_col].dropna()
    if len(rets) < 20:
        return None

    # Basic stats
    mean_ret = rets.mean()
    std_ret = rets.std()
    sharpe = mean_ret / std_ret * np.sqrt(252/10) if std_ret > 0 else 0  # Approximate
    sortino_std = rets[rets < 0].std()
    sortino = mean_ret / sortino_std * np.sqrt(252/10) if sortino_std > 0 else 0
    wr = (rets > 0).mean()
    pf = abs(rets[rets > 0].sum() / rets[rets < 0].sum()) if (rets < 0).any() else float('inf')

    # Permutation test
    n_perms = 200
    count_better = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(rets))
        if (rets.values * signs).mean() >= mean_ret:
            count_better += 1
    perm_p = count_better / n_perms

    # Regime analysis (simple: split by month SPY return)
    signal_df2 = signal_df.copy()
    signal_df2['month'] = pd.to_datetime(signal_df2['date']).dt.to_period('M')

    # Load SPY for regime
    spy_cache = '/home/jupiter/Lvl3Quant/output/flow_enhanced_signals_v1/spy_cache.parquet'
    if os.path.exists(spy_cache):
        spy = pd.read_parquet(spy_cache)
        col_map = {c: c.title() for c in spy.columns if c.lower() == 'close'}
        spy = spy.rename(columns=col_map)
        spy['date'] = pd.to_datetime(spy['date'])
        spy['month'] = spy['date'].dt.to_period('M')
        monthly_spy = spy.groupby('month')['Close'].agg(['first', 'last'])
        monthly_spy['regime'] = np.where(monthly_spy['last'] > monthly_spy['first'], 'green', 'red')
        regime_map = monthly_spy['regime'].to_dict()

        signal_df2['regime'] = signal_df2['month'].map(regime_map)
        green_rets = signal_df2[signal_df2['regime'] == 'green'][hold_col].dropna()
        red_rets = signal_df2[signal_df2['regime'] == 'red'][hold_col].dropna()

        if len(green_rets) > 10 and len(red_rets) > 10:
            green_sharpe = green_rets.mean() / green_rets.std() * np.sqrt(252/10) if green_rets.std() > 0 else 0
            red_sharpe = red_rets.mean() / red_rets.std() * np.sqrt(252/10) if red_rets.std() > 0 else 0
            regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.001)
        else:
            green_sharpe = red_sharpe = regime_gap = None
    else:
        green_sharpe = red_sharpe = regime_gap = None

    return {
        'trades': len(rets),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'wr': round(wr, 3),
        'pf': round(pf, 2),
        'perm_p': round(perm_p, 3),
        'regime_gap': round(regime_gap, 3) if regime_gap is not None else None,
        'green_sharpe': round(green_sharpe, 3) if green_sharpe is not None else None,
        'red_sharpe': round(red_sharpe, 3) if red_sharpe is not None else None,
    }


def passes_gates(result):
    """Check if result passes all gates."""
    if result is None:
        return False
    if result['perm_p'] > 0.05:
        return False
    if result['regime_gap'] is not None and result['regime_gap'] > 0.50:
        return False
    if result['trades'] < 20:
        return False
    return True


# ============== MAIN ==============

def main():
    print("=" * 70)
    print("POSITIONING ECOSYSTEM v2 — FULL HC #740 IMPLEMENTATION")
    print("Testing macro positioning filters on proven stock signals")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    stock_df = load_stock_data()
    macro_df = load_macro_etfs()

    # Compute stock features
    print("\n[2] Computing stock features...")
    stock_df = compute_stock_features(stock_df)
    print(f"  Done: {stock_df['ticker'].nunique()} stocks")

    # Compute positioning indicators
    print("\n[3] Computing positioning indicators...")
    indicators = {}

    indicators['cta_pressure'] = compute_cta_pressure(macro_df)
    print(f"  CTA pressure: {len(indicators['cta_pressure'])} days")

    indicators['fed_flow'] = compute_fed_flow(macro_df)
    print(f"  Fed flow: {len(indicators['fed_flow'])} days")

    indicators['risk_appetite'] = compute_risk_appetite(macro_df)
    print(f"  Risk appetite: {len(indicators['risk_appetite'])} days")

    indicators['sector_rotation'] = compute_sector_rotation(macro_df)
    print(f"  Sector rotation: {len(indicators['sector_rotation'])} days")

    indicators['breadth'] = compute_breadth_indicator(stock_df)
    print(f"  Breadth: {len(indicators['breadth'])} days")

    indicators['vix_regime'] = compute_vix_regime(macro_df)
    print(f"  VIX regime: {len(indicators['vix_regime'])} days")

    # Detect base signals
    print("\n[4] Detecting base signals...")
    base_signals = detect_base_signals(stock_df)
    for name, sdf in base_signals.items():
        print(f"  {name}: {len(sdf)} signals")

    # Define positioning filters
    print("\n[5] Testing positioning filters...")
    filters = {
        # CTA filters
        'cta_bullish': ('cta_pressure', lambda x: x > 0.6, "CTA > 60% bullish"),
        'cta_bearish': ('cta_pressure', lambda x: x < 0.4, "CTA < 40% (contrarian buy)"),
        'cta_extreme_bear': ('cta_pressure', lambda x: x < 0.25, "CTA < 25% (extreme fear)"),

        # Fed flow
        'fed_riskon': ('fed_flow', lambda x: x > 0.5, "Fed flow risk-on"),
        'fed_riskoff': ('fed_flow', lambda x: x < -0.5, "Fed flow risk-off (contrarian)"),
        'fed_neutral': ('fed_flow', lambda x: (x > -0.5) & (x < 0.5), "Fed flow neutral"),

        # Risk appetite
        'risk_high': ('risk_appetite', lambda x: x > 0.5, "High risk appetite"),
        'risk_low': ('risk_appetite', lambda x: x < -0.5, "Low risk appetite (contrarian)"),

        # Sector rotation
        'cyclical_leading': ('sector_rotation', lambda x: x > 0.01, "Cyclicals outperforming"),
        'defensive_leading': ('sector_rotation', lambda x: x < -0.01, "Defensives outperforming (contrarian)"),

        # Breadth
        'broad_participation': ('breadth', lambda x: x > 0.55, "Broad breadth > 55%"),
        'narrow_market': ('breadth', lambda x: x < 0.40, "Narrow breadth < 40% (contrarian)"),
        'extreme_narrow': ('breadth', lambda x: x < 0.30, "Extreme narrow < 30%"),

        # VIX
        'vix_calm': ('vix_regime', lambda x: x < -0.02, "VIX falling (calm)"),
        'vix_spike': ('vix_regime', lambda x: x > 0.05, "VIX spiking (contrarian)"),
    }

    # Test all combinations
    results = []
    total = 0
    passing = 0

    for sig_name, sig_df in base_signals.items():
        for hold_col, hold_label in [('fwd_ret_10d', '10d'), ('fwd_ret_21d', '21d')]:
            # Baseline (no filter)
            baseline = backtest(sig_df, hold_col)
            if baseline:
                variant = f"{sig_name}_baseline_{hold_label}"
                baseline['variant'] = variant
                baseline['filter'] = 'none'
                baseline['signal'] = sig_name
                baseline['hold'] = hold_label
                baseline['pass'] = passes_gates(baseline)
                results.append(baseline)
                total += 1
                if baseline['pass']:
                    passing += 1

            # Filtered variants
            for filt_name, (ind_key, cond_fn, desc) in filters.items():
                ind_series = indicators.get(ind_key)
                if ind_series is None or len(ind_series) == 0:
                    continue

                filtered = apply_positioning_filter(sig_df, ind_series, filt_name, cond_fn, hold_col)
                if filtered is None:
                    continue

                result = backtest(filtered, hold_col)
                if result is None:
                    continue

                variant = f"{sig_name}_{filt_name}_{hold_label}"
                result['variant'] = variant
                result['filter'] = filt_name
                result['signal'] = sig_name
                result['hold'] = hold_label
                result['description'] = desc
                result['pass'] = passes_gates(result)
                results.append(result)
                total += 1
                if result['pass']:
                    passing += 1

    # Report
    print(f"\n{'='*70}")
    print(f"RESULTS: {passing} of {total} variants pass all gates")
    print(f"{'='*70}")

    # Sort by passing, then by regime_gap
    results_df = pd.DataFrame(results)

    # Show passing
    passing_df = results_df[results_df['pass'] == True].sort_values('sharpe', ascending=False)
    if len(passing_df) > 0:
        print(f"\n✅ PASSING VARIANTS ({len(passing_df)}):")
        for _, r in passing_df.iterrows():
            mark = "⭐" if r.get('regime_gap') is not None and r['regime_gap'] < 0.20 else "✅"
            desc = r.get('description', '')
            print(f"  {mark} {r['variant']}: Sharpe {r['sharpe']}, Sortino {r['sortino']}, "
                  f"WR {r['wr']:.1%}, PF {r['pf']}, {r['trades']}t, "
                  f"perm p={r['perm_p']}, regime gap={r['regime_gap']}")
            if desc:
                print(f"      [{desc}]")

    # Value-add analysis: which positioning filter improves signals most?
    print(f"\n{'='*70}")
    print(f"POSITIONING FILTER VALUE-ADD ANALYSIS")
    print(f"{'='*70}")

    for filt_name in sorted(set(results_df['filter']) - {'none'}):
        filt_results = results_df[results_df['filter'] == filt_name]
        if len(filt_results) == 0:
            continue

        # Compare to baselines
        deltas = []
        for _, fr in filt_results.iterrows():
            base = results_df[(results_df['signal'] == fr['signal']) &
                            (results_df['hold'] == fr['hold']) &
                            (results_df['filter'] == 'none')]
            if len(base) == 0:
                continue
            base_sharpe = base.iloc[0]['sharpe']
            delta = fr['sharpe'] - base_sharpe
            deltas.append(delta)

        if deltas:
            avg_delta = np.mean(deltas)
            pct_positive = np.mean([d > 0 for d in deltas])
            n_pass = filt_results['pass'].sum()
            mark = "✅" if avg_delta > 0 and pct_positive > 0.5 else "❌"
            desc = filt_results.iloc[0].get('description', '')
            print(f"  {mark} {filt_name}: avg Sharpe Δ={avg_delta:+.3f}, "
                  f"positive {pct_positive:.0%}, {n_pass}/{len(filt_results)} pass")
            if desc:
                print(f"      [{desc}]")

    # Per-signal best filter
    print(f"\n{'='*70}")
    print(f"BEST FILTER PER SIGNAL")
    print(f"{'='*70}")

    for sig_name in base_signals.keys():
        sig_results = results_df[(results_df['signal'] == sig_name) & (results_df['pass'] == True)]
        if len(sig_results) == 0:
            print(f"  {sig_name}: NO passing variants")
            continue

        best = sig_results.sort_values('sharpe', ascending=False).iloc[0]
        print(f"  {sig_name}: best = {best['filter']} {best['hold']}, "
              f"Sharpe {best['sharpe']}, WR {best['wr']:.1%}, regime gap {best['regime_gap']}")

    # Save report
    report = {
        'total_variants': total,
        'passing_variants': passing,
        'results': results,
    }
    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    results_df.to_parquet(os.path.join(OUTPUT_DIR, 'results.parquet'))

    print(f"\n  Report saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
