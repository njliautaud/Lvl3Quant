#!/usr/bin/env python3
"""
Positioning Ecosystem Research v1 — HC #740
=============================================
MANDATE: "Breadth and flows of cash vs stock prices CTA positions fed positioning
everything has an effect" — The FULL flow ecosystem matters, not just volume.

This script builds POSITIONING PROXIES from available data and tests whether
they predict equity returns better than price-only signals.

POSITIONING FEATURES (derived from available OHLCV + ETF data):
  1. CTA PROXY — Multi-timeframe trend signals across assets. When price
     crosses 50/100/200 DMA on major assets, CTAs mechanically buy/sell.
     Aggregate "CTA pressure" = net trend alignment across assets.
  2. FED FLOW PROXY — Rate-sensitive ETF momentum (TLT, HYG, LQD).
     When rates are changing fast, positioning shifts are massive.
  3. CASH-VS-EQUITY PROXY — Relative volume/returns of defensive (SHY, BIL)
     vs risk (SPY, QQQ). Rising risk-off = outflows from equity.
  4. SECTOR ROTATION FLOW — Which sectors are getting inflows (rising
     relative volume + price) vs outflows. Identifies rotation before
     it becomes obvious.
  5. BREADTH AS FLOW — Advance/decline metrics, new highs vs lows.
     Narrow breadth = fragile/concentrated. Broad = durable.
  6. OPTIONS FLOW PROXY — Put/call ratios from overall market, VIX
     term structure as proxy for hedging demand.

STRATEGY: Use positioning ecosystem state to TIME entries on proven signals.
  - Same proven signals (oversold, vol compression, breadth collapse)
  - BUT only enter when positioning ecosystem confirms (not fighting flows)

Universe: Major ETFs + S&P 500 stocks, 2013-2026
Author: Claude (Head of Quant)
Date: 2026-07-22
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant") if os.path.exists("/home/jupiter") else Path(r"C:\Users\claude\Lvl3Quant") if os.path.exists(r"C:\Users\claude") else Path("/home/nick/Lvl3Quant")
OUTPUT = ROOT / "output" / "positioning_ecosystem_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

START_DATE = '2010-01-01'  # Longer history for macro signals
END_DATE = '2026-07-22'

SLIPPAGE_PCT = 0.001
N_PERMUTATIONS = 200
REGIME_GAP_MAX = 0.50

# ═════════════════════════════════════════════════════════════════════════════
# MACRO UNIVERSE (ETFs for positioning signals)
# ═════════════════════════════════════════════════════════════════════════════

# These ETFs give us cross-asset positioning information
MACRO_ETFS = {
    # Equity
    'SPY': 'sp500',
    'QQQ': 'nasdaq',
    'IWM': 'small_cap',
    'EFA': 'intl_dev',
    'EEM': 'emerging',
    # Bonds / Rates
    'TLT': 'long_bonds',
    'IEF': 'med_bonds',
    'SHY': 'short_bonds',
    'HYG': 'high_yield',
    'LQD': 'inv_grade',
    'TIP': 'tips',
    # Commodities
    'GLD': 'gold',
    'SLV': 'silver',
    'USO': 'oil',
    'DBA': 'agriculture',
    # Volatility
    'VXX': 'vix_short',  # May not have full history
    # Sectors
    'XLK': 'tech',
    'XLF': 'financials',
    'XLE': 'energy',
    'XLV': 'healthcare',
    'XLI': 'industrials',
    'XLP': 'staples',
    'XLU': 'utilities',
    'XLY': 'discretionary',
    'XLB': 'materials',
    'XLRE': 'real_estate',
    'XLC': 'communication',
}

# Stock universe for actual trading
STOCK_TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AVGO','ORCL',
    'CRM','AMD','ADBE','ACN','CSCO','INTC','IBM','TXN','QCOM','NOW',
    'INTU','AMAT','ADI','LRCX','KLAC','SNPS','CDNS','MRVL','FTNT','PANW',
    'JPM','BAC','WFC','GS','MS','C','BLK','SCHW','AXP','BK',
    'UNH','JNJ','LLY','PFE','MRK','ABBV','ABT','TMO','DHR','BMY',
    'AMGN','MDT','ISRG','ELV','SYK','GILD','VRTX','REGN','BSX',
    'HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG','MAR','HLT',
    'PG','KO','PEP','COST','WMT','PM','MO','CL','KMB','GIS',
    'HON','UNP','UPS','CAT','RTX','DE','BA','LMT','GD','NOC',
    'XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','DVN',
    'NEE','DUK','SO','D','AEP','SRE','EXC','XEL','WEC',
    'PLD','AMT','CCI','EQIX','PSA','O','SPG','DLR','WELL',
    'DIS','NFLX','CMCSA','CHTR','TMUS','VZ','T',
    'GE','ETN','PH','IR','CARR','OTIS',
    'FDX','DAL','UAL','LUV',
    'LIN','APD','ECL','SHW','DD','NEM','FCX','NUE','STLD',
]
STOCK_TICKERS = list(dict.fromkeys(STOCK_TICKERS))


# ═════════════════════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════════════════════

def fetch_all_data(tickers, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached = set(df['ticker'].unique())
        missing = [t for t in tickers if t not in cached]
        if not missing:
            print(f"  Cache hit: {df['ticker'].nunique()} tickers")
            return df
        frames = [df]
    else:
        missing = tickers
        frames = []

    import yfinance as yf
    batch_size = 50
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i+batch_size]
        print(f"  Batch {i//batch_size+1}: {len(batch)} tickers...")
        try:
            data = yf.download(batch, start=START_DATE, end=END_DATE,
                               progress=False, auto_adjust=True, threads=True)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                for sym in batch:
                    try:
                        close_col = data['Close']
                        if sym in close_col.columns:
                            td = pd.DataFrame({
                                'date': close_col.index,
                                'close': close_col[sym].values,
                                'open': data['Open'][sym].values if sym in data['Open'].columns else close_col[sym].values,
                                'high': data['High'][sym].values if sym in data['High'].columns else close_col[sym].values,
                                'low': data['Low'][sym].values if sym in data['Low'].columns else close_col[sym].values,
                                'volume': data['Volume'][sym].values if sym in data['Volume'].columns else 0,
                                'ticker': sym,
                            })
                            td = td.dropna(subset=['close'])
                            if len(td) > 50:
                                frames.append(td)
                    except:
                        pass
            else:
                data = data.reset_index()
                data.columns = [c.lower() for c in data.columns]
                data['ticker'] = batch[0]
                if len(data.dropna(subset=['close'])) > 50:
                    frames.append(data[['date','open','high','low','close','volume','ticker']].dropna(subset=['close']))
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)

    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date','ticker'])
    df.to_parquet(cache_path, index=False)
    print(f"  Final: {df['ticker'].nunique()} tickers, {len(df)} rows")
    return df


def classify_regime(spy_close):
    """Classify daily regime from SPY close series."""
    ret_20d = spy_close.pct_change(20)
    regime = {}
    for dt, r in ret_20d.items():
        dt_norm = pd.Timestamp(dt).normalize()
        if pd.isna(r):
            regime[dt_norm] = 'unknown'
        elif r > 0.02:
            regime[dt_norm] = 'green'
        elif r < -0.02:
            regime[dt_norm] = 'red'
        else:
            regime[dt_norm] = 'flat'
    return regime


# ═════════════════════════════════════════════════════════════════════════════
# POSITIONING ECOSYSTEM FEATURES (HC #740)
# ═════════════════════════════════════════════════════════════════════════════

def compute_cta_pressure(macro_prices):
    """
    CTA Pressure Index: aggregate trend-following signal across assets.
    CTAs mechanically buy when price > MA, sell when below.
    Aggregate across all major assets = net CTA positioning pressure.

    Returns: daily Series of CTA pressure (-1 to +1).
    """
    cta_assets = ['SPY','QQQ','IWM','EFA','EEM','TLT','GLD','USO']

    signals = pd.DataFrame(index=macro_prices.index)
    for asset in cta_assets:
        if asset not in macro_prices.columns:
            continue
        price = macro_prices[asset]
        # Multi-timeframe trend alignment
        ma50 = price.rolling(50).mean()
        ma100 = price.rolling(100).mean()
        ma200 = price.rolling(200).mean()

        # +1 if above MA, -1 if below, averaged across timeframes
        sig_50 = (price > ma50).astype(float) * 2 - 1
        sig_100 = (price > ma100).astype(float) * 2 - 1
        sig_200 = (price > ma200).astype(float) * 2 - 1

        signals[f'{asset}_trend'] = (sig_50 + sig_100 + sig_200) / 3

    if signals.empty:
        return pd.Series(0, index=macro_prices.index)

    # Average across all assets
    cta_pressure = signals.mean(axis=1)
    return cta_pressure


def compute_fed_flow(macro_prices):
    """
    Fed Flow Proxy: momentum of rate-sensitive assets.
    When TLT is falling fast = rates rising = tightening = risk-off flows.
    When HYG is falling = credit stress = risk-off.
    """
    fed_assets = {'TLT': 1.0, 'HYG': 0.7, 'LQD': 0.5, 'TIP': 0.3}

    components = []
    weights = []
    for asset, w in fed_assets.items():
        if asset in macro_prices.columns:
            # 20-day momentum, z-scored
            ret_20d = macro_prices[asset].pct_change(20)
            z = (ret_20d - ret_20d.rolling(252).mean()) / (ret_20d.rolling(252).std() + 1e-10)
            components.append(z)
            weights.append(w)

    if not components:
        return pd.Series(0, index=macro_prices.index)

    fed_flow = sum(c * w for c, w in zip(components, weights)) / sum(weights)
    return fed_flow


def compute_risk_appetite(macro_prices):
    """
    Cash-vs-Equity Proxy: relative strength of risk assets vs safe havens.
    Rising = money flowing into risk. Falling = flowing to safety.
    """
    risk_assets = ['SPY', 'QQQ', 'IWM', 'HYG']
    safe_assets = ['TLT', 'SHY', 'GLD']

    risk_rets = []
    for a in risk_assets:
        if a in macro_prices.columns:
            risk_rets.append(macro_prices[a].pct_change(10))

    safe_rets = []
    for a in safe_assets:
        if a in macro_prices.columns:
            safe_rets.append(macro_prices[a].pct_change(10))

    if not risk_rets or not safe_rets:
        return pd.Series(0, index=macro_prices.index)

    avg_risk = pd.concat(risk_rets, axis=1).mean(axis=1)
    avg_safe = pd.concat(safe_rets, axis=1).mean(axis=1)

    # Risk appetite = risk outperformance over safe
    appetite = avg_risk - avg_safe
    # Z-score it
    z = (appetite - appetite.rolling(252).mean()) / (appetite.rolling(252).std() + 1e-10)
    return z


def compute_sector_rotation(macro_prices):
    """
    Sector Rotation Flow: which sectors are gaining vs losing relative strength.
    Returns DataFrame of sector z-scored momentum.
    """
    sector_etfs = ['XLK','XLF','XLE','XLV','XLI','XLP','XLU','XLY','XLB']
    available = [s for s in sector_etfs if s in macro_prices.columns]

    if len(available) < 3:
        return pd.DataFrame()

    # 20-day momentum for each sector, relative to SPY
    spy_ret = macro_prices['SPY'].pct_change(20) if 'SPY' in macro_prices.columns else 0

    sector_momentum = pd.DataFrame(index=macro_prices.index)
    for etf in available:
        abs_ret = macro_prices[etf].pct_change(20)
        rel_ret = abs_ret - spy_ret  # Relative to market
        sector_momentum[etf] = rel_ret

    return sector_momentum


def compute_breadth_flow(stock_returns):
    """
    Breadth as Flow Signal: how broadly is money flowing into stocks?
    """
    pct_adv = (stock_returns > 0).sum(axis=1) / stock_returns.count(axis=1)
    pct_adv_10d = pct_adv.ewm(span=10).mean()
    breadth_momentum = pct_adv.diff(5)

    # New highs vs new lows (20-day)
    # Proxy: % of stocks at 20-day high vs 20-day low
    new_highs = pd.DataFrame(index=stock_returns.index)
    for col in stock_returns.columns:
        prices = stock_returns[col].cumsum()  # approximate price from returns
        new_highs[col] = prices >= prices.rolling(20).max()

    pct_new_highs = new_highs.sum(axis=1) / new_highs.count(axis=1)

    return pd.DataFrame({
        'breadth_pct_adv': pct_adv,
        'breadth_ema10': pct_adv_10d,
        'breadth_momentum_5d': breadth_momentum,
        'pct_near_20d_high': pct_new_highs,
    })


def compute_vix_term_structure(macro_prices):
    """
    VIX proxy from SPY realized vol vs implied (we don't have VIX directly
    in all cases, so use realized vol regime as proxy).
    """
    if 'SPY' not in macro_prices.columns:
        return pd.Series(0, index=macro_prices.index)

    spy_ret = macro_prices['SPY'].pct_change()
    vol_5d = spy_ret.rolling(5).std() * np.sqrt(252) * 100
    vol_20d = spy_ret.rolling(20).std() * np.sqrt(252) * 100
    vol_60d = spy_ret.rolling(60).std() * np.sqrt(252) * 100

    # Short-term vol vs long-term = term structure proxy
    # Positive = backwardation (fear), Negative = contango (calm)
    term_structure = vol_5d / (vol_60d + 1e-10) - 1

    return term_structure


def build_positioning_dashboard(macro_df, stock_returns_df):
    """
    Build the complete positioning ecosystem dashboard.
    Returns daily DataFrame with all positioning features.
    """
    # Pivot macro data to wide format
    macro_wide = macro_df.pivot_table(index='date', columns='ticker', values='close')
    macro_wide = macro_wide.sort_index()

    print("  Computing CTA pressure...")
    cta = compute_cta_pressure(macro_wide)

    print("  Computing Fed flow proxy...")
    fed = compute_fed_flow(macro_wide)

    print("  Computing risk appetite...")
    risk_app = compute_risk_appetite(macro_wide)

    print("  Computing sector rotation...")
    sector_rot = compute_sector_rotation(macro_wide)

    print("  Computing breadth flow...")
    breadth = compute_breadth_flow(stock_returns_df)

    print("  Computing VIX term structure proxy...")
    vix_ts = compute_vix_term_structure(macro_wide)

    # Combine all into positioning dashboard
    dashboard = pd.DataFrame(index=macro_wide.index)
    dashboard['cta_pressure'] = cta
    dashboard['fed_flow'] = fed
    dashboard['risk_appetite'] = risk_app
    dashboard['vix_term_structure'] = vix_ts

    # Add breadth features
    for col in breadth.columns:
        if col in ['breadth_pct_adv', 'breadth_ema10', 'breadth_momentum_5d']:
            dashboard[col] = breadth[col].reindex(dashboard.index)

    # Add sector rotation leader/laggard
    if not sector_rot.empty:
        sector_rot = sector_rot.reindex(dashboard.index)
        dashboard['sector_dispersion'] = sector_rot.std(axis=1)
        dashboard['sector_leader'] = sector_rot.idxmax(axis=1)
        dashboard['sector_laggard'] = sector_rot.idxmin(axis=1)

    # COMPOSITE POSITIONING SCORE
    # Positive = favorable positioning (flows into risk, CTAs long, Fed easy, breadth broad)
    # Negative = hostile positioning (flows to safety, CTAs short, Fed tight, breadth narrow)
    dashboard['positioning_score'] = (
        dashboard['cta_pressure'].fillna(0) * 0.25 +
        dashboard['fed_flow'].clip(-3, 3).fillna(0) / 3 * 0.25 +
        dashboard['risk_appetite'].clip(-3, 3).fillna(0) / 3 * 0.25 +
        (dashboard['breadth_ema10'].fillna(0.5) - 0.5) * 2 * 0.25
    )

    return dashboard


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY: POSITIONING-FILTERED ENTRY ON PROVEN SIGNALS
# ═════════════════════════════════════════════════════════════════════════════

def detect_oversold_signals(stock_df, rsi_period=5, rsi_thresh=20):
    """Detect oversold stocks."""
    signals = []
    for ticker, gdf in stock_df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        close = g['close']
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        for i in range(max(rsi_period + 5, 20), len(g)):
            if rsi.iloc[i] < rsi_thresh:
                signals.append({
                    'date': g['date'].iloc[i],
                    'ticker': ticker,
                    'rsi': float(rsi.iloc[i]),
                })
    return pd.DataFrame(signals)


def detect_drop_signals(stock_df, drop_pct=0.03):
    """Detect single-day drops > threshold."""
    signals = []
    for ticker, gdf in stock_df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        g['ret'] = g['close'].pct_change()
        for i in range(20, len(g)):
            if g['ret'].iloc[i] < -drop_pct:
                signals.append({
                    'date': g['date'].iloc[i],
                    'ticker': ticker,
                    'drop': float(g['ret'].iloc[i]),
                })
    return pd.DataFrame(signals)


def backtest_with_positioning(signals_df, stock_df, positioning_df, hold_days,
                               regime_map, pos_filter=None, pos_col='positioning_score',
                               pos_thresh=0):
    """
    Backtest signals with optional positioning filter.
    """
    prices_pivot = stock_df.pivot_table(index='date', columns='ticker', values='close')
    dates = prices_pivot.index

    # Merge positioning data with signals
    if pos_filter and not positioning_df.empty:
        signals_with_pos = signals_df.copy()
        signals_with_pos['date_norm'] = pd.to_datetime(signals_with_pos['date']).dt.normalize()

        pos_vals = positioning_df[pos_col].to_dict() if pos_col in positioning_df.columns else {}
        signals_with_pos['pos_value'] = signals_with_pos['date_norm'].map(pos_vals)

        if pos_filter == 'favorable':
            signals_with_pos = signals_with_pos[signals_with_pos['pos_value'] > pos_thresh]
        elif pos_filter == 'hostile':
            signals_with_pos = signals_with_pos[signals_with_pos['pos_value'] < pos_thresh]
        elif pos_filter == 'strong_favorable':
            signals_with_pos = signals_with_pos[signals_with_pos['pos_value'] > 0.3]
        elif pos_filter == 'strong_hostile':
            signals_with_pos = signals_with_pos[signals_with_pos['pos_value'] < -0.3]
        elif pos_filter == 'cta_long':
            cta_vals = positioning_df['cta_pressure'].to_dict()
            signals_with_pos['cta'] = signals_with_pos['date_norm'].map(cta_vals)
            signals_with_pos = signals_with_pos[signals_with_pos['cta'] > 0.2]
        elif pos_filter == 'cta_short':
            cta_vals = positioning_df['cta_pressure'].to_dict()
            signals_with_pos['cta'] = signals_with_pos['date_norm'].map(cta_vals)
            signals_with_pos = signals_with_pos[signals_with_pos['cta'] < -0.2]
        elif pos_filter == 'fed_easy':
            fed_vals = positioning_df['fed_flow'].to_dict()
            signals_with_pos['fed'] = signals_with_pos['date_norm'].map(fed_vals)
            signals_with_pos = signals_with_pos[signals_with_pos['fed'] > 0]
        elif pos_filter == 'fed_tight':
            fed_vals = positioning_df['fed_flow'].to_dict()
            signals_with_pos['fed'] = signals_with_pos['date_norm'].map(fed_vals)
            signals_with_pos = signals_with_pos[signals_with_pos['fed'] < 0]
        elif pos_filter == 'broad_breadth':
            breadth_vals = positioning_df['breadth_ema10'].to_dict()
            signals_with_pos['breadth'] = signals_with_pos['date_norm'].map(breadth_vals)
            signals_with_pos = signals_with_pos[signals_with_pos['breadth'] > 0.55]
        elif pos_filter == 'narrow_breadth':
            breadth_vals = positioning_df['breadth_ema10'].to_dict()
            signals_with_pos['breadth'] = signals_with_pos['date_norm'].map(breadth_vals)
            signals_with_pos = signals_with_pos[signals_with_pos['breadth'] < 0.45]
        elif pos_filter == 'risk_on':
            risk_vals = positioning_df['risk_appetite'].to_dict()
            signals_with_pos['risk'] = signals_with_pos['date_norm'].map(risk_vals)
            signals_with_pos = signals_with_pos[signals_with_pos['risk'] > 0]
        elif pos_filter == 'risk_off':
            risk_vals = positioning_df['risk_appetite'].to_dict()
            signals_with_pos['risk'] = signals_with_pos['date_norm'].map(risk_vals)
            signals_with_pos = signals_with_pos[signals_with_pos['risk'] < 0]

        signals_df = signals_with_pos

    if len(signals_df) < 10:
        return [], 0

    trades = []
    for _, sig in signals_df.iterrows():
        sig_date = pd.Timestamp(sig['date']).normalize()
        ticker = sig['ticker']

        if sig_date not in dates:
            continue
        date_idx = dates.get_loc(sig_date)
        if date_idx + 1 + hold_days >= len(dates):
            continue

        entry_date = dates[date_idx + 1]
        exit_date = dates[min(date_idx + 1 + hold_days, len(dates) - 1)]

        if ticker not in prices_pivot.columns:
            continue

        try:
            entry_price = prices_pivot.loc[entry_date, ticker]
            exit_price = prices_pivot.loc[exit_date, ticker]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price

            entry_dt = pd.Timestamp(entry_date).normalize()
            regime = regime_map.get(entry_dt, 'unknown')

            trades.append({
                'entry_date': str(entry_dt.date()),
                'ticker': ticker,
                'return_pct': float(ret),
                'regime': regime,
            })
        except:
            pass

    return trades, len(signals_df)


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_sharpe(returns):
    if len(returns) < 5: return 0.0
    return float(np.mean(returns) / (np.std(returns) + 1e-10))

def compute_sortino(returns):
    if len(returns) < 5: return 0.0
    downside = returns[returns < 0]
    ds_std = np.std(downside) if len(downside) > 1 else np.std(returns)
    return float(np.mean(returns) / (ds_std + 1e-10))

def validate(trades_df, n_perms=N_PERMUTATIONS):
    returns = trades_df['return_pct'].values
    real_sharpe = compute_sharpe(returns)
    real_sortino = compute_sortino(returns)

    count_better = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(returns))
        perm_sharpe = compute_sharpe(np.abs(returns) * signs)
        if perm_sharpe >= real_sharpe:
            count_better += 1
    perm_p = count_better / n_perms

    sharpes = {}
    for r in ['green', 'red', 'flat']:
        sub = trades_df[trades_df['regime'] == r]
        sharpes[r] = compute_sharpe(sub['return_pct'].values) if len(sub) >= 3 else 0
    sg, sr = sharpes.get('green', 0), sharpes.get('red', 0)
    denom = max(abs(sg), abs(sr), 1e-10)
    regime_gap = abs(sg - sr) / denom

    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    wr = len(wins) / len(returns) * 100
    pf = float(np.sum(wins)) / (abs(float(np.sum(losses))) + 1e-10)

    trades_df_copy = trades_df.copy()
    trades_df_copy['year'] = pd.to_datetime(trades_df_copy['entry_date']).dt.year
    yearly = trades_df_copy.groupby('year')['return_pct'].mean()
    prof_years = (yearly > 0).sum()

    return {
        'sharpe': float(real_sharpe), 'sortino': float(real_sortino),
        'mean_return_pct': float(np.mean(returns) * 100),
        'wr_pct': float(wr), 'pf': float(pf), 'n_trades': len(returns),
        'perm_p': float(perm_p), 'perm_pass': perm_p < 0.05,
        'regime_gap': float(regime_gap),
        'regime_sharpes': {k: float(v) for k, v in sharpes.items()},
        'regime_pass': regime_gap <= REGIME_GAP_MAX,
        'profitable_years': f"{prof_years}/{len(yearly)}",
        'all_pass': perm_p < 0.05 and regime_gap <= REGIME_GAP_MAX and wr > 50 and pf > 1.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("POSITIONING ECOSYSTEM RESEARCH v1 (HC #740)")
    print("Testing: CTA pressure, Fed flow, risk appetite, sector rotation,")
    print("         breadth, VIX structure — as filters on proven signals")
    print(f"Macro ETFs: {len(MACRO_ETFS)} | Stocks: {len(STOCK_TICKERS)}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("=" * 70)

    # Load data
    print("\n[1] Loading macro ETF data...")
    all_tickers = list(MACRO_ETFS.keys()) + STOCK_TICKERS
    all_tickers = list(dict.fromkeys(all_tickers))
    all_data = fetch_all_data(all_tickers, OUTPUT / "all_prices_cache.parquet")

    macro_data = all_data[all_data['ticker'].isin(MACRO_ETFS.keys())]
    stock_data = all_data[all_data['ticker'].isin(STOCK_TICKERS)]
    print(f"  Macro: {macro_data['ticker'].nunique()} ETFs")
    print(f"  Stocks: {stock_data['ticker'].nunique()} tickers")

    # Build stock returns
    stock_returns = stock_data.pivot_table(index='date', columns='ticker', values='close').pct_change()

    # Regime
    spy_pivot = macro_data[macro_data['ticker'] == 'SPY'].set_index('date')['close'].sort_index()
    regime_map = classify_regime(spy_pivot)

    # Build positioning dashboard
    print("\n[2] Building positioning ecosystem dashboard...")
    positioning = build_positioning_dashboard(macro_data, stock_returns)
    print(f"  Dashboard: {len(positioning)} days, {len(positioning.columns)} features")
    print(f"  Features: {list(positioning.columns)}")

    # Show positioning stats
    print("\n  Positioning Score Distribution:")
    ps = positioning['positioning_score'].dropna()
    print(f"    Mean: {ps.mean():.3f}, Std: {ps.std():.3f}")
    print(f"    Min: {ps.min():.3f}, Max: {ps.max():.3f}")
    print(f"    % Favorable (>0): {(ps > 0).mean()*100:.1f}%")

    # Detect signals
    print("\n[3] Detecting signals...")
    oversold_signals = detect_oversold_signals(stock_data)
    drop_signals = detect_drop_signals(stock_data)
    print(f"  Oversold (RSI<20): {len(oversold_signals)} signals")
    print(f"  Drop (>3%): {len(drop_signals)} signals")

    # Positioning filters to test
    pos_filters = [
        ('no_filter', None),
        ('favorable', 'favorable'),           # Composite positioning score > 0
        ('strong_favorable', 'strong_favorable'), # Score > 0.3
        ('hostile', 'hostile'),               # Score < 0
        ('strong_hostile', 'strong_hostile'), # Score < -0.3
        ('cta_long', 'cta_long'),             # CTAs are net long (trend aligned)
        ('cta_short', 'cta_short'),           # CTAs are net short
        ('fed_easy', 'fed_easy'),             # Rate-sensitive assets rising
        ('fed_tight', 'fed_tight'),           # Rate-sensitive assets falling
        ('risk_on', 'risk_on'),               # Risk appetite positive
        ('risk_off', 'risk_off'),             # Risk appetite negative
        ('broad_breadth', 'broad_breadth'),   # >55% advancing
        ('narrow_breadth', 'narrow_breadth'), # <45% advancing
    ]

    hold_periods = [5, 10, 21]
    signal_sets = [
        ('oversold', oversold_signals),
        ('drop3pct', drop_signals),
    ]

    # Run backtests
    print("\n[4] Running backtests...")
    all_results = {}

    for sig_name, sig_df in signal_sets:
        if sig_df.empty:
            print(f"\n  {sig_name}: NO SIGNALS")
            continue

        for pos_name, pos_filter in pos_filters:
            for hold in hold_periods:
                variant = f"{sig_name}_{pos_name}_hold{hold}d"

                trades, n_sigs = backtest_with_positioning(
                    sig_df.copy(), stock_data, positioning, hold,
                    regime_map, pos_filter=pos_filter
                )

                if len(trades) < 10:
                    all_results[variant] = {'status': 'SKIP', 'n_trades': len(trades)}
                    continue

                trades_df = pd.DataFrame(trades)
                result = validate(trades_df)
                all_results[variant] = {
                    **result,
                    'status': 'ALL_PASS' if result['all_pass'] else 'FAIL',
                    'n_signals': n_sigs,
                }

                if result['all_pass'] or pos_name == 'no_filter':
                    print(f"\n  {'✅' if result['all_pass'] else '⬜'} {variant}: "
                          f"Sharpe {result['sharpe']:.3f}, Sortino {result['sortino']:.3f}, "
                          f"WR {result['wr_pct']:.1f}%, PF {result['pf']:.2f}, "
                          f"{result['n_trades']} trades, perm p={result['perm_p']:.3f}, "
                          f"regime gap={result['regime_gap']:.2f}")

    # Positioning value-add analysis
    print(f"\n{'='*70}")
    print("POSITIONING VALUE-ADD ANALYSIS")
    print(f"{'='*70}")

    for sig_name, _ in signal_sets:
        print(f"\n  === {sig_name.upper()} ===")
        for hold in hold_periods:
            baseline_key = f"{sig_name}_no_filter_hold{hold}d"
            baseline = all_results.get(baseline_key, {})
            if not baseline.get('sharpe'):
                continue

            print(f"\n  Hold {hold}d — Baseline: Sharpe {baseline['sharpe']:.3f}, "
                  f"WR {baseline.get('wr_pct',0):.1f}%, {baseline.get('n_trades',0)} trades")

            for pos_name, _ in pos_filters:
                if pos_name == 'no_filter':
                    continue
                key = f"{sig_name}_{pos_name}_hold{hold}d"
                r = all_results.get(key, {})
                if r.get('status') == 'SKIP':
                    continue
                if not r.get('sharpe'):
                    continue

                delta = r['sharpe'] - baseline['sharpe']
                d = "+" if delta > 0 else ""
                p = "✅" if r.get('all_pass') else "❌"
                print(f"    {p} {pos_name}: Sharpe {r['sharpe']:.3f} ({d}{delta:.3f}), "
                      f"WR {r['wr_pct']:.1f}%, {r['n_trades']} trades, "
                      f"regime gap {r['regime_gap']:.2f}")

    # Key insight: which positioning filter matters most?
    print(f"\n{'='*70}")
    print("WHICH POSITIONING FACTOR MATTERS MOST?")
    print(f"{'='*70}")

    for pos_name, _ in pos_filters:
        if pos_name == 'no_filter':
            continue
        deltas = []
        for sig_name, _ in signal_sets:
            for hold in hold_periods:
                b = all_results.get(f"{sig_name}_no_filter_hold{hold}d", {})
                r = all_results.get(f"{sig_name}_{pos_name}_hold{hold}d", {})
                if b.get('sharpe') and r.get('sharpe'):
                    deltas.append(r['sharpe'] - b['sharpe'])
        if deltas:
            print(f"  {pos_name}: avg Sharpe delta {np.mean(deltas):+.3f}, "
                  f"positive {np.mean([1 if d > 0 else 0 for d in deltas])*100:.0f}%")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    passing = {k: v for k, v in all_results.items() if v.get('status') == 'ALL_PASS'}
    failing = {k: v for k, v in all_results.items() if v.get('status') == 'FAIL'}
    skipped = {k: v for k, v in all_results.items() if v.get('status') == 'SKIP'}

    print(f"\n  Total: {len(all_results)} | Pass: {len(passing)} | Fail: {len(failing)} | Skip: {len(skipped)}")

    if passing:
        print(f"\n  ✅ PASSING:")
        for name, r in sorted(passing.items(), key=lambda x: -x[1].get('sharpe', 0)):
            print(f"    {name}: Sharpe {r['sharpe']:.3f}, WR {r['wr_pct']:.1f}%, "
                  f"PF {r['pf']:.2f}, {r['n_trades']} trades")

    # Save
    report = {
        'timestamp': datetime.now().isoformat(),
        'mandate': 'HC #740: Full positioning ecosystem — CTA, Fed, risk appetite, breadth, sectors',
        'universe': {'macro_etfs': len(MACRO_ETFS), 'stocks': len(STOCK_TICKERS)},
        'period': f"{START_DATE} to {END_DATE}",
        'summary': {
            'total': len(all_results), 'passing': len(passing),
            'failing': len(failing), 'skipped': len(skipped),
        },
        'results': all_results,
    }

    with open(OUTPUT / "report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
