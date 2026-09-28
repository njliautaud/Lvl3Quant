#!/usr/bin/env python3
"""
Flow-Enhanced Signal Research v1 — HC #737 + HC #739
=====================================================
MANDATE: All research must incorporate actual flow data (institutional volume,
accumulation/distribution, money flow) — not just price action.

This script takes our PROVEN signals (oversold bounce, vol compression, volume
climax, breadth collapse contrarian) and tests whether adding flow filters
improves them. Flow signals derived from OHLCV:

FLOW FEATURES:
  1. OBV (On-Balance Volume) — cumulative volume direction
  2. MFI (Money Flow Index) — volume-weighted RSI
  3. A/D Line (Accumulation/Distribution) — close location value × volume
  4. VWAP deviation — price vs volume-weighted average
  5. Volume Profile — relative volume vs 20d average
  6. Price-Volume Divergence — price up but volume down (or vice versa)

HYPOTHESIS: Flow confirmation improves signal quality:
  - Oversold + HIGH institutional accumulation → stronger bounce
  - Vol compression + POSITIVE flow divergence → better breakout
  - Breadth collapse + flow reversal (accumulation starting) → better contrarian entry

Universe: 360+ S&P 500 stocks, 2013-2026
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
OUTPUT = ROOT / "output" / "flow_enhanced_signals_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# UNIVERSE
# ═════════════════════════════════════════════════════════════════════════════

TICKERS = [
    'AAPL','MSFT','GOOGL','GOOG','AMZN','META','NVDA','TSLA','AVGO','ORCL',
    'CRM','AMD','ADBE','ACN','CSCO','INTC','IBM','TXN','QCOM','NOW',
    'INTU','AMAT','ADI','LRCX','KLAC','SNPS','CDNS','MRVL','FTNT','PANW',
    'CRWD','WDAY','TEAM','ZS','DDOG','HUBS','NET','MDB','SNOW','PLTR',
    'ABNB','DASH','COIN','PYPL','SHOP','U','RBLX','PINS','SNAP',
    'JPM','BAC','WFC','GS','MS','C','BLK','SCHW','AXP','BK',
    'USB','PNC','TFC','COF','CME','ICE','MCO','SPGI','MSCI','FIS',
    'FISV','ADP','NDAQ','MMC','AON','CB','AFL','MET','PRU','ALL',
    'TRV','AIG','HIG','GL','BRO','WRB','CINF','RJF','NTRS',
    'UNH','JNJ','LLY','PFE','MRK','ABBV','ABT','TMO','DHR','BMY',
    'AMGN','MDT','ISRG','ELV','SYK','GILD','VRTX','REGN','BSX','ZBH',
    'BDX','IQV','A','DXCM','IDXX','PODD','ALGN','HOLX','MTD','WAT',
    'HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG','MAR','HLT',
    'ORLY','AZO','ROST','DG','DLTR','BBY','POOL','DHI','LEN','PHM',
    'NVR','GPC','GRMN','EBAY','ETSY','LULU','DECK','ON','TPR','RL',
    'PG','KO','PEP','COST','WMT','PM','MO','CL','KMB','GIS',
    'K','SJM','HSY','MNST','STZ','TSN','HRL','CPB','CAG','MKC',
    'CHD','CLX','EL','KHC','KDP','MDLZ','SYY','KR','TGT','WBA',
    'HON','UNP','UPS','CAT','RTX','DE','BA','LMT','GD','NOC',
    'GE','MMM','EMR','ROK','ITW','PH','ETN','IR','CARR','OTIS',
    'AME','DOV','NDSN','SWK','XYL','GNRC','TT','WAB','CSX','NSC',
    'FDX','DAL','UAL','LUV','AAL','JBHT','CHRW','EXPD','ODFL','SAIA',
    'XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','DVN',
    'HAL','BKR','FANG','TRGP','WMB',
    'LIN','APD','ECL','SHW','DD','NEM','FCX','NUE','STLD','CF',
    'VMC','MLM','ALB','PPG','DOW','IP','PKG','AVY','EMN',
    'NEE','DUK','SO','D','AEP','SRE','EXC','XEL','WEC',
    'ED','AEE','DTE','CMS','CNP','PNW','EVRG','NI','ATO','OGE',
    'PLD','AMT','CCI','EQIX','PSA','O','SPG','DLR','WELL','AVB',
    'EQR','VTR','IRM','ARE','MAA','UDR','KIM','REG','CPT','HST',
    'DIS','NFLX','CMCSA','CHTR','TMUS','VZ','T','FOX','FOXA','OMC',
    'MTCH','LYV','WBD','EA','TTWO','YELP',
    'RIVN','LCID','SOFI','HOOD','DKNG','PENN','MGM','CZR','WYNN','LVS',
    'MELI','SE','GRAB','BABA','JD','PDD','BIDU','NIO','LI','XPEV',
    'SPOT','ROKU','ZM','DOCU','OKTA','TWLO','PATH','BILL','FOUR','GTLB',
    'RRC','AR','EQT','CHRD','SM','MTDR','MGY',
    'CLF','AA','VALE','RIO','BHP','SCCO','TECK','WPM','GOLD',
    'MOS','FMC','CTVA','AGCO','CNH','FSLR','ENPH','SEDG',
    'RUN','JKS','CSIQ','ARRY','CWEN','AES','ORA',
]
TICKERS = list(dict.fromkeys(TICKERS))

START_DATE = '2013-01-01'
END_DATE = '2026-07-22'

SLIPPAGE_PCT = 0.001
N_PERMUTATIONS = 200
REGIME_GAP_MAX = 0.50


# ═════════════════════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════════════════════

def fetch_prices(tickers, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached = set(df['ticker'].unique())
        missing = [t for t in tickers if t not in cached]
        if not missing:
            print(f"  Cache hit: {df['ticker'].nunique()} tickers")
            return df
        print(f"  Partial cache, downloading {len(missing)} more...")
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
                            if len(td) > 100:
                                frames.append(td)
                    except:
                        pass
            else:
                data = data.reset_index()
                data.columns = [c.lower() for c in data.columns]
                data['ticker'] = batch[0]
                if len(data.dropna(subset=['close'])) > 100:
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


def fetch_spy(cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    import yfinance as yf
    spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True).reset_index()
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    spy.rename(columns={'Date':'date','Close':'close'}, inplace=True)
    spy.columns = [c.lower() for c in spy.columns]
    spy = spy[['date','close']].copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy.to_parquet(cache_path, index=False)
    return spy


def classify_regime(spy_df):
    spy = spy_df.sort_values('date').copy()
    spy['ret_20d'] = spy['close'].pct_change(20)
    regime = {}
    for _, row in spy.iterrows():
        dt = pd.Timestamp(row['date']).normalize()
        r = row['ret_20d']
        if pd.isna(r):
            regime[dt] = 'unknown'
        elif r > 0.02:
            regime[dt] = 'green'
        elif r < -0.02:
            regime[dt] = 'red'
        else:
            regime[dt] = 'flat'
    return regime


# ═════════════════════════════════════════════════════════════════════════════
# FLOW FEATURES (HC #737 / HC #739)
# ═════════════════════════════════════════════════════════════════════════════

def compute_flow_features(df):
    """
    Compute flow features per ticker. Input df must have:
    date, open, high, low, close, volume, ticker
    Returns df with additional flow columns.
    """
    results = []

    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        if len(g) < 50:
            continue

        close = g['close'].values.astype(float)
        high = g['high'].values.astype(float)
        low = g['low'].values.astype(float)
        opn = g['open'].values.astype(float)
        volume = g['volume'].values.astype(float)

        # 1. OBV (On-Balance Volume)
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        g['obv'] = obv

        # OBV trend (20-day slope normalized)
        obv_series = pd.Series(obv)
        g['obv_slope_20d'] = obv_series.rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x / (np.abs(x).mean() + 1e-10), 1)[0] if len(x) == 20 else 0,
            raw=True
        )

        # 2. MFI (Money Flow Index) — 14-period
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        pos_flow = np.zeros(len(close))
        neg_flow = np.zeros(len(close))
        for i in range(1, len(close)):
            if typical_price[i] > typical_price[i-1]:
                pos_flow[i] = raw_money_flow[i]
            else:
                neg_flow[i] = raw_money_flow[i]

        pos_series = pd.Series(pos_flow)
        neg_series = pd.Series(neg_flow)
        pos_14 = pos_series.rolling(14).sum()
        neg_14 = neg_series.rolling(14).sum()
        mfi = 100 - (100 / (1 + pos_14 / (neg_14 + 1e-10)))
        g['mfi'] = mfi.values

        # 3. A/D Line (Accumulation/Distribution)
        clv = np.where(high != low,
                       ((close - low) - (high - close)) / (high - low),
                       0)
        ad_volume = clv * volume
        ad_line = np.cumsum(ad_volume)
        g['ad_line'] = ad_line

        # A/D trend (20-day)
        ad_series = pd.Series(ad_line)
        g['ad_slope_20d'] = ad_series.rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x / (np.abs(x).mean() + 1e-10), 1)[0] if len(x) == 20 else 0,
            raw=True
        )

        # 4. Relative Volume (vs 20d average)
        vol_series = pd.Series(volume)
        g['rel_volume'] = volume / (vol_series.rolling(20).mean().values + 1e-10)

        # 5. Price-Volume Divergence
        # Price up but volume declining = distribution (bearish divergence)
        # Price down but volume rising = accumulation (bullish for contrarian)
        price_ret_5d = pd.Series(close).pct_change(5).values
        vol_ret_5d = vol_series.pct_change(5).values
        g['price_vol_divergence'] = price_ret_5d * -vol_ret_5d  # positive = divergence

        # 6. VWAP deviation (20-day)
        cum_vol = vol_series.rolling(20).sum()
        cum_pv = (pd.Series(close) * vol_series).rolling(20).sum()
        vwap_20d = cum_pv / (cum_vol + 1e-10)
        g['vwap_dev'] = (close - vwap_20d.values) / (vwap_20d.values + 1e-10)

        # 7. Flow Score (composite: normalized OBV slope + MFI direction + AD slope)
        # Positive = accumulation, Negative = distribution
        obv_z = (g['obv_slope_20d'] - g['obv_slope_20d'].rolling(60).mean()) / (g['obv_slope_20d'].rolling(60).std() + 1e-10)
        mfi_z = (g['mfi'] - 50) / 50  # -1 to +1
        ad_z = (g['ad_slope_20d'] - g['ad_slope_20d'].rolling(60).mean()) / (g['ad_slope_20d'].rolling(60).std() + 1e-10)
        g['flow_score'] = (obv_z + mfi_z + ad_z) / 3

        results.append(g)

    return pd.concat(results, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION (WITH AND WITHOUT FLOW FILTERS)
# ═════════════════════════════════════════════════════════════════════════════

def detect_oversold_signals(df, rsi_period=5, rsi_thresh=20):
    """RSI oversold signals per stock."""
    signals = []
    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        close = g['close']
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        g['rsi'] = rsi.values

        for i in range(rsi_period + 20, len(g)):
            if g['rsi'].iloc[i] < rsi_thresh:
                signals.append({
                    'date': g['date'].iloc[i],
                    'ticker': ticker,
                    'rsi': float(g['rsi'].iloc[i]),
                    'flow_score': float(g['flow_score'].iloc[i]) if not pd.isna(g['flow_score'].iloc[i]) else 0,
                    'mfi': float(g['mfi'].iloc[i]) if not pd.isna(g['mfi'].iloc[i]) else 50,
                    'obv_slope': float(g['obv_slope_20d'].iloc[i]) if not pd.isna(g['obv_slope_20d'].iloc[i]) else 0,
                    'ad_slope': float(g['ad_slope_20d'].iloc[i]) if not pd.isna(g['ad_slope_20d'].iloc[i]) else 0,
                    'rel_volume': float(g['rel_volume'].iloc[i]) if not pd.isna(g['rel_volume'].iloc[i]) else 1,
                    'vwap_dev': float(g['vwap_dev'].iloc[i]) if not pd.isna(g['vwap_dev'].iloc[i]) else 0,
                    'signal_type': 'oversold',
                })
    return signals


def detect_drop_signals(df, drop_pct=0.03):
    """Single-day drop signals per stock."""
    signals = []
    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        g['daily_ret'] = g['close'].pct_change()

        for i in range(20, len(g)):
            if g['daily_ret'].iloc[i] < -drop_pct:
                signals.append({
                    'date': g['date'].iloc[i],
                    'ticker': ticker,
                    'drop_pct': float(g['daily_ret'].iloc[i]),
                    'flow_score': float(g['flow_score'].iloc[i]) if not pd.isna(g['flow_score'].iloc[i]) else 0,
                    'mfi': float(g['mfi'].iloc[i]) if not pd.isna(g['mfi'].iloc[i]) else 50,
                    'obv_slope': float(g['obv_slope_20d'].iloc[i]) if not pd.isna(g['obv_slope_20d'].iloc[i]) else 0,
                    'ad_slope': float(g['ad_slope_20d'].iloc[i]) if not pd.isna(g['ad_slope_20d'].iloc[i]) else 0,
                    'rel_volume': float(g['rel_volume'].iloc[i]) if not pd.isna(g['rel_volume'].iloc[i]) else 1,
                    'vwap_dev': float(g['vwap_dev'].iloc[i]) if not pd.isna(g['vwap_dev'].iloc[i]) else 0,
                    'signal_type': 'drop',
                })
    return signals


def detect_vol_compression_signals(df, vol_pctile=10, lookback=20):
    """Vol compression: realized vol at <Nth percentile of its own 1-year history."""
    signals = []
    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        g['daily_ret'] = g['close'].pct_change()
        g['vol_20d'] = g['daily_ret'].rolling(lookback).std()
        g['vol_pctile'] = g['vol_20d'].rolling(252).rank(pct=True) * 100

        for i in range(270, len(g)):
            if pd.notna(g['vol_pctile'].iloc[i]) and g['vol_pctile'].iloc[i] < vol_pctile:
                signals.append({
                    'date': g['date'].iloc[i],
                    'ticker': ticker,
                    'vol_pctile': float(g['vol_pctile'].iloc[i]),
                    'flow_score': float(g['flow_score'].iloc[i]) if not pd.isna(g['flow_score'].iloc[i]) else 0,
                    'mfi': float(g['mfi'].iloc[i]) if not pd.isna(g['mfi'].iloc[i]) else 50,
                    'obv_slope': float(g['obv_slope_20d'].iloc[i]) if not pd.isna(g['obv_slope_20d'].iloc[i]) else 0,
                    'ad_slope': float(g['ad_slope_20d'].iloc[i]) if not pd.isna(g['ad_slope_20d'].iloc[i]) else 0,
                    'rel_volume': float(g['rel_volume'].iloc[i]) if not pd.isna(g['rel_volume'].iloc[i]) else 1,
                    'vwap_dev': float(g['vwap_dev'].iloc[i]) if not pd.isna(g['vwap_dev'].iloc[i]) else 0,
                    'signal_type': 'vol_compression',
                })
    return signals


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def backtest_signals(signals_df, prices_df, hold_days, regime_map, flow_filter=None, flow_col='flow_score', flow_thresh=0):
    """
    Backtest a set of signals with optional flow filter.
    flow_filter: 'positive' (flow_score > thresh), 'negative' (< thresh), 'high_mfi' (MFI > 50), etc.
    """
    prices_pivot = prices_df.pivot_table(index='date', columns='ticker', values='close')
    dates = prices_pivot.index

    # Apply flow filter
    if flow_filter == 'positive_flow':
        signals_df = signals_df[signals_df[flow_col] > flow_thresh]
    elif flow_filter == 'negative_flow':
        signals_df = signals_df[signals_df[flow_col] < flow_thresh]
    elif flow_filter == 'accumulation':
        # Both OBV and AD trending up
        signals_df = signals_df[(signals_df['obv_slope'] > 0) & (signals_df['ad_slope'] > 0)]
    elif flow_filter == 'distribution':
        signals_df = signals_df[(signals_df['obv_slope'] < 0) & (signals_df['ad_slope'] < 0)]
    elif flow_filter == 'mfi_oversold':
        signals_df = signals_df[signals_df['mfi'] < 20]
    elif flow_filter == 'mfi_neutral_up':
        signals_df = signals_df[(signals_df['mfi'] > 30) & (signals_df['mfi'] < 70)]
    elif flow_filter == 'high_rel_volume':
        signals_df = signals_df[signals_df['rel_volume'] > 1.5]
    elif flow_filter == 'below_vwap':
        signals_df = signals_df[signals_df['vwap_dev'] < -0.02]
    elif flow_filter == 'flow_divergence_bullish':
        # Price dropping but accumulation happening (flow positive)
        signals_df = signals_df[signals_df['flow_score'] > 0.3]
    elif flow_filter == 'strong_accumulation':
        signals_df = signals_df[signals_df['flow_score'] > 0.5]

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
                'exit_date': str(pd.Timestamp(exit_date).date()),
                'ticker': ticker,
                'return_pct': float(ret),
                'regime': regime,
                'hold_days': hold_days,
            })
        except:
            pass

    return trades, len(signals_df)


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_sharpe(returns):
    if len(returns) < 5:
        return 0.0
    return float(np.mean(returns) / (np.std(returns) + 1e-10))

def compute_sortino(returns):
    if len(returns) < 5:
        return 0.0
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
        'sharpe': float(real_sharpe),
        'sortino': float(real_sortino),
        'mean_return_pct': float(np.mean(returns) * 100),
        'wr_pct': float(wr),
        'pf': float(pf),
        'n_trades': len(returns),
        'perm_p': float(perm_p),
        'perm_pass': perm_p < 0.05,
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
    print("FLOW-ENHANCED SIGNAL RESEARCH v1 (HC #737 + HC #739)")
    print(f"Testing whether flow data improves our proven signals")
    print(f"Universe: {len(TICKERS)} tickers | Period: {START_DATE} to {END_DATE}")
    print("=" * 70)

    # Data
    print("\n[1] Loading data...")
    prices = fetch_prices(TICKERS, OUTPUT / "prices_cache.parquet")
    spy = fetch_spy(OUTPUT / "spy_cache.parquet")
    regime_map = classify_regime(spy)

    # Compute flow features
    print("\n[2] Computing flow features (OBV, MFI, A/D, VWAP, rel volume)...")
    prices = compute_flow_features(prices)
    n_tickers = prices['ticker'].nunique()
    print(f"  Flow features computed for {n_tickers} tickers")

    # Detect signals
    print("\n[3] Detecting signals...")
    oversold_signals = detect_oversold_signals(prices)
    drop_signals = detect_drop_signals(prices, drop_pct=0.03)
    vol_comp_signals = detect_vol_compression_signals(prices)

    print(f"  Oversold (RSI<20): {len(oversold_signals)} signals")
    print(f"  Drop (>3%): {len(drop_signals)} signals")
    print(f"  Vol compression (<10th pctile): {len(vol_comp_signals)} signals")

    oversold_df = pd.DataFrame(oversold_signals)
    drop_df = pd.DataFrame(drop_signals)
    vol_comp_df = pd.DataFrame(vol_comp_signals)

    # Flow filter variants
    flow_filters = [
        ('no_filter', None),
        ('positive_flow', 'positive_flow'),          # Flow score > 0
        ('strong_accumulation', 'strong_accumulation'), # Flow score > 0.5
        ('accumulation', 'accumulation'),              # OBV + AD both up
        ('distribution', 'distribution'),              # OBV + AD both down (anti-filter)
        ('mfi_oversold', 'mfi_oversold'),             # MFI < 20
        ('high_rel_volume', 'high_rel_volume'),       # Volume > 1.5x average
        ('below_vwap', 'below_vwap'),                 # Price below 20d VWAP
        ('flow_divergence_bullish', 'flow_divergence_bullish'),  # Dropping price + rising accumulation
    ]

    hold_periods = [5, 10, 21]

    # Run all combinations
    print("\n[4] Running backtests...")
    all_results = {}

    signal_sets = [
        ('oversold', oversold_df),
        ('drop3pct', drop_df),
        ('vol_compression', vol_comp_df),
    ]

    for sig_name, sig_df in signal_sets:
        if sig_df.empty:
            print(f"\n  {sig_name}: NO SIGNALS, skipping")
            continue

        for flow_name, flow_filter in flow_filters:
            for hold in hold_periods:
                variant = f"{sig_name}_{flow_name}_hold{hold}d"

                trades, n_signals = backtest_signals(
                    sig_df.copy(), prices, hold, regime_map,
                    flow_filter=flow_filter
                )

                if len(trades) < 10:
                    all_results[variant] = {'status': 'SKIP', 'n_trades': len(trades), 'n_signals': n_signals}
                    continue

                trades_df = pd.DataFrame(trades)
                result = validate(trades_df)
                status = "PASS" if result['all_pass'] else "FAIL"

                # Compare to no-filter baseline
                all_results[variant] = {
                    **result,
                    'status': 'ALL_PASS' if result['all_pass'] else 'FAIL',
                    'n_signals': n_signals,
                    'flow_filter': flow_name,
                }

                if result['all_pass'] or flow_name == 'no_filter':
                    print(f"\n  {'✅' if result['all_pass'] else '⬜'} {variant}: "
                          f"Sharpe {result['sharpe']:.3f}, Sortino {result['sortino']:.3f}, "
                          f"WR {result['wr_pct']:.1f}%, PF {result['pf']:.2f}, "
                          f"{result['n_trades']} trades, perm p={result['perm_p']:.3f}, "
                          f"regime gap={result['regime_gap']:.2f}")

    # Analysis: Does flow improve signals?
    print(f"\n{'='*70}")
    print("FLOW VALUE-ADD ANALYSIS")
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

            for flow_name, _ in flow_filters:
                if flow_name == 'no_filter':
                    continue
                key = f"{sig_name}_{flow_name}_hold{hold}d"
                r = all_results.get(key, {})
                if r.get('status') == 'SKIP':
                    print(f"    {flow_name}: SKIP (too few trades)")
                    continue
                if not r.get('sharpe'):
                    continue

                delta_sharpe = r['sharpe'] - baseline['sharpe']
                direction = "+" if delta_sharpe > 0 else ""
                pass_str = "✅" if r.get('all_pass') else "❌"
                print(f"    {pass_str} {flow_name}: Sharpe {r['sharpe']:.3f} ({direction}{delta_sharpe:.3f}), "
                      f"WR {r['wr_pct']:.1f}%, {r['n_trades']} trades, "
                      f"regime gap {r['regime_gap']:.2f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    passing = {k: v for k, v in all_results.items() if v.get('status') == 'ALL_PASS'}
    failing = {k: v for k, v in all_results.items() if v.get('status') == 'FAIL'}
    skipped = {k: v for k, v in all_results.items() if v.get('status') == 'SKIP'}

    print(f"\n  Total variants: {len(all_results)}")
    print(f"  Passing: {len(passing)} | Failing: {len(failing)} | Skipped: {len(skipped)}")

    if passing:
        print(f"\n  ✅ PASSING VARIANTS:")
        for name, r in sorted(passing.items(), key=lambda x: -x[1].get('sharpe', 0)):
            print(f"    {name}: Sharpe {r['sharpe']:.3f}, Sortino {r['sortino']:.3f}, "
                  f"WR {r['wr_pct']:.1f}%, PF {r['pf']:.2f}, {r['n_trades']} trades, "
                  f"perm p={r['perm_p']:.3f}, regime gap={r['regime_gap']:.2f}")

    # Key question: which flow filter adds the most value?
    print(f"\n  KEY QUESTION: Which flow filter improves Sharpe most consistently?")
    flow_deltas = {}
    for flow_name, _ in flow_filters:
        if flow_name == 'no_filter':
            continue
        deltas = []
        for sig_name, _ in signal_sets:
            for hold in hold_periods:
                baseline_key = f"{sig_name}_no_filter_hold{hold}d"
                key = f"{sig_name}_{flow_name}_hold{hold}d"
                b = all_results.get(baseline_key, {})
                r = all_results.get(key, {})
                if b.get('sharpe') and r.get('sharpe'):
                    deltas.append(r['sharpe'] - b['sharpe'])
        if deltas:
            flow_deltas[flow_name] = {
                'avg_delta': float(np.mean(deltas)),
                'pct_positive': float(np.mean([1 if d > 0 else 0 for d in deltas]) * 100),
                'n_comparisons': len(deltas),
            }
            print(f"    {flow_name}: avg Sharpe delta {np.mean(deltas):+.3f}, "
                  f"positive {np.mean([1 if d > 0 else 0 for d in deltas])*100:.0f}% of time")

    # Save
    report = {
        'timestamp': datetime.now().isoformat(),
        'observation': 'Testing whether flow data (OBV, MFI, A/D, VWAP, rel volume) improves proven signals',
        'mandate': 'HC #737 + HC #739: All research must incorporate actual flow data',
        'universe_size': n_tickers,
        'period': f"{START_DATE} to {END_DATE}",
        'signal_counts': {
            'oversold': len(oversold_signals),
            'drop3pct': len(drop_signals),
            'vol_compression': len(vol_comp_signals),
        },
        'summary': {
            'total_variants': len(all_results),
            'passing': len(passing),
            'failing': len(failing),
            'skipped': len(skipped),
        },
        'flow_value_add': flow_deltas,
        'results': all_results,
    }

    with open(OUTPUT / "report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed/60:.1f} minutes")
    print(f"  Report saved to {OUTPUT / 'report.json'}")


if __name__ == '__main__':
    main()
