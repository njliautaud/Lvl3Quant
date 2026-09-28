#!/usr/bin/env python3
"""
Breadth Thrust Strategy v1 — Observation-First (HC #735 R1)
=============================================================
OBSERVATION: Market breadth extremes predict subsequent returns asymmetrically.
- Zweig Breadth Thrust: when % advancing goes from <40% to >61.5% within 10 days
- McClellan Oscillator: extreme readings revert
- Advance-Decline breadth thrusts: sudden "everything rallies" days are rare and predictive

HYPOTHESIS: Rare breadth thrust events (rapid shift from oversold breadth to
overbought breadth) signal regime shifts. We test:
  1. LONG broad market after breadth thrust (classic Zweig)
  2. LONG laggards specifically (they haven't caught up yet)
  3. SHORT after breadth collapses (opposite signal — market weakness)

Also tests breadth divergence: market making new highs but breadth narrowing.

Universe: 400+ S&P 500 stocks, 2013-2026
Author: Claude (Head of Quant)
Date: 2026-07-22
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant") if os.path.exists("/home/jupiter") else Path("/home/nick/Lvl3Quant")
OUTPUT = ROOT / "output" / "breadth_thrust_v1"
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
# BREADTH METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_breadth_series(returns_df):
    """
    For each date, compute:
    - pct_advancing: fraction of stocks with positive return
    - pct_advancing_10d: rolling 10-day EMA of pct_advancing
    - net_advance: (advancers - decliners) / total
    - mcclellan: 19-day EMA - 39-day EMA of net_advance (simplified)
    """
    pct_adv = (returns_df > 0).sum(axis=1) / returns_df.count(axis=1)
    pct_dec = (returns_df < 0).sum(axis=1) / returns_df.count(axis=1)
    net_adv = pct_adv - pct_dec

    breadth = pd.DataFrame({
        'pct_advancing': pct_adv,
        'pct_declining': pct_dec,
        'net_advance': net_adv,
        'pct_adv_ema10': pct_adv.ewm(span=10).mean(),
        'mcclellan_fast': net_adv.ewm(span=19).mean(),
        'mcclellan_slow': net_adv.ewm(span=39).mean(),
    })
    breadth['mcclellan_osc'] = breadth['mcclellan_fast'] - breadth['mcclellan_slow']

    # Rolling breadth metrics
    breadth['adv_10d_min'] = pct_adv.rolling(10).min()
    breadth['adv_10d_max'] = pct_adv.rolling(10).max()

    # Breadth momentum: change in breadth over N days
    breadth['breadth_momentum_5d'] = pct_adv.diff(5)
    breadth['breadth_momentum_10d'] = pct_adv.diff(10)

    return breadth


def detect_zweig_thrust(breadth_df, low_thresh=0.40, high_thresh=0.615, window=10):
    """
    Classic Zweig Breadth Thrust: pct_advancing goes from <low_thresh to >high_thresh
    within `window` trading days. Extremely rare (happened ~15 times since 1926).
    We relax thresholds slightly to get more data points.
    """
    events = []
    pct_adv = breadth_df['pct_advancing']

    for i in range(window, len(breadth_df)):
        current = pct_adv.iloc[i]
        lookback_min = pct_adv.iloc[i-window:i].min()

        if current > high_thresh and lookback_min < low_thresh:
            # Verify this isn't duplicate (within 20 days of prior)
            date = breadth_df.index[i]
            if events and (date - events[-1]['date']).days < 20:
                continue
            events.append({
                'date': date,
                'pct_advancing': float(current),
                'prior_min': float(lookback_min),
                'thrust_magnitude': float(current - lookback_min),
            })

    return events


def detect_breadth_surge(breadth_df, threshold=0.80, min_prior_low=0.50):
    """
    Breadth Surge: >80% of stocks advancing on a single day, after
    prior breadth was weak (<50% advancing within last 5 days).
    """
    events = []
    pct_adv = breadth_df['pct_advancing']

    for i in range(5, len(breadth_df)):
        current = pct_adv.iloc[i]
        prior_5d_min = pct_adv.iloc[i-5:i].min()

        if current > threshold and prior_5d_min < min_prior_low:
            date = breadth_df.index[i]
            if events and (date - events[-1]['date']).days < 10:
                continue
            events.append({
                'date': date,
                'pct_advancing': float(current),
                'prior_5d_min': float(prior_5d_min),
            })

    return events


def detect_breadth_collapse(breadth_df, threshold=0.20, max_prior_high=0.50):
    """
    Breadth Collapse (opposite of thrust): <20% advancing after being >50%.
    Potential short signal or regime shift warning.
    """
    events = []
    pct_adv = breadth_df['pct_advancing']

    for i in range(10, len(breadth_df)):
        current = pct_adv.iloc[i]
        prior_10d_max = pct_adv.iloc[i-10:i].max()

        if current < threshold and prior_10d_max > max_prior_high:
            date = breadth_df.index[i]
            if events and (date - events[-1]['date']).days < 10:
                continue
            events.append({
                'date': date,
                'pct_advancing': float(current),
                'prior_10d_max': float(prior_10d_max),
            })

    return events


def detect_mcclellan_extreme(breadth_df, low_pctile=5, high_pctile=95):
    """
    McClellan Oscillator at extreme reading — mean reversion expected.
    """
    osc = breadth_df['mcclellan_osc'].dropna()
    low_val = osc.quantile(low_pctile / 100)
    high_val = osc.quantile(high_pctile / 100)

    events_low = []
    events_high = []

    for i in range(len(breadth_df)):
        if pd.isna(breadth_df['mcclellan_osc'].iloc[i]):
            continue

        date = breadth_df.index[i]
        val = breadth_df['mcclellan_osc'].iloc[i]

        if val <= low_val:
            if events_low and (date - events_low[-1]['date']).days < 10:
                continue
            events_low.append({'date': date, 'mcclellan': float(val), 'type': 'oversold'})

        if val >= high_val:
            if events_high and (date - events_high[-1]['date']).days < 10:
                continue
            events_high.append({'date': date, 'mcclellan': float(val), 'type': 'overbought'})

    return events_low, events_high


def detect_breadth_divergence(breadth_df, spy_df, lookback=20):
    """
    Breadth Divergence: SPY making new 20-day high but breadth weakening
    (pct_advancing trending down). Classic distribution signal.
    """
    spy = spy_df.set_index('date').sort_index()
    spy = spy.reindex(breadth_df.index, method='ffill')

    events = []
    pct_adv = breadth_df['pct_advancing']

    for i in range(lookback + 5, len(breadth_df)):
        date = breadth_df.index[i]

        # SPY at new 20-day high?
        if date not in spy.index:
            continue
        spy_price = spy.loc[date, 'close'] if date in spy.index else None
        if spy_price is None or pd.isna(spy_price):
            continue

        spy_window = spy['close'].iloc[max(0, i-lookback):i]
        spy_window = spy_window.dropna()
        if len(spy_window) < lookback - 5:
            continue
        if spy_price <= spy_window.max():
            continue  # not at 20-day high

        # Breadth weakening? Compare current 5d avg to prior 5d avg
        breadth_current = pct_adv.iloc[i-4:i+1].mean()
        breadth_prior = pct_adv.iloc[i-14:i-9].mean()

        if breadth_current < breadth_prior - 0.05:  # breadth dropped 5%+
            if events and (date - events[-1]['date']).days < 15:
                continue
            events.append({
                'date': date,
                'spy_price': float(spy_price),
                'breadth_current': float(breadth_current),
                'breadth_prior': float(breadth_prior),
                'breadth_drop': float(breadth_prior - breadth_current),
            })

    return events


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def backtest_broad_long(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name):
    """
    After breadth thrust event: LONG equal-weight ALL stocks.
    Tests whether broad market rallies after thrust signals.
    """
    trades = []

    for event in events:
        date = event['date']
        date_idx = returns_df.index.get_loc(date)
        if date_idx + 1 + hold_days >= len(returns_df.index):
            continue

        entry_date = returns_df.index[date_idx + 1]
        exit_date = returns_df.index[min(date_idx + 1 + hold_days, len(returns_df.index) - 1)]

        # Equal-weight long all available stocks
        target_returns = []
        for sym in prices_pivot.columns:
            try:
                entry_price = prices_pivot.loc[entry_date, sym] * (1 + SLIPPAGE_PCT)
                exit_price = prices_pivot.loc[exit_date, sym] * (1 - SLIPPAGE_PCT)
                if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                    ret = (exit_price - entry_price) / entry_price
                    target_returns.append(ret)
            except:
                pass

        if not target_returns:
            continue

        avg_return = float(np.mean(target_returns))
        entry_dt = pd.Timestamp(entry_date).normalize()
        regime = regime_map.get(entry_dt, 'unknown')

        trades.append({
            'entry_date': str(entry_dt.date()),
            'exit_date': str(pd.Timestamp(exit_date).date()),
            'signal_date': str(pd.Timestamp(date).date()),
            'return_pct': avg_return,
            'n_stocks': len(target_returns),
            'regime': regime,
            'signal': strategy_name,
            'hold_days': hold_days,
        })

    return trades


def backtest_laggard_long(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name):
    """
    After breadth thrust: LONG the laggards (stocks that DIDN'T rally on the thrust day).
    These should catch up if the thrust is genuine.
    """
    trades = []

    for event in events:
        date = event['date']
        date_idx = returns_df.index.get_loc(date)
        if date_idx + 1 + hold_days >= len(returns_df.index):
            continue

        entry_date = returns_df.index[date_idx + 1]
        exit_date = returns_df.index[min(date_idx + 1 + hold_days, len(returns_df.index) - 1)]

        # Find laggards: bottom 25% performers on thrust day
        day_returns = returns_df.loc[date].dropna()
        q25 = day_returns.quantile(0.25)
        laggards = day_returns[day_returns <= q25].index.tolist()

        target_returns = []
        for sym in laggards:
            if sym in prices_pivot.columns:
                try:
                    entry_price = prices_pivot.loc[entry_date, sym] * (1 + SLIPPAGE_PCT)
                    exit_price = prices_pivot.loc[exit_date, sym] * (1 - SLIPPAGE_PCT)
                    if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                        ret = (exit_price - entry_price) / entry_price
                        target_returns.append(ret)
                except:
                    pass

        if not target_returns:
            continue

        avg_return = float(np.mean(target_returns))
        entry_dt = pd.Timestamp(entry_date).normalize()
        regime = regime_map.get(entry_dt, 'unknown')

        trades.append({
            'entry_date': str(entry_dt.date()),
            'exit_date': str(pd.Timestamp(exit_date).date()),
            'signal_date': str(pd.Timestamp(date).date()),
            'return_pct': avg_return,
            'n_stocks': len(target_returns),
            'regime': regime,
            'signal': strategy_name,
            'hold_days': hold_days,
        })

    return trades


def backtest_short_collapse(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name):
    """
    After breadth collapse: SHORT broad market (weakness should continue).
    """
    trades = []

    for event in events:
        date = event['date']
        date_idx = returns_df.index.get_loc(date)
        if date_idx + 1 + hold_days >= len(returns_df.index):
            continue

        entry_date = returns_df.index[date_idx + 1]
        exit_date = returns_df.index[min(date_idx + 1 + hold_days, len(returns_df.index) - 1)]

        target_returns = []
        for sym in prices_pivot.columns:
            try:
                entry_price = prices_pivot.loc[entry_date, sym] * (1 - SLIPPAGE_PCT)
                exit_price = prices_pivot.loc[exit_date, sym] * (1 + SLIPPAGE_PCT)
                if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                    ret = -(exit_price - entry_price) / entry_price  # short
                    target_returns.append(ret)
            except:
                pass

        if not target_returns:
            continue

        avg_return = float(np.mean(target_returns))
        entry_dt = pd.Timestamp(entry_date).normalize()
        regime = regime_map.get(entry_dt, 'unknown')

        trades.append({
            'entry_date': str(entry_dt.date()),
            'exit_date': str(pd.Timestamp(exit_date).date()),
            'signal_date': str(pd.Timestamp(date).date()),
            'return_pct': avg_return,
            'n_stocks': len(target_returns),
            'regime': regime,
            'signal': strategy_name,
            'hold_days': hold_days,
        })

    return trades


def backtest_divergence_short(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name):
    """
    After breadth divergence (new SPY high but breadth weakening): SHORT leaders.
    Distribution pattern = potential top.
    """
    trades = []

    for event in events:
        date = event['date']
        date_idx = returns_df.index.get_loc(date)
        if date_idx + 1 + hold_days >= len(returns_df.index):
            continue

        entry_date = returns_df.index[date_idx + 1]
        exit_date = returns_df.index[min(date_idx + 1 + hold_days, len(returns_df.index) - 1)]

        # Short the leaders (top 25% performers over last 20 days)
        lookback_start = max(0, date_idx - 20)
        period_returns = returns_df.iloc[lookback_start:date_idx+1].sum()
        q75 = period_returns.quantile(0.75)
        leaders = period_returns[period_returns >= q75].index.tolist()

        target_returns = []
        for sym in leaders:
            if sym in prices_pivot.columns:
                try:
                    entry_price = prices_pivot.loc[entry_date, sym] * (1 - SLIPPAGE_PCT)
                    exit_price = prices_pivot.loc[exit_date, sym] * (1 + SLIPPAGE_PCT)
                    if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                        ret = -(exit_price - entry_price) / entry_price
                        target_returns.append(ret)
                except:
                    pass

        if not target_returns:
            continue

        avg_return = float(np.mean(target_returns))
        entry_dt = pd.Timestamp(entry_date).normalize()
        regime = regime_map.get(entry_dt, 'unknown')

        trades.append({
            'entry_date': str(entry_dt.date()),
            'exit_date': str(pd.Timestamp(exit_date).date()),
            'signal_date': str(pd.Timestamp(date).date()),
            'return_pct': avg_return,
            'n_stocks': len(target_returns),
            'regime': regime,
            'signal': strategy_name,
            'hold_days': hold_days,
        })

    return trades


def backtest_mcclellan_long(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name):
    """McClellan oversold → long (mean reversion)."""
    return backtest_broad_long(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name)


def backtest_mcclellan_short(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name):
    """McClellan overbought → short (mean reversion)."""
    return backtest_short_collapse(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name)


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
    """Run all validation gates."""
    returns = trades_df['return_pct'].values
    real_sharpe = compute_sharpe(returns)
    real_sortino = compute_sortino(returns)

    # Permutation test (flip directions)
    count_better = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(returns))
        perm_sharpe = compute_sharpe(np.abs(returns) * signs)
        if perm_sharpe >= real_sharpe:
            count_better += 1
    perm_p = count_better / n_perms

    # Regime
    sharpes = {}
    for r in ['green', 'red', 'flat']:
        sub = trades_df[trades_df['regime'] == r]
        sharpes[r] = compute_sharpe(sub['return_pct'].values) if len(sub) >= 3 else 0
    sg, sr = sharpes.get('green', 0), sharpes.get('red', 0)
    denom = max(abs(sg), abs(sr), 1e-10)
    regime_gap = abs(sg - sr) / denom

    # Win rate / PF
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    wr = len(wins) / len(returns) * 100
    pf = float(np.sum(wins)) / (abs(float(np.sum(losses))) + 1e-10)

    # Yearly
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
        'yearly_detail': {int(y): float(v*100) for y, v in yearly.items()},
        'all_pass': perm_p < 0.05 and regime_gap <= REGIME_GAP_MAX and wr > 50 and pf > 1.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("BREADTH THRUST STRATEGY v1 — OBSERVATION-FIRST")
    print(f"Observation: Rare breadth extremes predict subsequent market direction")
    print(f"Hypothesis: Breadth thrust = regime shift → trade accordingly")
    print(f"Universe: {len(TICKERS)} tickers | Period: {START_DATE} to {END_DATE}")
    print("=" * 70)

    # Data
    print("\n[1] Loading data...")
    prices = fetch_prices(TICKERS, OUTPUT / "prices_cache.parquet")
    spy = fetch_spy(OUTPUT / "spy_cache.parquet")
    regime_map = classify_regime(spy)

    # Build returns matrix
    print("\n[2] Building returns and breadth series...")
    returns = prices.pivot_table(index='date', columns='ticker', values='close').pct_change()
    prices_pivot = prices.pivot_table(index='date', columns='ticker', values='close')
    breadth = compute_breadth_series(returns)
    print(f"  Returns: {returns.shape[0]} days × {returns.shape[1]} tickers")

    # Detect events
    print("\n[3] Detecting breadth events...")

    # Zweig Breadth Thrust variants
    zweig_classic = detect_zweig_thrust(breadth, low_thresh=0.40, high_thresh=0.615, window=10)
    zweig_relaxed = detect_zweig_thrust(breadth, low_thresh=0.42, high_thresh=0.58, window=10)
    zweig_wide = detect_zweig_thrust(breadth, low_thresh=0.40, high_thresh=0.55, window=15)

    # Breadth surges
    surge_80 = detect_breadth_surge(breadth, threshold=0.80, min_prior_low=0.50)
    surge_85 = detect_breadth_surge(breadth, threshold=0.85, min_prior_low=0.45)
    surge_75 = detect_breadth_surge(breadth, threshold=0.75, min_prior_low=0.45)

    # Breadth collapses
    collapse_20 = detect_breadth_collapse(breadth, threshold=0.20, max_prior_high=0.50)
    collapse_15 = detect_breadth_collapse(breadth, threshold=0.15, max_prior_high=0.50)

    # McClellan extremes
    mcclellan_low, mcclellan_high = detect_mcclellan_extreme(breadth, low_pctile=5, high_pctile=95)

    # Breadth divergence
    divergence = detect_breadth_divergence(breadth, spy, lookback=20)

    print(f"  Zweig classic:     {len(zweig_classic)} events")
    print(f"  Zweig relaxed:     {len(zweig_relaxed)} events")
    print(f"  Zweig wide:        {len(zweig_wide)} events")
    print(f"  Breadth surge 80%: {len(surge_80)} events")
    print(f"  Breadth surge 85%: {len(surge_85)} events")
    print(f"  Breadth surge 75%: {len(surge_75)} events")
    print(f"  Collapse <20%:     {len(collapse_20)} events")
    print(f"  Collapse <15%:     {len(collapse_15)} events")
    print(f"  McClellan low:     {len(mcclellan_low)} events")
    print(f"  McClellan high:    {len(mcclellan_high)} events")
    print(f"  Divergence:        {len(divergence)} events")

    # Run backtests
    print("\n[4] Running backtests...")
    all_results = {}

    # Define all test configurations
    test_configs = [
        # Breadth thrust → long
        ('zweig_classic_broad_long', zweig_classic, backtest_broad_long),
        ('zweig_relaxed_broad_long', zweig_relaxed, backtest_broad_long),
        ('zweig_wide_broad_long', zweig_wide, backtest_broad_long),
        ('surge80_broad_long', surge_80, backtest_broad_long),
        ('surge85_broad_long', surge_85, backtest_broad_long),
        ('surge75_broad_long', surge_75, backtest_broad_long),

        # Laggard long (catch-up play)
        ('zweig_classic_laggard_long', zweig_classic, backtest_laggard_long),
        ('zweig_relaxed_laggard_long', zweig_relaxed, backtest_laggard_long),
        ('surge80_laggard_long', surge_80, backtest_laggard_long),

        # Collapse → short
        ('collapse20_short', collapse_20, backtest_short_collapse),
        ('collapse15_short', collapse_15, backtest_short_collapse),

        # Collapse → contrarian long (our meta-finding: contrarian works)
        ('collapse20_contrarian_long', collapse_20, backtest_broad_long),
        ('collapse15_contrarian_long', collapse_15, backtest_broad_long),

        # McClellan extremes
        ('mcclellan_oversold_long', mcclellan_low, backtest_mcclellan_long),
        ('mcclellan_overbought_short', mcclellan_high, backtest_mcclellan_short),

        # Divergence → short leaders
        ('divergence_short_leaders', divergence, backtest_divergence_short),
    ]

    hold_periods = [5, 10, 21]

    for cfg_name, events, backtest_fn in test_configs:
        if not events:
            print(f"\n  {cfg_name}: NO EVENTS, skipping")
            for hold in hold_periods:
                all_results[f"{cfg_name}_hold{hold}d"] = {'status': 'SKIP', 'n_trades': 0, 'reason': 'no events'}
            continue

        for hold in hold_periods:
            variant = f"{cfg_name}_hold{hold}d"
            print(f"\n  {variant}... ({len(events)} events)")

            trades = backtest_fn(events, returns, prices_pivot, hold, regime_map, cfg_name)

            if len(trades) < 10:
                print(f"    SKIP: only {len(trades)} trades")
                all_results[variant] = {'status': 'SKIP', 'n_trades': len(trades)}
                continue

            trades_df = pd.DataFrame(trades)
            result = validate(trades_df)
            status = "✅ ALL PASS" if result['all_pass'] else "❌ FAIL"
            print(f"    {status} | {result['n_trades']} trades, Sharpe {result['sharpe']:.3f}, "
                  f"Sortino {result['sortino']:.3f}, WR {result['wr_pct']:.1f}%, PF {result['pf']:.2f}")
            print(f"    Perm p={result['perm_p']:.3f}, Regime gap={result['regime_gap']:.2f} "
                  f"(green={result['regime_sharpes']['green']:.3f}, red={result['regime_sharpes']['red']:.3f})")

            if not result['all_pass']:
                failed = []
                if not result['perm_pass']: failed.append('permutation')
                if not result['regime_pass']: failed.append('regime')
                if result['wr_pct'] <= 50: failed.append('win_rate')
                if result['pf'] <= 1.0: failed.append('profit_factor')
                print(f"    Failed: {', '.join(failed)}")

            all_results[variant] = {**result, 'status': 'ALL_PASS' if result['all_pass'] else 'FAIL'}

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
                  f"WR {r['wr_pct']:.1f}%, PF {r['pf']:.2f}, "
                  f"{r['n_trades']} trades, perm p={r['perm_p']:.3f}, regime gap={r['regime_gap']:.2f}")

    if failing:
        print(f"\n  ❌ TOP FAILING (by Sharpe):")
        for name, r in sorted(failing.items(), key=lambda x: -x[1].get('sharpe', 0))[:5]:
            failed_reasons = []
            if not r.get('perm_pass', True): failed_reasons.append('perm')
            if not r.get('regime_pass', True): failed_reasons.append('regime')
            if r.get('wr_pct', 0) <= 50: failed_reasons.append('WR')
            if r.get('pf', 0) <= 1.0: failed_reasons.append('PF')
            print(f"    {name}: Sharpe {r.get('sharpe',0):.3f}, WR {r.get('wr_pct',0):.1f}%, "
                  f"PF {r.get('pf',0):.2f}, {r.get('n_trades',0)} trades — FAILED: {', '.join(failed_reasons)}")

    # Meta-analysis
    print(f"\n{'='*70}")
    print("META-ANALYSIS")
    print(f"{'='*70}")

    # Check: do contrarian (long after collapse) beat momentum (long after thrust)?
    thrust_variants = {k: v for k, v in all_results.items()
                       if 'broad_long' in k and ('zweig' in k or 'surge' in k) and v.get('sharpe') is not None}
    contrarian_variants = {k: v for k, v in all_results.items()
                           if 'contrarian' in k and v.get('sharpe') is not None}

    if thrust_variants:
        avg_thrust_sharpe = np.mean([v['sharpe'] for v in thrust_variants.values()])
        print(f"\n  Avg Sharpe (thrust → long):      {avg_thrust_sharpe:.3f}")

    if contrarian_variants:
        avg_contrarian_sharpe = np.mean([v['sharpe'] for v in contrarian_variants.values()])
        print(f"  Avg Sharpe (collapse → long):    {avg_contrarian_sharpe:.3f}")

    short_variants = {k: v for k, v in all_results.items()
                      if 'short' in k and v.get('sharpe') is not None}
    if short_variants:
        avg_short_sharpe = np.mean([v['sharpe'] for v in short_variants.values()])
        print(f"  Avg Sharpe (short variants):     {avg_short_sharpe:.3f}")

    print(f"\n  KEY QUESTION: Does breadth add value beyond simple oversold bounce?")
    print(f"  Compare against: oversold bounce v2 (Sharpe 0.24-0.83)")

    # Save
    report = {
        'timestamp': datetime.now().isoformat(),
        'observation': 'Rare breadth extremes (Zweig thrust, McClellan extremes, breadth divergence) predict market direction',
        'hypothesis': 'Breadth thrust = regime shift, breadth collapse = contrarian entry',
        'universe_size': int(returns.shape[1]),
        'period': f"{START_DATE} to {END_DATE}",
        'event_counts': {
            'zweig_classic': len(zweig_classic),
            'zweig_relaxed': len(zweig_relaxed),
            'zweig_wide': len(zweig_wide),
            'surge_80': len(surge_80),
            'surge_85': len(surge_85),
            'surge_75': len(surge_75),
            'collapse_20': len(collapse_20),
            'collapse_15': len(collapse_15),
            'mcclellan_low': len(mcclellan_low),
            'mcclellan_high': len(mcclellan_high),
            'divergence': len(divergence),
        },
        'summary': {
            'total_variants': len(all_results),
            'passing': len(passing),
            'failing': len(failing),
            'skipped': len(skipped),
        },
        'results': all_results,
    }

    with open(OUTPUT / "report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed/60:.1f} minutes")
    print(f"  Report saved to {OUTPUT / 'report.json'}")


if __name__ == '__main__':
    main()
