#!/usr/bin/env python3
"""
Tail Contagion Strategy v1 — Observation-First (HC #735 R1)
=============================================================
OBSERVATION (from scanner #736): Stocks crash together more than they rally
together. Lower tail dependence = 0.31, upper = 0.27, p=0.000.

HYPOTHESIS: When "bellwether" stocks (high network centrality) start crashing,
the contagion spreads to periphery stocks with a lag. We can:
  1. Detect early contagion from bellwethers (JPM, AAPL, MSFT)
  2. SHORT the laggards (periphery stocks that haven't dropped yet)
  3. Or HEDGE by going long VIX/protective puts when contagion starts

STRATEGY TESTED HERE:
  - Define "crash contagion" as: ≥5 of top-20 bellwethers down >2% on same day
  - When contagion fires: SHORT equal-weight basket of laggards (stocks NOT yet down)
  - Hold for 1, 3, 5 days
  - Compare against: (a) going long everything (contrarian bounce), (b) random

Also tests a simpler variant:
  - When market breadth collapses (>70% of stocks down): short the survivors
  - The survivors are likely next to fall (contagion lag)

Universe: 400+ stocks, 2015-2026
Author: Claude (Head of Quant)
Date: 2026-07-22
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant") if os.path.exists("/home/jupiter") else Path("/home/nick/Lvl3Quant")
OUTPUT = ROOT / "output" / "tail_contagion_v1"
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

# Bellwethers (from scanner: high network centrality)
BELLWETHERS = ['JPM','AAPL','MSFT','GOOGL','AMZN','META','NVDA','GS','BAC',
               'UNH','HD','CAT','HON','PG','JNJ','XOM','SRE','SBUX','BLK','C']

START_DATE = '2015-01-01'
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
# CONTAGION DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def build_daily_returns(prices_df):
    """Build a date × ticker returns matrix."""
    pivot = prices_df.pivot_table(index='date', columns='ticker', values='close')
    returns = pivot.pct_change()
    return returns


def detect_contagion_events(returns_df, bellwethers, threshold=-0.02, min_bellwethers=5):
    """
    Detect days where >= min_bellwethers from the bellwether list are down > threshold.
    Returns list of (date, bellwethers_down, laggards) tuples.
    """
    available_bellwethers = [b for b in bellwethers if b in returns_df.columns]
    all_tickers = returns_df.columns.tolist()

    events = []
    for date in returns_df.index:
        day_returns = returns_df.loc[date]

        # Count bellwethers that are crashing
        bellwether_rets = day_returns[available_bellwethers].dropna()
        crashing_bellwethers = bellwether_rets[bellwether_rets < threshold].index.tolist()

        if len(crashing_bellwethers) >= min_bellwethers:
            # Find laggards: stocks that are NOT down (survivors)
            all_rets = day_returns.dropna()
            survivors = all_rets[all_rets >= 0].index.tolist()  # flat or up
            # Also include stocks that are only slightly down (< |threshold/2|)
            mild = all_rets[(all_rets >= threshold/2) & (all_rets < 0)].index.tolist()
            laggards = survivors + mild

            if len(laggards) >= 5:  # need enough laggards to trade
                events.append({
                    'date': date,
                    'n_bellwethers_down': len(crashing_bellwethers),
                    'bellwethers_down': crashing_bellwethers,
                    'n_laggards': len(laggards),
                    'laggards': laggards,
                    'market_breadth_down': float((all_rets < 0).mean()),
                })

    return events


def detect_breadth_collapse(returns_df, breadth_threshold=0.70):
    """
    Detect days where > breadth_threshold of all stocks are down.
    Survivors are the short targets (contagion will reach them).
    """
    events = []
    for date in returns_df.index:
        day_returns = returns_df.loc[date].dropna()
        if len(day_returns) < 50:
            continue

        pct_down = (day_returns < 0).mean()
        if pct_down >= breadth_threshold:
            survivors = day_returns[day_returns >= 0].index.tolist()
            mild_up = day_returns[(day_returns > 0) & (day_returns < 0.02)].index.tolist()

            if len(survivors) >= 3:
                events.append({
                    'date': date,
                    'pct_down': float(pct_down),
                    'n_survivors': len(survivors),
                    'survivors': survivors,
                })

    return events


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def backtest_contagion_short(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name):
    """
    For each contagion event, SHORT the laggards/survivors equal-weight.
    """
    trades = []

    for event in events:
        date = event['date']
        targets = event.get('laggards', event.get('survivors', []))

        if not targets:
            continue

        # Get next trading day for entry
        date_idx = returns_df.index.get_loc(date)
        if date_idx + 1 + hold_days >= len(returns_df.index):
            continue

        entry_date = returns_df.index[date_idx + 1]
        exit_date = returns_df.index[min(date_idx + 1 + hold_days, len(returns_df.index) - 1)]

        # Equal-weight short basket
        target_returns = []
        for sym in targets:
            if sym in prices_pivot.columns:
                try:
                    entry_price = prices_pivot.loc[entry_date, sym]
                    exit_price = prices_pivot.loc[exit_date, sym]
                    if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                        # SHORT: profit when price drops
                        entry_price *= (1 - SLIPPAGE_PCT)  # sell at slightly worse price
                        exit_price *= (1 + SLIPPAGE_PCT)   # buy back at slightly worse price
                        ret = -(exit_price - entry_price) / entry_price  # negative for short
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
            'n_stocks_shorted': len(target_returns),
            'regime': regime,
            'signal': strategy_name,
            'hold_days': hold_days,
        })

    return trades


def backtest_contrarian_long(events, returns_df, prices_pivot, hold_days, regime_map, strategy_name):
    """
    Contrarian alternative: LONG the crashed bellwethers (bounce play).
    """
    trades = []

    for event in events:
        date = event['date']
        crashed = event.get('bellwethers_down', [])
        if not crashed:
            # For breadth events, long the most-crashed stocks
            day_returns = returns_df.loc[date].dropna()
            crashed = day_returns.nsmallest(20).index.tolist()

        date_idx = returns_df.index.get_loc(date)
        if date_idx + 1 + hold_days >= len(returns_df.index):
            continue

        entry_date = returns_df.index[date_idx + 1]
        exit_date = returns_df.index[min(date_idx + 1 + hold_days, len(returns_df.index) - 1)]

        target_returns = []
        for sym in crashed:
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
            'n_stocks_bought': len(target_returns),
            'regime': regime,
            'signal': strategy_name,
            'hold_days': hold_days,
        })

    return trades


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_sharpe(returns):
    if len(returns) < 5:
        return 0.0
    return float(np.mean(returns) / (np.std(returns) + 1e-10))


def validate(trades_df, n_perms=N_PERMUTATIONS):
    """Run all validation gates."""
    returns = trades_df['return_pct'].values
    real_sharpe = compute_sharpe(returns)

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
    print("TAIL CONTAGION STRATEGY v1 — OBSERVATION-FIRST")
    print(f"Observation: Lower tail dep (0.31) > upper (0.27), p=0.000")
    print(f"Hypothesis: Short survivors after bellwether crash = contagion lag")
    print(f"Universe: {len(TICKERS)} tickers | Period: {START_DATE} to {END_DATE}")
    print("=" * 70)

    # Data
    print("\n[1] Loading data...")
    prices = fetch_prices(TICKERS, OUTPUT / "prices_cache.parquet")
    spy = fetch_spy(OUTPUT / "spy_cache.parquet")
    regime_map = classify_regime(spy)

    # Build returns matrix
    print("\n[2] Building returns matrix...")
    returns = build_daily_returns(prices)
    prices_pivot = prices.pivot_table(index='date', columns='ticker', values='close')
    print(f"  Returns matrix: {returns.shape[0]} days × {returns.shape[1]} tickers")

    # Detect events
    print("\n[3] Detecting contagion events...")

    # Strategy A: Bellwether contagion (≥5 bellwethers down >2%)
    contagion_events_strict = detect_contagion_events(returns, BELLWETHERS, threshold=-0.02, min_bellwethers=5)
    contagion_events_loose = detect_contagion_events(returns, BELLWETHERS, threshold=-0.01, min_bellwethers=7)
    contagion_events_crash = detect_contagion_events(returns, BELLWETHERS, threshold=-0.03, min_bellwethers=3)

    # Strategy B: Breadth collapse
    breadth_events_70 = detect_breadth_collapse(returns, breadth_threshold=0.70)
    breadth_events_80 = detect_breadth_collapse(returns, breadth_threshold=0.80)

    print(f"  Bellwether contagion (strict): {len(contagion_events_strict)} events")
    print(f"  Bellwether contagion (loose):  {len(contagion_events_loose)} events")
    print(f"  Bellwether contagion (crash):  {len(contagion_events_crash)} events")
    print(f"  Breadth collapse (70%):        {len(breadth_events_70)} events")
    print(f"  Breadth collapse (80%):        {len(breadth_events_80)} events")

    # Run backtests
    print("\n[4] Running backtests...")
    all_results = {}

    test_configs = [
        # (name, events, function, hold_days)
        ('bellwether_strict_short', contagion_events_strict, 'short'),
        ('bellwether_loose_short', contagion_events_loose, 'short'),
        ('bellwether_crash_short', contagion_events_crash, 'short'),
        ('breadth70_short', breadth_events_70, 'short'),
        ('breadth80_short', breadth_events_80, 'short'),
        # Contrarian alternatives
        ('bellwether_strict_long', contagion_events_strict, 'long'),
        ('bellwether_crash_long', contagion_events_crash, 'long'),
        ('breadth70_long', breadth_events_70, 'long'),
        ('breadth80_long', breadth_events_80, 'long'),
    ]

    hold_periods = [1, 3, 5, 10]

    for cfg_name, events, direction in test_configs:
        if not events:
            print(f"\n  {cfg_name}: NO EVENTS, skipping")
            continue

        for hold in hold_periods:
            variant = f"{cfg_name}_hold{hold}d"
            print(f"\n  {variant}...")

            if direction == 'short':
                trades = backtest_contagion_short(events, returns, prices_pivot, hold, regime_map, cfg_name)
            else:
                trades = backtest_contrarian_long(events, returns, prices_pivot, hold, regime_map, cfg_name)

            if len(trades) < 10:
                print(f"    SKIP: only {len(trades)} trades")
                all_results[variant] = {'status': 'SKIP', 'n_trades': len(trades)}
                continue

            trades_df = pd.DataFrame(trades)
            result = validate(trades_df)
            status = "✅ ALL PASS" if result['all_pass'] else "❌ FAIL"
            print(f"    {status} | {result['n_trades']} trades, Sharpe {result['sharpe']:.3f}, "
                  f"WR {result['wr_pct']:.1f}%, PF {result['pf']:.2f}")
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

    print(f"\n  Passing: {len(passing)} | Failing: {len(failing)} | Skipped: {len(skipped)}")

    if passing:
        print(f"\n  PASSING VARIANTS:")
        for name, r in sorted(passing.items(), key=lambda x: -x[1].get('sharpe', 0)):
            print(f"    {name}: Sharpe {r['sharpe']:.3f}, WR {r['wr_pct']:.1f}%, PF {r['pf']:.2f}, "
                  f"{r['n_trades']} trades, perm p={r['perm_p']:.3f}, regime gap={r['regime_gap']:.2f}")

    if failing:
        print(f"\n  TOP FAILING (by Sharpe):")
        for name, r in sorted(failing.items(), key=lambda x: -x[1].get('sharpe', 0))[:5]:
            print(f"    {name}: Sharpe {r.get('sharpe',0):.3f}, WR {r.get('wr_pct',0):.1f}%, "
                  f"PF {r.get('pf',0):.2f}, {r.get('n_trades',0)} trades")

    # Save
    report = {
        'timestamp': datetime.now().isoformat(),
        'observation': 'Lower tail dependence (0.31) > upper (0.27), stocks crash together more than rally',
        'hypothesis': 'Short survivors after bellwether crash = exploit contagion lag',
        'universe_size': len(TICKERS),
        'period': f"{START_DATE} to {END_DATE}",
        'event_counts': {
            'bellwether_strict': len(contagion_events_strict),
            'bellwether_loose': len(contagion_events_loose),
            'bellwether_crash': len(contagion_events_crash),
            'breadth_70': len(breadth_events_70),
            'breadth_80': len(breadth_events_80),
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
    print("=" * 70)


if __name__ == '__main__':
    main()
