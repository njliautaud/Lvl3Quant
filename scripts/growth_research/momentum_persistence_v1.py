#!/usr/bin/env python3
"""
Momentum Persistence v1 — Observation-First (HC #735 R1)
==========================================================
OBSERVATION (from scanner #735): RSI>80 → +0.78% over next month.
Overbought stocks DON'T mean-revert — they PERSIST. This contradicts
the classic "overbought = sell" wisdom.

HYPOTHESIS: Strong momentum (RSI>80) signals institutional buying that
hasn't finished. The stock continues upward. Combined with the oversold
bounce finding (RSI<20 → buy), this gives us BOTH sides of the momentum
spectrum.

STRATEGIES TESTED:
  A) Simple RSI momentum: buy RSI>80, hold N days
  B) Breakout + momentum: RSI>70 AND new 20-day high
  C) Sector momentum: buy stocks in top-performing sector when RSI>70
  D) Multi-factor: RSI>80 AND volume surge (>2x avg) AND positive trend (above 50MA)

COMPLEMENTARY TO:
  - Oversold bounce (RSI<20 → buy) = mean reversion on downside
  - Vol compression breakout = volatility regime signal
  - Tail contagion long = crash recovery

If this works, we get a diversified portfolio: momentum + mean-reversion + vol-regime.

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
OUTPUT = ROOT / "output" / "momentum_persistence_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# UNIVERSE (same as oversold bounce v2 — consistency)
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

START_DATE = '2015-01-01'
END_DATE = '2026-07-22'
SLIPPAGE_PCT = 0.001
N_PERMUTATIONS = 200
REGIME_GAP_MAX = 0.50
MIN_TRADES = 50


# ═════════════════════════════════════════════════════════════════════════════
# DATA (reuse cache from oversold bounce v2 if available)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_prices(tickers, cache_path):
    cache_path = Path(cache_path)
    # Try reusing oversold bounce cache first
    alt_cache = ROOT / "output" / "oversold_bounce_v2" / "prices_cache_v2.parquet"
    if alt_cache.exists() and not cache_path.exists():
        print(f"  Reusing cache from oversold bounce v2...")
        df = pd.read_parquet(alt_cache)
        df.to_parquet(cache_path, index=False)
        return df

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Cache hit: {df['ticker'].nunique()} tickers")
        return df

    import yfinance as yf
    frames = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Batch {i//batch_size+1}: {len(batch)} tickers...")
        try:
            data = yf.download(batch, start=START_DATE, end=END_DATE,
                               progress=False, auto_adjust=True, threads=True)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                for sym in batch:
                    try:
                        if sym in data['Close'].columns:
                            td = pd.DataFrame({
                                'date': data.index,
                                'close': data['Close'][sym].values,
                                'open': data['Open'][sym].values,
                                'high': data['High'][sym].values,
                                'low': data['Low'][sym].values,
                                'volume': data['Volume'][sym].values,
                                'ticker': sym,
                            }).dropna(subset=['close'])
                            if len(td) > 100:
                                frames.append(td)
                    except:
                        pass
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
    alt_cache = ROOT / "output" / "oversold_bounce_v2" / "spy_cache_v2.parquet"
    if alt_cache.exists() and not cache_path.exists():
        import shutil
        shutil.copy2(alt_cache, cache_path)

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
# INDICATORS
# ═════════════════════════════════════════════════════════════════════════════

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_indicators(df_ticker):
    """Compute all indicators for a single ticker."""
    df = df_ticker.sort_values('date').copy()
    df['ret_1d'] = df['close'].pct_change()
    df['rsi14'] = compute_rsi(df['close'], 14)
    df['rsi5'] = compute_rsi(df['close'], 5)
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['high_20d'] = df['high'].rolling(20).max()
    df['vol_avg'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg'].replace(0, np.nan)
    df['above_ma50'] = (df['close'] > df['ma50']).astype(int)
    df['new_20d_high'] = (df['close'] >= df['high_20d']).astype(int)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY SIGNALS
# ═════════════════════════════════════════════════════════════════════════════

# Signal check functions (NOT lambdas — must be picklable for multiprocessing)
def sig_rsi14_gt80(row):
    return row.get('rsi14', 0) > 80

def sig_rsi14_gt70(row):
    return row.get('rsi14', 0) > 70

def sig_rsi5_gt90(row):
    return row.get('rsi5', 0) > 90

def sig_breakout_rsi70(row):
    return row.get('rsi14', 0) > 70 and row.get('new_20d_high', 0) == 1

def sig_breakout_rsi80(row):
    return row.get('rsi14', 0) > 80 and row.get('new_20d_high', 0) == 1

def sig_multifactor_strong(row):
    return (row.get('rsi14', 0) > 80 and
            row.get('vol_ratio', 0) > 1.5 and
            row.get('above_ma50', 0) == 1)

def sig_multifactor_moderate(row):
    return (row.get('rsi14', 0) > 70 and
            row.get('vol_ratio', 0) > 1.2 and
            row.get('above_ma50', 0) == 1)

def sig_extreme_momentum(row):
    return row.get('rsi5', 0) > 95 and row.get('above_ma50', 0) == 1

SIGNALS = {
    'rsi14_gt80': sig_rsi14_gt80,
    'rsi14_gt70': sig_rsi14_gt70,
    'rsi5_gt90': sig_rsi5_gt90,
    'breakout_rsi70': sig_breakout_rsi70,
    'breakout_rsi80': sig_breakout_rsi80,
    'multifactor_strong': sig_multifactor_strong,
    'multifactor_moderate': sig_multifactor_moderate,
    'extreme_momentum': sig_extreme_momentum,
}

HOLD_PERIODS = [3, 5, 10, 21]  # days


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def backtest_ticker(args):
    """Backtest a single ticker for a given signal/hold."""
    ticker, df_ticker, signal_name, signal_fn, hold_days, regime_map = args

    df = compute_indicators(df_ticker)
    trades = []
    i = 50  # skip warmup
    while i < len(df) - hold_days - 1:
        row = df.iloc[i]
        try:
            if signal_fn(row):
                entry_price = df.iloc[i+1]['open'] * (1 + SLIPPAGE_PCT)
                exit_idx = min(i + 1 + hold_days, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close'] * (1 - SLIPPAGE_PCT)

                ret = (exit_price - entry_price) / entry_price
                entry_dt = pd.Timestamp(row['date']).normalize()
                regime = regime_map.get(entry_dt, 'unknown')

                trades.append({
                    'ticker': ticker,
                    'entry_date': str(entry_dt.date()),
                    'exit_date': str(pd.Timestamp(df.iloc[exit_idx]['date']).date()),
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'return_pct': float(ret),
                    'regime': regime,
                    'signal': signal_name,
                    'hold_days': hold_days,
                    'rsi14': float(row.get('rsi14', 0)),
                })
                i = exit_idx + 1
            else:
                i += 1
        except:
            i += 1

    return trades


def run_backtest(all_data, signal_name, signal_fn, hold_days, regime_map):
    tickers = all_data['ticker'].unique()
    args_list = []
    for ticker in tickers:
        df_t = all_data[all_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(df_t) >= 60:
            args_list.append((ticker, df_t, signal_name, signal_fn, hold_days, regime_map))

    all_trades = []
    n_workers = min(16, len(args_list))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(backtest_ticker, a): a[0] for a in args_list}
        for f in as_completed(futures):
            try:
                all_trades.extend(f.result())
            except:
                pass

    return all_trades


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_sharpe(returns):
    if len(returns) < 5:
        return 0.0
    return float(np.mean(returns) / (np.std(returns) + 1e-10))


def validate(trades_df):
    returns = trades_df['return_pct'].values
    real_sharpe = compute_sharpe(returns)

    # Permutation
    count = 0
    for _ in range(N_PERMUTATIONS):
        signs = np.random.choice([-1, 1], size=len(returns))
        if compute_sharpe(np.abs(returns) * signs) >= real_sharpe:
            count += 1
    perm_p = count / N_PERMUTATIONS

    # Regime
    sharpes = {}
    for r in ['green', 'red', 'flat']:
        sub = trades_df[trades_df['regime'] == r]
        sharpes[r] = compute_sharpe(sub['return_pct'].values) if len(sub) >= 5 else 0
    sg, sr = sharpes.get('green', 0), sharpes.get('red', 0)
    regime_gap = abs(sg - sr) / max(abs(sg), abs(sr), 1e-10)

    # Win rate / PF
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    wr = len(wins) / len(returns) * 100
    pf = float(np.sum(wins)) / (abs(float(np.sum(losses))) + 1e-10)

    # Sub-period
    td = trades_df.copy()
    td['entry_dt'] = pd.to_datetime(td['entry_date'])
    pre = td[td['entry_dt'] < '2020-06-01']['return_pct']
    post = td[td['entry_dt'] >= '2020-06-01']['return_pct']
    sub_pass = (pre.mean() > 0 if len(pre) >= 10 else True) and (post.mean() > 0 if len(post) >= 10 else True)

    # Yearly
    td['year'] = td['entry_dt'].dt.year
    yearly = td.groupby('year')['return_pct'].mean()
    prof_years = (yearly > 0).sum()
    yearly_pass = prof_years / len(yearly) >= 0.70 if len(yearly) > 0 else False

    # Ticker concentration
    conc = trades_df['ticker'].value_counts(normalize=True)
    max_conc = float(conc.iloc[0]) if len(conc) > 0 else 0
    conc_pass = max_conc <= 0.15

    # Outlier robustness
    lo, hi = np.percentile(returns, [5, 95])
    trimmed = returns[(returns >= lo) & (returns <= hi)]
    trimmed_sharpe = compute_sharpe(trimmed)
    outlier_drop = (1 - trimmed_sharpe / (real_sharpe + 1e-10)) * 100
    outlier_pass = outlier_drop <= 50

    all_pass = (perm_p < 0.05 and regime_gap <= REGIME_GAP_MAX and
                wr > 50 and pf > 1.0 and sub_pass and yearly_pass and
                conc_pass and outlier_pass)

    return {
        'sharpe': float(real_sharpe),
        'mean_return_pct': float(np.mean(returns) * 100),
        'wr_pct': float(wr),
        'pf': float(pf),
        'n_trades': len(returns),
        'n_tickers': int(trades_df['ticker'].nunique()),
        'perm_p': float(perm_p),
        'regime_gap': float(regime_gap),
        'regime_sharpes': {k: float(v) for k, v in sharpes.items()},
        'sub_period_pass': sub_pass,
        'yearly_pass': yearly_pass,
        'profitable_years': f"{prof_years}/{len(yearly)}",
        'yearly_detail': {int(y): float(v*100) for y, v in yearly.items()},
        'ticker_conc': float(max_conc),
        'outlier_drop_pct': float(outlier_drop),
        'all_pass': all_pass,
        'failed_gates': [g for g, p in [
            ('permutation', perm_p < 0.05), ('regime', regime_gap <= REGIME_GAP_MAX),
            ('win_rate', wr > 50), ('profit_factor', pf > 1.0),
            ('sub_period', sub_pass), ('yearly', yearly_pass),
            ('ticker_conc', conc_pass), ('outlier', outlier_pass),
        ] if not p],
    }


# ═════════════════════════════════════════════════════════════════════════════
# CORRELATION WITH OTHER STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════

def check_correlation_with_oversold(momentum_trades_df, prices_df, regime_map):
    """
    Check if momentum signals fire at different times than oversold signals.
    Low correlation = great for portfolio diversification.
    """
    # Build simple date-level activity indicator
    mom_dates = set(pd.to_datetime(momentum_trades_df['entry_date']).dt.normalize())

    # Build oversold signals (RSI5 < 20)
    oversold_dates = set()
    for ticker in prices_df['ticker'].unique():
        df_t = prices_df[prices_df['ticker'] == ticker].sort_values('date')
        if len(df_t) < 10:
            continue
        rsi5 = compute_rsi(df_t['close'], 5)
        for i, (_, row) in enumerate(df_t.iterrows()):
            if i < 5:
                continue
            if rsi5.iloc[i] < 20:
                oversold_dates.add(pd.Timestamp(row['date']).normalize())

    both = mom_dates & oversold_dates
    overlap_pct = len(both) / max(len(mom_dates), 1) * 100

    return {
        'momentum_signal_days': len(mom_dates),
        'oversold_signal_days': len(oversold_dates),
        'overlap_days': len(both),
        'overlap_pct': float(overlap_pct),
        'complementary': overlap_pct < 30,  # <30% overlap = good diversification
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("MOMENTUM PERSISTENCE v1 — OBSERVATION-FIRST")
    print(f"Observation: RSI>80 → +0.78% over next month (momentum persists)")
    print(f"Universe: {len(TICKERS)} tickers | Period: {START_DATE} to {END_DATE}")
    print(f"Signals: {list(SIGNALS.keys())} | Holds: {HOLD_PERIODS}")
    print("=" * 70)

    # Data
    print("\n[1] Loading data...")
    prices = fetch_prices(TICKERS, OUTPUT / "prices_cache.parquet")
    spy = fetch_spy(OUTPUT / "spy_cache.parquet")
    regime_map = classify_regime(spy)
    print(f"  {prices['ticker'].nunique()} tickers loaded")

    # Backtests
    print("\n[2] Running backtests...")
    all_results = {}
    best_trades_df = None
    best_sharpe = -999

    for sig_name, sig_fn in SIGNALS.items():
        for hold in HOLD_PERIODS:
            variant = f"{sig_name}_hold{hold}d"
            print(f"\n  {variant}...")

            trades = run_backtest(prices, sig_name, sig_fn, hold, regime_map)

            if len(trades) < MIN_TRADES:
                print(f"    SKIP: {len(trades)} trades (need {MIN_TRADES})")
                all_results[variant] = {'status': 'SKIP', 'n_trades': len(trades)}
                continue

            trades_df = pd.DataFrame(trades)
            result = validate(trades_df)

            status = "✅ ALL PASS" if result['all_pass'] else "❌ FAIL"
            print(f"    {status} | {result['n_trades']} trades, {result['n_tickers']} tickers")
            print(f"    Sharpe {result['sharpe']:.4f}, WR {result['wr_pct']:.1f}%, PF {result['pf']:.2f}")
            print(f"    Perm p={result['perm_p']:.3f}, Regime gap={result['regime_gap']:.2f}")
            if result['failed_gates']:
                print(f"    Failed: {', '.join(result['failed_gates'])}")

            all_results[variant] = {**result, 'status': 'ALL_PASS' if result['all_pass'] else 'FAIL'}

            if result['all_pass'] and result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_trades_df = trades_df

    # Correlation check
    corr_result = {}
    if best_trades_df is not None:
        print("\n[3] Checking correlation with oversold bounce...")
        corr_result = check_correlation_with_oversold(best_trades_df, prices, regime_map)
        print(f"  Momentum signal days: {corr_result['momentum_signal_days']}")
        print(f"  Oversold signal days: {corr_result['oversold_signal_days']}")
        print(f"  Overlap: {corr_result['overlap_pct']:.1f}%")
        print(f"  Complementary: {'YES ✅' if corr_result['complementary'] else 'NO ❌'}")

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
            print(f"    {name}: Sharpe {r['sharpe']:.4f}, WR {r['wr_pct']:.1f}%, PF {r['pf']:.2f}, "
                  f"{r['n_trades']} trades, perm p={r['perm_p']:.3f}, regime gap={r['regime_gap']:.2f}")
            print(f"      Yearly: {r['profitable_years']} profitable, "
                  f"regimes: G={r['regime_sharpes']['green']:.3f} R={r['regime_sharpes']['red']:.3f}")

    if failing:
        print(f"\n  TOP FAILING:")
        for name, r in sorted(failing.items(), key=lambda x: -x[1].get('sharpe', 0))[:5]:
            print(f"    {name}: Sharpe {r.get('sharpe',0):.4f}, WR {r.get('wr_pct',0):.1f}%, "
                  f"failed: {', '.join(r.get('failed_gates', []))}")

    # Save
    report = {
        'timestamp': datetime.now().isoformat(),
        'observation': 'RSI>80 → +0.78% next month (momentum persists, contradicts mean-reversion)',
        'hypothesis': 'Strong momentum signals institutional buying that hasnt finished',
        'universe_size': int(prices['ticker'].nunique()),
        'period': f"{START_DATE} to {END_DATE}",
        'summary': {'passing': len(passing), 'failing': len(failing), 'skipped': len(skipped)},
        'results': all_results,
        'correlation_with_oversold': corr_result,
    }

    with open(OUTPUT / "report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed/60:.1f} minutes")
    print(f"  Report saved to {OUTPUT / 'report.json'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
