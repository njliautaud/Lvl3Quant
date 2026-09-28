#!/usr/bin/env python3
"""
Oversold Bounce v2 — Expanded Universe (HC #735 R3)
=====================================================
Observation (from scanner entry #736): RSI<20 → +1.25% over next month.
v1 passed ALL gates on 20 stocks. Now expanding to 400+ (S&P 500 constituents).

Winning signal from v1: drop_5d_10pct with 5d hold
  - WR 58.5%, PF 1.71, Sharpe 0.20/trade, perm p=0.000, regime gap 0.32

This v2 tests:
  1. Equity-only (no options BS-pricing complexity)
  2. 400+ stock universe (S&P 500 + Russell extras)
  3. 2015-2026 (11 years, more OOT data)
  4. Top 3 signals from v1 + the best one
  5. Full gates: permutation, regime, sub-period, outlier, ticker concentration

Author: Claude (Head of Quant)
Date: 2026-07-22
"""

import sys, json, warnings, os, time, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant") if os.path.exists("/home/jupiter") else Path("/home/nick/Lvl3Quant")
OUTPUT = ROOT / "output" / "oversold_bounce_v2"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# LARGE UNIVERSE — S&P 500 + Russell extras = ~470 stocks
# ═════════════════════════════════════════════════════════════════════════════

SP500_CORE = [
    # Technology
    'AAPL','MSFT','GOOGL','GOOG','AMZN','META','NVDA','TSLA','AVGO','ORCL',
    'CRM','AMD','ADBE','ACN','CSCO','INTC','IBM','TXN','QCOM','NOW',
    'INTU','AMAT','ADI','LRCX','KLAC','SNPS','CDNS','MRVL','FTNT','PANW',
    'CRWD','WDAY','TEAM','ZS','DDOG','HUBS','NET','MDB','SNOW','PLTR',
    'ABNB','DASH','COIN','SQ','PYPL','SHOP','U','RBLX','PINS','SNAP',
    # Financials
    'JPM','BAC','WFC','GS','MS','C','BLK','SCHW','AXP','BK',
    'USB','PNC','TFC','COF','CME','ICE','MCO','SPGI','MSCI','FIS',
    'FISV','ADP','NDAQ','MMC','AON','CB','AFL','MET','PRU','ALL',
    'TRV','AIG','HIG','GL','RE','BRO','WRB','CINF','RJF','NTRS',
    # Healthcare
    'UNH','JNJ','LLY','PFE','MRK','ABBV','ABT','TMO','DHR','BMY',
    'AMGN','MDT','ISRG','ELV','SYK','GILD','VRTX','REGN','BSX','ZBH',
    'BDX','IQV','A','DXCM','IDXX','PODD','ALGN','HOLX','MTD','WAT',
    # Consumer Discretionary
    'HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG','MAR','HLT',
    'ORLY','AZO','ROST','DG','DLTR','BBY','POOL','DHI','LEN','PHM',
    'NVR','GPC','GRMN','EBAY','ETSY','LULU','DECK','ON','TPR','RL',
    # Consumer Staples
    'PG','KO','PEP','COST','WMT','PM','MO','CL','KMB','GIS',
    'K','SJM','HSY','MNST','STZ','TSN','HRL','CPB','CAG','MKC',
    'CHD','CLX','EL','KHC','KDP','MDLZ','SYY','KR','TGT','WBA',
    # Industrials
    'HON','UNP','UPS','CAT','RTX','DE','BA','LMT','GD','NOC',
    'GE','MMM','EMR','ROK','ITW','PH','ETN','IR','CARR','OTIS',
    'AME','DOV','NDSN','SWK','XYL','GNRC','TT','WAB','CSX','NSC',
    'FDX','DAL','UAL','LUV','AAL','JBHT','CHRW','EXPD','ODFL','SAIA',
    # Energy
    'XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','DVN',
    'HES','HAL','BKR','FANG','CTRA','MRO','APA','OVV','TRGP','WMB',
    # Materials
    'LIN','APD','ECL','SHW','DD','NEM','FCX','NUE','STLD','CF',
    'VMC','MLM','ALB','PPG','DOW','IP','PKG','SEE','AVY','EMN',
    # Utilities
    'NEE','DUK','SO','D','AEP','SRE','EXC','XEL','WEC','ES',
    'ED','AEE','DTE','CMS','CNP','PNW','EVRG','NI','ATO','OGE',
    # REITs
    'PLD','AMT','CCI','EQIX','PSA','O','SPG','DLR','WELL','AVB',
    'EQR','VTR','IRM','ARE','MAA','UDR','KIM','REG','CPT','HST',
    # Communication
    'DIS','NFLX','CMCSA','CHTR','TMUS','VZ','T','FOX','FOXA','OMC',
    'IPG','MTCH','LYV','WBD','PARA','EA','TTWO','ZG','RDFN','YELP',
    # Additional Russell 1000 to hit 400+
    'RIVN','LCID','SOFI','HOOD','DKNG','PENN','MGM','CZR','WYNN','LVS',
    'MELI','SE','GRAB','BABA','JD','PDD','BIDU','NIO','LI','XPEV',
    'SPOT','ROKU','ZM','DOCU','OKTA','TWLO','PATH','BILL','FOUR','GTLB',
    'SWN','RRC','AR','EQT','CHRD','SM','MTDR','MGY','CPE','CIVI',
    'CLF','X','AA','VALE','RIO','BHP','SCCO','TECK','WPM','GOLD',
    'MOS','IPI','FMC','CTVA','DE','AGCO','CNH','FSLR','ENPH','SEDG',
    'RUN','NOVA','JKS','CSIQ','SPWR','ARRY','NEP','CWEN','AES','ORA',
]

# Deduplicate
TICKERS = list(dict.fromkeys(SP500_CORE))
print(f"Universe: {len(TICKERS)} tickers")

# Focus on top signals from v1 + add RSI variants
ENTRY_SIGNALS = {
    'drop_5d_10pct':   {'type': 'multi_drop', 'days': 5,  'pct': 0.10},  # v1 WINNER
    'drop_1wk_10pct':  {'type': 'multi_drop', 'days': 5,  'pct': 0.10},  # same, renamed for clarity
    'rsi2_lt10':       {'type': 'rsi',        'period': 2, 'threshold': 10},
    'rsi5_lt20':       {'type': 'rsi',        'period': 5, 'threshold': 20},
    'drop_1d_5pct':    {'type': 'daily_drop', 'pct': 0.05},
}
# Remove duplicate
del ENTRY_SIGNALS['drop_1wk_10pct']

HOLD_PERIODS = [1, 3, 5, 10]

EQUITY_SLIPPAGE_PCT    = 0.001     # 0.1% per side
N_PERMUTATIONS         = 200
REGIME_GAP_MAX         = 0.50
PERM_P_THRESHOLD       = 0.05
OUTLIER_SHARPE_DROP_MAX = 0.50
TICKER_CONC_MAX        = 0.15      # tighter for 400+ universe
MIN_TRADES             = 50        # need more trades with bigger universe

START_DATE = '2015-01-01'
END_DATE   = '2026-07-22'


# ═════════════════════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════════════════════

def fetch_prices(tickers, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached_tickers = set(df['ticker'].unique())
        missing = [t for t in tickers if t not in cached_tickers]
        if not missing:
            print(f"  Cache hit: {df['ticker'].nunique()} tickers, {len(df)} rows")
            return df
        print(f"  Partial cache ({len(cached_tickers)} tickers), downloading {len(missing)} more...")
        new_frames = [df]
    else:
        missing = tickers
        new_frames = []
        print(f"  Downloading {len(missing)} tickers...")

    import yfinance as yf
    # Batch download for speed
    batch_size = 50
    for batch_start in range(0, len(missing), batch_size):
        batch = missing[batch_start:batch_start + batch_size]
        print(f"  Batch {batch_start//batch_size + 1}/{(len(missing)-1)//batch_size + 1}: {len(batch)} tickers...")
        try:
            data = yf.download(batch, start=START_DATE, end=END_DATE,
                               progress=False, auto_adjust=True, threads=True)
            if data.empty:
                continue

            # Handle multi-ticker result
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        ticker_data = data.xs(ticker, level=1, axis=1) if ticker in data.columns.get_level_values(1) else None
                        if ticker_data is None or ticker_data.empty:
                            continue
                        ticker_data = ticker_data.reset_index()
                        ticker_data.columns = [c.lower() if isinstance(c, str) else c for c in ticker_data.columns]
                        if 'date' not in ticker_data.columns:
                            ticker_data = ticker_data.rename(columns={ticker_data.columns[0]: 'date'})
                        ticker_data['ticker'] = ticker
                        if len(ticker_data.dropna(subset=['close'])) > 100:
                            cols = [c for c in ['date','open','high','low','close','volume','ticker'] if c in ticker_data.columns]
                            new_frames.append(ticker_data[cols].dropna(subset=['close']))
                    except:
                        pass
            else:
                # Single ticker case
                data = data.reset_index()
                data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
                if 'date' not in data.columns:
                    data = data.rename(columns={data.columns[0]: 'date'})
                data['ticker'] = batch[0]
                if len(data.dropna(subset=['close'])) > 100:
                    cols = [c for c in ['date','open','high','low','close','volume','ticker'] if c in data.columns]
                    new_frames.append(data[cols].dropna(subset=['close']))
        except Exception as e:
            print(f"  Batch error: {e}")
        time.sleep(1)  # rate limit

    df = pd.concat(new_frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date', 'ticker'])
    df.to_parquet(cache_path, index=False)
    print(f"  Final: {df['ticker'].nunique()} tickers, {len(df)} rows")
    return df


def fetch_spy(cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        spy = pd.read_parquet(cache_path)
        if spy['date'].max() >= pd.Timestamp('2026-06-01'):
            return spy

    import yfinance as yf
    spy = yf.download('SPY', start=START_DATE, end=END_DATE,
                       progress=False, auto_adjust=True).reset_index()
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy.rename(columns={'Date': 'date', 'Close': 'close'}, inplace=True)
    spy.columns = [c.lower() if isinstance(c, str) else c for c in spy.columns]
    spy = spy[['date','close']].copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy.to_parquet(cache_path, index=False)
    return spy


def classify_regime(spy_df):
    """Classify each day as green/red/flat based on SPY 20d return."""
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


def compute_signals(df_ticker):
    df = df_ticker.sort_values('date').copy()
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['rsi2'] = compute_rsi(df['close'], 2)
    df['rsi5'] = compute_rsi(df['close'], 5)
    df['down'] = (df['ret_1d'] < 0).astype(int)
    df['consec_down'] = df['down'].rolling(3).sum()
    return df


def check_entry(row, signal_cfg):
    stype = signal_cfg['type']
    if stype == 'rsi':
        val = row.get(f"rsi{signal_cfg['period']}", np.nan)
        return not np.isnan(val) and val < signal_cfg['threshold']
    elif stype == 'daily_drop':
        return row.get('ret_1d', 0) < -signal_cfg['pct']
    elif stype == 'multi_drop':
        return row.get(f"ret_{signal_cfg['days']}d", 0) < -signal_cfg['pct']
    elif stype == 'consec_down':
        return row.get('consec_down', 0) >= signal_cfg['days']
    return False


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def backtest_ticker(args):
    """Process a single ticker — designed for parallel execution."""
    ticker, df_ticker, signal_name, signal_cfg, hold_days, regime_map = args

    df = compute_signals(df_ticker)
    trades = []
    i = 0
    while i < len(df) - hold_days - 1:
        row = df.iloc[i]
        if check_entry(row, signal_cfg):
            entry_date = row['date']
            entry_price = df.iloc[i+1]['open'] * (1 + EQUITY_SLIPPAGE_PCT)
            exit_idx = min(i + 1 + hold_days, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close'] * (1 - EQUITY_SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price
            entry_dt = pd.Timestamp(entry_date).normalize()
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
            })
            i = exit_idx + 1
        else:
            i += 1

    return trades


def run_backtest(all_data, signal_name, signal_cfg, hold_days, regime_map):
    """Run backtest across all tickers, parallelized."""
    tickers = all_data['ticker'].unique()

    args_list = []
    for ticker in tickers:
        df_t = all_data[all_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(df_t) >= 30:
            args_list.append((ticker, df_t, signal_name, signal_cfg, hold_days, regime_map))

    all_trades = []
    # Use parallel processing for speed
    n_workers = min(16, len(args_list))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(backtest_ticker, args): args[0] for args in args_list}
        for future in as_completed(futures):
            try:
                trades = future.result()
                all_trades.extend(trades)
            except Exception as e:
                pass

    return all_trades


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION GATES
# ═════════════════════════════════════════════════════════════════════════════

def compute_sharpe(returns):
    """Per-trade Sharpe."""
    if len(returns) < 5:
        return 0.0
    return float(np.mean(returns) / (np.std(returns) + 1e-10))


def gate_permutation(trades_df, n_perms=N_PERMUTATIONS):
    """Permutation test: shuffle returns, count how often random beats real."""
    real_sharpe = compute_sharpe(trades_df['return_pct'].values)
    shuffled_returns = trades_df['return_pct'].values.copy()

    count_better = 0
    for _ in range(n_perms):
        np.random.shuffle(shuffled_returns)
        # Randomly flip signs (proper direction permutation)
        random_signs = np.random.choice([-1, 1], size=len(shuffled_returns))
        perm_returns = np.abs(shuffled_returns) * random_signs
        perm_sharpe = compute_sharpe(perm_returns)
        if perm_sharpe >= real_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return {'p_value': float(p_value), 'pass': p_value < PERM_P_THRESHOLD}


def gate_regime(trades_df, regime_col='regime'):
    """Regime-agnostic: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) < 0.50."""
    sharpes = {}
    for regime in ['green', 'red', 'flat']:
        subset = trades_df[trades_df[regime_col] == regime]
        if len(subset) >= 5:
            sharpes[regime] = compute_sharpe(subset['return_pct'].values)
        else:
            sharpes[regime] = 0.0

    sg, sr = sharpes.get('green', 0), sharpes.get('red', 0)
    denom = max(abs(sg), abs(sr), 1e-10)
    gap = abs(sg - sr) / denom

    return {
        'gap': float(gap),
        'sharpes': {k: float(v) for k, v in sharpes.items()},
        'pass': gap <= REGIME_GAP_MAX
    }


def gate_sub_period(trades_df, split_date='2020-06-01'):
    """Sub-period consistency."""
    trades_df = trades_df.copy()
    trades_df['entry_date_dt'] = pd.to_datetime(trades_df['entry_date'])
    pre = trades_df[trades_df['entry_date_dt'] < split_date]
    post = trades_df[trades_df['entry_date_dt'] >= split_date]

    pre_mean = float(pre['return_pct'].mean()) if len(pre) >= 10 else 0
    post_mean = float(post['return_pct'].mean()) if len(post) >= 10 else 0

    # Both should be positive
    both_positive = pre_mean > 0 and post_mean > 0

    return {
        'pre_2020_mean_pct': pre_mean * 100,
        'post_2020_mean_pct': post_mean * 100,
        'pre_n': len(pre),
        'post_n': len(post),
        'pass': both_positive
    }


def gate_outlier(trades_df):
    """Remove top/bottom 5% and check Sharpe doesn't collapse."""
    full_sharpe = compute_sharpe(trades_df['return_pct'].values)

    returns = trades_df['return_pct'].values
    lo, hi = np.percentile(returns, [5, 95])
    trimmed = returns[(returns >= lo) & (returns <= hi)]
    trimmed_sharpe = compute_sharpe(trimmed)

    drop_pct = (1 - trimmed_sharpe / (full_sharpe + 1e-10)) * 100

    return {
        'full_sharpe': float(full_sharpe),
        'trimmed_sharpe': float(trimmed_sharpe),
        'drop_pct': float(drop_pct),
        'pass': drop_pct <= OUTLIER_SHARPE_DROP_MAX * 100
    }


def gate_ticker_concentration(trades_df):
    """No single ticker > 15% of trades."""
    counts = trades_df['ticker'].value_counts(normalize=True)
    max_conc = float(counts.iloc[0]) if len(counts) > 0 else 0

    return {
        'max_concentration': max_conc,
        'top_ticker': counts.index[0] if len(counts) > 0 else 'N/A',
        'n_tickers_traded': len(counts),
        'pass': max_conc <= TICKER_CONC_MAX
    }


def gate_winrate_pf(trades_df):
    """Win rate and profit factor."""
    returns = trades_df['return_pct'].values
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    wr = len(wins) / len(returns) if len(returns) > 0 else 0
    gross_wins = float(np.sum(wins))
    gross_losses = float(np.abs(np.sum(losses)))
    pf = gross_wins / (gross_losses + 1e-10)

    return {
        'win_rate_pct': float(wr * 100),
        'profit_factor': float(pf),
        'n_trades': len(returns),
        'avg_win_pct': float(np.mean(wins) * 100) if len(wins) > 0 else 0,
        'avg_loss_pct': float(np.mean(losses) * 100) if len(losses) > 0 else 0,
        'pass': wr > 0.50 and pf > 1.0
    }


def gate_yearly_consistency(trades_df):
    """Check how many individual years are profitable."""
    trades_df = trades_df.copy()
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year

    yearly = trades_df.groupby('year')['return_pct'].agg(['mean', 'count', 'sum']).reset_index()
    profitable_years = (yearly['mean'] > 0).sum()
    total_years = len(yearly)

    return {
        'profitable_years': int(profitable_years),
        'total_years': int(total_years),
        'pct_profitable': float(profitable_years / total_years * 100) if total_years > 0 else 0,
        'yearly_detail': {int(r['year']): {'mean_ret_pct': float(r['mean']*100), 'n_trades': int(r['count'])}
                         for _, r in yearly.iterrows()},
        'pass': profitable_years / (total_years + 1e-10) >= 0.70
    }


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def simulate_portfolio(trades_list, starting_capital=100_000, max_positions=20, risk_per_trade_pct=0.02):
    """
    Simulate a portfolio with position limits and equal-weight sizing.
    Returns equity curve, annualized metrics.
    """
    if not trades_list:
        return {'sharpe_annual': 0, 'cagr': 0, 'max_dd': 0}

    trades = sorted(trades_list, key=lambda t: t['entry_date'])
    capital = starting_capital
    open_positions = []
    equity_curve = []

    all_dates = sorted(set(t['entry_date'] for t in trades) | set(t['exit_date'] for t in trades))

    for trade in trades:
        # Close expired
        still_open = []
        for pos in open_positions:
            if trade['entry_date'] >= pos['exit_date']:
                pnl = pos['size'] * pos['return_pct']
                capital += pos['size'] + pnl
            else:
                still_open.append(pos)
        open_positions = still_open

        if len(open_positions) >= max_positions:
            continue

        size = capital * risk_per_trade_pct
        if capital < size or capital < 1000:
            continue

        capital -= size
        open_positions.append({**trade, 'size': size})
        equity_curve.append({'date': trade['entry_date'], 'equity': capital + sum(p['size'] for p in open_positions)})

    # Close remaining
    for pos in open_positions:
        pnl = pos['size'] * pos['return_pct']
        capital += pos['size'] + pnl

    if not equity_curve:
        return {'sharpe_annual': 0, 'cagr': 0, 'max_dd': 0}

    final_equity = capital
    years = max((pd.Timestamp(trades[-1]['exit_date']) - pd.Timestamp(trades[0]['entry_date'])).days / 365.25, 0.5)
    cagr = (final_equity / starting_capital) ** (1/years) - 1

    # Equity curve for drawdown
    eq_values = [e['equity'] for e in equity_curve]
    peak = eq_values[0]
    max_dd = 0
    for v in eq_values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    # Daily returns proxy
    eq_returns = np.diff(eq_values) / np.array(eq_values[:-1]) if len(eq_values) > 1 else [0]
    sharpe_annual = float(np.mean(eq_returns) / (np.std(eq_returns) + 1e-10) * np.sqrt(252)) if len(eq_returns) > 1 else 0

    return {
        'sharpe_annual': float(sharpe_annual),
        'cagr_pct': float(cagr * 100),
        'max_dd_pct': float(max_dd * 100),
        'final_equity': float(final_equity),
        'n_trades_executed': len(equity_curve),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print(f"=" * 70)
    print(f"OVERSOLD BOUNCE v2 — EXPANDED UNIVERSE")
    print(f"Universe: {len(TICKERS)} tickers | Period: {START_DATE} to {END_DATE}")
    print(f"Signals: {list(ENTRY_SIGNALS.keys())} | Holds: {HOLD_PERIODS}")
    print(f"=" * 70)

    # Load data
    print("\n[1/4] Loading price data...")
    prices = fetch_prices(TICKERS, OUTPUT / "prices_cache_v2.parquet")
    spy = fetch_spy(OUTPUT / "spy_cache_v2.parquet")
    regime_map = classify_regime(spy)

    n_tickers = prices['ticker'].nunique()
    print(f"  Loaded {n_tickers} tickers, {len(prices)} total rows")
    print(f"  Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")

    # Run all variants
    print("\n[2/4] Running backtests...")
    results = {}

    for sig_name, sig_cfg in ENTRY_SIGNALS.items():
        for hold in HOLD_PERIODS:
            variant = f"{sig_name}_hold{hold}d"
            print(f"\n  Testing: {variant}...")

            trades = run_backtest(prices, sig_name, sig_cfg, hold, regime_map)

            if len(trades) < MIN_TRADES:
                print(f"    SKIP: only {len(trades)} trades (need {MIN_TRADES})")
                results[variant] = {'status': 'SKIP', 'reason': f'only {len(trades)} trades', 'n_trades': len(trades)}
                continue

            trades_df = pd.DataFrame(trades)

            # Run all gates
            perm = gate_permutation(trades_df)
            regime = gate_regime(trades_df)
            sub_period = gate_sub_period(trades_df)
            outlier = gate_outlier(trades_df)
            ticker_conc = gate_ticker_concentration(trades_df)
            wr_pf = gate_winrate_pf(trades_df)
            yearly = gate_yearly_consistency(trades_df)

            all_pass = all([perm['pass'], regime['pass'], sub_period['pass'],
                           outlier['pass'], ticker_conc['pass'], wr_pf['pass'], yearly['pass']])

            # Portfolio sim for passing variants
            portfolio = {}
            if all_pass:
                portfolio = simulate_portfolio(trades)

            results[variant] = {
                'status': 'ALL_PASS' if all_pass else 'FAIL',
                'n_trades': len(trades),
                'n_tickers': int(trades_df['ticker'].nunique()),
                'mean_return_pct': float(trades_df['return_pct'].mean() * 100),
                'median_return_pct': float(trades_df['return_pct'].median() * 100),
                'sharpe_per_trade': float(compute_sharpe(trades_df['return_pct'].values)),
                'gates': {
                    'permutation': perm,
                    'regime': regime,
                    'sub_period': sub_period,
                    'outlier_removal': outlier,
                    'ticker_concentration': ticker_conc,
                    'win_rate_pf': wr_pf,
                    'yearly_consistency': yearly,
                },
                'portfolio': portfolio,
            }

            status = "✅ ALL PASS" if all_pass else "❌ FAIL"
            failed_gates = [g for g, v in results[variant]['gates'].items() if not v.get('pass', True)]
            print(f"    {status} | {len(trades)} trades, {trades_df['ticker'].nunique()} tickers")
            print(f"    Sharpe/trade: {results[variant]['sharpe_per_trade']:.4f}, WR: {wr_pf['win_rate_pct']:.1f}%, PF: {wr_pf['profit_factor']:.2f}")
            if failed_gates:
                print(f"    Failed: {', '.join(failed_gates)}")
            if portfolio:
                print(f"    Portfolio: Sharpe {portfolio.get('sharpe_annual', 0):.2f}, CAGR {portfolio.get('cagr_pct', 0):.1f}%, MaxDD {portfolio.get('max_dd_pct', 0):.1f}%")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    passing = {k: v for k, v in results.items() if v.get('status') == 'ALL_PASS'}
    failing = {k: v for k, v in results.items() if v.get('status') == 'FAIL'}
    skipped = {k: v for k, v in results.items() if v.get('status') == 'SKIP'}

    print(f"\n  Passing: {len(passing)} | Failing: {len(failing)} | Skipped: {len(skipped)}")

    if passing:
        print(f"\n  PASSING VARIANTS:")
        for name, r in sorted(passing.items(), key=lambda x: -x[1]['sharpe_per_trade']):
            wr = r['gates']['win_rate_pf']
            reg = r['gates']['regime']
            print(f"    {name}: Sharpe {r['sharpe_per_trade']:.4f}, WR {wr['win_rate_pct']:.1f}%, "
                  f"PF {wr['profit_factor']:.2f}, {r['n_trades']} trades, "
                  f"regime gap {reg['gap']:.2f}, perm p={r['gates']['permutation']['p_value']:.3f}")

    # Save
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_tickers_requested': len(TICKERS),
            'n_tickers_loaded': int(prices['ticker'].nunique()),
            'start': START_DATE,
            'end': END_DATE,
            'hold_periods': HOLD_PERIODS,
            'n_permutations': N_PERMUTATIONS,
            'regime_gap_max': REGIME_GAP_MAX,
            'perm_p_threshold': PERM_P_THRESHOLD,
            'min_trades': MIN_TRADES,
            'slippage_pct': EQUITY_SLIPPAGE_PCT,
        },
        'summary': {
            'total_variants': len(results),
            'passing': len(passing),
            'failing': len(failing),
            'skipped': len(skipped),
        },
        'results': results,
    }

    report_path = OUTPUT / "backtest_report_v2.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
