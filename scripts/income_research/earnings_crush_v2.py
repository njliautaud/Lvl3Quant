#!/usr/bin/env python3
"""
Earnings IV Crush Income Strategy Backtest — v2
================================================
Changes from v1:
  1. IRON CONDORS instead of naked strangles (buy 5-delta wings to cap loss)
  2. REAL BID/ASK fills: sell short legs at BID, buy long wings at ASK,
     close short legs at ASK, close long wings at BID (worst-case, no mid)
  3. Tracks data quality: what % of close trades have REAL chain prices
     vs. falling back to intrinsic value
  4. Uses refreshed Dolt data (now through Jul 13 2026; Mon/Wed/Fri pre-Oct 2024,
     daily since Oct 2024) — this should dramatically cut the 52% zero-buyback rate
  5. Year-by-year breakdown
  6. Worst 10 trades analysis
  7. All v1 validations retained: HC #428 R1 regime gap test, permutation test

Iron Condor structure (1 unit = 1 iron condor = 4 legs):
  SELL put  @~20-delta    → collect bid
  BUY  put  @~5-delta     → pay ask  (wing)
  SELL call @~20-delta    → collect bid
  BUY  call @~5-delta     → pay ask  (wing)
Net credit = (put_bid - wing_put_ask) + (call_bid - wing_call_ask)
Max loss    = wing_width * 100 - net_credit * 100

Close (day after earnings):
  BUY  put  @~20-delta close → pay ask
  SELL put  @~5-delta close  → get bid  (close wing)
  BUY  call @~20-delta close → pay ask
  SELL call @~5-delta close  → get bid  (close wing)
Net cost to close = (put_ask_close - wing_put_bid_close) + (call_ask_close - wing_call_bid_close)

P&L per contract = (net_credit - net_cost_to_close) * 100

Commission: $0 (Robinhood — HC #694)
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
import yfinance as yf
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT        = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR  = ROOT / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
PRICES_PATH = ROOT / "wheel_strategy_v1" / "data" / "cache" / "prices.parquet"
OUTPUT      = ROOT / "output" / "earnings_crush_v2"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL   = 100_000
RISK_PER_TRADE     = 0.02        # 2% of portfolio per trade
SHORT_DELTA        = 0.20        # sell ~20-delta options on each side
WING_DELTA         = 0.05        # buy ~5-delta protection wings
MAX_DTE_ENTRY      = 35          # max DTE at entry
MIN_DTE_ENTRY      = 2           # min DTE at entry
MIN_BID            = 0.05        # skip options with penny/zero bid
MIN_NET_CREDIT     = 0.10        # min net credit to bother (per side)

# Tickers with real parquet data
TICKERS = [
    'AAPL','ABBV','ABNB','ADBE','AMD','AMZN','AXP','BA','BAC','BLK',
    'BRK-B','C','CAT','CL','COIN','COST','CRM','CRWD','CVX','DDOG',
    'DE','DIS','F','GE','GM','GOOGL','GS','HD','HOOD','INTC',
    'JNJ','JPM','KO','LLY','LOW','MA','MCD','META','MRNA','MS',
    'MSFT','NFLX','NOW','NVDA','ORCL','OXY','PANW','PEP','PFE','PG',
    'PLTR','PYPL','RTX','SBUX','SCHW','SLB','SMCI','SPY','T','TGT',
    'TMUS','TSLA','UBER','UNH','V','VZ','WFC','WMT','XOM',
]
TRADING_TICKERS = [t for t in TICKERS if t != 'SPY']


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Fetch Earnings Dates (same as v1, reuse cache if available)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_earnings_dates(tickers, cache_path):
    """Fetch earnings dates for all tickers, with a local JSON cache."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  Loaded earnings dates from cache ({len(cached)} tickers)")
        return {k: [pd.Timestamp(d) for d in v] for k, v in cached.items()}

    # Try v1 cache first (no need to re-fetch)
    v1_cache = ROOT / "output" / "earnings_crush_v1" / "earnings_dates_cache.json"
    if v1_cache.exists():
        print(f"  Reusing v1 earnings cache from {v1_cache}")
        with open(v1_cache) as f:
            cached = json.load(f)
        # Save to v2 cache
        with open(cache_path, 'w') as f:
            json.dump(cached, f)
        return {k: [pd.Timestamp(d) for d in v] for k, v in cached.items()}

    print(f"  Fetching earnings dates from yfinance for {len(tickers)} tickers...")
    all_dates = {}
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            df = stock.get_earnings_dates(limit=80)
            if df is not None and len(df) > 0:
                dates = df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None)
                past = [d.normalize() for d in dates if d.normalize() <= pd.Timestamp.now().normalize()]
                all_dates[ticker] = [str(d.date()) for d in sorted(past)]
                print(f"    {ticker}: {len(past)} earnings dates")
            else:
                all_dates[ticker] = []
        except Exception as e:
            print(f"    {ticker}: error — {e}")
            all_dates[ticker] = []
        time.sleep(0.3)

    with open(cache_path, 'w') as f:
        json.dump(all_dates, f, default=str)
    return {k: [pd.Timestamp(d) for d in v] for k, v in all_dates.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Load Options Chains
# ═══════════════════════════════════════════════════════════════════════════════

def load_chains(ticker):
    p = CHAINS_DIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    df = df[df['vol'] > 0].copy()
    return df


def get_snapshot_date(chain_dates, target_date, direction='before', max_days=5):
    arr = np.array(chain_dates)
    td  = np.datetime64(target_date)
    if direction == 'before':
        mask = arr <= td
        if not mask.any():
            return None
        return pd.Timestamp(arr[mask].max())
    else:
        mask = arr >= td
        if not mask.any():
            return None
        return pd.Timestamp(arr[mask].min())


def chain_implied_underlying(chain_snap):
    """Infer underlying price from ATM strike (avoids split-adjust mismatch)."""
    snap = chain_snap[chain_snap['vol'] > 0].copy()
    if len(snap) == 0:
        return None
    snap['dist_atm'] = (snap['delta'].abs() - 0.50).abs()
    best = snap.nsmallest(3, 'dist_atm')
    if len(best) == 0:
        return None
    return float(best['strike'].mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Iron Condor Leg Selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_ic_legs(chain_snap, earnings_date):
    """
    Select all 4 legs of an iron condor:
      short put  @~20-delta  → sell at BID
      long  put  @~5-delta   → buy  at ASK (wing — OTM from short put)
      short call @~20-delta  → sell at BID
      long  call @~5-delta   → buy  at ASK (wing — OTM from short call)

    Returns dict with leg details, or None if setup fails.
    """
    # Valid expirations: after earnings, within DTE window
    valid_exp = chain_snap[
        (chain_snap['expiration'] > earnings_date) &
        (chain_snap['dte'] >= MIN_DTE_ENTRY) &
        (chain_snap['dte'] <= MAX_DTE_ENTRY)
    ]['expiration'].unique()

    if len(valid_exp) == 0:
        return None

    exp = pd.Timestamp(min(valid_exp))
    snap = chain_snap[chain_snap['expiration'] == exp].copy()

    # ── Put side ──────────────────────────────────────────────────────
    puts = snap[(snap['type'] == 'p') & (snap['delta'] < 0)].copy()
    if len(puts) < 2:
        return None

    # Short put: closest to -SHORT_DELTA, must have real bid
    puts_short = puts[puts['bid'] >= MIN_BID].copy()
    if len(puts_short) == 0:
        return None
    puts_short['dist'] = (puts_short['delta'].abs() - SHORT_DELTA).abs()
    short_put = puts_short.nsmallest(1, 'dist').iloc[0]

    # Long put wing: closest to -WING_DELTA, must be OTM (lower strike than short put)
    puts_wing = puts[
        (puts['strike'] < short_put['strike']) &
        (puts['bid'] >= 0)  # wing can have very low bid
    ].copy()
    if len(puts_wing) == 0:
        # fallback: try anything with lower strike
        puts_wing = puts[puts['strike'] < short_put['strike']].copy()
    if len(puts_wing) == 0:
        return None
    puts_wing['dist'] = (puts_wing['delta'].abs() - WING_DELTA).abs()
    long_put = puts_wing.nsmallest(1, 'dist').iloc[0]

    # ── Call side ─────────────────────────────────────────────────────
    calls = snap[(snap['type'] == 'c') & (snap['delta'] > 0)].copy()
    if len(calls) < 2:
        return None

    calls_short = calls[calls['bid'] >= MIN_BID].copy()
    if len(calls_short) == 0:
        return None
    calls_short['dist'] = (calls_short['delta'].abs() - SHORT_DELTA).abs()
    short_call = calls_short.nsmallest(1, 'dist').iloc[0]

    calls_wing = calls[
        (calls['strike'] > short_call['strike']) &
        (calls['bid'] >= 0)
    ].copy()
    if len(calls_wing) == 0:
        calls_wing = calls[calls['strike'] > short_call['strike']].copy()
    if len(calls_wing) == 0:
        return None
    calls_wing['dist'] = (calls_wing['delta'].abs() - WING_DELTA).abs()
    long_call = calls_wing.nsmallest(1, 'dist').iloc[0]

    # ── Net credit at entry (REAL BID/ASK fills) ──────────────────────
    # Sell short legs at BID, buy wing legs at ASK
    put_credit  = float(short_put['bid']) - float(long_put['ask'])
    call_credit = float(short_call['bid']) - float(long_call['ask'])
    net_credit  = put_credit + call_credit

    if net_credit <= 0:
        return None  # debit structure — skip (no edge collected)

    # ── Wing widths ───────────────────────────────────────────────────
    put_width  = float(short_put['strike']) - float(long_put['strike'])
    call_width = float(long_call['strike']) - float(short_call['strike'])
    max_loss_per_contract = (max(put_width, call_width) - net_credit) * 100

    return {
        'expiration'         : exp,
        'dte_entry'          : int(short_put['dte']),
        # Short put
        'short_put_strike'   : float(short_put['strike']),
        'short_put_delta'    : float(short_put['delta']),
        'short_put_iv'       : float(short_put['vol']),
        'short_put_bid'      : float(short_put['bid']),
        'short_put_ask'      : float(short_put['ask']),
        # Long put wing
        'long_put_strike'    : float(long_put['strike']),
        'long_put_delta'     : float(long_put['delta']),
        'long_put_ask'       : float(long_put['ask']),
        'long_put_bid'       : float(long_put['bid']),
        # Short call
        'short_call_strike'  : float(short_call['strike']),
        'short_call_delta'   : float(short_call['delta']),
        'short_call_iv'      : float(short_call['vol']),
        'short_call_bid'     : float(short_call['bid']),
        'short_call_ask'     : float(short_call['ask']),
        # Long call wing
        'long_call_strike'   : float(long_call['strike']),
        'long_call_delta'    : float(long_call['delta']),
        'long_call_ask'      : float(long_call['ask']),
        'long_call_bid'      : float(long_call['bid']),
        # Summary
        'put_credit'         : round(put_credit, 3),
        'call_credit'        : round(call_credit, 3),
        'net_credit'         : round(net_credit, 3),
        'put_width'          : put_width,
        'call_width'         : call_width,
        'max_loss_per_contract': round(max_loss_per_contract, 2),
        'avg_iv_entry'       : (float(short_put['vol']) + float(short_call['vol'])) / 2,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Close IC Position — Real Bid/Ask
# ═══════════════════════════════════════════════════════════════════════════════

def get_ic_close_prices(close_snap, legs):
    """
    Close all 4 IC legs at REAL bid/ask from post-earnings chain snapshot.

    Buy to close short legs at ASK.
    Sell to close long wing legs at BID.

    Returns (net_cost_to_close, close_quality, details_dict) or (None, None, None).

    data_quality:
      'real'        = real chain price found for ALL 4 legs
      'partial'     = some real, some intrinsic
      'intrinsic'   = all legs fell back to intrinsic (expiration not in close snap)
    """
    exp = legs['expiration']
    snap = close_snap[close_snap['expiration'] == exp]

    close_undl = chain_implied_underlying(close_snap)

    def get_leg_price(strike, opt_type, side):
        """
        side: 'buy' → pay ASK (to close short leg)
              'sell' → get BID (to close long wing)
        Returns (price, data_quality)
        """
        rows = snap[(snap['type'] == opt_type) & (snap['strike'] == strike)]
        if len(rows) > 0:
            row = rows.iloc[0]
            price = float(row['ask']) if side == 'buy' else float(row['bid'])
            return price, 'real'

        # Fallback: intrinsic value
        if close_undl is not None:
            if opt_type == 'p':
                intrinsic = max(strike - close_undl, 0.0)
            else:
                intrinsic = max(close_undl - strike, 0.0)
            # For buy-to-close: add a small premium over intrinsic (bid/ask of intrinsic ~ 0)
            price = intrinsic
            return price, 'intrinsic'

        return None, 'missing'

    short_put_close, sq1 = get_leg_price(legs['short_put_strike'], 'p', 'buy')
    long_put_close,  sq2 = get_leg_price(legs['long_put_strike'],  'p', 'sell')
    short_call_close,sq3 = get_leg_price(legs['short_call_strike'], 'c', 'buy')
    long_call_close, sq4 = get_leg_price(legs['long_call_strike'],  'c', 'sell')

    if any(x is None for x in [short_put_close, long_put_close, short_call_close, long_call_close]):
        return None, None, None

    # Net cost to close (we pay for short legs, collect from wing legs)
    net_cost = (short_put_close - long_put_close) + (short_call_close - long_call_close)

    quality = {
        'short_put' : sq1,
        'long_put'  : sq2,
        'short_call': sq3,
        'long_call' : sq4,
    }

    all_real = all(v == 'real' for v in quality.values())
    any_real = any(v == 'real' for v in quality.values())

    if all_real:
        close_quality = 'real'
    elif any_real:
        close_quality = 'partial'
    else:
        close_quality = 'intrinsic'

    details = {
        'short_put_close_ask' : round(short_put_close, 3),
        'long_put_close_bid'  : round(long_put_close, 3),
        'short_call_close_ask': round(short_call_close, 3),
        'long_call_close_bid' : round(long_call_close, 3),
        'net_cost_to_close'   : round(net_cost, 3),
        'close_quality'       : close_quality,
    }

    return net_cost, close_quality, details


def get_avg_iv_close(chain_snap_close, expiration):
    snap = chain_snap_close[chain_snap_close['expiration'] == expiration].copy()
    near = snap[(snap['delta'].abs() >= 0.10) & (snap['delta'].abs() <= 0.35) & (snap['vol'] > 0)]
    if len(near) == 0:
        return None
    return float(near['vol'].mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Run the Backtest
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(earnings_by_ticker, prices_df):
    print("\nRunning v2 backtest (iron condors, real bid/ask)...")

    all_trades = []
    skip_stats = defaultdict(int)

    for ticker in sorted(TRADING_TICKERS):
        e_dates = earnings_by_ticker.get(ticker, [])
        if not e_dates:
            continue

        chain = load_chains(ticker)
        if chain is None:
            print(f"  {ticker}: no chain data, skipping")
            skip_stats['no_chain'] += 1
            continue

        chain_dates = sorted(chain['date'].unique())
        trade_count = 0

        for earnings_date in sorted(e_dates):
            # Entry: last chain snapshot before earnings day
            entry_snap_date = get_snapshot_date(chain_dates, earnings_date - timedelta(days=1), 'before', max_days=5)
            if entry_snap_date is None:
                skip_stats['no_entry_snap'] += 1
                continue
            if (earnings_date - entry_snap_date).days > 4:
                skip_stats['entry_too_stale'] += 1
                continue

            entry_snap = chain[chain['date'] == entry_snap_date].copy()

            undl_price = chain_implied_underlying(entry_snap)
            if undl_price is None or undl_price <= 0:
                skip_stats['no_underlying'] += 1
                continue

            # Select IC legs
            legs = select_ic_legs(entry_snap, earnings_date)
            if legs is None:
                skip_stats['no_ic_legs'] += 1
                continue

            # Close: first chain snapshot on or after earnings day
            close_snap_date = get_snapshot_date(chain_dates, earnings_date, 'after', max_days=5)
            if close_snap_date is None:
                skip_stats['no_close_snap'] += 1
                continue
            if (close_snap_date - earnings_date).days > 4:
                skip_stats['close_too_stale'] += 1
                continue

            close_snap = chain[chain['date'] == close_snap_date].copy()

            net_cost, close_quality, close_details = get_ic_close_prices(close_snap, legs)
            if net_cost is None:
                skip_stats['no_close_price'] += 1
                continue

            close_undl = chain_implied_underlying(close_snap)

            # P&L per contract: collected premium - cost to close
            pnl_per_contract = (legs['net_credit'] - net_cost) * 100

            # IV measurements
            iv_entry = legs['avg_iv_entry']
            iv_close = get_avg_iv_close(close_snap, legs['expiration'])
            iv_crush = (iv_entry - iv_close) if iv_close is not None else None

            # Stock move
            stock_move_pct = None
            if close_undl is not None and undl_price is not None and undl_price > 0:
                stock_move_pct = (close_undl - undl_price) / undl_price * 100

            trade_count += 1
            record = {
                'ticker'               : ticker,
                'earnings_date'        : earnings_date,
                'entry_date'           : entry_snap_date,
                'close_date'           : close_snap_date,
                'days_before_earnings' : (earnings_date - entry_snap_date).days,
                'expiration'           : legs['expiration'],
                'dte_entry'            : legs['dte_entry'],
                'underlying_entry'     : round(float(undl_price), 2),
                'underlying_close'     : round(float(close_undl), 2) if close_undl is not None else None,
                'stock_move_pct'       : round(stock_move_pct, 2) if stock_move_pct is not None else None,
                # IC legs — entry
                'short_put_strike'     : legs['short_put_strike'],
                'short_put_delta'      : round(legs['short_put_delta'], 3),
                'short_put_bid'        : legs['short_put_bid'],
                'long_put_strike'      : legs['long_put_strike'],
                'long_put_delta'       : round(legs['long_put_delta'], 3),
                'long_put_ask'         : legs['long_put_ask'],
                'short_call_strike'    : legs['short_call_strike'],
                'short_call_delta'     : round(legs['short_call_delta'], 3),
                'short_call_bid'       : legs['short_call_bid'],
                'long_call_strike'     : legs['long_call_strike'],
                'long_call_delta'      : round(legs['long_call_delta'], 3),
                'long_call_ask'        : legs['long_call_ask'],
                'put_credit'           : legs['put_credit'],
                'call_credit'          : legs['call_credit'],
                'net_credit'           : legs['net_credit'],
                'put_width'            : legs['put_width'],
                'call_width'           : legs['call_width'],
                'max_loss_per_contract': legs['max_loss_per_contract'],
                # IC legs — close
                'short_put_close_ask'  : close_details['short_put_close_ask'],
                'long_put_close_bid'   : close_details['long_put_close_bid'],
                'short_call_close_ask' : close_details['short_call_close_ask'],
                'long_call_close_bid'  : close_details['long_call_close_bid'],
                'net_cost_to_close'    : close_details['net_cost_to_close'],
                'close_quality'        : close_quality,
                'pnl_per_contract'     : round(pnl_per_contract, 2),
                # IV
                'avg_iv_entry'         : round(iv_entry, 4),
                'avg_iv_close'         : round(iv_close, 4) if iv_close is not None else None,
                'iv_crush'             : round(iv_crush, 4) if iv_crush is not None else None,
                'iv_crush_pct'         : round(iv_crush / iv_entry * 100, 1) if iv_crush is not None and iv_entry > 0 else None,
            }
            all_trades.append(record)

        if trade_count > 0:
            print(f"  {ticker}: {trade_count} trades")

    print(f"\nTotal trades: {len(all_trades)}")
    print(f"Skip reasons: {dict(skip_stats)}")
    return all_trades


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Data Quality Report
# ═══════════════════════════════════════════════════════════════════════════════

def data_quality_report(trades_df):
    """Report how many closes had REAL chain data vs intrinsic fallback."""
    print("\n── Data Quality Report ───────────────────────────────────────")
    total = len(trades_df)
    for q in ['real', 'partial', 'intrinsic']:
        n = (trades_df['close_quality'] == q).sum()
        pct = n / total * 100
        label = {
            'real'     : 'Real chain prices (both sides)',
            'partial'  : 'Partial (some real, some intrinsic)',
            'intrinsic': 'All-intrinsic fallback (no chain data)',
        }[q]
        print(f"  {label}: {n:4d} / {total} = {pct:.1f}%")

    # How does close quality affect P&L?
    print("\n  P&L by close data quality:")
    for q in ['real', 'partial', 'intrinsic']:
        sub = trades_df[trades_df['close_quality'] == q]
        if len(sub) == 0:
            continue
        wr = (sub['trade_pnl'] > 0).mean() * 100
        avg = sub['trade_pnl'].mean()
        print(f"  {q:10s}: n={len(sub):4d}  WR={wr:.1f}%  AvgPnL=${avg:.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7: Portfolio Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_portfolio(trades_df, starting_capital=STARTING_CAPITAL):
    """
    Size each trade by risk budget.
    For iron condors, max risk per contract = max_loss_per_contract.
    n_contracts = max(1, floor(risk_budget / max_loss_per_contract))
    """
    trades_df = trades_df.sort_values('earnings_date').copy()
    equity = starting_capital
    sized_trades = []
    equity_curve = []

    for _, trade in trades_df.iterrows():
        max_loss = trade['max_loss_per_contract']
        if max_loss <= 0 or np.isnan(max_loss):
            continue

        risk_budget = equity * RISK_PER_TRADE
        n_contracts = max(1, int(risk_budget / max_loss))

        trade_pnl = trade['pnl_per_contract'] * n_contracts
        equity += trade_pnl

        t = trade.to_dict()
        t['n_contracts']  = n_contracts
        t['trade_pnl']    = round(trade_pnl, 2)
        t['equity_after'] = round(equity, 2)
        sized_trades.append(t)

        equity_curve.append({
            'date'  : trade['earnings_date'],
            'equity': equity,
        })

    return pd.DataFrame(sized_trades), pd.DataFrame(equity_curve)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 8: Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(sized_df, equity_df, label="Strategy"):
    if len(sized_df) == 0:
        print("No trades to evaluate.")
        return {}

    returns = sized_df['trade_pnl'].values
    wins    = returns[returns > 0]
    losses  = returns[returns < 0]
    wr      = len(wins) / len(returns)
    avg_win  = wins.mean()  if len(wins) > 0  else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    pf       = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf

    eq = equity_df.copy()
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.sort_values('date')
    years = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days / 365.25
    end_equity = eq['equity'].iloc[-1]
    cagr = (end_equity / STARTING_CAPITAL) ** (1 / max(years, 0.1)) - 1 if years > 0 else 0

    peak   = eq['equity'].cummax()
    dd     = (eq['equity'] - peak) / peak
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    trades_per_year  = len(returns) / max(years, 0.1)
    annualize_factor = np.sqrt(trades_per_year)

    ret_pct = returns / STARTING_CAPITAL
    sharpe  = (ret_pct.mean() / ret_pct.std() * annualize_factor) if ret_pct.std() > 0 else 0

    downside = ret_pct[ret_pct < 0]
    down_std = downside.std() if len(downside) > 0 else 1e-9
    sortino  = (ret_pct.mean() / down_std * annualize_factor) if down_std > 0 else 0

    return {
        'label'         : label,
        'n_trades'      : len(returns),
        'years'         : round(years, 1),
        'total_pnl'     : round(returns.sum(), 2),
        'cagr'          : round(cagr * 100, 2),
        'sharpe'        : round(sharpe, 3),
        'sortino'       : round(sortino, 3),
        'max_dd'        : round(max_dd * 100, 2),
        'calmar'        : round(calmar, 3),
        'win_rate'      : round(wr * 100, 1),
        'avg_win'       : round(avg_win, 2),
        'avg_loss'      : round(avg_loss, 2),
        'profit_factor' : round(pf, 3),
        'end_equity'    : round(end_equity, 2),
    }


def print_metrics(m):
    print(f"\n{'='*58}")
    print(f"  {m.get('label','Strategy')}")
    print(f"  {m.get('n_trades',0)} trades, {m.get('years',0)} years")
    print(f"{'='*58}")
    print(f"  CAGR:            {m.get('cagr','n/a')}%")
    print(f"  Sharpe:          {m.get('sharpe','n/a')}")
    print(f"  Sortino:         {m.get('sortino','n/a')}")
    print(f"  Max Drawdown:    {m.get('max_dd','n/a')}%")
    print(f"  Calmar:          {m.get('calmar','n/a')}")
    print(f"  Win Rate:        {m.get('win_rate','n/a')}%")
    print(f"  Avg Win:         ${m.get('avg_win','n/a'):,.2f}")
    print(f"  Avg Loss:        ${m.get('avg_loss','n/a'):,.2f}")
    print(f"  Profit Factor:   {m.get('profit_factor','n/a')}")
    print(f"  Total P&L:       ${m.get('total_pnl','n/a'):,.0f}")
    print(f"  End Equity:      ${m.get('end_equity','n/a'):,.0f}")
    print(f"{'='*58}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 9: Year-by-Year Breakdown
# ═══════════════════════════════════════════════════════════════════════════════

def year_by_year(sized_df):
    print("\n── Year-by-Year Breakdown ────────────────────────────────────")
    sized_df = sized_df.copy()
    sized_df['year'] = pd.to_datetime(sized_df['earnings_date']).dt.year
    sized_df['ret_pct'] = sized_df['trade_pnl'] / STARTING_CAPITAL

    print(f"  {'Year':>6}  {'N':>5}  {'WR%':>6}  {'PF':>6}  {'AvgPnL':>8}  {'Sharpe':>7}  {'YrPnL':>9}")
    for yr, grp in sorted(sized_df.groupby('year')):
        n   = len(grp)
        wr  = (grp['trade_pnl'] > 0).mean() * 100
        w   = grp[grp['trade_pnl'] > 0]['trade_pnl']
        l   = grp[grp['trade_pnl'] < 0]['trade_pnl']
        pf  = abs(w.sum() / l.sum()) if l.sum() != 0 else np.inf
        avg = grp['trade_pnl'].mean()
        r   = grp['ret_pct']
        sh  = (r.mean() / r.std() * np.sqrt(n)) if r.std() > 0 else 0
        yr_pnl = grp['trade_pnl'].sum()
        print(f"  {yr:>6}  {n:>5}  {wr:>6.1f}  {pf:>6.3f}  {avg:>8.1f}  {sh:>7.3f}  {yr_pnl:>9,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 10: Regime Analysis (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════════════════

def regime_analysis(sized_df, prices_df):
    spy = prices_df[prices_df['ticker'] == 'SPY'][['date', 'close']].set_index('date').sort_index()
    spy['spy_ret_30'] = spy['close'].pct_change(30)

    def classify_regime(date):
        try:
            candidates = spy.index[spy.index <= pd.Timestamp(date)]
            if len(candidates) == 0:
                return 'unknown'
            r = spy.loc[candidates[-1], 'spy_ret_30']
            if np.isnan(r):
                return 'unknown'
            if r > 0.03:
                return 'bull'
            elif r < -0.03:
                return 'bear'
            else:
                return 'flat'
        except:
            return 'unknown'

    sized_df = sized_df.copy()
    sized_df['regime'] = sized_df['earnings_date'].apply(
        lambda d: classify_regime(pd.Timestamp(d))
    )

    print("\n── Regime Analysis (HC #428 R1) ─────────────────────────────")
    regime_sharpes = {}
    for regime in ['bull', 'bear', 'flat', 'unknown']:
        sub = sized_df[sized_df['regime'] == regime]
        if len(sub) == 0:
            continue
        ret_pct = sub['trade_pnl'].values / STARTING_CAPITAL
        avg_r  = ret_pct.mean()
        std_r  = ret_pct.std()
        sharpe = (avg_r / std_r * np.sqrt(len(ret_pct))) if std_r > 0 else 0
        wins   = (sub['trade_pnl'] > 0).sum()
        wr     = wins / len(sub) * 100
        l_sum  = sub[sub['trade_pnl'] < 0]['trade_pnl'].sum()
        pf_val = abs(sub[sub['trade_pnl']>0]['trade_pnl'].sum() / l_sum) if l_sum != 0 else np.inf
        print(f"  {regime.upper():7s}: n={len(sub):4d}  WR={wr:.1f}%  Sharpe={sharpe:.3f}  PF={pf_val:.3f}")
        if regime in ('bull', 'bear'):
            regime_sharpes[regime] = sharpe

    if 'bull' in regime_sharpes and 'bear' in regime_sharpes:
        sg, sr = regime_sharpes['bull'], regime_sharpes['bear']
        denom  = max(abs(sg), abs(sr))
        ratio  = abs(sg - sr) / denom if denom > 0 else 0
        flag   = "REJECT" if ratio > 0.50 else "PASS"
        print(f"\n  Regime gap test: |Sharpe_bull - Sharpe_bear| / max = {ratio:.3f}  [{flag}]")

    return sized_df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 11: Gap / Worst Trades Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def gap_analysis(sized_df):
    print("\n── Earnings Move Analysis ────────────────────────────────────")
    df = sized_df.dropna(subset=['stock_move_pct']).copy()
    df['abs_move'] = df['stock_move_pct'].abs()

    for thr in [5, 10, 15, 20]:
        big   = df[df['abs_move'] >= thr]
        small = df[df['abs_move'] <  thr]
        if len(big) == 0:
            continue
        print(f"  |Move|≥{thr:3d}%: n={len(big):4d}  WR={( big['trade_pnl']>0).mean()*100:.1f}%  "
              f"Avg=${big['trade_pnl'].mean():.0f}   "
              f"|Move|<{thr:3d}%: n={len(small):4d}  WR={(small['trade_pnl']>0).mean()*100:.1f}%  "
              f"Avg=${small['trade_pnl'].mean():.0f}")

    print("\n── Worst 10 Trades ───────────────────────────────────────────")
    worst_cols = ['ticker','earnings_date','stock_move_pct',
                  'short_put_strike','long_put_strike',
                  'short_call_strike','long_call_strike',
                  'net_credit','net_cost_to_close','close_quality',
                  'trade_pnl','n_contracts']
    worst = sized_df.nsmallest(10, 'trade_pnl')[worst_cols]
    print(worst.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Step 12: Permutation Test
# ═══════════════════════════════════════════════════════════════════════════════

def permutation_test(sized_df, n_trials=100):
    print(f"\n── Permutation Test ({n_trials} trials) ─────────────────────────")
    pnls = sized_df['trade_pnl'].values.copy()
    real_mean = pnls.mean()

    rng = np.random.default_rng(42)
    perm_means = []
    for _ in range(n_trials):
        signs = rng.choice([-1, 1], size=len(pnls))
        perm_means.append((np.abs(pnls) * signs).mean())

    perm_means = np.array(perm_means)
    p_value = (perm_means >= real_mean).mean()

    print(f"  Real mean P&L per trade:     ${real_mean:.2f}")
    print(f"  Permuted mean (avg):         ${perm_means.mean():.2f}")
    print(f"  Permuted mean (p95):         ${np.percentile(perm_means, 95):.2f}")
    print(f"  p-value (1-sided):           {p_value:.4f}")
    flag = "SIGNIFICANT (p<0.05)" if p_value < 0.05 else "NOT SIGNIFICANT"
    print(f"  Result:                      {flag}")
    return float(p_value)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 13: IV Crush Summary
# ═══════════════════════════════════════════════════════════════════════════════

def iv_crush_summary(sized_df):
    df = sized_df.dropna(subset=['iv_crush']).copy()
    print(f"\n── IV Crush Summary ─────────────────────────────────────────")
    print(f"  Trades with IV data: {len(df)}")
    print(f"  Avg IV entry:   {df['avg_iv_entry'].mean():.3f}")
    print(f"  Avg IV close:   {df['avg_iv_close'].mean():.3f}")
    print(f"  Avg IV crush:   {df['iv_crush'].mean():.3f}  ({df['iv_crush_pct'].mean():.1f}%)")
    print(f"  % events crush: {(df['iv_crush'] > 0).mean()*100:.1f}%")

    print("\n  Sample — 10 largest IV crushes:")
    cols = ['ticker','earnings_date','avg_iv_entry','avg_iv_close','iv_crush_pct',
            'stock_move_pct','net_credit','net_cost_to_close','close_quality','trade_pnl']
    top10 = df.nlargest(10, 'iv_crush')[cols]
    print(top10.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("EARNINGS IV CRUSH BACKTEST — v2 (Iron Condors, Real Bid/Ask)")
    print("="*65)

    # Load prices
    print("\nLoading prices...")
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = pd.to_datetime(prices_df['date'])

    # Fetch/load earnings dates
    earnings_cache = OUTPUT / "earnings_dates_cache.json"
    print("\nStep 1: Earnings dates")
    earnings_by_ticker = fetch_earnings_dates(TRADING_TICKERS, earnings_cache)

    # Run backtest
    print("\nStep 2: Backtest")
    trades = run_backtest(earnings_by_ticker, prices_df)

    if not trades:
        print("No trades found. Check data.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df[trades_df['earnings_date'] >= pd.Timestamp('2019-01-01')]
    trades_df = trades_df.dropna(subset=['pnl_per_contract'])

    print(f"\nTrades after filtering: {len(trades_df)}")
    print(f"Date range: {trades_df['earnings_date'].min().date()} – {trades_df['earnings_date'].max().date()}")

    # Portfolio simulation (must happen before data_quality_report which needs trade_pnl)
    print("\nStep 3: Portfolio simulation")
    sized_df, equity_df = simulate_portfolio(trades_df)

    # Data quality report (key new metric for v2)
    data_quality_report(sized_df)

    # Save raw outputs
    sized_df.to_csv(OUTPUT / "trades.csv", index=False)
    equity_df.to_csv(OUTPUT / "equity_curve.csv", index=False)

    # Performance metrics
    m = compute_metrics(sized_df, equity_df,
                        label="Earnings IV Crush — Iron Condor (real bid/ask)")
    print_metrics(m)

    with open(OUTPUT / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)

    # IV crush summary
    iv_crush_summary(sized_df)

    # Year-by-year
    year_by_year(sized_df)

    # Regime analysis (HC #428 R1)
    sized_df = regime_analysis(sized_df, prices_df)
    sized_df.to_csv(OUTPUT / "trades_with_regime.csv", index=False)

    # Gap + worst trades
    gap_analysis(sized_df)

    # Permutation test
    p_val = permutation_test(sized_df, n_trials=100)
    m['permutation_p_value'] = p_val
    with open(OUTPUT / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)

    # Per-ticker summary
    print("\n── Per-Ticker Summary ───────────────────────────────────────")
    ticker_stats = sized_df.groupby('ticker').agg(
        n_trades       = ('trade_pnl', 'count'),
        total_pnl      = ('trade_pnl', 'sum'),
        win_rate       = ('trade_pnl', lambda x: (x > 0).mean() * 100),
        avg_pnl        = ('trade_pnl', 'mean'),
        avg_iv_entry   = ('avg_iv_entry', 'mean'),
        avg_iv_crush_pct = ('iv_crush_pct', 'mean'),
        pct_real_close = ('close_quality', lambda x: (x == 'real').mean() * 100),
    ).round(2).sort_values('total_pnl', ascending=False)
    print(ticker_stats.to_string())
    ticker_stats.to_csv(OUTPUT / "per_ticker.csv")

    print(f"\nAll outputs saved to {OUTPUT}/")
    print("\nDONE.")


if __name__ == "__main__":
    main()
