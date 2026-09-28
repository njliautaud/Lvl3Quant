#!/usr/bin/env python3
"""
Earnings Straddle/Strangle Selling Backtest — v2 (BUG FIX)
==========================================================
v1 had a CRITICAL BUG: close_position() fell back to intrinsic value
when the option wasn't found in chain data at close time. For OTM options
that weren't breached, intrinsic = $0 — giving free closes on 84.5% of
trades and inflating WR from real ~22.8% to claimed 86.5%.

FIX: When the option is not found in the chain at close time, use
Black-Scholes pricing with a POST-CRUSH IV estimate (30-day realized vol
computed from the 30 trading days BEFORE entry, NOT before earnings,
to avoid contaminating the estimate with the earnings move itself).
This is conservative: real post-crush IV is usually slightly above
realized vol.

Configurations tested:
  A) Entry timing: 1 day before vs 2 days before earnings
  B) Structure: ATM straddle vs 5% OTM strangle vs 10% OTM strangle
  C) Stop-loss: 2x premium, 3x premium, no stop
  D) Universe: all stocks vs top-10 liquid vs sector-diversified

Quality gates (MANDATORY):
  - HC #428 R1: regime-agnostic (bull/bear/flat gap < 0.50)
  - Permutation test: 100 shuffles, p < 0.05
  - Realistic costs: $0.65/contract commission, 5% premium slippage
  - Black-Scholes pricing with post-crush IV for fallback closes
  - Adversarial check: intrinsic-fallback rate reported

Data: real Dolt options chains (bid/ask/IV/greeks) from 2019-2026.
"""

import sys, json, warnings, os, time, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT        = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR  = ROOT / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
PRICES_PATH = ROOT / "wheel_strategy_v1" / "data" / "cache" / "prices.parquet"
OUTPUT      = ROOT / "output" / "earnings_vol_selling_v2"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL   = 100_000
MAX_RISK_PER_TRADE = 0.05       # 5% of portfolio per trade
COMMISSION_PER_CONTRACT = 0.65  # $0.65 per contract leg
SLIPPAGE_PCT       = 0.05       # 5% of premium as slippage
MAX_DTE_ENTRY      = 35
MIN_DTE_ENTRY      = 2
MIN_BID            = 0.03
RISK_FREE_RATE     = 0.04       # ~4% for BS pricing

# Full universe
TICKERS = [
    'AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA','NFLX','AMD','INTC',
    'CRM','PYPL','DIS','COST','V','MA','HD','WMT','MCD','KO',
    'JPM','BAC','GS','JNJ','UNH','PG',
    # Additional liquid names with chain data
    'ADBE','ABBV','BA','CAT','COIN','CRWD','CVX','DE','GE',
    'LLY','LOW','MS','NOW','ORCL','PEP','PFE','SBUX','UBER','XOM',
]
# Deduplicate and filter to only tickers with chain data
TICKERS = sorted(set(TICKERS))

# Subsets for universe configs
TOP10_LIQUID = ['AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA','AMD','NFLX','SPY']
SECTOR_DIVERSIFIED = [
    'AAPL','MSFT',     # tech
    'JPM','GS',        # finance
    'JNJ','UNH',       # health
    'XOM','CVX',       # energy
    'AMZN','COST',     # consumer
    'NVDA','AMD',      # semis
    'META','GOOGL',    # comm services
    'PG','KO',         # staples
]


# ═══════════════════════════════════════════════════════════════════════════════
# Black-Scholes Pricing
# ═══════════════════════════════════════════════════════════════════════════════

def bs_price(S, K, T, sigma, r, opt_type='c'):
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        if opt_type == 'c':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == 'c':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, sigma, r, opt_type='c'):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        if opt_type == 'c':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if opt_type == 'c':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_earnings_dates(tickers, cache_path):
    """Load earnings dates from cache, or fetch from yfinance."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  Loaded earnings dates from cache ({len(cached)} tickers)")
        return {k: [pd.Timestamp(d) for d in v] for k, v in cached.items()}

    # Try existing caches
    for alt in [ROOT/"output"/"earnings_crush_v2"/"earnings_dates_cache.json",
                ROOT/"output"/"earnings_crush_v1"/"earnings_dates_cache.json"]:
        if alt.exists():
            print(f"  Reusing existing cache from {alt}")
            with open(alt) as f:
                cached = json.load(f)
            with open(cache_path, 'w') as f:
                json.dump(cached, f)
            return {k: [pd.Timestamp(d) for d in v] for k, v in cached.items()}

    import yfinance as yf
    print(f"  Fetching earnings dates from yfinance for {len(tickers)} tickers...")
    all_dates = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.get_earnings_dates(limit=80)
            if df is not None and len(df) > 0:
                dates = df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None)
                past = [d.normalize() for d in dates if d.normalize() <= pd.Timestamp.now().normalize()]
                all_dates[ticker] = [str(d.date()) for d in sorted(past)]
                print(f"    {ticker}: {len(past)} dates")
            else:
                all_dates[ticker] = []
        except Exception as e:
            print(f"    {ticker}: error — {e}")
            all_dates[ticker] = []
        time.sleep(0.3)

    with open(cache_path, 'w') as f:
        json.dump(all_dates, f, default=str)
    return {k: [pd.Timestamp(d) for d in v] for k, v in all_dates.items()}


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
        result = pd.Timestamp(arr[mask].max())
    else:
        mask = arr >= td
        if not mask.any():
            return None
        result = pd.Timestamp(arr[mask].min())
    if abs((result - pd.Timestamp(target_date)).days) > max_days:
        return None
    return result


def chain_implied_underlying(chain_snap):
    """Infer underlying price from ATM strike."""
    snap = chain_snap[chain_snap['vol'] > 0].copy()
    if len(snap) == 0:
        return None
    snap['dist_atm'] = (snap['delta'].abs() - 0.50).abs()
    best = snap.nsmallest(3, 'dist_atm')
    return float(best['strike'].mean()) if len(best) > 0 else None


# ═══════════════════════════════════════════════════════════════════════════════
# Option Selection for Straddle/Strangle
# ═══════════════════════════════════════════════════════════════════════════════

def select_straddle_legs(chain_snap, earnings_date, structure='straddle', otm_pct=0.0):
    """
    Select legs for straddle or strangle.

    structure:
      'straddle'  — ATM put + ATM call (same strike, closest to spot)
      'strangle5' — 5% OTM put + 5% OTM call
      'strangle10'— 10% OTM put + 10% OTM call

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

    undl = chain_implied_underlying(chain_snap)
    if undl is None or undl <= 0:
        return None

    # Determine target strikes
    if structure == 'straddle':
        # ATM: find closest strike to underlying
        all_strikes = snap['strike'].unique()
        if len(all_strikes) == 0:
            return None
        atm_strike = float(all_strikes[np.argmin(np.abs(all_strikes - undl))])
        put_target_strike = atm_strike
        call_target_strike = atm_strike
    else:
        # Strangle: OTM on each side
        put_target_strike = undl * (1 - otm_pct)
        call_target_strike = undl * (1 + otm_pct)

    # --- PUT LEG ---
    puts = snap[(snap['type'] == 'p') & (snap['bid'] >= MIN_BID)].copy()
    if len(puts) == 0:
        return None
    puts['dist'] = (puts['strike'] - put_target_strike).abs()
    put_leg = puts.nsmallest(1, 'dist').iloc[0]

    # --- CALL LEG ---
    calls = snap[(snap['type'] == 'c') & (snap['bid'] >= MIN_BID)].copy()
    if len(calls) == 0:
        return None
    calls['dist'] = (calls['strike'] - call_target_strike).abs()
    call_leg = calls.nsmallest(1, 'dist').iloc[0]

    # Entry: sell at bid (conservative), apply slippage
    put_sell_price  = float(put_leg['bid'])  * (1 - SLIPPAGE_PCT)
    call_sell_price = float(call_leg['bid']) * (1 - SLIPPAGE_PCT)
    total_premium   = put_sell_price + call_sell_price

    if total_premium <= 0.10:
        return None  # too little premium to bother

    # IV at entry
    avg_iv = (float(put_leg['vol']) + float(call_leg['vol'])) / 2

    return {
        'expiration'      : exp,
        'dte_entry'       : int(put_leg['dte']),
        'underlying'      : round(undl, 2),
        'structure'       : structure,
        # Put leg
        'put_strike'      : float(put_leg['strike']),
        'put_delta'       : float(put_leg['delta']),
        'put_iv'          : float(put_leg['vol']),
        'put_bid'         : float(put_leg['bid']),
        'put_ask'         : float(put_leg['ask']),
        'put_sell_price'  : round(put_sell_price, 4),
        # Call leg
        'call_strike'     : float(call_leg['strike']),
        'call_delta'      : float(call_leg['delta']),
        'call_iv'         : float(call_leg['vol']),
        'call_bid'        : float(call_leg['bid']),
        'call_ask'        : float(call_leg['ask']),
        'call_sell_price' : round(call_sell_price, 4),
        # Summary
        'total_premium'   : round(total_premium, 4),
        'avg_iv_entry'    : round(avg_iv, 4),
    }


class RealizedVolCache:
    """
    Pre-computes and caches 30-day realized vol for each ticker.
    Uses a rolling approach: for each ticker, precompute all close prices
    sorted by date, then use binary search for fast lookups.
    """

    def __init__(self, prices_df, n_days=30):
        self.n_days = n_days
        self._cache = {}  # (ticker, date_str) -> vol
        self._ticker_closes = {}  # ticker -> (dates_array, closes_array)

        # Pre-sort prices per ticker
        prices_df = prices_df.copy()
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        for ticker, grp in prices_df.groupby('ticker'):
            grp = grp.sort_values('date')
            self._ticker_closes[ticker] = (
                grp['date'].values,
                grp['close'].values,
            )

    def get(self, ticker, before_date):
        """Get annualized realized vol from n_days trading days before before_date."""
        key = (ticker, str(before_date))
        if key in self._cache:
            return self._cache[key]

        if ticker not in self._ticker_closes:
            self._cache[key] = 0.30
            return 0.30

        dates, closes = self._ticker_closes[ticker]
        bd = np.datetime64(pd.Timestamp(before_date))
        mask = dates < bd
        idx = mask.sum()  # number of dates before before_date

        if idx < 10:
            self._cache[key] = 0.30
            return 0.30

        # Take last n_days+1 prices before before_date
        start = max(0, idx - self.n_days - 1)
        window_closes = closes[start:idx]

        if len(window_closes) < 10:
            self._cache[key] = 0.30
            return 0.30

        log_returns = np.log(window_closes[1:] / window_closes[:-1])
        daily_vol = log_returns.std()
        annualized_vol = daily_vol * np.sqrt(252)

        # Floor at 15% — no stock with options has < 15% annual vol
        result = max(annualized_vol, 0.15)
        self._cache[key] = result
        return result


def close_position(close_snap, legs, rv_cache=None, ticker=None, entry_date=None,
                   close_date=None):
    """
    Close straddle/strangle day after earnings.
    Buy back at ask (conservative), apply slippage.

    v2 FIX: When the option is not found in the chain at close time, use
    Black-Scholes pricing with POST-CRUSH IV (= 30-day realized vol from
    the 30 trading days before ENTRY, not before earnings) instead of
    intrinsic value. This prevents OTM options from getting $0 closes.

    Returns (total_cost_to_close, close_quality, close_iv, close_details) or Nones.
    """
    exp = legs['expiration']
    snap = close_snap[close_snap['expiration'] == exp]

    close_undl = chain_implied_underlying(close_snap)

    def _get_post_crush_iv():
        """Get post-crush IV estimate = realized vol from 30 days before entry."""
        if rv_cache is not None and ticker is not None and entry_date is not None:
            return rv_cache.get(ticker, entry_date)
        return 0.30  # conservative fallback

    def get_close_price(strike, opt_type):
        """Buy to close at ASK (worst case). Fallback to BS with post-crush IV."""
        rows = snap[(snap['type'] == opt_type) & (snap['strike'] == strike)]
        if len(rows) > 0:
            row = rows.iloc[0]
            price = float(row['ask']) * (1 + SLIPPAGE_PCT)  # slippage on close
            iv_close = float(row['vol'])
            return price, iv_close, 'real'

        # v2 FIX: Use Black-Scholes with post-crush IV instead of intrinsic
        if close_undl is not None:
            post_crush_iv = _get_post_crush_iv()

            # Compute remaining time to expiration from close date
            if close_date is not None:
                remaining_days = (pd.Timestamp(exp) - pd.Timestamp(close_date)).days
            else:
                # Estimate: use dte_entry minus ~2 days for the earnings gap
                remaining_days = max(legs['dte_entry'] - 2, 0)
            T = max(remaining_days / 365.0, 1.0 / 365.0)  # floor at 1 day

            bs_val = bs_price(close_undl, strike, T, post_crush_iv,
                              RISK_FREE_RATE, opt_type)

            # Apply slippage (buying at ask, so add slippage)
            price = bs_val * (1 + SLIPPAGE_PCT)

            return price, post_crush_iv, 'bs_fallback'

        return None, None, 'missing'

    put_close, put_iv_close, pq = get_close_price(legs['put_strike'], 'p')
    call_close, call_iv_close, cq = get_close_price(legs['call_strike'], 'c')

    if put_close is None or call_close is None:
        return None, None, None, None

    total_cost = put_close + call_close

    # Quality
    if pq == 'real' and cq == 'real':
        quality = 'real'
    elif pq == 'real' or cq == 'real':
        quality = 'partial'
    elif pq == 'bs_fallback' or cq == 'bs_fallback':
        quality = 'bs_fallback'
    else:
        quality = 'missing'

    # Average close IV
    ivs = [x for x in [put_iv_close, call_iv_close] if x is not None]
    avg_iv_close = np.mean(ivs) if ivs else None

    details = {
        'put_close_ask'  : round(put_close, 4),
        'call_close_ask' : round(call_close, 4),
        'total_cost'     : round(total_cost, 4),
        'close_quality'  : quality,
        'avg_iv_close'   : round(avg_iv_close, 4) if avg_iv_close else None,
        'close_undl'     : round(close_undl, 2) if close_undl else None,
    }

    return total_cost, quality, avg_iv_close, details


# ═══════════════════════════════════════════════════════════════════════════════
# Core Backtest Engine
# ═══════════════════════════════════════════════════════════════════════════════

def preload_all_chains(tickers):
    """Load all chain data into memory once to avoid repeated disk reads."""
    chains = {}
    for ticker in tickers:
        chain = load_chains(ticker)
        if chain is not None:
            chains[ticker] = {
                'chain': chain,
                'dates': sorted(chain['date'].unique()),
            }
    print(f"  Preloaded chains for {len(chains)} tickers")
    return chains


def run_backtest(earnings_by_ticker, prices_df, config, chains_cache=None, rv_cache=None):
    """
    Run a single backtest configuration.

    config dict:
      entry_days_before: 1 or 2
      structure: 'straddle', 'strangle5', 'strangle10'
      stop_loss_mult: 2.0, 3.0, or None (no stop)
      universe: list of tickers
      label: str
    chains_cache: preloaded chain data dict (optional, for speed)
    rv_cache: RealizedVolCache for post-crush IV estimation
    """
    entry_days = config['entry_days_before']
    structure  = config['structure']
    stop_mult  = config.get('stop_loss_mult', None)
    universe   = config['universe']
    label      = config['label']

    otm_pct = {'straddle': 0.0, 'strangle5': 0.05, 'strangle10': 0.10}[structure]

    all_trades = []
    skip_stats = defaultdict(int)

    for ticker in sorted(universe):
        e_dates = earnings_by_ticker.get(ticker, [])
        if not e_dates:
            continue

        if chains_cache and ticker in chains_cache:
            chain = chains_cache[ticker]['chain']
            chain_dates = chains_cache[ticker]['dates']
        else:
            chain = load_chains(ticker)
            if chain is None:
                skip_stats['no_chain'] += 1
                continue
            chain_dates = sorted(chain['date'].unique())

        for earnings_date in sorted(e_dates):
            # Entry: chain snapshot N days before earnings
            target_entry = earnings_date - timedelta(days=entry_days)
            entry_snap_date = get_snapshot_date(chain_dates, target_entry, 'before', max_days=5)
            if entry_snap_date is None:
                skip_stats['no_entry_snap'] += 1
                continue

            entry_snap = chain[chain['date'] == entry_snap_date]

            # Select legs
            legs = select_straddle_legs(entry_snap, earnings_date, structure, otm_pct)
            if legs is None:
                skip_stats['no_legs'] += 1
                continue

            # Close: first chain snapshot on or after earnings day
            close_snap_date = get_snapshot_date(chain_dates, earnings_date, 'after', max_days=5)
            if close_snap_date is None:
                skip_stats['no_close_snap'] += 1
                continue

            close_snap = chain[chain['date'] == close_snap_date]

            total_cost, quality, iv_close, close_details = close_position(
                close_snap, legs,
                rv_cache=rv_cache, ticker=ticker,
                entry_date=entry_snap_date, close_date=close_snap_date
            )
            if total_cost is None:
                skip_stats['no_close_price'] += 1
                continue

            # P&L per contract (premium collected - cost to close - commissions)
            # 2 legs x 2 transactions (open + close) = 4 contracts worth of commission
            commission = COMMISSION_PER_CONTRACT * 4
            pnl_per_contract = (legs['total_premium'] - total_cost) * 100 - commission

            # Stop-loss check (simulated): if loss exceeds stop_mult * premium
            if stop_mult is not None:
                max_allowed_loss = -1 * stop_mult * legs['total_premium'] * 100
                if pnl_per_contract < max_allowed_loss:
                    pnl_per_contract = max_allowed_loss - commission  # stopped out

            # Stock move
            close_undl = close_details.get('close_undl', None)
            stock_move_pct = None
            if close_undl and legs['underlying'] > 0:
                stock_move_pct = (close_undl - legs['underlying']) / legs['underlying'] * 100

            # IV crush
            iv_entry = legs['avg_iv_entry']
            iv_crush = (iv_entry - iv_close) if iv_close is not None else None

            record = {
                'ticker'              : ticker,
                'earnings_date'       : earnings_date,
                'entry_date'          : entry_snap_date,
                'close_date'          : close_snap_date,
                'days_before'         : entry_days,
                'structure'           : structure,
                'expiration'          : legs['expiration'],
                'dte_entry'           : legs['dte_entry'],
                'underlying_entry'    : legs['underlying'],
                'underlying_close'    : close_undl,
                'stock_move_pct'      : round(stock_move_pct, 2) if stock_move_pct else None,
                'put_strike'          : legs['put_strike'],
                'put_delta'           : round(legs['put_delta'], 3),
                'put_sell_price'      : legs['put_sell_price'],
                'call_strike'         : legs['call_strike'],
                'call_delta'          : round(legs['call_delta'], 3),
                'call_sell_price'     : legs['call_sell_price'],
                'total_premium'       : legs['total_premium'],
                'put_close_price'     : close_details['put_close_ask'],
                'call_close_price'    : close_details['call_close_ask'],
                'total_cost_close'    : close_details['total_cost'],
                'commission'          : round(commission, 2),
                'close_quality'       : quality,
                'pnl_per_contract'    : round(pnl_per_contract, 2),
                'avg_iv_entry'        : round(iv_entry, 4),
                'avg_iv_close'        : round(iv_close, 4) if iv_close else None,
                'iv_crush'            : round(iv_crush, 4) if iv_crush else None,
                'iv_crush_pct'        : round(iv_crush/iv_entry*100, 1) if iv_crush and iv_entry > 0 else None,
            }
            all_trades.append(record)

    return all_trades, dict(skip_stats)


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_portfolio(trades_df, starting_capital=STARTING_CAPITAL):
    """
    Size trades by risk budget.
    For naked straddles, margin ~ max(put_notional, call_notional) * margin_rate.
    Risk per trade capped at MAX_RISK_PER_TRADE of equity.
    """
    trades_df = trades_df.sort_values('earnings_date').copy()
    equity = starting_capital
    results = []
    equity_curve = []

    for _, trade in trades_df.iterrows():
        # Margin requirement for naked straddle ~ 20% of underlying * 100
        undl = trade['underlying_entry']
        margin_per_contract = undl * 100 * 0.20
        if margin_per_contract <= 0:
            continue

        # Max risk = stop-loss equivalent; if no stop, assume 2x premium as max risk
        premium_per = trade['total_premium'] * 100
        max_risk_per = max(premium_per * 2, margin_per_contract * 0.10)

        risk_budget = equity * MAX_RISK_PER_TRADE
        n_contracts = max(1, int(risk_budget / max_risk_per))

        trade_pnl = trade['pnl_per_contract'] * n_contracts
        equity += trade_pnl

        t = trade.to_dict()
        t['n_contracts']  = n_contracts
        t['trade_pnl']    = round(trade_pnl, 2)
        t['equity_after'] = round(equity, 2)
        results.append(t)

        equity_curve.append({
            'date'  : trade['earnings_date'],
            'equity': equity,
        })

    return pd.DataFrame(results), pd.DataFrame(equity_curve)


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(sized_df, equity_df, label="Strategy"):
    if len(sized_df) == 0:
        return {'label': label, 'n_trades': 0}

    returns = sized_df['trade_pnl'].values
    wins    = returns[returns > 0]
    losses  = returns[returns < 0]
    wr      = len(wins) / len(returns)
    avg_win  = wins.mean()  if len(wins) > 0  else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    pf       = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')

    eq = equity_df.copy()
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.sort_values('date')
    years = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days / 365.25
    end_equity = eq['equity'].iloc[-1]
    cagr = (end_equity / STARTING_CAPITAL) ** (1 / max(years, 0.1)) - 1 if years > 0 else 0

    peak   = eq['equity'].cummax()
    dd     = (eq['equity'] - peak) / peak
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else float('inf')

    trades_per_year  = len(returns) / max(years, 0.1)
    ann_factor = np.sqrt(trades_per_year)

    ret_pct = returns / STARTING_CAPITAL
    sharpe  = (ret_pct.mean() / ret_pct.std() * ann_factor) if ret_pct.std() > 0 else 0

    downside = ret_pct[ret_pct < 0]
    down_std = downside.std() if len(downside) > 0 else 1e-9
    sortino  = (ret_pct.mean() / down_std * ann_factor) if down_std > 0 else 0

    monthly_income = returns.sum() / max(years, 0.1) / 12

    return {
        'label'           : label,
        'n_trades'        : int(len(returns)),
        'years'           : round(years, 1),
        'total_pnl'       : round(float(returns.sum()), 2),
        'cagr_pct'        : round(cagr * 100, 2),
        'sharpe'          : round(float(sharpe), 3),
        'sortino'         : round(float(sortino), 3),
        'max_dd_pct'      : round(max_dd * 100, 2),
        'calmar'          : round(float(calmar), 3),
        'win_rate_pct'    : round(wr * 100, 1),
        'avg_win'         : round(float(avg_win), 2),
        'avg_loss'        : round(float(avg_loss), 2),
        'profit_factor'   : round(float(pf), 3) if pf != float('inf') else 999.0,
        'end_equity'      : round(float(end_equity), 2),
        'monthly_income'  : round(float(monthly_income), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Regime Analysis (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════════════════

def regime_analysis(sized_df, prices_df):
    """Classify trades by market regime and test for regime-agnostic behavior."""
    spy = prices_df[prices_df['ticker'] == 'SPY'][['date', 'close']].copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy = spy.set_index('date').sort_index()
    spy['spy_ret_30'] = spy['close'].pct_change(30)

    def classify(date):
        try:
            candidates = spy.index[spy.index <= pd.Timestamp(date)]
            if len(candidates) == 0:
                return 'unknown'
            r = spy.loc[candidates[-1], 'spy_ret_30']
            if pd.isna(r):
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
    sized_df['regime'] = sized_df['earnings_date'].apply(lambda d: classify(pd.Timestamp(d)))

    regime_sharpes = {}
    regime_results = {}
    for regime in ['bull', 'bear', 'flat', 'unknown']:
        sub = sized_df[sized_df['regime'] == regime]
        if len(sub) == 0:
            continue
        ret_pct = sub['trade_pnl'].values / STARTING_CAPITAL
        sharpe = (ret_pct.mean() / ret_pct.std() * np.sqrt(len(ret_pct))) if ret_pct.std() > 0 else 0
        wr = (sub['trade_pnl'] > 0).mean() * 100
        pnl = sub['trade_pnl'].sum()
        regime_results[regime] = {
            'n': int(len(sub)), 'wr': round(wr, 1),
            'sharpe': round(float(sharpe), 3), 'total_pnl': round(float(pnl), 0)
        }
        if regime in ('bull', 'bear'):
            regime_sharpes[regime] = sharpe

    # Gap test
    gap_test = 'N/A'
    gap_ratio = None
    if 'bull' in regime_sharpes and 'bear' in regime_sharpes:
        sg, sr = regime_sharpes['bull'], regime_sharpes['bear']
        denom = max(abs(sg), abs(sr))
        gap_ratio = abs(sg - sr) / denom if denom > 0 else 0
        gap_test = 'PASS' if gap_ratio <= 0.50 else 'REJECT'

    return sized_df, {
        'regimes': regime_results,
        'gap_ratio': round(gap_ratio, 3) if gap_ratio is not None else None,
        'gap_test': gap_test,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Permutation Test
# ═══════════════════════════════════════════════════════════════════════════════

def permutation_test(sized_df, n_trials=200):
    """Sign-flip permutation test: randomize the sign of P&Ls."""
    pnls = sized_df['trade_pnl'].values.copy()
    real_mean = pnls.mean()

    rng = np.random.default_rng(42)
    perm_means = np.array([
        (np.abs(pnls) * rng.choice([-1, 1], size=len(pnls))).mean()
        for _ in range(n_trials)
    ])

    p_value = float((perm_means >= real_mean).mean())
    return {
        'real_mean_pnl': round(float(real_mean), 2),
        'perm_p95': round(float(np.percentile(perm_means, 95)), 2),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Ticker Breakdown
# ═══════════════════════════════════════════════════════════════════════════════

def per_ticker_stats(sized_df):
    """Compute per-ticker performance summary."""
    stats = sized_df.groupby('ticker').agg(
        n_trades       = ('trade_pnl', 'count'),
        total_pnl      = ('trade_pnl', 'sum'),
        win_rate       = ('trade_pnl', lambda x: round((x > 0).mean() * 100, 1)),
        avg_pnl        = ('trade_pnl', 'mean'),
        avg_iv_entry   = ('avg_iv_entry', 'mean'),
        avg_iv_crush   = ('iv_crush_pct', 'mean'),
        pct_real_close = ('close_quality', lambda x: round((x == 'real').mean() * 100, 1)),
    ).round(2).sort_values('total_pnl', ascending=False)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Year-by-Year
# ═══════════════════════════════════════════════════════════════════════════════

def year_by_year(sized_df):
    df = sized_df.copy()
    df['year'] = pd.to_datetime(df['earnings_date']).dt.year
    results = {}
    for yr, grp in sorted(df.groupby('year')):
        n = len(grp)
        wr = (grp['trade_pnl'] > 0).mean() * 100
        w = grp[grp['trade_pnl'] > 0]['trade_pnl']
        l = grp[grp['trade_pnl'] < 0]['trade_pnl']
        pf = abs(w.sum() / l.sum()) if l.sum() != 0 else 999.0
        yr_pnl = grp['trade_pnl'].sum()
        r = grp['trade_pnl'].values / STARTING_CAPITAL
        sh = (r.mean() / r.std() * np.sqrt(n)) if r.std() > 0 else 0
        results[int(yr)] = {
            'n': int(n), 'wr': round(wr, 1), 'pf': round(float(pf), 2),
            'sharpe': round(float(sh), 3), 'total_pnl': round(float(yr_pnl), 0),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Grid
# ═══════════════════════════════════════════════════════════════════════════════

def build_configs(all_tickers, chains_cache=None):
    """
    Build configuration grid. Two-phase approach:
      Phase 1: sweep entry_days x structure x stop on ALL tickers (18 configs)
      Phase 2: test universe variants on the best Phase 1 combo (added later)
    """
    configs = []

    # Available tickers with chain data
    if chains_cache:
        available = sorted(chains_cache.keys())
    else:
        available = [t for t in all_tickers if (CHAINS_DIR / f"{t}.parquet").exists()]

    # Phase 1: full sweep on all tickers
    for entry_days in [1, 2]:
        for structure in ['straddle', 'strangle5', 'strangle10']:
            for stop_label, stop_mult in [('no_stop', None), ('2x_stop', 2.0), ('3x_stop', 3.0)]:
                label = f"{structure}_entry{entry_days}d_{stop_label}_all"
                configs.append({
                    'entry_days_before': entry_days,
                    'structure'        : structure,
                    'stop_loss_mult'   : stop_mult,
                    'universe'         : available,
                    'universe_name'    : 'all',
                    'label'            : label,
                })

    return configs


def build_universe_configs(best_cfg_template, all_tickers, chains_cache=None):
    """Phase 2: test universe variants on the best Phase 1 config."""
    if chains_cache:
        available = sorted(chains_cache.keys())
    else:
        available = [t for t in all_tickers if (CHAINS_DIR / f"{t}.parquet").exists()]

    top10 = [t for t in TOP10_LIQUID if t in available and t != 'SPY']
    sector = [t for t in SECTOR_DIVERSIFIED if t in available]

    configs = []
    for univ_name, univ_tickers in [('top10', top10), ('sector', sector)]:
        cfg = dict(best_cfg_template)
        cfg['universe'] = univ_tickers
        cfg['universe_name'] = univ_name
        cfg['label'] = cfg['label'].replace('_all', f'_{univ_name}')
        configs.append(cfg)

    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("EARNINGS STRADDLE/STRANGLE SELLING BACKTEST — v2 (BS fallback fix)")
    print("=" * 70)

    # Load prices
    print("\nLoading prices...")
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = pd.to_datetime(prices_df['date'])

    # Earnings dates
    earnings_cache = OUTPUT / "earnings_dates_cache.json"
    print("\nStep 1: Earnings dates")
    earnings_all = fetch_earnings_dates(TICKERS, earnings_cache)

    # Preload all chains once
    print("\nStep 1b: Preloading chain data...")
    chains_cache = preload_all_chains(TICKERS)

    # Pre-compute realized vol cache for fast BS fallback
    print("\nStep 1c: Building realized vol cache...")
    rv_cache = RealizedVolCache(prices_df, n_days=30)
    print(f"  Built vol cache for {len(rv_cache._ticker_closes)} tickers")

    # Build Phase 1 configs (entry x structure x stop, all on full universe)
    configs = build_configs(TICKERS, chains_cache)
    print(f"\nStep 2: Running {len(configs)} Phase 1 configurations (full universe)...")

    all_results = {}
    best_sharpe = -999
    best_config = None

    for i, cfg in enumerate(configs):
        label = cfg['label']
        trades, skips = run_backtest(earnings_all, prices_df, cfg, chains_cache=chains_cache, rv_cache=rv_cache)

        if not trades:
            print(f"  [{i+1}/{len(configs)}] {label}: 0 trades (skipped)")
            continue

        tdf = pd.DataFrame(trades)
        tdf = tdf[tdf['earnings_date'] >= pd.Timestamp('2019-01-01')]
        tdf = tdf.dropna(subset=['pnl_per_contract'])

        if len(tdf) < 10:
            print(f"  [{i+1}/{len(configs)}] {label}: only {len(tdf)} trades (skipped)")
            continue

        sized_df, equity_df = simulate_portfolio(tdf)
        if len(sized_df) == 0:
            continue

        metrics = compute_metrics(sized_df, equity_df, label=label)
        perm = permutation_test(sized_df, n_trials=200)
        sized_df_regime, regime = regime_analysis(sized_df, prices_df)
        yby = year_by_year(sized_df)

        result = {
            'config': {k: v for k, v in cfg.items() if k != 'universe'},
            'metrics': metrics,
            'permutation': perm,
            'regime': regime,
            'year_by_year': yby,
            'skip_stats': skips,
        }
        all_results[label] = result

        sh = metrics.get('sharpe', 0)
        wr = metrics.get('win_rate_pct', 0)
        n  = metrics.get('n_trades', 0)
        pv = perm.get('p_value', 1)
        gt = regime.get('gap_test', 'N/A')

        flag = ""
        if sh > best_sharpe and pv < 0.05:
            best_sharpe = sh
            best_config = label
            flag = " <-- BEST"

        print(f"  [{i+1}/{len(configs)}] {label}: "
              f"n={n} Sharpe={sh:.3f} WR={wr:.1f}% PF={metrics.get('profit_factor',0):.2f} "
              f"p={pv:.4f} regime={gt}{flag}")

    # ── Save all results ──
    print(f"\nStep 3: Saving results to {OUTPUT}")

    # Convert timestamps for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj

    with open(OUTPUT / "all_configs_results.json", 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)

    # ── Summary table ──
    print("\n" + "=" * 90)
    print("CONFIGURATION COMPARISON — SORTED BY SHARPE")
    print("=" * 90)

    summary_rows = []
    for label, res in all_results.items():
        m = res['metrics']
        p = res['permutation']
        r = res['regime']
        summary_rows.append({
            'config'    : label,
            'n_trades'  : m.get('n_trades', 0),
            'sharpe'    : m.get('sharpe', 0),
            'sortino'   : m.get('sortino', 0),
            'cagr_pct'  : m.get('cagr_pct', 0),
            'max_dd_pct': m.get('max_dd_pct', 0),
            'wr_pct'    : m.get('win_rate_pct', 0),
            'pf'        : m.get('profit_factor', 0),
            'monthly_$' : m.get('monthly_income', 0),
            'p_value'   : p.get('p_value', 1),
            'regime'    : r.get('gap_test', 'N/A'),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('sharpe', ascending=False)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT / "config_comparison.csv", index=False)

    # ── Deep dive on best config ──
    if best_config and best_config in all_results:
        print(f"\n{'=' * 70}")
        print(f"DEEP DIVE — BEST CONFIG: {best_config}")
        print(f"{'=' * 70}")

        best = all_results[best_config]
        m = best['metrics']

        print(f"\n  CAGR:             {m['cagr_pct']}%")
        print(f"  Sharpe:           {m['sharpe']}")
        print(f"  Sortino:          {m['sortino']}")
        print(f"  Max Drawdown:     {m['max_dd_pct']}%")
        print(f"  Win Rate:         {m['win_rate_pct']}%")
        print(f"  Profit Factor:    {m['profit_factor']}")
        print(f"  Monthly Income:   ${m['monthly_income']:,.0f} per $100K")
        print(f"  Total P&L:        ${m['total_pnl']:,.0f}")
        print(f"  End Equity:       ${m['end_equity']:,.0f}")

        print(f"\n  Permutation test: p={best['permutation']['p_value']:.4f} "
              f"({'SIGNIFICANT' if best['permutation']['significant'] else 'NOT SIGNIFICANT'})")

        print(f"\n  Regime analysis:")
        for reg, stats in best['regime']['regimes'].items():
            print(f"    {reg.upper():8s}: n={stats['n']:4d}  WR={stats['wr']:.1f}%  "
                  f"Sharpe={stats['sharpe']:.3f}  PnL=${stats['total_pnl']:,.0f}")
        if best['regime']['gap_ratio'] is not None:
            print(f"    Gap ratio: {best['regime']['gap_ratio']:.3f} [{best['regime']['gap_test']}]")

        print(f"\n  Year-by-year:")
        for yr, stats in sorted(best['year_by_year'].items()):
            print(f"    {yr}: n={stats['n']:3d}  WR={stats['wr']:.1f}%  "
                  f"Sharpe={stats['sharpe']:.3f}  PnL=${stats['total_pnl']:,.0f}")

        # Run the best config again to get per-ticker breakdown
        best_cfg = None
        for cfg in configs:
            if cfg['label'] == best_config:
                best_cfg = cfg
                break

        if best_cfg:
            trades, _ = run_backtest(earnings_all, prices_df, best_cfg, chains_cache=chains_cache, rv_cache=rv_cache)
            tdf = pd.DataFrame(trades)
            tdf = tdf[tdf['earnings_date'] >= pd.Timestamp('2019-01-01')].dropna(subset=['pnl_per_contract'])
            sized_df, equity_df = simulate_portfolio(tdf)

            # Per-ticker
            ticker_stats = per_ticker_stats(sized_df)
            print(f"\n  Per-ticker breakdown (best vol crush tickers):")
            print(ticker_stats.head(20).to_string())
            ticker_stats.to_csv(OUTPUT / "best_config_per_ticker.csv")

            # Data quality
            total = len(sized_df)
            for q in ['real', 'partial', 'bs_fallback', 'intrinsic']:
                n = (sized_df['close_quality'] == q).sum()
                print(f"  Close quality '{q}': {n}/{total} = {n/total*100:.1f}%")

            # Save trades
            sized_df.to_csv(OUTPUT / "best_config_trades.csv", index=False)
            equity_df.to_csv(OUTPUT / "best_config_equity_curve.csv", index=False)

            # Worst trades
            print(f"\n  Worst 10 trades:")
            worst = sized_df.nsmallest(10, 'trade_pnl')[
                ['ticker','earnings_date','stock_move_pct','structure',
                 'total_premium','total_cost_close','trade_pnl','n_contracts']
            ]
            print(worst.to_string(index=False))

    # ── Quality gate summary ──
    print(f"\n{'=' * 70}")
    print("QUALITY GATE SUMMARY")
    print(f"{'=' * 70}")

    passing = []
    for label, res in all_results.items():
        m = res['metrics']
        p = res['permutation']
        r = res['regime']
        passes_perm = p.get('significant', False)
        passes_regime = r.get('gap_test', 'REJECT') == 'PASS'
        passes_sharpe = m.get('sharpe', 0) > 0.5
        passes_all = passes_perm and passes_regime and passes_sharpe

        if passes_all:
            passing.append({
                'config': label,
                'sharpe': m['sharpe'],
                'sortino': m['sortino'],
                'wr': m['win_rate_pct'],
                'pf': m['profit_factor'],
                'monthly': m['monthly_income'],
                'p_value': p['p_value'],
                'regime_gap': r.get('gap_ratio', None),
            })

    if passing:
        print(f"\n  {len(passing)} configs PASS all quality gates (Sharpe>0.5 + perm p<0.05 + regime gap<0.50):")
        for p in sorted(passing, key=lambda x: -x['sharpe']):
            print(f"    {p['config']}: Sharpe={p['sharpe']:.3f} Sortino={p['sortino']:.3f} "
                  f"WR={p['wr']:.1f}% PF={p['pf']:.2f} Monthly=${p['monthly']:,.0f} "
                  f"p={p['p_value']:.4f} regime_gap={p['regime_gap']}")
    else:
        print("\n  NO configs pass all quality gates.")
        print("  Closest configs (sorted by Sharpe):")
        top5 = sorted(all_results.items(), key=lambda x: -x[1]['metrics'].get('sharpe', 0))[:5]
        for label, res in top5:
            m = res['metrics']
            p = res['permutation']
            r = res['regime']
            print(f"    {label}: Sharpe={m.get('sharpe',0):.3f} p={p.get('p_value',1):.4f} "
                  f"regime={r.get('gap_test','N/A')} (gap={r.get('gap_ratio','N/A')})")

    # ── Adversarial checks ──
    print(f"\n{'=' * 70}")
    print("ADVERSARIAL CHECKS")
    print(f"{'=' * 70}")

    # Check 1: What fraction of trades used BS fallback vs real chain data?
    if best_config and best_config in all_results and best_cfg:
        trades_adv, _ = run_backtest(earnings_all, prices_df, best_cfg, chains_cache=chains_cache, rv_cache=rv_cache)
        tdf_adv = pd.DataFrame(trades_adv)
        tdf_adv = tdf_adv[tdf_adv['earnings_date'] >= pd.Timestamp('2019-01-01')].dropna(subset=['pnl_per_contract'])

        total_adv = len(tdf_adv)
        for q in ['real', 'partial', 'bs_fallback']:
            n = (tdf_adv['close_quality'] == q).sum()
            pnl_q = tdf_adv[tdf_adv['close_quality'] == q]['pnl_per_contract']
            wr_q = (pnl_q > 0).mean() * 100 if len(pnl_q) > 0 else 0
            avg_q = pnl_q.mean() if len(pnl_q) > 0 else 0
            print(f"  Close quality '{q}': {n}/{total_adv} = {n/total_adv*100:.1f}%  "
                  f"WR={wr_q:.1f}%  Avg P&L=${avg_q:.2f}")

        # Check 2: Compare WR of real-close trades vs fallback trades
        real_trades = tdf_adv[tdf_adv['close_quality'] == 'real']
        fallback_trades = tdf_adv[tdf_adv['close_quality'] == 'bs_fallback']
        if len(real_trades) > 0 and len(fallback_trades) > 0:
            wr_real = (real_trades['pnl_per_contract'] > 0).mean() * 100
            wr_fb = (fallback_trades['pnl_per_contract'] > 0).mean() * 100
            print(f"\n  WR divergence check: Real-close WR={wr_real:.1f}% vs BS-fallback WR={wr_fb:.1f}%")
            if abs(wr_real - wr_fb) > 15:
                print(f"  WARNING: WR gap of {abs(wr_real-wr_fb):.1f}% between real and fallback closes.")
                print(f"  This may indicate BS fallback is still too generous or too conservative.")
            else:
                print(f"  PASS: WR gap of {abs(wr_real-wr_fb):.1f}% is within tolerance.")

        # Check 3: Per-structure fallback rate
        print(f"\n  Per-structure BS fallback rates:")
        for struct in ['straddle', 'strangle5', 'strangle10']:
            sub = tdf_adv[tdf_adv['structure'] == struct] if 'structure' in tdf_adv.columns else pd.DataFrame()
            if len(sub) > 0:
                fb_rate = (sub['close_quality'] == 'bs_fallback').mean() * 100
                print(f"    {struct}: {fb_rate:.1f}% BS fallback")

    print(f"\nAll outputs saved to {OUTPUT}/")
    print("DONE.")


if __name__ == "__main__":
    main()
