#!/usr/bin/env python3
"""
Earnings IV Crush Income Strategy Backtest — v1
================================================
Hypothesis: IV spikes before earnings, then collapses ("IV crush") after
the announcement regardless of directional outcome. This creates edge
for option sellers who sell pre-earnings and close post-earnings.

Strategy:
  - 1-2 days before earnings: sell a short strangle (~20-delta put + ~20-delta call)
    on the nearest expiration AFTER earnings (to capture maximum crush)
  - Day after earnings: close the position at mid-price
  - Commission: $0 (Robinhood)
  - Position sizing: risk 2% of portfolio; for strangles margin ~20% of underlying

Metrics (HC #69): Sharpe, Sortino, CAGR, MaxDD, Calmar, WR, PF
Regime analysis: HC #428 R1 — must hold across bull/bear/flat regimes.
Validation: permutation test (shuffle earnings dates, 100 trials).

Universe: 69 tickers with real options chain data (Dolt).
Data: /home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/options_real/chains/
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
OUTPUT      = ROOT / "output" / "earnings_crush_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL  = 100_000
RISK_PER_TRADE    = 0.02       # 2% of portfolio per trade
STRANGLE_MARGIN   = 0.20       # margin ~20% of underlying for naked strangle
TARGET_DELTA      = 0.20       # sell ~20-delta options on each side
MAX_DTE_ENTRY     = 35         # only use expirations ≤ 35 DTE at entry
MIN_DTE_ENTRY     = 2          # at least 2 DTE at entry (avoid same-day expiry)
MIN_BID           = 0.05       # skip options with zero/penny bid
BA_HALF_SPREAD    = 0.10       # 10% of mid as conservative fill friction each way

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

# SPY is useful as a benchmark, exclude from trading
TRADING_TICKERS = [t for t in TICKERS if t != 'SPY']


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Fetch Earnings Dates
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_earnings_dates(tickers, cache_path):
    """Fetch earnings dates for all tickers, with a local JSON cache."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  Loaded earnings dates from cache ({len(cached)} tickers)")
        return {k: [pd.Timestamp(d) for d in v] for k, v in cached.items()}

    print(f"  Fetching earnings dates from yfinance for {len(tickers)} tickers...")
    all_dates = {}
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            df = stock.get_earnings_dates(limit=80)
            if df is not None and len(df) > 0:
                dates = df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None)
                # only past dates (exclude future)
                past = [d.normalize() for d in dates if d.normalize() <= pd.Timestamp.now().normalize()]
                all_dates[ticker] = [str(d.date()) for d in sorted(past)]
                print(f"    {ticker}: {len(past)} earnings dates ({sorted(past)[0].date()} – {sorted(past)[-1].date()})")
            else:
                print(f"    {ticker}: no data")
                all_dates[ticker] = []
        except Exception as e:
            print(f"    {ticker}: error — {e}")
            all_dates[ticker] = []
        time.sleep(0.3)  # be kind to yfinance

    with open(cache_path, 'w') as f:
        json.dump(all_dates, f, default=str)
    print(f"  Saved earnings cache to {cache_path}")

    return {k: [pd.Timestamp(d) for d in v] for k, v in all_dates.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Load Options Chains
# ═══════════════════════════════════════════════════════════════════════════════

def load_chains(ticker):
    """Load options chain parquet for one ticker, return DataFrame."""
    p = CHAINS_DIR / f"{ticker}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    df = df[df['vol'] > 0].copy()
    return df


def get_snapshot_date(chain_dates, target_date, direction='before', max_days=5):
    """
    Find nearest chain snapshot before (or after) target_date.
    direction: 'before' → latest snapshot <= target
               'after'  → earliest snapshot >= target
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# Underlying Price Helper — chain-implied (avoids split-adjust mismatch)
# ═══════════════════════════════════════════════════════════════════════════════

def chain_implied_underlying(chain_snap):
    """
    Infer the underlying price from the ATM option strike in the chain snapshot.
    This avoids the problem where prices.parquet uses post-split adjusted prices
    but the options chain retains pre-split strikes.
    We use the strike of the option with delta closest to ±0.50.
    """
    snap = chain_snap[chain_snap['vol'] > 0].copy()
    if len(snap) == 0:
        return None
    snap['dist_atm'] = (snap['delta'].abs() - 0.50).abs()
    best = snap.nsmallest(3, 'dist_atm')
    if len(best) == 0:
        return None
    # Average the top-3 ATM strike prices weighted by how close they are
    return float(best['strike'].mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Select Strangle Legs
# ═══════════════════════════════════════════════════════════════════════════════

def select_strangle_legs(chain_snap, earnings_date, target_delta=TARGET_DELTA):
    """
    From a single snapshot, select:
      - nearest-expiration that is AFTER earnings_date AND within MAX_DTE_ENTRY
      - OTM put closest to -target_delta
      - OTM call closest to +target_delta
    Returns dict with leg info, or None if not found.
    """
    # Expirations that fall after earnings and within our DTE window
    valid_exp = chain_snap[
        (chain_snap['expiration'] > earnings_date) &
        (chain_snap['dte'] >= MIN_DTE_ENTRY) &
        (chain_snap['dte'] <= MAX_DTE_ENTRY)
    ]['expiration'].unique()

    if len(valid_exp) == 0:
        return None

    # Use the nearest expiration
    exp = pd.Timestamp(min(valid_exp))

    snap_exp = chain_snap[chain_snap['expiration'] == exp].copy()

    # Select put leg (~-target_delta)
    puts = snap_exp[snap_exp['type'] == 'p'].copy()
    puts = puts[(puts['delta'] < 0) & (puts['bid'] >= MIN_BID)]
    if len(puts) == 0:
        return None
    puts['delta_dist'] = (puts['delta'].abs() - target_delta).abs()
    put_leg = puts.nsmallest(1, 'delta_dist').iloc[0]

    # Select call leg (~+target_delta)
    calls = snap_exp[snap_exp['type'] == 'c'].copy()
    calls = calls[(calls['delta'] > 0) & (calls['bid'] >= MIN_BID)]
    if len(calls) == 0:
        return None
    calls['delta_dist'] = (calls['delta'].abs() - target_delta).abs()
    call_leg = calls.nsmallest(1, 'delta_dist').iloc[0]

    # Entry fill: sell at mid minus half-spread friction
    put_entry  = put_leg['mid']  * (1 - BA_HALF_SPREAD)
    call_entry = call_leg['mid'] * (1 - BA_HALF_SPREAD)

    return {
        'expiration'   : exp,
        'dte_entry'    : int(put_leg['dte']),
        'put_strike'   : float(put_leg['strike']),
        'put_delta'    : float(put_leg['delta']),
        'put_iv_entry' : float(put_leg['vol']),
        'put_mid_entry': float(put_leg['mid']),
        'put_fill'     : put_entry,
        'call_strike'  : float(call_leg['strike']),
        'call_delta'   : float(call_leg['delta']),
        'call_iv_entry': float(call_leg['vol']),
        'call_mid_entry': float(call_leg['mid']),
        'call_fill'    : call_entry,
        'total_premium': put_entry + call_entry,
        'avg_iv_entry' : (float(put_leg['vol']) + float(call_leg['vol'])) / 2,
    }


def get_close_prices(chain_snap_close, put_strike, call_strike, expiration):
    """
    From the post-earnings snapshot, get the close price for each leg.
    Returns (put_close_mid, call_close_mid) or None if legs not found.
    """
    snap = chain_snap_close[chain_snap_close['expiration'] == expiration]
    if len(snap) == 0:
        return None, None

    put_rows  = snap[(snap['type'] == 'p') & (snap['strike'] == put_strike)]
    call_rows = snap[(snap['type'] == 'c') & (snap['strike'] == call_strike)]

    put_mid  = float(put_rows['mid'].iloc[0])  if len(put_rows) > 0  else None
    call_mid = float(call_rows['mid'].iloc[0]) if len(call_rows) > 0 else None

    # If strikes disappeared from chain (deep ITM/OTM after gap), use intrinsic
    return put_mid, call_mid


def get_avg_iv_close(chain_snap_close, expiration, target_delta=TARGET_DELTA):
    """Average IV of ~20-delta options in the close snapshot for IV crush measurement."""
    snap = chain_snap_close[chain_snap_close['expiration'] == expiration].copy()
    near = snap[(snap['delta'].abs() >= 0.10) & (snap['delta'].abs() <= 0.35) & (snap['vol'] > 0)]
    if len(near) == 0:
        return None
    return float(near['vol'].mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Run the Backtest
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(earnings_by_ticker, prices_df):
    """
    Main backtest loop. Returns list of trade records.
    """
    print("\nRunning backtest...")

    # Index prices for quick lookup
    prices_idx = prices_df.set_index(['ticker', 'date'])['close']

    all_trades = []

    for ticker in sorted(TRADING_TICKERS):
        e_dates = earnings_by_ticker.get(ticker, [])
        if not e_dates:
            continue

        chain = load_chains(ticker)
        if chain is None:
            print(f"  {ticker}: no chain data, skipping")
            continue

        chain_dates = sorted(chain['date'].unique())

        trade_count = 0
        for earnings_date in sorted(e_dates):
            # ── Entry snapshot: last chain snapshot before earnings ──────────
            entry_snap_date = get_snapshot_date(chain_dates, earnings_date - timedelta(days=1), 'before', max_days=5)
            if entry_snap_date is None:
                continue

            # Must be within 3 calendar days of earnings
            days_before = (earnings_date - entry_snap_date).days
            if days_before > 4:
                continue

            entry_snap = chain[chain['date'] == entry_snap_date].copy()

            # Get underlying price from chain ATM strike (avoids split-adjust mismatch)
            undl_price = chain_implied_underlying(entry_snap)
            if undl_price is None or undl_price <= 0:
                continue

            # ── Select strangle legs ─────────────────────────────────────────
            legs = select_strangle_legs(entry_snap, earnings_date)
            if legs is None:
                continue

            # ── Close snapshot: first chain snapshot on or after earnings ────
            close_snap_date = get_snapshot_date(chain_dates, earnings_date, 'after', max_days=5)
            if close_snap_date is None:
                continue
            if (close_snap_date - earnings_date).days > 4:
                continue

            close_snap = chain[chain['date'] == close_snap_date].copy()

            # ── Close prices ─────────────────────────────────────────────────
            put_close_mid, call_close_mid = get_close_prices(
                close_snap, legs['put_strike'], legs['call_strike'], legs['expiration']
            )

            # Get close underlying from chain ATM (avoids split-adjust mismatch)
            close_undl = chain_implied_underlying(close_snap)

            if put_close_mid is None and close_undl is not None:
                put_close_mid = max(legs['put_strike'] - close_undl, 0.0)
            if call_close_mid is None and close_undl is not None:
                call_close_mid = max(close_undl - legs['call_strike'], 0.0)

            if put_close_mid is None or call_close_mid is None:
                continue

            # Add fill friction on close (buy to close = pay more)
            put_close_fill  = put_close_mid  * (1 + BA_HALF_SPREAD)
            call_close_fill = call_close_mid * (1 + BA_HALF_SPREAD)

            # ── P&L per contract ─────────────────────────────────────────────
            # We sold (collected) total_premium, paid (total_close) to close
            premium_collected = legs['total_premium']
            premium_paid_close = put_close_fill + call_close_fill
            pnl_per_contract = (premium_collected - premium_paid_close) * 100  # per contract

            # ── IV Crush measurement ─────────────────────────────────────────
            iv_entry = legs['avg_iv_entry']
            iv_close = get_avg_iv_close(close_snap, legs['expiration'])
            iv_crush = (iv_entry - iv_close) if iv_close is not None else None

            # ── Stock move ───────────────────────────────────────────────────
            stock_move_pct = None
            if close_undl is not None and undl_price is not None:
                stock_move_pct = (close_undl - undl_price) / undl_price * 100

            trade_count += 1
            all_trades.append({
                'ticker'              : ticker,
                'earnings_date'       : earnings_date,
                'entry_date'          : entry_snap_date,
                'close_date'          : close_snap_date,
                'days_before_earnings': days_before,
                'expiration'          : legs['expiration'],
                'dte_entry'           : legs['dte_entry'],
                'underlying_entry'    : round(float(undl_price), 2),
                'underlying_close'    : round(float(close_undl), 2) if close_undl is not None else None,
                'stock_move_pct'      : round(stock_move_pct, 2) if stock_move_pct is not None else None,
                'put_strike'          : legs['put_strike'],
                'put_delta_entry'     : round(legs['put_delta'], 3),
                'put_iv_entry'        : round(legs['put_iv_entry'], 4),
                'put_mid_entry'       : round(legs['put_mid_entry'], 3),
                'put_fill_entry'      : round(legs['put_fill'], 3),
                'put_close_fill'      : round(put_close_fill, 3),
                'call_strike'         : legs['call_strike'],
                'call_delta_entry'    : round(legs['call_delta'], 3),
                'call_iv_entry'       : round(legs['call_iv_entry'], 4),
                'call_mid_entry'      : round(legs['call_mid_entry'], 3),
                'call_fill_entry'     : round(legs['call_fill'], 3),
                'call_close_fill'     : round(call_close_fill, 3),
                'premium_collected'   : round(premium_collected, 3),
                'premium_paid_close'  : round(premium_paid_close, 3),
                'pnl_per_contract'    : round(pnl_per_contract, 2),
                'avg_iv_entry'        : round(iv_entry, 4),
                'avg_iv_close'        : round(iv_close, 4) if iv_close is not None else None,
                'iv_crush'            : round(iv_crush, 4) if iv_crush is not None else None,
                'iv_crush_pct'        : round(iv_crush / iv_entry * 100, 1) if iv_crush is not None and iv_entry > 0 else None,
            })

        if trade_count > 0:
            print(f"  {ticker}: {trade_count} trades")

    print(f"\nTotal trades found: {len(all_trades)}")
    return all_trades


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Portfolio Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_portfolio(trades_df, starting_capital=STARTING_CAPITAL):
    """
    Simulate portfolio equity curve.
    For each trade: size = 2% of current capital / (margin requirement)
    Margin = 20% of underlying × 100 (one contract).
    Number of contracts = max(1, floor(risk_budget / margin_per_contract))
    """
    trades_df = trades_df.sort_values('earnings_date').copy()
    equity = starting_capital
    equity_curve = []
    sized_trades = []

    for _, trade in trades_df.iterrows():
        undl = trade['underlying_entry']
        if undl <= 0 or np.isnan(undl):
            continue

        margin_per_contract = undl * STRANGLE_MARGIN * 100
        risk_budget = equity * RISK_PER_TRADE
        n_contracts = max(1, int(risk_budget / margin_per_contract))

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
# Step 6: Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(sized_df, equity_df, label="Strategy"):
    """Compute full risk-adjusted performance metrics."""
    if len(sized_df) == 0:
        print("No trades to evaluate.")
        return {}

    returns = sized_df['trade_pnl'].values

    # Win/loss stats
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    wr     = len(wins) / len(returns)
    avg_win  = wins.mean()  if len(wins) > 0  else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    pf       = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf

    # CAGR
    eq = equity_df.copy()
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.sort_values('date')
    years = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days / 365.25
    start_equity = STARTING_CAPITAL
    end_equity   = eq['equity'].iloc[-1]
    cagr = (end_equity / start_equity) ** (1 / max(years, 0.1)) - 1 if years > 0 else 0

    # Max drawdown
    peak = eq['equity'].cummax()
    dd   = (eq['equity'] - peak) / peak
    max_dd = float(dd.min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    # Sharpe / Sortino (per-trade, annualized assuming ~70 trades/year)
    trades_per_year = len(returns) / max(years, 0.1)
    annualize_factor = np.sqrt(trades_per_year)

    ret_pct = returns / STARTING_CAPITAL
    sharpe  = (ret_pct.mean() / ret_pct.std() * annualize_factor) if ret_pct.std() > 0 else 0

    downside = ret_pct[ret_pct < 0]
    down_std = downside.std() if len(downside) > 0 else 1e-9
    sortino  = (ret_pct.mean() / down_std * annualize_factor) if down_std > 0 else 0

    total_pnl = returns.sum()

    metrics = {
        'label'       : label,
        'n_trades'    : len(returns),
        'years'       : round(years, 1),
        'total_pnl'   : round(total_pnl, 2),
        'cagr'        : round(cagr * 100, 2),
        'sharpe'      : round(sharpe, 3),
        'sortino'     : round(sortino, 3),
        'max_dd'      : round(max_dd * 100, 2),
        'calmar'      : round(calmar, 3),
        'win_rate'    : round(wr * 100, 1),
        'avg_win'     : round(avg_win, 2),
        'avg_loss'    : round(avg_loss, 2),
        'profit_factor': round(pf, 3),
        'end_equity'  : round(end_equity, 2),
    }
    return metrics


def print_metrics(m):
    print(f"\n{'='*55}")
    print(f"  {m.get('label','Strategy')} — {m.get('n_trades',0)} trades, {m.get('years',0)} years")
    print(f"{'='*55}")
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
    print(f"{'='*55}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7: Regime Analysis (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════════════════

def regime_analysis(sized_df, prices_df):
    """
    Classify each trade date as bull/bear/flat based on SPY 30-day momentum.
    Per HC #428 R1: test across all regimes, compute stratified Sharpe.
    """
    spy = prices_df[prices_df['ticker'] == 'SPY'][['date', 'close']].set_index('date').sort_index()
    spy['spy_ret_30'] = spy['close'].pct_change(30)

    def classify_regime(date):
        try:
            r = spy.loc[date, 'spy_ret_30'] if date in spy.index else None
            if r is None:
                # find nearest
                candidates = spy.index[spy.index <= date]
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
        pf_val = abs(sub[sub['trade_pnl']>0]['trade_pnl'].sum() /
                     sub[sub['trade_pnl']<0]['trade_pnl'].sum()) if (sub['trade_pnl']<0).any() else np.inf
        print(f"  {regime.upper():7s}: n={len(sub):4d}  WR={wr:.1f}%  Sharpe={sharpe:.3f}  PF={pf_val:.3f}")

    # HC #428 R1 reject gate: |Sharpe_green - Sharpe_red| / max(|Sg|,|Sr|) > 0.50
    regime_sharpes = {}
    for regime in ['bull', 'bear']:
        sub = sized_df[sized_df['regime'] == regime]
        if len(sub) < 5:
            continue
        ret_pct = sub['trade_pnl'].values / STARTING_CAPITAL
        std_r  = ret_pct.std()
        regime_sharpes[regime] = (ret_pct.mean() / std_r * np.sqrt(len(ret_pct))) if std_r > 0 else 0

    if 'bull' in regime_sharpes and 'bear' in regime_sharpes:
        sg, sr = regime_sharpes['bull'], regime_sharpes['bear']
        denom = max(abs(sg), abs(sr))
        ratio = abs(sg - sr) / denom if denom > 0 else 0
        flag  = "REJECT" if ratio > 0.50 else "PASS"
        print(f"\n  Regime gap test: |Sharpe_bull - Sharpe_bear| / max = {ratio:.3f}  [{flag}]")

    return sized_df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 8: Big Gap Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def gap_analysis(sized_df):
    """Analyze what happens when stock gaps ≥10% on earnings."""
    print("\n── Gap Analysis ─────────────────────────────────────────────")
    df = sized_df.dropna(subset=['stock_move_pct']).copy()
    df['abs_move'] = df['stock_move_pct'].abs()

    thresholds = [5, 10, 15, 20]
    for thr in thresholds:
        big = df[df['abs_move'] >= thr]
        small = df[df['abs_move'] < thr]
        if len(big) == 0:
            continue
        big_wr  = (big['trade_pnl'] > 0).mean() * 100
        small_wr = (small['trade_pnl'] > 0).mean() * 100
        big_avg  = big['trade_pnl'].mean()
        small_avg = small['trade_pnl'].mean()
        print(f"  |Move| ≥{thr:3d}%: n={len(big):4d}  WR={big_wr:.1f}%  AvgPnL=${big_avg:.0f}  "
              f"  |Move| <{thr:3d}%: n={len(small):4d}  WR={small_wr:.1f}%  AvgPnL=${small_avg:.0f}")

    # Top 5 worst trades
    print("\n  Top 5 worst trades:")
    worst = sized_df.nsmallest(5, 'trade_pnl')[
        ['ticker','earnings_date','stock_move_pct','put_strike','call_strike',
         'premium_collected','premium_paid_close','trade_pnl','n_contracts']
    ]
    print(worst.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Step 9: Permutation Test
# ═══════════════════════════════════════════════════════════════════════════════

def permutation_test(sized_df, n_trials=100):
    """
    Permutation test: randomly assign +1/-1 signs to trade PnLs.
    If the strategy has real edge, the true mean return should be
    significantly above zero vs the sign-permuted distribution.

    This tests whether the DIRECTION of the PnL is systematic
    (i.e., we are consistently right about the IV crush direction),
    not just whether the magnitudes happen to work out.
    """
    print(f"\n── Permutation Test ({n_trials} trials) ─────────────────────────")
    pnls = sized_df['trade_pnl'].values.copy()
    real_mean = pnls.mean()

    perm_means = []
    rng = np.random.default_rng(42)

    for _ in range(n_trials):
        # Randomly flip signs
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


# ═══════════════════════════════════════════════════════════════════════════════
# Step 10: IV Crush Summary
# ═══════════════════════════════════════════════════════════════════════════════

def iv_crush_summary(sized_df):
    """Print a sample of trades showing IV crush and best overall stats."""
    df = sized_df.dropna(subset=['iv_crush']).copy()
    print(f"\n── IV Crush Summary ─────────────────────────────────────────")
    print(f"  Trades with IV crush data: {len(df)}")
    print(f"  Avg IV entry:  {df['avg_iv_entry'].mean():.3f}")
    print(f"  Avg IV close:  {df['avg_iv_close'].mean():.3f}")
    print(f"  Avg crush:     {df['iv_crush'].mean():.3f}  ({df['iv_crush_pct'].mean():.1f}%)")
    print(f"  % events with IV crush (IV fell): {(df['iv_crush'] > 0).mean()*100:.1f}%")

    # Sample of 10 trades with good IV crush
    print("\n  Sample of 10 trades (highest IV crush):")
    sample_cols = ['ticker','earnings_date','avg_iv_entry','avg_iv_close','iv_crush_pct',
                   'stock_move_pct','premium_collected','premium_paid_close','trade_pnl']
    top10 = df.nlargest(10, 'iv_crush')[sample_cols]
    print(top10.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("EARNINGS IV CRUSH BACKTEST — v1")
    print("="*60)

    # Load prices
    print("\nLoading prices...")
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = pd.to_datetime(prices_df['date'])

    # Fetch earnings dates
    earnings_cache = OUTPUT / "earnings_dates_cache.json"
    print("\nStep 1: Earnings dates")
    earnings_by_ticker = fetch_earnings_dates(TRADING_TICKERS, earnings_cache)

    # Run backtest
    print("\nStep 2: Running backtest")
    trades = run_backtest(earnings_by_ticker, prices_df)

    if not trades:
        print("No trades found. Check data.")
        return

    trades_df = pd.DataFrame(trades)
    # Filter to dates within our data range (chains data starts 2019-02)
    trades_df = trades_df[trades_df['earnings_date'] >= pd.Timestamp('2019-01-01')]
    # Drop trades with NaN pnl
    trades_df = trades_df.dropna(subset=['pnl_per_contract'])

    print(f"\nTrades after filtering: {len(trades_df)}")
    print(f"Date range: {trades_df['earnings_date'].min().date()} – {trades_df['earnings_date'].max().date()}")

    # Portfolio simulation
    print("\nStep 3: Portfolio simulation")
    sized_df, equity_df = simulate_portfolio(trades_df)

    # Save raw trades
    sized_df.to_csv(OUTPUT / "trades.csv", index=False)
    equity_df.to_csv(OUTPUT / "equity_curve.csv", index=False)
    print(f"Saved trades to {OUTPUT}/trades.csv")

    # Performance metrics
    m = compute_metrics(sized_df, equity_df, label="Earnings IV Crush — Short Strangle")
    print_metrics(m)
    with open(OUTPUT / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)

    # IV crush summary
    iv_crush_summary(sized_df)

    # Regime analysis (HC #428 R1)
    sized_df = regime_analysis(sized_df, prices_df)

    # Gap analysis (big movers)
    gap_analysis(sized_df)

    # Permutation test
    permutation_test(sized_df, n_trials=100)

    # Per-ticker summary
    print("\n── Per-Ticker Summary ───────────────────────────────────────")
    ticker_stats = sized_df.groupby('ticker').agg(
        n_trades=('trade_pnl', 'count'),
        total_pnl=('trade_pnl', 'sum'),
        win_rate=('trade_pnl', lambda x: (x > 0).mean() * 100),
        avg_pnl=('trade_pnl', 'mean'),
        avg_iv_entry=('avg_iv_entry', 'mean'),
        avg_iv_crush_pct=('iv_crush_pct', 'mean'),
    ).round(2).sort_values('total_pnl', ascending=False)
    print(ticker_stats.to_string())
    ticker_stats.to_csv(OUTPUT / "per_ticker.csv")

    print(f"\nAll outputs saved to {OUTPUT}/")
    print("\nDONE.")


if __name__ == "__main__":
    main()
