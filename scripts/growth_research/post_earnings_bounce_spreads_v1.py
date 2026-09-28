#!/usr/bin/env python3
"""
Post-Earnings Bounce — Debit Spreads v1
==========================================

Equity bounce after large earnings drops is VALIDATED (Sharpe 1.50, WR 62.7%,
all 5 adversarial gates pass, regime gap 0.33). Single-leg options FAILED
(theta kills the 10-day hold). This version tests DEBIT SPREADS:
- Bull call spread: limited theta, defined risk, affordable at $645.

HYPOTHESIS:
- Stocks that drop 8%+ on earnings tend to bounce over next 5-10 trading days
- The bounce is real (validated in equity) but single-leg options bleed theta
- Debit spreads cap both theta and max loss while capturing the bounce
- Lower cost per trade = more positions = better diversification

VARIANTS:
A — $5 wide bull call spread, entry T+1 after 8% drop, 10d hold, 30 DTE
B — $5 wide bull call spread, entry T+1, 5d hold, 14 DTE (faster)
C — $10 wide bull call spread, entry T+1, 10d hold, 30 DTE (wider)
D — $5 wide spread + VIX filter (VIX > 20 = sell vol, VIX < 20 = buy vol)
E — $5 wide spread + RSI oversold filter (RSI < 30 at entry)
F — LGBM-scored (features: gap size, vol, momentum, VIX) top-3 picks only

UNIVERSE: 52 growth stocks + 30 S&P 500 large caps
CAPITAL: $645 (agentic account)
MAX POS: $200 per trade, max 3 concurrent
PERIOD: 2022-01-01 to 2026-07-01 (4.5 years)
"""

import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = '.'

LOG_DIR = os.path.join(LVL3_ROOT, 'scripts', 'growth_research', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'post_earnings_bounce_spreads_v1.log')),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ==================== CONFIG ====================

# Combined universe: growth stocks + large-cap S&P names
STOCK_UNIVERSE = [
    # Growth stocks (from PEAD universe)
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
    # Additional large-caps (from original bounce study)
    'JPM', 'V', 'UNH', 'HD', 'MA', 'PG', 'JNJ', 'XOM', 'CVX', 'ABBV',
    'LLY', 'MRK', 'PEP', 'COST', 'WMT', 'CRM', 'INTC', 'BA', 'DIS', 'GS',
]

INITIAL_CAPITAL = 645.0
MAX_POS_COST = 200.0
MAX_CONCURRENT = 3
COMMISSION_PER_LEG = 0.65  # RH
COMMISSION_RT = 2.60       # 4 legs (open long, open short, close long, close short)
RISK_FREE_RATE = 0.05
BS_HAIRCUT = 0.85

OOT_START = '2022-01-01'
OOT_END = '2026-07-01'

MIN_DROP_PCT = -8.0  # 8% minimum earnings drop to trigger
HOLD_DAYS_DEFAULT = 10
DTE_DEFAULT = 30

VARIANTS = {
    'A_Spread5_10d': {
        'desc': '$5 wide bull call spread, 10d hold, 30 DTE',
        'spread_width': 5,
        'hold_days': 10,
        'dte': 30,
        'filters': {},
    },
    'B_Spread5_5d': {
        'desc': '$5 wide bull call spread, 5d hold, 14 DTE',
        'spread_width': 5,
        'hold_days': 5,
        'dte': 14,
        'filters': {},
    },
    'C_Spread10_10d': {
        'desc': '$10 wide bull call spread, 10d hold, 30 DTE',
        'spread_width': 10,
        'hold_days': 10,
        'dte': 30,
        'filters': {},
    },
    'D_VIX_Filter': {
        'desc': '$5 spread + VIX > 20 filter (higher IV = cheaper entry)',
        'spread_width': 5,
        'hold_days': 10,
        'dte': 30,
        'filters': {'vix_min': 20},
    },
    'E_RSI_Filter': {
        'desc': '$5 spread + RSI < 30 filter (deeply oversold)',
        'spread_width': 5,
        'hold_days': 10,
        'dte': 30,
        'filters': {'rsi_max': 30},
    },
    'F_Selective': {
        'desc': '$5 spread, only stocks with >60% historical bounce rate',
        'spread_width': 5,
        'hold_days': 10,
        'dte': 30,
        'filters': {'min_hist_bounce_rate': 0.60},
    },
}


# ==================== BLACK-SCHOLES ====================

def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bull_call_spread_price(S, K_long, K_short, T, r, sigma):
    """Price of a bull call spread (buy K_long, sell K_short where K_short > K_long)."""
    long_call = bs_call(S, K_long, T, r, sigma)
    short_call = bs_call(S, K_short, T, r, sigma)
    return (long_call - short_call) * BS_HAIRCUT

def bull_call_spread_max_profit(K_long, K_short, entry_cost):
    """Max profit = spread width - net debit."""
    return (K_short - K_long) * 100 - entry_cost

def bull_call_spread_breakeven(K_long, entry_premium):
    """Breakeven = long strike + net debit per share."""
    return K_long + entry_premium


# ==================== DATA ====================

def load_data():
    """Load price data and earnings dates."""
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'bounce_spreads_prices_cache.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'bounce_spreads_earnings_cache.json')

    if os.path.exists(cache_path):
        prices = pd.read_parquet(cache_path)
        log.info(f"Loaded cached prices: {len(prices)} rows")
    else:
        log.info("Downloading price data...")
        all_tickers = list(set(STOCK_UNIVERSE + ['SPY', '^VIX']))
        frames = []
        for t in all_tickers:
            try:
                df = yf.download(t, start='2020-01-01', progress=False, auto_adjust=True)
                if len(df) < 50: continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                df['ticker'] = t
                df.index.name = 'date'
                frames.append(df.reset_index())
            except:
                pass
        prices = pd.concat(frames, ignore_index=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_parquet(cache_path)
        log.info(f"  Saved {len(prices)} rows")

    if os.path.exists(earnings_cache):
        with open(earnings_cache) as f:
            earnings = json.load(f)
        log.info(f"Loaded cached earnings: {len(earnings)} tickers")
    else:
        log.info("Fetching earnings dates...")
        earnings = {}
        for t in STOCK_UNIVERSE:
            try:
                stock = yf.Ticker(t)
                dates = stock.get_earnings_dates(limit=40)
                if dates is not None and len(dates) > 0:
                    earnings[t] = [str(d.date()) if hasattr(d, 'date') else str(d)[:10]
                                   for d in dates.index]
            except:
                pass
        with open(earnings_cache, 'w') as f:
            json.dump(earnings, f)
        log.info(f"  Cached {len(earnings)} tickers")

    return prices, earnings


def compute_rsi(close_values, period=14):
    """Compute RSI for the last value in close_values."""
    if len(close_values) < period + 1:
        return 50
    rets = np.diff(close_values[-(period+1):])
    gains = np.mean(np.maximum(rets, 0))
    losses = np.mean(np.maximum(-rets, 0))
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - 100 / (1 + rs)


# ==================== BACKTESTING ====================

def run_variant(vname, config, prices, earnings):
    """Run a single variant backtest."""
    log.info(f"\n{'='*60}")
    log.info(f"  VARIANT {vname}: {config['desc']}")
    log.info(f"{'='*60}")

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    equity_dates = [pd.Timestamp(OOT_START)]
    trades = []
    open_positions = []

    spread_width = config['spread_width']
    hold_days = config['hold_days']
    dte = config['dte']
    filters = config.get('filters', {})

    # Get VIX data
    vix_df = prices[prices['ticker'] == '^VIX'].sort_values('date').reset_index(drop=True)

    # Pre-compute historical bounce rates per ticker (for variant F)
    bounce_rates = {}
    if filters.get('min_hist_bounce_rate'):
        for ticker in STOCK_UNIVERSE:
            if ticker not in earnings:
                continue
            tdf = prices[prices['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            if len(tdf) < 50:
                continue
            trading_days = tdf['date'].values
            bounces = 0
            drops = 0
            for edate_str in earnings[ticker]:
                edate = pd.Timestamp(edate_str)
                if edate >= pd.Timestamp(OOT_START):
                    continue  # only pre-OOT for training
                idx = np.searchsorted(trading_days, np.datetime64(edate))
                if idx < 1 or idx + hold_days >= len(tdf):
                    continue
                pre_close = float(tdf['close'].iloc[idx - 1])
                post_open = float(tdf['open'].iloc[idx])
                if pre_close <= 0:
                    continue
                gap = (post_open / pre_close - 1) * 100
                if gap <= MIN_DROP_PCT:
                    drops += 1
                    # Did it bounce within hold_days?
                    entry_price = float(tdf['close'].iloc[idx])
                    future_prices = tdf['close'].iloc[idx+1:idx+1+hold_days].values
                    if len(future_prices) > 0:
                        max_bounce = (np.max(future_prices) / entry_price - 1) * 100
                        if max_bounce >= 3.0:  # 3% bounce = success
                            bounces += 1
            if drops > 0:
                bounce_rates[ticker] = bounces / drops
            else:
                bounce_rates[ticker] = 0

    # Build event list
    events = []
    for ticker in STOCK_UNIVERSE:
        if ticker not in earnings:
            continue
        tdf = prices[prices['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(tdf) < 50:
            continue
        trading_days = tdf['date'].values

        for edate_str in earnings[ticker]:
            edate = pd.Timestamp(edate_str)
            if edate < pd.Timestamp(OOT_START) or edate > pd.Timestamp(OOT_END):
                continue

            idx = np.searchsorted(trading_days, np.datetime64(edate))
            if idx < 1 or idx + hold_days + 1 >= len(tdf):
                continue

            pre_close = float(tdf['close'].iloc[idx - 1])
            post_open = float(tdf['open'].iloc[idx])
            if pre_close <= 0:
                continue

            gap_pct = (post_open / pre_close - 1) * 100

            # Only large negative gaps (earnings misses / guidance cuts)
            if gap_pct > MIN_DROP_PCT:
                continue

            entry_idx = idx  # enter on earnings day close (or T+1 open)
            exit_idx = min(idx + hold_days, len(tdf) - 1)

            events.append({
                'ticker': ticker,
                'earn_date': edate,
                'gap_pct': gap_pct,
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'pre_close': pre_close,
                'post_open': post_open,
            })

    events.sort(key=lambda x: x['earn_date'])
    log.info(f"  Total drop events (>= {abs(MIN_DROP_PCT)}% drop): {len(events)}")

    # Process events
    for event in events:
        ticker = event['ticker']
        tdf = prices[prices['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        entry_idx = event['entry_idx']
        exit_idx = event['exit_idx']

        # Concurrency check
        current_open = [p for p in open_positions
                       if p['exit_date'] > event['earn_date']]
        if len(current_open) >= MAX_CONCURRENT:
            continue

        # Apply filters
        entry_price = float(tdf['close'].iloc[entry_idx])

        if filters.get('vix_min'):
            vix_mask = vix_df['date'] <= tdf['date'].iloc[entry_idx]
            if vix_mask.any():
                current_vix = float(vix_df.loc[vix_mask, 'close'].iloc[-1])
                if current_vix < filters['vix_min']:
                    continue

        if filters.get('rsi_max'):
            rsi = compute_rsi(tdf['close'].iloc[:entry_idx+1].values)
            if rsi > filters['rsi_max']:
                continue

        if filters.get('min_hist_bounce_rate'):
            if bounce_rates.get(ticker, 0) < filters['min_hist_bounce_rate']:
                continue

        # Price the bull call spread
        # Long call at ATM, short call at ATM + spread_width
        strike_long = round(entry_price)
        strike_short = strike_long + spread_width

        # Estimate IV (post-earnings = crushed, use 0.8x realized vol)
        if entry_idx >= 22:
            rets = np.diff(np.log(tdf['close'].values[entry_idx-21:entry_idx+1]))
            realized_vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 5 else 0.3
        else:
            realized_vol = 0.3
        entry_iv = realized_vol * 0.9  # post-earnings, IV slightly above realized
        entry_iv = max(entry_iv, 0.15)

        T_entry = dte / 252.0
        spread_premium = bull_call_spread_price(
            entry_price, strike_long, strike_short, T_entry, RISK_FREE_RATE, entry_iv
        )
        entry_cost = spread_premium * 100 + COMMISSION_RT

        if entry_cost <= 0 or entry_cost > MAX_POS_COST or entry_cost > capital:
            continue

        # Price at exit
        exit_price = float(tdf['close'].iloc[exit_idx])
        remaining_dte = max(dte - (exit_idx - entry_idx), 1)
        T_exit = remaining_dte / 252.0

        # IV at exit (should normalize somewhat)
        exit_iv = realized_vol * 0.95  # IV normalizing slightly after earnings drop
        exit_iv = max(exit_iv, 0.15)

        spread_exit = bull_call_spread_price(
            exit_price, strike_long, strike_short, T_exit, RISK_FREE_RATE, exit_iv
        )
        exit_value = spread_exit * 100 - COMMISSION_RT

        # P&L
        pnl = exit_value - entry_cost
        pnl_pct = pnl / entry_cost if entry_cost > 0 else 0
        capital += pnl

        # Stock move during hold
        stock_move = (exit_price / entry_price - 1) * 100

        trades.append({
            'ticker': ticker,
            'earn_date': str(event['earn_date'].date()),
            'entry_date': str(tdf['date'].iloc[entry_idx])[:10],
            'exit_date': str(tdf['date'].iloc[exit_idx])[:10],
            'gap_pct': round(event['gap_pct'], 2),
            'entry_stock': round(entry_price, 2),
            'exit_stock': round(exit_price, 2),
            'stock_move_pct': round(stock_move, 2),
            'strike_long': strike_long,
            'strike_short': strike_short,
            'entry_iv': round(entry_iv, 4),
            'spread_entry': round(spread_premium, 2),
            'spread_exit': round(spread_exit, 2),
            'entry_cost': round(entry_cost, 2),
            'exit_value': round(exit_value, 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct * 100, 2),
            'capital_after': round(capital, 2),
        })

        equity_curve.append(capital)
        equity_dates.append(event['earn_date'])

        open_positions.append({
            'ticker': ticker,
            'exit_date': pd.Timestamp(tdf['date'].iloc[exit_idx]),
        })

    return trades, equity_curve, equity_dates


# ==================== EVALUATION ====================

def evaluate(name, config, trades, equity_curve, equity_dates):
    n = len(trades)
    if n == 0:
        log.info(f"  {name}: NO TRADES")
        return {'name': name, 'n_trades': 0, 'gates_passed': 0}

    pnls = [t['pnl'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    total_pnl = sum(pnls)
    final = equity_curve[-1]
    total_ret = (final / INITIAL_CAPITAL - 1) * 100

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    years = max((equity_dates[-1] - equity_dates[0]).days / 365.25, 0.5) if len(equity_dates) > 1 else 1

    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(n / years)
    else:
        sharpe = 0

    downside = [p for p in pnl_pcts if p < 0]
    if downside and np.std(downside) > 0:
        sortino = np.mean(pnl_pcts) / np.std(downside) * np.sqrt(n / years)
    else:
        sortino = sharpe

    # MDD
    peak = INITIAL_CAPITAL
    mdd = 0
    for eq in equity_curve:
        if eq > peak: peak = eq
        dd = (eq - peak) / peak
        if dd < mdd: mdd = dd
    mdd_pct = mdd * 100

    # Gates
    gates = 0
    gate_results = {}

    # G1: Sharpe > 1.0
    gate_results['sharpe'] = sharpe >= 1.0
    if gate_results['sharpe']: gates += 1

    # G2: Permutation test
    if n >= 8:
        obs = np.mean(pnl_pcts)
        count = 0
        for _ in range(1000):
            shuf = np.random.choice([-1, 1], size=n) * np.abs(pnl_pcts)
            if np.mean(shuf) >= obs: count += 1
        perm_p = count / 1000
    else:
        perm_p = 1.0
    gate_results['perm'] = perm_p < 0.05
    if gate_results['perm']: gates += 1

    # G3: Regime balance
    first_half = [t['pnl_pct'] for t in trades if t['entry_date'] < '2023-07-01']
    second_half = [t['pnl_pct'] for t in trades if t['entry_date'] >= '2023-07-01']
    if first_half and second_half:
        s1 = np.mean(first_half) / max(np.std(first_half), 0.01)
        s2 = np.mean(second_half) / max(np.std(second_half), 0.01)
        regime_gap = abs(s1 - s2) / max(abs(s1), abs(s2), 0.01)
    else:
        regime_gap = 1.0
    gate_results['regime'] = regime_gap < 0.50
    if gate_results['regime']: gates += 1

    # G4: Random baseline
    rand_sharpes = []
    for _ in range(100):
        rand_pnl = np.random.choice([-1, 1], size=n) * np.abs(pnl_pcts)
        if np.std(rand_pnl) > 0:
            rand_sharpes.append(np.mean(rand_pnl) / np.std(rand_pnl) * np.sqrt(n / years))
    rand_sharpe = np.mean(rand_sharpes) if rand_sharpes else 0
    gate_results['random'] = sharpe > rand_sharpe * 1.2
    if gate_results['random']: gates += 1

    # G5: MDD < 50%
    gate_results['mdd'] = abs(mdd_pct) < 50
    if gate_results['mdd']: gates += 1

    # Concentration
    ticker_pnl = {}
    for t in trades:
        ticker_pnl[t['ticker']] = ticker_pnl.get(t['ticker'], 0) + t['pnl']
    if total_pnl > 0:
        top3 = sorted(ticker_pnl.values(), reverse=True)[:3]
        conc = sum(top3) / total_pnl * 100
    else:
        conc = 0

    result = {
        'name': name,
        'desc': config['desc'],
        'n_trades': n,
        'wins': wins,
        'wr': round(wr, 1),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 2),
        'final': round(final, 2),
        'total_return': round(total_ret, 1),
        'mdd': round(mdd_pct, 1),
        'perm_p': round(perm_p, 4),
        'regime_gap': round(regime_gap, 3),
        'random_sharpe': round(rand_sharpe, 3),
        'concentration_top3': round(conc, 1),
        'gates_passed': gates,
        'gate_results': gate_results,
        'avg_stock_bounce': round(np.mean([t['stock_move_pct'] for t in trades]), 2),
    }

    log.info(f"\n  --- {name} ---")
    log.info(f"  Trades: {n} (W:{wins} L:{n-wins} WR:{wr:.0f}%)")
    log.info(f"  Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | PF: {pf:.2f}")
    log.info(f"  Final: ${final:.0f} ({total_ret:+.1f}%) | MDD: {mdd_pct:.1f}%")
    log.info(f"  Perm p: {perm_p:.4f} | Regime: {regime_gap:.3f} | Random: {rand_sharpe:.3f}")
    log.info(f"  Avg stock bounce: {result['avg_stock_bounce']:+.2f}%")
    log.info(f"  GATES: {gates}/5 {'✅' if gates >= 4 else '❌'}")
    for g, v in gate_results.items():
        log.info(f"    {g}: {'PASS' if v else 'FAIL'}")

    return result


# ==================== MAIN ====================

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("  Post-Earnings Bounce — Debit Spreads v1")
    log.info("=" * 60)

    prices, earnings = load_data()
    all_results = []

    for vname, vconfig in VARIANTS.items():
        try:
            trades, eq_curve, eq_dates = run_variant(vname, vconfig, prices, earnings)
            result = evaluate(vname, vconfig, trades, eq_curve, eq_dates)
            all_results.append(result)
        except Exception as e:
            log.error(f"  {vname}: FAILED ({e})")
            import traceback
            traceback.print_exc()
            all_results.append({'name': vname, 'gates_passed': 0, 'error': str(e)})

    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"  SUMMARY — Post-Earnings Bounce Spreads v1 ({elapsed:.0f}s)")
    log.info(f"{'='*60}")

    for r in sorted(all_results, key=lambda x: x.get('gates_passed', 0), reverse=True):
        gates = r.get('gates_passed', 0)
        sharpe = r.get('sharpe', 0)
        n = r.get('n_trades', 0)
        final = r.get('final', INITIAL_CAPITAL)
        wr = r.get('wr', 0)
        log.info(f"  {r['name']}: {gates}/5 gates | Sharpe {sharpe:.3f} | "
                 f"{n} trades | WR {wr:.0f}% | ${final:.0f}")

    # Save
    output_path = os.path.join(LVL3_ROOT, 'research', 'findings', 'post_earnings_bounce_spreads_v1.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nSaved to {output_path}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment('post_earnings_bounce_spreads_v1')
        for r in all_results:
            with mlflow.start_run(run_name=r.get('name', 'unknown')):
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_param('variant', r.get('name', ''))
                mlflow.log_param('desc', r.get('desc', ''))
        log.info("Logged to MLflow")
    except Exception as e:
        log.warning(f"MLflow: {e}")


if __name__ == '__main__':
    main()
