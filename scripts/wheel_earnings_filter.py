#!/usr/bin/env python3
"""
wheel_earnings_filter.py — Earnings Avoidance Filter for Wheel Strategy

HYPOTHESIS: Selling puts when earnings fall within the option's DTE window
exposes the strategy to gap risk (3-10% overnight moves). By skipping those
weeks, we should reduce max drawdown and improve Sharpe without sacrificing
much total return (since we have 220+ tickers to rotate into).

APPROACH:
1. Download historical quarterly earnings dates for all tickers via yfinance
2. Run the portfolio backtest WITH and WITHOUT earnings avoidance
3. Compare metrics: Sharpe, MaxDD, Calmar, CAGR

The filter: on any day we'd open a new CSP for ticker T, check if T has
an earnings date within [today, today + DTE]. If yes, SKIP that ticker.
"""
import sys
import time
import math
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_earnings_filter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    format='%(asctime)s [EARN-FILT] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('EARN-FILT')


def download_earnings_dates(tickers, cache_file=None):
    """Download historical earnings dates for all tickers via yfinance."""
    if cache_file is None:
        cache_file = CACHE / "earnings_dates.parquet"

    if cache_file.exists():
        log.info(f"Loading cached earnings dates from {cache_file}")
        return pd.read_parquet(cache_file)

    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed")
        sys.exit(1)

    all_earnings = []
    failed = []

    log.info(f"Downloading earnings dates for {len(tickers)} tickers...")

    for i, ticker in enumerate(sorted(tickers)):
        if (i + 1) % 20 == 0:
            log.info(f"  Progress: {i+1}/{len(tickers)}")

        try:
            t = yf.Ticker(ticker)
            # Get earnings dates (past + future)
            ed = t.earnings_dates
            if ed is not None and len(ed) > 0:
                df = pd.DataFrame({
                    'ticker': ticker,
                    'earnings_date': ed.index.tz_localize(None) if ed.index.tz else ed.index,
                })
                all_earnings.append(df)
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)

        # Rate limit
        if (i + 1) % 5 == 0:
            time.sleep(0.5)

    if not all_earnings:
        log.error("No earnings dates downloaded!")
        return pd.DataFrame(columns=['ticker', 'earnings_date'])

    earnings = pd.concat(all_earnings, ignore_index=True)
    earnings['earnings_date'] = pd.to_datetime(earnings['earnings_date'])
    earnings = earnings.sort_values(['ticker', 'earnings_date']).reset_index(drop=True)

    # Save cache
    earnings.to_parquet(cache_file, index=False)
    log.info(f"Downloaded earnings dates: {len(earnings)} dates for {earnings['ticker'].nunique()} tickers")
    log.info(f"Failed/no-data tickers: {len(failed)}: {failed[:20]}")

    return earnings


def build_earnings_lookup(earnings_df):
    """Build efficient lookup: for each ticker, sorted array of earnings dates."""
    lookup = {}
    for ticker, group in earnings_df.groupby('ticker'):
        dates = np.sort(group['earnings_date'].values)
        lookup[ticker] = dates
    return lookup


def has_earnings_in_window(ticker, trade_date, dte_days, earnings_lookup, buffer_days=2):
    """
    Check if ticker has earnings within [trade_date - buffer, trade_date + dte + buffer].
    buffer_days accounts for earnings being announced slightly before/after expected.
    """
    if ticker not in earnings_lookup:
        return False  # No data = assume safe

    dates = earnings_lookup[ticker]
    window_start = np.datetime64(trade_date) - np.timedelta64(buffer_days, 'D')
    window_end = np.datetime64(trade_date) + np.timedelta64(dte_days + buffer_days, 'D')

    # Binary search for efficiency
    idx_start = np.searchsorted(dates, window_start, side='left')
    idx_end = np.searchsorted(dates, window_end, side='right')

    return idx_end > idx_start  # True if any earnings date falls in window


def run_backtest_with_filter(prices_df, spy_regime, sector_map, earnings_lookup,
                             use_earnings_filter=True,
                             starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
                             put_delta=0.30, dte_target=14, profit_take=0.65,
                             bear_mode="liq_csp_only", max_assignments_5d=3,
                             max_share_positions=5, loss_cut_pct=-0.15,
                             min_price=10.0, max_price=500.0, buffer_days=2):
    """
    Run wheel portfolio backtest with optional earnings avoidance filter.
    Adapted from wheel_universe_v3_expand.run_portfolio_v3.
    """
    from collections import deque
    from dataclasses import dataclass

    MARGIN_REQ_PCT = 0.20
    COST_PER_CONTRACT = 0.65
    SLIPPAGE_FRAC = 0.025
    SLIPPAGE_MIN = 0.03
    VIX_MAX = 35.0
    RISK_FREE = 0.04
    CALL_DELTA = 0.30

    def _Phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(0, (K - S) if kind == "put" else (S - K))
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if kind == "put":
            return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
        else:
            return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

    def find_strike(S, sigma, T, target_delta, r=RISK_FREE, kind="put"):
        """Find strike for target put delta using BS."""
        if T <= 0 or sigma <= 0:
            return S
        lo, hi = S * 0.5, S * 1.5
        for _ in range(50):
            K = (lo + hi) / 2
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            if kind == "put":
                delta_abs = abs(_Phi(d1) - 1)
            else:
                delta_abs = _Phi(d1)
            if delta_abs < target_delta:
                if kind == "put":
                    hi = K
                else:
                    lo = K
            else:
                if kind == "put":
                    lo = K
                else:
                    hi = K
        return round((lo + hi) / 2, 2)

    @dataclass
    class CSP:
        ticker: str
        open_date: pd.Timestamp
        strike: float
        premium: float
        dte: int
        expiry: pd.Timestamp
        margin: float

    @dataclass
    class SharePos:
        ticker: str
        assign_date: pd.Timestamp
        shares: int
        cost_basis: float
        cc_strike: float = 0
        cc_premium: float = 0
        cc_expiry: pd.Timestamp = None

    # Prepare data
    tickers = [t for t in prices_df['ticker'].unique()
               if t != 'SPY' and t not in ('VIX', '^VIX')]

    # Filter by price range
    last_prices = prices_df.groupby('ticker')['close'].last()
    tickers = [t for t in tickers if t in last_prices.index and min_price <= last_prices[t] <= max_price]

    dates = sorted(prices_df['date'].unique())
    spy_bear = dict(zip(spy_regime['date'], spy_regime['bear']))

    # Price lookup
    price_map = {}
    sigma_map = {}
    for _, row in prices_df.iterrows():
        key = (row['ticker'], row['date'])
        price_map[key] = row['close']
        if 'sigma' in row.index and not pd.isna(row.get('sigma', np.nan)):
            sigma_map[key] = row['sigma']

    cash = float(starting_cash)
    csps = []
    shares = []
    assignment_log = deque(maxlen=100)
    daily_equity = []
    trades = []
    earnings_skips = 0
    total_considered = 0

    for di, date in enumerate(dates):
        is_bear = spy_bear.get(date, 0)
        vix_val = 20.0

        # Mark to market
        equity = cash

        # Check CSP expiries
        expired_csps = [c for c in csps if date >= c.expiry]
        active_csps = [c for c in csps if date < c.expiry]

        for c in expired_csps:
            px = price_map.get((c.ticker, date))
            if px is None:
                cash += c.margin
                trades.append({'ticker': c.ticker, 'type': 'csp_expire', 'date': date, 'pnl': c.premium * 100})
                continue

            if px < c.strike:
                # ITM at expiry — assignment or force-close
                recent_assigns = sum(1 for ad in assignment_log if (date - ad).days <= 5)
                if recent_assigns >= max_assignments_5d or len(shares) >= max_share_positions:
                    intrinsic = (c.strike - px) * 100
                    cash += c.margin  # Get margin back
                    cash -= intrinsic  # Pay to close ITM put
                    cash -= COST_PER_CONTRACT
                    pnl = c.premium * 100 - intrinsic - COST_PER_CONTRACT
                    trades.append({'ticker': c.ticker, 'type': 'csp_force_close', 'date': date, 'pnl': pnl})
                else:
                    assignment_log.append(date)
                    cost_basis = c.strike - c.premium
                    cash += c.margin  # Get margin back
                    cash -= c.strike * 100  # Buy shares at strike (premium already received)
                    shares.append(SharePos(
                        ticker=c.ticker, assign_date=date,
                        shares=100, cost_basis=cost_basis
                    ))
                    trades.append({'ticker': c.ticker, 'type': 'assigned', 'date': date, 'pnl': 0})
            else:
                # Expired worthless — keep premium (already received at open)
                cash += c.margin
                trades.append({'ticker': c.ticker, 'type': 'csp_profit', 'date': date, 'pnl': c.premium * 100})

        csps = active_csps

        # Check profit-take on active CSPs
        new_csps = []
        for c in csps:
            px = price_map.get((c.ticker, date))
            sigma = sigma_map.get((c.ticker, date), 0.3)
            if px is not None:
                remaining = max((c.expiry - date).days / 365.0, 1/365)
                current_val = bs_price(px, c.strike, remaining, sigma)
                if current_val <= c.premium * (1 - profit_take):
                    buyback = current_val * 100 + COST_PER_CONTRACT
                    cash += c.margin
                    cash -= buyback
                    pnl = (c.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                    trades.append({'ticker': c.ticker, 'type': 'csp_profit_take', 'date': date, 'pnl': pnl})
                    continue
            new_csps.append(c)
        csps = new_csps

        # Bear mode: liquidate CSPs in bear
        if is_bear and bear_mode == "liq_csp_only":
            for c in csps:
                px = price_map.get((c.ticker, date))
                sigma = sigma_map.get((c.ticker, date), 0.3)
                if px is not None:
                    remaining = max((c.expiry - date).days / 365.0, 1/365)
                    current_val = bs_price(px, c.strike, remaining, sigma)
                    buyback = current_val * 100 + COST_PER_CONTRACT
                    cash += c.margin
                    cash -= buyback
                    pnl = (c.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                    trades.append({'ticker': c.ticker, 'type': 'bear_close', 'date': date, 'pnl': pnl})
                else:
                    cash += c.margin
            csps = []

        # Share positions: check loss-cut and CC management
        new_shares = []
        for s in shares:
            px = price_map.get((s.ticker, date))
            if px is None:
                new_shares.append(s)
                continue

            # Loss cut
            unrealized = (px - s.cost_basis) / s.cost_basis
            if unrealized < loss_cut_pct:
                pnl = (px - s.cost_basis) * s.shares
                cash += px * s.shares
                trades.append({'ticker': s.ticker, 'type': 'share_loss_cut', 'date': date, 'pnl': pnl})
                continue

            # CC expiry
            if s.cc_expiry and date >= s.cc_expiry:
                if px >= s.cc_strike and s.cc_strike > 0:
                    # Called away
                    pnl = (s.cc_strike - s.cost_basis) * s.shares + s.cc_premium * 100
                    cash += s.cc_strike * s.shares + s.cc_premium * 100
                    trades.append({'ticker': s.ticker, 'type': 'cc_called', 'date': date, 'pnl': pnl})
                    continue
                else:
                    s.cc_strike = 0
                    s.cc_premium = 0
                    s.cc_expiry = None

            # Sell CC if none active
            if s.cc_strike == 0:
                sigma = sigma_map.get((s.ticker, date), 0.3)
                T = dte_target / 365.0
                cc_strike = find_strike(px, sigma, T, CALL_DELTA, kind="call")
                cc_prem = bs_price(px, cc_strike, T, sigma, kind="call")
                slip = max(cc_prem * SLIPPAGE_FRAC, SLIPPAGE_MIN)
                cc_prem = max(cc_prem - slip, 0.01)
                s.cc_strike = cc_strike
                s.cc_premium = cc_prem
                s.cc_expiry = date + timedelta(days=dte_target)

            new_shares.append(s)
        shares = new_shares

        # Open new CSPs (only in bull regime or if bear mode doesn't block)
        if not is_bear or bear_mode != "liq_csp_only":
            current_margin_used = sum(c.margin for c in csps)
            csp_tickers = set(c.ticker for c in csps)
            share_tickers = set(s.ticker for s in shares)
            total_equity_est = cash + sum(
                price_map.get((s.ticker, date), s.cost_basis) * s.shares for s in shares
            )

            # Shuffle tickers for diversification
            import random
            random.seed(int(date.timestamp()) if hasattr(date, 'timestamp') else di)
            candidates = [t for t in tickers if t not in csp_tickers and t not in share_tickers]
            random.shuffle(candidates)

            for ticker in candidates:
                if current_margin_used >= total_equity_est * margin_cap:
                    break

                px = price_map.get((ticker, date))
                sigma = sigma_map.get((ticker, date))
                if px is None or sigma is None or px < min_price or px > max_price:
                    continue

                total_considered += 1

                # EARNINGS FILTER
                if use_earnings_filter:
                    if has_earnings_in_window(ticker, date, dte_target, earnings_lookup, buffer_days):
                        earnings_skips += 1
                        continue

                margin_needed = px * 100 * MARGIN_REQ_PCT

                if current_margin_used + margin_needed > total_equity_est * margin_cap:
                    continue
                if cash < margin_needed:
                    continue

                T = dte_target / 365.0
                strike = find_strike(px, sigma, T, put_delta)
                premium = bs_price(px, strike, T, sigma, kind="put")
                slip = max(premium * SLIPPAGE_FRAC, SLIPPAGE_MIN)
                premium = max(premium - slip, 0.01)

                if premium < 0.10:
                    continue

                csps.append(CSP(
                    ticker=ticker, open_date=date, strike=strike,
                    premium=premium, dte=dte_target,
                    expiry=date + timedelta(days=dte_target),
                    margin=margin_needed
                ))
                current_margin_used += margin_needed
                cash -= margin_needed  # Reserve margin from cash
                cash += premium * 100  # Receive premium
                cash -= COST_PER_CONTRACT  # Commission

        # Daily equity
        share_val = sum(
            price_map.get((s.ticker, date), s.cost_basis) * s.shares for s in shares
        )
        csp_margin = sum(c.margin for c in csps)
        equity = cash + share_val + csp_margin
        daily_equity.append({'date': date, 'equity': equity, 'n_csps': len(csps),
                            'n_shares': len(shares), 'cash': cash})

    # Compute metrics
    eq_df = pd.DataFrame(daily_equity)
    eq_df['ret'] = eq_df['equity'].pct_change()
    eq_df = eq_df.dropna(subset=['ret'])

    total_ret = (eq_df['equity'].iloc[-1] / starting_cash - 1) * 100
    years = len(eq_df) / 252
    cagr = ((eq_df['equity'].iloc[-1] / starting_cash) ** (1/years) - 1) * 100 if years > 0 else 0

    mu = eq_df['ret'].mean() * 252
    std = eq_df['ret'].std() * np.sqrt(252)
    sharpe = mu / std if std > 0 else 0

    downside = eq_df['ret'][eq_df['ret'] < 0].std() * np.sqrt(252)
    sortino = mu / downside if downside > 0 else 0

    cummax = eq_df['equity'].cummax()
    dd = (eq_df['equity'] / cummax - 1)
    max_dd = dd.min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    daily_wr = (eq_df['ret'] > 0).mean()

    pos_days = eq_df['ret'][eq_df['ret'] > 0].sum()
    neg_days = abs(eq_df['ret'][eq_df['ret'] < 0].sum())
    pf = pos_days / neg_days if neg_days > 0 else 999

    # Regime analysis
    eq_df['bear'] = eq_df['date'].map(spy_bear).fillna(0).astype(int)
    bull_ret = eq_df[eq_df['bear'] == 0]['ret']
    bear_ret = eq_df[eq_df['bear'] == 1]['ret']

    bull_sharpe = (bull_ret.mean() * 252) / (bull_ret.std() * np.sqrt(252)) if len(bull_ret) > 20 else 0
    bear_sharpe = (bear_ret.mean() * 252) / (bear_ret.std() * np.sqrt(252)) if len(bear_ret) > 20 else 0

    if max(abs(bull_sharpe), abs(bear_sharpe)) > 0:
        regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe))
    else:
        regime_gap = 0

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    result = {
        'metrics': {
            'total_return_pct': round(total_ret, 1),
            'cagr_pct': round(cagr, 1),
            'sharpe': round(sharpe, 2),
            'sortino': round(sortino, 2),
            'max_dd_pct': round(max_dd, 1),
            'calmar': round(calmar, 2),
            'daily_wr': round(daily_wr, 3),
            'profit_factor': round(pf, 2),
            'final_equity': round(eq_df['equity'].iloc[-1], 2),
            'starting_equity': starting_cash,
            'years': round(years, 2),
            'n_days': len(eq_df),
            'n_trades': len(trades),
        },
        'regime': {
            'bull_sharpe': round(bull_sharpe, 3),
            'bear_sharpe': round(bear_sharpe, 3),
            'regime_gap': round(regime_gap, 3),
        },
        'filter_stats': {
            'earnings_skips': earnings_skips,
            'total_considered': total_considered,
            'skip_rate': round(earnings_skips / max(total_considered, 1) * 100, 1),
        }
    }

    return result, eq_df, trades_df


def main():
    from wheel_universe_v3_expand import load_all_data

    log.info("=" * 70)
    log.info("WHEEL EARNINGS AVOIDANCE FILTER — A/B TEST")
    log.info("=" * 70)

    # Load price data
    log.info("Loading price data...")
    prices, spy_regime, sector_map = load_all_data(start_date="2019-01-01")
    tickers = [t for t in prices['ticker'].unique() if t not in ('SPY', 'VIX', '^VIX')]
    log.info(f"Universe: {len(tickers)} tickers")

    # Download/load earnings dates
    log.info("Loading earnings dates...")
    earnings_df = download_earnings_dates(tickers)
    earnings_lookup = build_earnings_lookup(earnings_df)
    log.info(f"Earnings data: {len(earnings_lookup)} tickers with dates")

    # Stats on earnings coverage
    tickers_with_earnings = set(earnings_lookup.keys()) & set(tickers)
    log.info(f"Coverage: {len(tickers_with_earnings)}/{len(tickers)} tickers have earnings dates")

    all_results = {}

    # ============ TEST 1: BASELINE (no filter) ============
    log.info("\n" + "=" * 50)
    log.info("TEST 1: BASELINE (no earnings filter)")
    log.info("=" * 50)

    t0 = time.time()
    result_base, eq_base, trades_base = run_backtest_with_filter(
        prices, spy_regime, sector_map, earnings_lookup,
        use_earnings_filter=False,
        put_delta=0.30, dte_target=14, profit_take=0.65,
        margin_cap=0.40, per_name_pct=0.03,
    )
    result_base['elapsed_s'] = round(time.time() - t0, 1)
    all_results['baseline_no_filter'] = result_base

    m = result_base['metrics']
    log.info(f"  CAGR: {m['cagr_pct']}% | Sharpe: {m['sharpe']} | MaxDD: {m['max_dd_pct']}% | Calmar: {m['calmar']}")
    log.info(f"  WR: {m['daily_wr']} | PF: {m['profit_factor']} | Trades: {m['n_trades']}")

    # ============ TEST 2: EARNINGS FILTER (2-day buffer) ============
    log.info("\n" + "=" * 50)
    log.info("TEST 2: EARNINGS FILTER (2-day buffer)")
    log.info("=" * 50)

    t0 = time.time()
    result_filt2, eq_filt2, trades_filt2 = run_backtest_with_filter(
        prices, spy_regime, sector_map, earnings_lookup,
        use_earnings_filter=True, buffer_days=2,
        put_delta=0.30, dte_target=14, profit_take=0.65,
        margin_cap=0.40, per_name_pct=0.03,
    )
    result_filt2['elapsed_s'] = round(time.time() - t0, 1)
    all_results['earnings_filter_2d_buffer'] = result_filt2

    m = result_filt2['metrics']
    f = result_filt2['filter_stats']
    log.info(f"  CAGR: {m['cagr_pct']}% | Sharpe: {m['sharpe']} | MaxDD: {m['max_dd_pct']}% | Calmar: {m['calmar']}")
    log.info(f"  WR: {m['daily_wr']} | PF: {m['profit_factor']} | Trades: {m['n_trades']}")
    log.info(f"  Earnings skips: {f['earnings_skips']}/{f['total_considered']} ({f['skip_rate']}%)")

    # ============ TEST 3: EARNINGS FILTER (5-day buffer) ============
    log.info("\n" + "=" * 50)
    log.info("TEST 3: EARNINGS FILTER (5-day buffer — conservative)")
    log.info("=" * 50)

    t0 = time.time()
    result_filt5, eq_filt5, trades_filt5 = run_backtest_with_filter(
        prices, spy_regime, sector_map, earnings_lookup,
        use_earnings_filter=True, buffer_days=5,
        put_delta=0.30, dte_target=14, profit_take=0.65,
        margin_cap=0.40, per_name_pct=0.03,
    )
    result_filt5['elapsed_s'] = round(time.time() - t0, 1)
    all_results['earnings_filter_5d_buffer'] = result_filt5

    m = result_filt5['metrics']
    f = result_filt5['filter_stats']
    log.info(f"  CAGR: {m['cagr_pct']}% | Sharpe: {m['sharpe']} | MaxDD: {m['max_dd_pct']}% | Calmar: {m['calmar']}")
    log.info(f"  WR: {m['daily_wr']} | PF: {m['profit_factor']} | Trades: {m['n_trades']}")
    log.info(f"  Earnings skips: {f['earnings_skips']}/{f['total_considered']} ({f['skip_rate']}%)")

    # ============ TEST 4: EARNINGS FILTER + 35-DELTA (best from IV sweep) ============
    log.info("\n" + "=" * 50)
    log.info("TEST 4: EARNINGS FILTER (2d buffer) + 35-DELTA")
    log.info("=" * 50)

    t0 = time.time()
    result_d35, eq_d35, trades_d35 = run_backtest_with_filter(
        prices, spy_regime, sector_map, earnings_lookup,
        use_earnings_filter=True, buffer_days=2,
        put_delta=0.35, dte_target=14, profit_take=0.65,
        margin_cap=0.40, per_name_pct=0.03,
    )
    result_d35['elapsed_s'] = round(time.time() - t0, 1)
    all_results['earnings_filter_2d_delta35'] = result_d35

    m = result_d35['metrics']
    f = result_d35['filter_stats']
    log.info(f"  CAGR: {m['cagr_pct']}% | Sharpe: {m['sharpe']} | MaxDD: {m['max_dd_pct']}% | Calmar: {m['calmar']}")
    log.info(f"  WR: {m['daily_wr']} | PF: {m['profit_factor']} | Trades: {m['n_trades']}")
    log.info(f"  Earnings skips: {f['earnings_skips']}/{f['total_considered']} ({f['skip_rate']}%)")

    # ============ SUMMARY ============
    log.info("\n" + "=" * 70)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 70)
    log.info(f"{'Config':<35} {'CAGR':>6} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} {'Skip%':>6}")
    log.info("-" * 70)
    for name, res in all_results.items():
        m = res['metrics']
        skip = res['filter_stats']['skip_rate']
        log.info(f"{name:<35} {m['cagr_pct']:>5.1f}% {m['sharpe']:>7.2f} {m['max_dd_pct']:>6.1f}% {m['calmar']:>7.2f} {skip:>5.1f}%")

    # Save results
    out_file = OUT_DIR / "earnings_filter_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_file}")

    # Save equity curves for comparison
    eq_base['config'] = 'baseline'
    eq_filt2['config'] = 'filter_2d'
    eq_filt5['config'] = 'filter_5d'
    eq_d35['config'] = 'filter_2d_d35'

    eq_all = pd.concat([eq_base, eq_filt2, eq_filt5, eq_d35], ignore_index=True)
    eq_all.to_parquet(OUT_DIR / "equity_curves.parquet", index=False)

    return all_results


if __name__ == "__main__":
    main()
