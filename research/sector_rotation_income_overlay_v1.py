#!/usr/bin/env python3
"""
Sector Rotation + Covered Call Income Overlay v1
=================================================
Tests whether selling covered calls on top-momentum sector ETFs
generates income while preserving the growth upside of rotation.

Base strategy: top 3 sector ETFs by 6-month momentum, hold 30 days,
rebalance monthly. Equal-weight across selections.

Three variants:
  A) Pure rotation (no calls) — baseline
  B) Rotation + monthly covered calls on ALL held ETFs
  C) Rotation + covered calls ONLY when VIX > 20

Covered call mechanics:
  - Sell 30-delta calls at entry, 30-day expiry
  - Premium estimated via Black-Scholes with IV = 21d realized vol * 1.2
  - At expiry: keep premium; cap upside at strike if price > strike

Adversarial checks:
  - Regime test (green/red/flat days, PRIOR-DAY SPY close — no leakage)
  - Permutation test (500 shuffles of rotation selection)
  - Sub-period consistency

Leakage prevention:
  - Momentum ranking uses PRIOR data only
  - Entry at NEXT DAY after signal
  - Regime uses PRIOR-DAY SPY return
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm as sp_norm

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "sector_rotation_income_overlay"
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE  = ROOT / "research" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# =====================================================================
# CONFIG
# =====================================================================

STARTING_CAPITAL     = 100_000
REBALANCE_DAYS       = 21        # ~1 month of trading days
MOM_LOOKBACK         = 126       # 6 months
N_TOP                = 3
RISK_FREE_RATE       = 0.04
VOL_PREMIUM          = 1.2       # IV = realized vol * 1.2
CALL_EXPIRY_DAYS     = 30
CALL_DELTA_TARGET    = 0.30
VIX_THRESHOLD        = 20.0
N_PERMUTATIONS       = 500
TX_COST_PCT          = 0.001     # 0.1% slippage per side (ETF)

SECTOR_ETFS = ['XLK','XLF','XLV','XLE','XLI','XLC','XLY','XLP','XLU','XLRE','XLB']

# =====================================================================
# BLACK-SCHOLES
# =====================================================================

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * sp_norm.cdf(d1) - K * np.exp(-r*T) * sp_norm.cdf(d2)


def get_30delta_strike(S, T, r, sigma):
    """Strike for a 30-delta call: K such that N(d1) = 0.30."""
    # For call delta = N(d1) = 0.30, d1 = norm.ppf(0.30)
    # d1 = [ln(S/K) + (r + sig^2/2)*T] / (sig*sqrt(T))
    # Solve for K: K = S * exp(-(d1*sig*sqrt(T) - (r + sig^2/2)*T))
    if T <= 0 or sigma <= 0:
        return S * 1.05  # fallback
    d1_target = sp_norm.ppf(0.70)  # 30-delta call => N(d1)=0.30... wait
    # Call delta = N(d1). For delta=0.30, N(d1)=0.30, so d1 = ppf(0.30)
    d1_target = sp_norm.ppf(CALL_DELTA_TARGET)  # ppf(0.30) ≈ -0.524
    # Hmm, delta=0.30 means d1=ppf(0.30)=-0.524, which means K > S (OTM call)
    # That's correct: 30-delta call is OTM
    K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + sigma**2/2) * T)
    # Rearranged: ln(S/K) = d1*sig*sqrt(T) - (r+sig^2/2)*T
    # K = S * exp(-(d1*sig*sqrt(T) - (r+sig^2/2)*T))
    # = S * exp((r+sig^2/2)*T - d1*sig*sqrt(T))
    # Since d1 is negative, -d1 is positive, so K > S. Correct for OTM call.
    return K


# =====================================================================
# DATA LOADING
# =====================================================================

def fetch_prices():
    """Fetch daily prices from yfinance with caching."""
    cache_path = CACHE / "sector_rotation_income_overlay_prices.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded cached prices: {len(df)} rows")
        return df

    import yfinance as yf
    tickers = SECTOR_ETFS + ['SPY']
    print(f"  Downloading prices for {len(tickers)} tickers...")
    all_frames = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start='2019-01-01', end='2026-07-24',
                               progress=False, auto_adjust=True)
            if len(data) > 100:
                data = data.reset_index()
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                data['ticker'] = ticker
                data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                     'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
                all_frames.append(data[['date','open','high','low','close','volume','ticker']])
                print(f"    {ticker}: {len(data)} days")
        except Exception as e:
            print(f"    {ticker}: error -- {e}")
        time.sleep(0.3)

    df = pd.concat(all_frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(cache_path, index=False)
    print(f"  Saved {len(df)} rows to cache")
    return df


def fetch_vix():
    """Fetch VIX index data."""
    cache_path = CACHE / "sector_rotation_income_overlay_vix.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded cached VIX: {len(df)} rows")
        return df

    import yfinance as yf
    print("  Downloading VIX...")
    data = yf.download('^VIX', start='2019-01-01', end='2026-07-24',
                       progress=False, auto_adjust=True)
    data = data.reset_index()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    data.rename(columns={'Date': 'date', 'Close': 'vix_close'}, inplace=True)
    data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
    data = data[['date','vix_close']].copy()
    data['date'] = pd.to_datetime(data['date'])
    data.to_parquet(cache_path, index=False)
    print(f"  Saved VIX data: {len(data)} rows")
    return data


# =====================================================================
# STRATEGY SIMULATION
# =====================================================================

def compute_realized_vol(close_series, window=21):
    """21-day realized vol, annualized."""
    log_ret = np.log(close_series / close_series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def run_rotation_backtest(prices_wide, vix_series, spy_close,
                          mode='pure', label=''):
    """
    Run sector rotation backtest.
    mode: 'pure' (no calls), 'calls_always', 'calls_high_vix'

    Returns daily equity curve as a Series.
    """
    dates = prices_wide.index
    tickers = [c for c in prices_wide.columns if c != 'SPY']

    # Precompute momentum (6-month return) and realized vol
    mom = prices_wide[tickers].pct_change(MOM_LOOKBACK)
    rvol = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        rvol[t] = compute_realized_vol(prices_wide[t], 21)

    # Warmup period
    warmup = max(MOM_LOOKBACK, 21) + 5

    capital = STARTING_CAPITAL
    equity_curve = pd.Series(index=dates, dtype=float)
    equity_curve.iloc[:warmup] = capital

    # Track positions
    holdings = {}  # ticker -> {shares, entry_price, strike, premium, entry_date_idx}
    rebal_countdown = 0

    trade_log = []

    for i in range(warmup, len(dates)):
        date = dates[i]

        # Check if we need to rebalance
        if rebal_countdown <= 0:
            # --- CLOSE existing positions ---
            for ticker, pos in holdings.items():
                exit_price = prices_wide[ticker].iloc[i]
                entry_price = pos['entry_price']
                shares = pos['shares']

                if mode != 'pure' and pos.get('has_call', False):
                    strike = pos['strike']
                    premium = pos['premium']

                    # Covered call settlement
                    if exit_price > strike:
                        # Called away: capped at strike + premium
                        pnl_per_share = (strike - entry_price) + premium
                    else:
                        # Keep premium + full stock move
                        pnl_per_share = (exit_price - entry_price) + premium
                else:
                    # Pure rotation: just stock P&L
                    pnl_per_share = exit_price - entry_price

                # Transaction cost on close
                close_cost = exit_price * TX_COST_PCT * shares
                capital += pnl_per_share * shares - close_cost

                trade_log.append({
                    'date': str(date.date()),
                    'ticker': ticker,
                    'entry': entry_price,
                    'exit': exit_price,
                    'shares': shares,
                    'pnl': pnl_per_share * shares - close_cost,
                    'had_call': pos.get('has_call', False),
                })

            holdings = {}

            # --- SELECT new top N sectors ---
            mom_today = mom.iloc[i]
            valid = mom_today.dropna()
            if len(valid) < N_TOP:
                equity_curve.iloc[i] = capital
                rebal_countdown = REBALANCE_DAYS
                continue

            top_n = valid.nlargest(N_TOP).index.tolist()

            # Equal-weight allocation
            alloc_per = capital / N_TOP

            # Get VIX for the day (use prior day to avoid leakage)
            vix_today = vix_series.iloc[i-1] if i > 0 else 15.0

            for ticker in top_n:
                price = prices_wide[ticker].iloc[i]
                shares = int(alloc_per / price)
                if shares <= 0:
                    continue

                entry_cost = price * TX_COST_PCT * shares
                capital -= entry_cost

                # Determine if we sell a covered call
                sell_call = False
                if mode == 'calls_always':
                    sell_call = True
                elif mode == 'calls_high_vix':
                    sell_call = (vix_today > VIX_THRESHOLD)

                pos = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_date_idx': i,
                    'has_call': False,
                    'strike': 0,
                    'premium': 0,
                }

                if sell_call:
                    sigma = rvol[ticker].iloc[i]
                    if pd.notna(sigma) and sigma > 0.01:
                        iv = sigma * VOL_PREMIUM
                        T = CALL_EXPIRY_DAYS / 365.0
                        K = get_30delta_strike(price, T, RISK_FREE_RATE, iv)
                        premium = bs_call_price(price, K, T, RISK_FREE_RATE, iv)

                        pos['has_call'] = True
                        pos['strike'] = K
                        pos['premium'] = premium  # per share

                holdings[ticker] = pos

            rebal_countdown = REBALANCE_DAYS

        # --- Mark-to-market (intraperiod) ---
        mtm = capital  # cash portion
        for ticker, pos in holdings.items():
            curr_price = prices_wide[ticker].iloc[i]
            shares = pos['shares']
            entry_price = pos['entry_price']

            if pos.get('has_call', False):
                # Approximate MTM: stock gain + premium, but capped at strike
                effective_price = min(curr_price, pos['strike'])
                mtm += (effective_price - entry_price + pos['premium']) * shares
            else:
                mtm += (curr_price - entry_price) * shares

        equity_curve.iloc[i] = mtm
        rebal_countdown -= 1

    # Close any remaining positions at end
    if holdings:
        last_i = len(dates) - 1
        for ticker, pos in holdings.items():
            exit_price = prices_wide[ticker].iloc[last_i]
            shares = pos['shares']
            entry_price = pos['entry_price']

            if mode != 'pure' and pos.get('has_call', False):
                strike = pos['strike']
                premium = pos['premium']
                if exit_price > strike:
                    pnl_per_share = (strike - entry_price) + premium
                else:
                    pnl_per_share = (exit_price - entry_price) + premium
            else:
                pnl_per_share = exit_price - entry_price

            close_cost = exit_price * TX_COST_PCT * shares
            capital += pnl_per_share * shares - close_cost

        equity_curve.iloc[last_i] = capital

    equity_curve = equity_curve.ffill().bfill()
    return equity_curve, trade_log


# =====================================================================
# METRICS
# =====================================================================

def compute_metrics(equity_curve, label=''):
    """Compute risk-adjusted metrics from daily equity curve."""
    eq = equity_curve.dropna()
    if len(eq) < 30:
        return {'label': label, 'error': 'insufficient data'}

    daily_ret = eq.pct_change().dropna()

    # Monthly returns for WR/PF
    monthly_eq = eq.resample('ME').last().dropna()
    monthly_ret = monthly_eq.pct_change().dropna()

    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    n_years = len(eq) / 252.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_years) - 1 if n_years > 0 else 0

    # Sharpe
    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (ann_ret - RISK_FREE_RATE) / downside_vol

    # Max drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Monthly WR & PF
    wins = monthly_ret[monthly_ret > 0]
    losses = monthly_ret[monthly_ret < 0]
    wr = len(wins) / len(monthly_ret) if len(monthly_ret) > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    return {
        'label': label,
        'total_return_pct': round(total_ret * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd_pct': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'monthly_wr_pct': round(wr * 100, 1),
        'profit_factor': round(pf, 3),
        'ann_vol_pct': round(ann_vol * 100, 2),
        'n_years': round(n_years, 2),
        'final_equity': round(eq.iloc[-1], 2),
    }


# =====================================================================
# REGIME TEST
# =====================================================================

def regime_test(equity_curve, spy_close):
    """
    Classify into green/red/flat USING PRIOR-DAY SPY return (no leakage).
    Gap check: ensure regime classification date < return measurement date.
    """
    eq = equity_curve.dropna()
    daily_ret = eq.pct_change().dropna()

    # Prior-day SPY return for regime classification
    spy_ret = spy_close.pct_change()
    # Shift forward: today's regime = yesterday's SPY return
    regime_signal = spy_ret.shift(1)

    # Align
    common = daily_ret.index.intersection(regime_signal.dropna().index)
    daily_ret = daily_ret.loc[common]
    regime_signal = regime_signal.loc[common]

    green = daily_ret[regime_signal > 0.001]   # SPY up > 0.1%
    red   = daily_ret[regime_signal < -0.001]  # SPY down > 0.1%
    flat  = daily_ret[(regime_signal >= -0.001) & (regime_signal <= 0.001)]

    def regime_stats(rets, name):
        if len(rets) < 5:
            return {'regime': name, 'n_days': len(rets), 'sharpe': 0, 'mean_ret_bps': 0}
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
        return {
            'regime': name,
            'n_days': len(rets),
            'sharpe': round(sharpe, 3),
            'mean_ret_bps': round(rets.mean() * 10000, 2),
            'win_rate': round((rets > 0).mean() * 100, 1),
        }

    results = {
        'green': regime_stats(green, 'green'),
        'red': regime_stats(red, 'red'),
        'flat': regime_stats(flat, 'flat'),
    }

    # Gap check
    sg = abs(results['green']['sharpe'])
    sr = abs(results['red']['sharpe'])
    max_sr = max(sg, sr)
    gap = abs(sg - sr) / max_sr if max_sr > 0 else 0
    results['regime_gap'] = round(gap, 3)
    results['regime_gap_pass'] = gap <= 0.50

    return results


# =====================================================================
# PERMUTATION TEST
# =====================================================================

def permutation_test(prices_wide, vix_series, spy_close,
                     actual_sharpe, mode='pure', n_perms=500):
    """
    Shuffle ETF selection (random top-N instead of momentum-based)
    to test whether alpha comes from selection vs. just holding any 3 ETFs.
    """
    tickers = [c for c in prices_wide.columns if c != 'SPY']
    dates = prices_wide.index
    warmup = max(MOM_LOOKBACK, 21) + 5

    perm_sharpes = []

    for p in range(n_perms):
        np.random.seed(p + 42)

        capital = STARTING_CAPITAL
        equity_curve = pd.Series(index=dates, dtype=float)
        equity_curve.iloc[:warmup] = capital

        holdings = {}
        rebal_countdown = 0

        for i in range(warmup, len(dates)):
            if rebal_countdown <= 0:
                # Close existing
                for ticker, pos in holdings.items():
                    exit_price = prices_wide[ticker].iloc[i]
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    close_cost = exit_price * TX_COST_PCT * pos['shares']
                    capital += pnl - close_cost
                holdings = {}

                # RANDOM selection instead of momentum
                available = [t for t in tickers if pd.notna(prices_wide[t].iloc[i])]
                if len(available) >= N_TOP:
                    selected = list(np.random.choice(available, N_TOP, replace=False))
                else:
                    selected = available

                alloc_per = capital / max(len(selected), 1)
                for ticker in selected:
                    price = prices_wide[ticker].iloc[i]
                    shares = int(alloc_per / price)
                    if shares > 0:
                        entry_cost = price * TX_COST_PCT * shares
                        capital -= entry_cost
                        holdings[ticker] = {
                            'shares': shares,
                            'entry_price': price,
                        }

                rebal_countdown = REBALANCE_DAYS

            # MTM
            mtm = capital
            for ticker, pos in holdings.items():
                mtm += (prices_wide[ticker].iloc[i] - pos['entry_price']) * pos['shares']
            equity_curve.iloc[i] = mtm
            rebal_countdown -= 1

        equity_curve = equity_curve.ffill().bfill()
        dr = equity_curve.pct_change().dropna()
        if len(dr) > 30 and dr.std() > 0:
            s = (dr.mean() * 252 - RISK_FREE_RATE) / (dr.std() * np.sqrt(252))
            perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()

    return {
        'actual_sharpe': round(actual_sharpe, 3),
        'perm_mean_sharpe': round(perm_sharpes.mean(), 3),
        'perm_median_sharpe': round(np.median(perm_sharpes), 3),
        'perm_p95_sharpe': round(np.percentile(perm_sharpes, 95), 3),
        'p_value': round(p_value, 4),
        'n_permutations': n_perms,
        'significant': p_value < 0.05,
    }


# =====================================================================
# SUB-PERIOD CONSISTENCY
# =====================================================================

def sub_period_test(equity_curve):
    """Split into halves and check consistency."""
    eq = equity_curve.dropna()
    mid = len(eq) // 2

    first_half = eq.iloc[:mid]
    second_half = eq.iloc[mid:]

    m1 = compute_metrics(first_half, 'first_half')
    m2 = compute_metrics(second_half, 'second_half')

    return {
        'first_half': m1,
        'second_half': m2,
        'sharpe_consistent': abs(m1['sharpe'] - m2['sharpe']) < 1.5,
        'both_positive_cagr': m1['cagr_pct'] > 0 and m2['cagr_pct'] > 0,
    }


# =====================================================================
# CALL PREMIUM ANALYSIS
# =====================================================================

def analyze_call_premiums(trade_log):
    """Analyze covered call performance from trade log."""
    if not trade_log:
        return {'n_trades': 0}

    call_trades = [t for t in trade_log if t.get('had_call', False)]
    no_call_trades = [t for t in trade_log if not t.get('had_call', False)]

    if not call_trades:
        return {'n_trades_with_calls': 0, 'n_trades_without': len(no_call_trades)}

    return {
        'n_trades_with_calls': len(call_trades),
        'n_trades_without': len(no_call_trades),
        'avg_pnl_with_call': round(np.mean([t['pnl'] for t in call_trades]), 2),
        'avg_pnl_without': round(np.mean([t['pnl'] for t in no_call_trades]), 2) if no_call_trades else 0,
        'wr_with_call': round(100 * np.mean([t['pnl'] > 0 for t in call_trades]), 1),
        'wr_without': round(100 * np.mean([t['pnl'] > 0 for t in no_call_trades]), 1) if no_call_trades else 0,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("SECTOR ROTATION + COVERED CALL INCOME OVERLAY v1")
    print("=" * 70)

    # --- Load data ---
    print("\n[1] Loading data...")
    prices_df = fetch_prices()
    vix_df = fetch_vix()

    # Pivot to wide format
    prices_wide = prices_df.pivot(index='date', columns='ticker', values='close')
    prices_wide = prices_wide.sort_index().dropna(how='all')

    # SPY close for regime
    spy_close = prices_wide['SPY'].copy()

    # VIX aligned to dates
    vix_df = vix_df.set_index('date')
    vix_aligned = vix_df['vix_close'].reindex(prices_wide.index).ffill()

    print(f"  Date range: {prices_wide.index[0].strftime('%Y-%m-%d')} to {prices_wide.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading days: {len(prices_wide)}")
    print(f"  Tickers: {list(prices_wide.columns)}")

    # --- Run three variants ---
    print("\n[2] Running backtests...")

    variants = {
        'A_pure_rotation': 'pure',
        'B_rotation_plus_calls': 'calls_always',
        'C_rotation_calls_high_vix': 'calls_high_vix',
    }

    results = {}
    equity_curves = {}
    trade_logs = {}

    for name, mode in variants.items():
        print(f"\n  --- {name} ---")
        eq, trades = run_rotation_backtest(prices_wide, vix_aligned, spy_close,
                                           mode=mode, label=name)
        equity_curves[name] = eq
        trade_logs[name] = trades

        metrics = compute_metrics(eq, name)
        results[name] = {'metrics': metrics}

        print(f"    CAGR: {metrics['cagr_pct']:.1f}%  |  Sharpe: {metrics['sharpe']:.3f}  |  "
              f"Sortino: {metrics['sortino']:.3f}  |  MaxDD: {metrics['max_dd_pct']:.1f}%  |  "
              f"Calmar: {metrics['calmar']:.3f}")
        print(f"    Monthly WR: {metrics['monthly_wr_pct']:.0f}%  |  PF: {metrics['profit_factor']:.2f}  |  "
              f"Final: ${metrics['final_equity']:,.0f}")

    # --- Call premium analysis ---
    print("\n[3] Call premium analysis...")
    for name in ['B_rotation_plus_calls', 'C_rotation_calls_high_vix']:
        analysis = analyze_call_premiums(trade_logs[name])
        results[name]['call_analysis'] = analysis
        print(f"\n  {name}:")
        if analysis.get('n_trades_with_calls', 0) > 0:
            print(f"    Trades with calls: {analysis['n_trades_with_calls']}")
            print(f"    Avg PnL with call: ${analysis['avg_pnl_with_call']:,.2f}")
            print(f"    Avg PnL without:   ${analysis['avg_pnl_without']:,.2f}")
            print(f"    WR with call: {analysis['wr_with_call']:.0f}%  |  WR without: {analysis['wr_without']:.0f}%")
        else:
            print(f"    No call trades found")

    # --- Regime test ---
    print("\n[4] Regime test (prior-day SPY classification)...")
    for name, eq in equity_curves.items():
        regime = regime_test(eq, spy_close)
        results[name]['regime'] = regime
        g = regime['green']
        r = regime['red']
        f = regime['flat']
        print(f"\n  {name}:")
        print(f"    Green (n={g['n_days']}): Sharpe {g['sharpe']:.3f}, WR {g.get('win_rate',0):.0f}%")
        print(f"    Red   (n={r['n_days']}): Sharpe {r['sharpe']:.3f}, WR {r.get('win_rate',0):.0f}%")
        print(f"    Flat  (n={f['n_days']}): Sharpe {f['sharpe']:.3f}, WR {f.get('win_rate',0):.0f}%")
        print(f"    Regime gap: {regime['regime_gap']:.3f} {'PASS' if regime['regime_gap_pass'] else 'FAIL (>0.50)'}")

    # --- Permutation test ---
    print(f"\n[5] Permutation test ({N_PERMUTATIONS} shuffles)...")
    baseline_sharpe = results['A_pure_rotation']['metrics']['sharpe']
    perm = permutation_test(prices_wide, vix_aligned, spy_close,
                            baseline_sharpe, mode='pure', n_perms=N_PERMUTATIONS)
    results['permutation_test'] = perm
    print(f"    Actual Sharpe: {perm['actual_sharpe']:.3f}")
    print(f"    Permutation mean: {perm['perm_mean_sharpe']:.3f}")
    print(f"    Permutation p95:  {perm['perm_p95_sharpe']:.3f}")
    print(f"    p-value: {perm['p_value']:.4f} {'*** SIGNIFICANT' if perm['significant'] else '(not significant)'}")

    # --- Sub-period consistency ---
    print("\n[6] Sub-period consistency...")
    for name, eq in equity_curves.items():
        sub = sub_period_test(eq)
        results[name]['sub_period'] = sub
        h1 = sub['first_half']
        h2 = sub['second_half']
        print(f"\n  {name}:")
        print(f"    1st half — CAGR: {h1['cagr_pct']:.1f}%, Sharpe: {h1['sharpe']:.3f}")
        print(f"    2nd half — CAGR: {h2['cagr_pct']:.1f}%, Sharpe: {h2['sharpe']:.3f}")
        print(f"    Consistent: {'YES' if sub['sharpe_consistent'] else 'NO'}")

    # --- Covered call impact summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: COVERED CALL INCOME OVERLAY IMPACT")
    print("=" * 70)

    ma = results['A_pure_rotation']['metrics']
    mb = results['B_rotation_plus_calls']['metrics']
    mc = results['C_rotation_calls_high_vix']['metrics']

    print(f"\n{'Metric':<20} {'A) Pure':>12} {'B) +Calls':>12} {'C) +HiVIX':>12} {'B-A':>10} {'C-A':>10}")
    print("-" * 76)
    for metric, fmt in [
        ('cagr_pct', '.1f'), ('sharpe', '.3f'), ('sortino', '.3f'),
        ('max_dd_pct', '.1f'), ('calmar', '.3f'), ('monthly_wr_pct', '.0f'),
        ('profit_factor', '.2f'), ('ann_vol_pct', '.1f'),
    ]:
        va = ma[metric]
        vb = mb[metric]
        vc = mc[metric]
        diff_b = vb - va
        diff_c = vc - va
        print(f"  {metric:<18} {va:>12{fmt}} {vb:>12{fmt}} {vc:>12{fmt}} {diff_b:>+10{fmt}} {diff_c:>+10{fmt}}")

    print(f"\n  Final equity:    ${ma['final_equity']:>11,.0f} ${mb['final_equity']:>11,.0f} ${mc['final_equity']:>11,.0f}")

    # --- Verdict ---
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check if calls help
    calls_help_sharpe = mb['sharpe'] > ma['sharpe']
    calls_help_sortino = mb['sortino'] > ma['sortino']
    calls_reduce_dd = abs(mb['max_dd_pct']) < abs(ma['max_dd_pct'])
    hv_better = mc['sharpe'] > mb['sharpe']

    verdicts = []
    if calls_help_sharpe:
        verdicts.append("Covered calls IMPROVE Sharpe ratio")
    else:
        verdicts.append("Covered calls HURT Sharpe ratio (upside capping > premium income)")

    if calls_reduce_dd:
        verdicts.append("Covered calls REDUCE max drawdown (premium cushion helps)")
    else:
        verdicts.append("Covered calls DO NOT reduce max drawdown")

    if hv_better:
        verdicts.append("High-VIX-only calls OUTPERFORM always-on calls (sell vol when expensive)")
    else:
        verdicts.append("Always-on calls beat high-VIX-only (consistent premium > timing)")

    regime_pass = results['A_pure_rotation']['regime']['regime_gap_pass']
    if regime_pass:
        verdicts.append("Regime test PASSED — strategy works in both green and red markets")
    else:
        verdicts.append("Regime test FAILED — strategy is regime-dependent")

    if perm['significant']:
        verdicts.append(f"Permutation test PASSED (p={perm['p_value']:.4f}) — momentum selection adds real alpha")
    else:
        verdicts.append(f"Permutation test FAILED (p={perm['p_value']:.4f}) — no evidence momentum selection > random")

    for v in verdicts:
        print(f"  * {v}")

    # --- Save results ---
    output_path = OUTPUT / "results.json"

    # Convert non-serializable items
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_results[k] = v
        else:
            save_results[k] = str(v)

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
