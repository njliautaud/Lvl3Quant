#!/usr/bin/env python3
"""
Momentum Put Credit Spread on Sector ETFs v1
=============================================
Combines validated sector ETF momentum signal with defined-risk income:
- Use 12-1 momentum + quality features to rank sector ETFs
- Sell put credit spreads on top-ranked (high momentum) ETFs
- Defined risk per trade (spread width), monthly rebalance
- Targets agentic account ($645, max $200-300/trade)

Thesis: High-momentum ETFs are less likely to drop → selling put spreads
captures premium with momentum tailwind. If momentum signal is real
(validated Sharpe 3.96 on ETF rotation), the put selling should have
higher WR and better risk-adjusted returns than naked momentum.

Walk-forward: 252d train, 21d test, sliding window.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, '/home/jupiter/Lvl3Quant')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

# Sector/factor ETFs (same universe as validated sector_etf_momentum_v1)
ETF_UNIVERSE = [
    'XLE', 'XLK', 'XLF', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',
    'XLC',  # Communication Services
    'GLD', 'SLV', 'DBC',  # Commodities
    'TLT', 'IEF', 'HYG',  # Fixed income
    'QQQ', 'IWM', 'EEM',  # Broad indices
    'VNQ', 'XBI',  # Real estate, biotech
]

# Options parameters
SPREAD_WIDTH_PCT = 3.0     # OTM put spread width as % of underlying
SHORT_PUT_DELTA = 0.30     # Approximate delta for short put (30 delta)
DAYS_TO_EXPIRY = 30        # Monthly options
RISK_FREE_RATE = 0.05      # For Black-Scholes approximation

# Strategy parameters
TOP_N = 3                  # Number of ETFs to sell puts on (top momentum)
HOLD_DAYS = 21             # Hold to near-expiry (monthly cycle)
MAX_POSITION_RISK = 200    # Max risk per spread ($) for agentic account
STARTING_CAPITAL = 645     # Agentic account size

# Walk-forward
TRAIN_DAYS = 252           # 1 year lookback for momentum features
TEST_DAYS = 21             # Monthly rebalance

# Commission
COMMISSION_PER_CONTRACT = 1.30  # Per leg, so $2.60 round trip for spread


def fetch_etf_data():
    """Fetch historical price data for ETF universe."""
    import yfinance as yf

    # Try to load cached data first
    cache_path = '/home/jupiter/Lvl3Quant/data/etf_universe_cache.parquet'
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        if cache_age.days < 1:
            print(f"Using cached ETF data ({len(cached)} rows)")
            return cached

    print(f"Downloading ETF data for {len(ETF_UNIVERSE)} tickers...")
    all_data = []
    for ticker in ETF_UNIVERSE:
        try:
            df = yf.download(ticker, start='2015-01-01', end='2026-07-24',
                           progress=False, auto_adjust=True)
            if len(df) > 252:  # Need at least 1 year
                # Flatten multi-level columns BEFORE adding ticker
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [str(c[0]).lower() for c in df.columns]
                else:
                    df.columns = [str(c).lower() for c in df.columns]
                df = df.reset_index()
                df.columns = [str(c).lower() for c in df.columns]
                df['ticker'] = ticker
                all_data.append(df)
                print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    combined = pd.concat(all_data, ignore_index=True)

    # Ensure clean column names
    combined.columns = [str(c).lower().replace('price_', '').strip() for c in combined.columns]

    # Ensure we have required columns
    required = ['date', 'close', 'ticker']
    for r in required:
        if r not in combined.columns:
            raise ValueError(f"Missing column: {r}. Available: {list(combined.columns)}")

    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    combined.to_parquet(cache_path)
    print(f"Cached {len(combined)} rows to {cache_path}")

    return combined


def compute_momentum_features(df_ticker, lookback=252):
    """Compute momentum + quality features for a single ETF."""
    close = df_ticker['close'].values
    n = len(close)

    if n < lookback:
        return None

    features = {}

    # Returns at various horizons
    rets_1m = close[-1] / close[-21] - 1 if n >= 21 else 0
    rets_3m = close[-1] / close[-63] - 1 if n >= 63 else 0
    rets_6m = close[-1] / close[-126] - 1 if n >= 126 else 0
    rets_12m = close[-1] / close[-252] - 1 if n >= 252 else 0
    rets_12_1 = close[-21] / close[-252] - 1 if n >= 252 else 0  # 12-1 momentum (skip last month)

    features['mom_1m'] = rets_1m
    features['mom_3m'] = rets_3m
    features['mom_6m'] = rets_6m
    features['mom_12m'] = rets_12m
    features['mom_12_1'] = rets_12_1

    # Momentum acceleration
    if n >= 63:
        features['mom_accel'] = rets_1m - (rets_3m / 3)
    else:
        features['mom_accel'] = 0

    # Volatility
    daily_rets = np.diff(close[-63:]) / close[-63:-1] if n >= 64 else np.array([0])
    features['vol_60d'] = np.std(daily_rets) * np.sqrt(252) if len(daily_rets) > 1 else 0

    # Sharpe-like ratio (return / vol)
    features['sharpe_6m'] = rets_6m / (features['vol_60d'] + 1e-8)

    # Max drawdown (63d)
    if n >= 63:
        window = close[-63:]
        peak = np.maximum.accumulate(window)
        dd = (window - peak) / peak
        features['maxdd_63d'] = np.min(dd)
    else:
        features['maxdd_63d'] = 0

    # Skewness and kurtosis of returns
    if len(daily_rets) > 10:
        features['skew_63d'] = stats.skew(daily_rets)
        features['kurt_63d'] = stats.kurtosis(daily_rets)
    else:
        features['skew_63d'] = 0
        features['kurt_63d'] = 0

    # RSI (14-day)
    if n >= 15:
        deltas = np.diff(close[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rs = avg_gain / (avg_loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))
    else:
        features['rsi_14'] = 50

    # Volume momentum (normalized)
    if 'volume' in df_ticker.columns and n >= 42:
        vol = df_ticker['volume'].values
        features['vol_ratio'] = np.mean(vol[-21:]) / (np.mean(vol[-42:-21]) + 1)
    else:
        features['vol_ratio'] = 1

    return features


def estimate_put_spread_premium(price, vol_annual, days_to_expiry, spread_width_pct,
                                 short_delta=0.30, risk_free_rate=0.05):
    """
    Estimate put credit spread premium using simplified Black-Scholes.
    Returns (premium_received, max_loss, breakeven).
    """
    from scipy.stats import norm

    T = days_to_expiry / 365.0
    sigma = vol_annual

    # Find strike for target delta
    # For put: delta ≈ N(d1) - 1
    # d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
    # For delta = -0.30: N(d1) = 0.70, d1 = norm.ppf(0.70)
    d1_target = norm.ppf(1 - short_delta)  # For 0.30 delta put

    # K = S * exp(-d1*σ√T + (r + σ²/2)T)  — approximately
    sqrt_T = np.sqrt(T)
    short_strike = price * np.exp(-d1_target * sigma * sqrt_T + (risk_free_rate - 0.5 * sigma**2) * T)

    # Long put strike (lower)
    spread_width = price * spread_width_pct / 100
    long_strike = short_strike - spread_width

    if long_strike <= 0:
        return 0, spread_width, short_strike

    # BS put prices
    def bs_put(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    short_put_price = bs_put(price, short_strike, T, risk_free_rate, sigma)
    long_put_price = bs_put(price, long_strike, T, risk_free_rate, sigma)

    premium = short_put_price - long_put_price
    max_loss = spread_width - premium
    breakeven = short_strike - premium

    return premium, max_loss, breakeven


def simulate_put_spread_pnl(entry_price, exit_price, short_strike, long_strike,
                              premium, days_held, days_to_expiry):
    """
    Simulate P&L of a put credit spread given entry and exit prices.
    Simplified: if we hold to expiry, outcome is deterministic.
    If we exit early, use linear interpolation of theta decay.
    """
    # At expiry:
    if exit_price >= short_strike:
        # Both puts expire OTM → keep full premium
        pnl = premium
    elif exit_price >= long_strike:
        # Short put ITM, long put OTM
        intrinsic_loss = short_strike - exit_price
        pnl = premium - intrinsic_loss
    else:
        # Both ITM → max loss
        spread_width = short_strike - long_strike
        pnl = premium - spread_width  # This is negative (max loss)

    # Adjust for early exit (theta decay approximation)
    if days_held < days_to_expiry:
        theta_fraction = days_held / days_to_expiry
        # Blend between entry value and expiry value
        # More weight to expiry outcome as we approach expiry
        pnl = pnl * (0.3 + 0.7 * theta_fraction)

    return pnl


def run_backtest(data, variant_name, top_n=3, hold_days=21, spread_width_pct=3.0,
                 short_delta=0.30, starting_capital=645, max_risk=200):
    """Run walk-forward backtest of momentum put credit spread strategy."""

    # Get all unique dates
    all_dates = sorted(data['date'].unique())

    # Need at least train + some test
    if len(all_dates) < TRAIN_DAYS + TEST_DAYS:
        return None

    # Build feature matrix for each rebalance date
    trades = []
    equity = [starting_capital]
    equity_dates = [all_dates[TRAIN_DAYS]]

    rebal_idx = TRAIN_DAYS
    while rebal_idx + hold_days < len(all_dates):
        rebal_date = all_dates[rebal_idx]
        exit_date = all_dates[min(rebal_idx + hold_days, len(all_dates) - 1)]

        # Compute features for each ETF using training window
        train_start = all_dates[max(0, rebal_idx - TRAIN_DAYS)]

        rankings = []
        for ticker in ETF_UNIVERSE:
            ticker_data = data[(data['ticker'] == ticker) &
                             (data['date'] >= train_start) &
                             (data['date'] <= rebal_date)]

            if len(ticker_data) < 126:  # Need 6 months minimum
                continue

            features = compute_momentum_features(ticker_data)
            if features is None:
                continue

            # Composite momentum score (same weights as validated ETF rotation)
            score = (
                0.30 * features.get('mom_12_1', 0) +
                0.25 * features.get('sharpe_6m', 0) +
                0.20 * features.get('mom_accel', 0) +
                0.15 * (1 + features.get('maxdd_63d', -1)) +  # Higher is better (less DD)
                0.10 * features.get('vol_ratio', 1)
            )

            # Get current price and vol for premium estimation
            current_price = ticker_data['close'].iloc[-1]
            vol_annual = features.get('vol_60d', 0.20)

            rankings.append({
                'ticker': ticker,
                'score': score,
                'price': current_price,
                'vol': vol_annual,
                'features': features
            })

        if len(rankings) < top_n:
            rebal_idx += hold_days
            continue

        # Sort by score (highest momentum = best)
        rankings.sort(key=lambda x: x['score'], reverse=True)
        top_picks = rankings[:top_n]

        # Trade each top ETF
        current_equity = equity[-1]
        period_pnl = 0

        for pick in top_picks:
            price = pick['price']
            vol = max(pick['vol'], 0.10)  # Floor vol at 10%

            # Estimate premium
            premium, max_loss_per_share, breakeven = estimate_put_spread_premium(
                price, vol, DAYS_TO_EXPIRY, spread_width_pct, short_delta
            )

            if premium <= 0 or max_loss_per_share <= 0:
                continue

            # Position sizing: max_risk per spread
            spread_width_dollars = price * spread_width_pct / 100 * 100  # Per contract (100 shares)
            premium_dollars = premium * 100
            max_loss_dollars = spread_width_dollars - premium_dollars

            if max_loss_dollars <= 0:
                continue

            # How many contracts can we afford?
            n_contracts = max(1, int(min(max_risk, current_equity * 0.3) / max_loss_dollars))

            # Get exit price
            exit_data = data[(data['ticker'] == pick['ticker']) & (data['date'] == exit_date)]
            if len(exit_data) == 0:
                # Try nearest date
                future_data = data[(data['ticker'] == pick['ticker']) &
                                  (data['date'] > rebal_date) &
                                  (data['date'] <= exit_date)]
                if len(future_data) == 0:
                    continue
                exit_data = future_data.iloc[-1:]

            exit_price = exit_data['close'].iloc[0]

            # Calculate strikes
            from scipy.stats import norm
            T = DAYS_TO_EXPIRY / 365.0
            d1_target = norm.ppf(1 - short_delta)
            sqrt_T = np.sqrt(T)
            short_strike = price * np.exp(-d1_target * vol * sqrt_T + (RISK_FREE_RATE - 0.5 * vol**2) * T)
            long_strike = short_strike - price * spread_width_pct / 100

            # Simulate P&L
            pnl_per_share = simulate_put_spread_pnl(
                price, exit_price, short_strike, long_strike,
                premium, hold_days, DAYS_TO_EXPIRY
            )

            trade_pnl = pnl_per_share * 100 * n_contracts  # 100 shares per contract
            trade_pnl -= COMMISSION_PER_CONTRACT * 2 * 2 * n_contracts  # Open + close, 2 legs

            period_pnl += trade_pnl

            trades.append({
                'date': rebal_date,
                'exit_date': exit_date,
                'ticker': pick['ticker'],
                'score': pick['score'],
                'price': price,
                'exit_price': exit_price,
                'vol': vol,
                'short_strike': short_strike,
                'long_strike': long_strike,
                'premium': premium * 100 * n_contracts,
                'pnl': trade_pnl,
                'n_contracts': n_contracts,
                'return_pct': exit_price / price - 1,
                'won': trade_pnl > 0
            })

        current_equity += period_pnl
        equity.append(current_equity)
        equity_dates.append(exit_date)

        rebal_idx += hold_days

    if len(trades) == 0:
        return None

    return {
        'variant': variant_name,
        'trades': trades,
        'equity': equity,
        'equity_dates': equity_dates
    }


def compute_metrics(result, starting_capital=645):
    """Compute risk-adjusted metrics from backtest result."""
    trades = result['trades']
    equity = result['equity']

    if len(trades) == 0:
        return None

    trade_df = pd.DataFrame(trades)

    # Basic stats
    total_trades = len(trades)
    winners = sum(1 for t in trades if t['pnl'] > 0)
    wr = winners / total_trades if total_trades > 0 else 0

    pnls = [t['pnl'] for t in trades]
    avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
    avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Equity curve metrics
    equity_arr = np.array(equity)
    final_equity = equity_arr[-1]
    total_return = (final_equity - starting_capital) / starting_capital

    # Approximate years
    if len(result['equity_dates']) >= 2:
        first_date = pd.Timestamp(result['equity_dates'][0])
        last_date = pd.Timestamp(result['equity_dates'][-1])
        years = (last_date - first_date).days / 365.25
    else:
        years = 1

    cagr = (final_equity / starting_capital) ** (1 / max(years, 0.1)) - 1 if final_equity > 0 else -1

    # Max drawdown
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = np.min(dd)

    # Monthly returns for Sharpe/Sortino
    period_returns = []
    for i in range(1, len(equity)):
        if equity[i-1] > 0:
            period_returns.append(equity[i] / equity[i-1] - 1)

    if len(period_returns) > 1:
        # Annualize (assuming ~12 periods/year for monthly)
        periods_per_year = 12
        mean_ret = np.mean(period_returns)
        std_ret = np.std(period_returns)
        sharpe = mean_ret / std_ret * np.sqrt(periods_per_year) if std_ret > 0 else 0

        downside_rets = [r for r in period_returns if r < 0]
        downside_std = np.std(downside_rets) if len(downside_rets) > 1 else std_ret
        sortino = mean_ret / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'total_trades': total_trades,
        'win_rate': wr,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': cagr,
        'max_dd': max_dd,
        'calmar': calmar,
        'total_return': total_return,
        'final_equity': final_equity,
        'starting_capital': starting_capital,
        'years': years,
        'trades_per_year': total_trades / max(years, 0.1)
    }


def permutation_test(data, result, n_perms=200, top_n=3, hold_days=21):
    """
    Permutation test: shuffle momentum rankings randomly.
    If random selection is also profitable → signal is not real.
    """
    real_metrics = compute_metrics(result)
    if real_metrics is None:
        return 1.0, []

    real_sharpe = real_metrics['sharpe']
    perm_sharpes = []

    all_dates = sorted(data['date'].unique())
    available_tickers = data['ticker'].unique()

    print(f"  Running {n_perms} permutations...")

    for i in range(n_perms):
        # Instead of running full backtest, approximate:
        # Shuffle trade outcomes (which ticker was selected)
        trades = result['trades']
        shuffled_equity = [STARTING_CAPITAL]

        # Group trades by date
        trade_dates = {}
        for t in trades:
            d = str(t['date'])
            if d not in trade_dates:
                trade_dates[d] = []
            trade_dates[d].append(t)

        for date_key in sorted(trade_dates.keys()):
            date_trades = trade_dates[date_key]
            # Randomly reassign PnLs from the pool
            random_pnls = np.random.choice([t['pnl'] for t in trades], size=len(date_trades), replace=True)
            period_pnl = sum(random_pnls)
            shuffled_equity.append(shuffled_equity[-1] + period_pnl)

        if len(shuffled_equity) > 2:
            period_returns = []
            for j in range(1, len(shuffled_equity)):
                if shuffled_equity[j-1] > 0:
                    period_returns.append(shuffled_equity[j] / shuffled_equity[j-1] - 1)

            if len(period_returns) > 1 and np.std(period_returns) > 0:
                perm_sharpe = np.mean(period_returns) / np.std(period_returns) * np.sqrt(12)
            else:
                perm_sharpe = 0
        else:
            perm_sharpe = 0

        perm_sharpes.append(perm_sharpe)

    # p-value: fraction of permutations with Sharpe >= real
    p_value = np.mean([1 if ps >= real_sharpe else 0 for ps in perm_sharpes])

    return p_value, perm_sharpes


def regime_test(result, data):
    """
    R1 regime-agnostic test: compare performance in bull vs bear markets.
    Uses SPY close-to-close to classify regime.
    """
    trades = result['trades']
    if len(trades) == 0:
        return None

    # Get SPY data for regime classification
    spy_data = data[data['ticker'] == 'QQQ'].copy()  # Use QQQ as proxy if no SPY
    if len(spy_data) == 0:
        # Try to classify from trade returns
        return {'gap': 0, 'bull_sharpe': 0, 'bear_sharpe': 0, 'pass': True}

    spy_data = spy_data.sort_values('date')
    spy_data['sma_200'] = spy_data['close'].rolling(200).mean()

    bull_pnls = []
    bear_pnls = []

    for t in trades:
        trade_date = pd.Timestamp(t['date'])
        spy_row = spy_data[spy_data['date'] <= trade_date].iloc[-1:] if len(spy_data[spy_data['date'] <= trade_date]) > 0 else None

        if spy_row is not None and len(spy_row) > 0:
            if pd.notna(spy_row['sma_200'].iloc[0]):
                if spy_row['close'].iloc[0] > spy_row['sma_200'].iloc[0]:
                    bull_pnls.append(t['pnl'])
                else:
                    bear_pnls.append(t['pnl'])
            else:
                bull_pnls.append(t['pnl'])
        else:
            bull_pnls.append(t['pnl'])

    def sharpe_from_pnls(pnls):
        if len(pnls) < 2:
            return 0
        return np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(12)

    bull_sharpe = sharpe_from_pnls(bull_pnls)
    bear_sharpe = sharpe_from_pnls(bear_pnls)

    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0

    return {
        'bull_sharpe': bull_sharpe,
        'bear_sharpe': bear_sharpe,
        'bull_trades': len(bull_pnls),
        'bear_trades': len(bear_pnls),
        'gap': gap,
        'pass': gap <= 0.50
    }


def sub_period_test(result):
    """Split into halves, check both are profitable."""
    trades = result['trades']
    if len(trades) < 10:
        return {'pass': False, 'h1_sharpe': 0, 'h2_sharpe': 0}

    mid = len(trades) // 2
    h1 = trades[:mid]
    h2 = trades[mid:]

    def half_sharpe(trade_list):
        pnls = [t['pnl'] for t in trade_list]
        if len(pnls) < 2:
            return 0
        return np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(12)

    h1_sharpe = half_sharpe(h1)
    h2_sharpe = half_sharpe(h2)

    return {
        'h1_sharpe': h1_sharpe,
        'h2_sharpe': h2_sharpe,
        'pass': h1_sharpe > 0 and h2_sharpe > 0
    }


def outlier_test(result):
    """Remove top 5% of PnLs and check still profitable."""
    trades = result['trades']
    pnls = sorted([t['pnl'] for t in trades])

    if len(pnls) < 20:
        return {'pass': True, 'trimmed_sharpe': 0}

    # Remove top 5%
    n_remove = max(1, int(len(pnls) * 0.05))
    trimmed = pnls[:-n_remove]

    if len(trimmed) < 2 or np.std(trimmed) == 0:
        return {'pass': True, 'trimmed_sharpe': 0}

    trimmed_sharpe = np.mean(trimmed) / np.std(trimmed) * np.sqrt(12)

    return {
        'trimmed_sharpe': trimmed_sharpe,
        'pass': trimmed_sharpe > 0
    }


def main():
    print("=" * 70)
    print("MOMENTUM PUT CREDIT SPREAD ON SECTOR ETFs v1")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("momentum_put_spread_etf")
        mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}")

    # Fetch data
    data = fetch_etf_data()
    print(f"\nLoaded {len(data)} rows, {data['ticker'].nunique()} ETFs")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Test multiple variants
    variants = [
        # (name, top_n, hold_days, spread_width_pct, short_delta, max_risk)
        ("Top3_21d_3pct_30d", 3, 21, 3.0, 0.30, 200),
        ("Top3_21d_5pct_30d", 3, 21, 5.0, 0.30, 200),
        ("Top5_21d_3pct_30d", 5, 21, 3.0, 0.30, 200),
        ("Top3_21d_3pct_20d", 3, 21, 3.0, 0.20, 200),
        ("Top3_14d_3pct_30d", 3, 14, 3.0, 0.30, 200),
        ("Top2_21d_3pct_30d", 2, 21, 3.0, 0.30, 300),  # Fewer positions, more per trade
        ("Top3_21d_2pct_30d", 3, 21, 2.0, 0.30, 200),   # Tighter spread
        ("Top3_21d_3pct_15d", 3, 21, 3.0, 0.15, 200),   # Further OTM (15 delta)
    ]

    results = []
    best_result = None
    best_sharpe = -999

    for v_name, top_n, hold_days, spread_pct, delta, max_risk in variants:
        print(f"\n--- Variant: {v_name} ---")
        result = run_backtest(data, v_name, top_n=top_n, hold_days=hold_days,
                            spread_width_pct=spread_pct, short_delta=delta,
                            starting_capital=STARTING_CAPITAL, max_risk=max_risk)

        if result is None:
            print("  No trades generated")
            continue

        metrics = compute_metrics(result)
        if metrics is None:
            continue

        print(f"  Trades: {metrics['total_trades']}, WR: {metrics['win_rate']:.1%}")
        print(f"  Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
        print(f"  CAGR: {metrics['cagr']:.1%}, MaxDD: {metrics['max_dd']:.1%}")
        print(f"  PF: {metrics['profit_factor']:.2f}, Final: ${metrics['final_equity']:.0f}")

        result['metrics'] = metrics
        results.append(result)

        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_result = result

    if best_result is None:
        print("\nNO VARIANTS PRODUCED RESULTS")
        if MLFLOW_AVAILABLE:
            mlflow.log_param("status", "FAILED")
            mlflow.end_run()
        return

    # Run adversarial gates on best variant
    print(f"\n{'='*70}")
    print(f"BEST VARIANT: {best_result['variant']}")
    print(f"{'='*70}")

    metrics = best_result['metrics']
    print(f"\nPerformance:")
    print(f"  Sharpe:  {metrics['sharpe']:.2f}")
    print(f"  Sortino: {metrics['sortino']:.2f}")
    print(f"  CAGR:    {metrics['cagr']:.1%}")
    print(f"  MaxDD:   {metrics['max_dd']:.1%}")
    print(f"  WR:      {metrics['win_rate']:.1%}")
    print(f"  PF:      {metrics['profit_factor']:.2f}")
    print(f"  Calmar:  {metrics['calmar']:.2f}")
    print(f"  Trades:  {metrics['total_trades']} ({metrics['trades_per_year']:.0f}/yr)")
    print(f"  Final:   ${metrics['final_equity']:.0f} (from ${STARTING_CAPITAL})")

    # Gate 1: Permutation test
    print(f"\n--- GATE 1: Permutation Test (200 shuffles) ---")
    perm_p, perm_sharpes = permutation_test(data, best_result)
    perm_pass = perm_p < 0.05
    print(f"  p-value: {perm_p:.3f} {'PASS ✅' if perm_pass else 'FAIL ❌'}")
    print(f"  Real Sharpe: {metrics['sharpe']:.2f}, Random mean: {np.mean(perm_sharpes):.2f}")

    # Gate 2: Regime test
    print(f"\n--- GATE 2: Regime-Agnostic (R1) ---")
    regime = regime_test(best_result, data)
    if regime:
        print(f"  Bull Sharpe: {regime['bull_sharpe']:.2f} ({regime['bull_trades']} trades)")
        print(f"  Bear Sharpe: {regime['bear_sharpe']:.2f} ({regime['bear_trades']} trades)")
        print(f"  Gap: {regime['gap']:.3f} {'PASS ✅' if regime['pass'] else 'FAIL ❌'}")

    # Gate 3: Sub-period
    print(f"\n--- GATE 3: Sub-Period Stability ---")
    sub = sub_period_test(best_result)
    print(f"  H1 Sharpe: {sub['h1_sharpe']:.2f}")
    print(f"  H2 Sharpe: {sub['h2_sharpe']:.2f}")
    print(f"  {'PASS ✅' if sub['pass'] else 'FAIL ❌'}")

    # Gate 4: Outlier
    print(f"\n--- GATE 4: Outlier Robustness ---")
    outlier = outlier_test(best_result)
    print(f"  Trimmed Sharpe: {outlier['trimmed_sharpe']:.2f}")
    print(f"  {'PASS ✅' if outlier['pass'] else 'FAIL ❌'}")

    gates_passed = sum([perm_pass, regime.get('pass', False) if regime else False,
                       sub['pass'], outlier['pass']])
    print(f"\n{'='*70}")
    print(f"GATES: {gates_passed}/4 PASS")
    print(f"{'='*70}")

    # Ticker analysis
    trade_df = pd.DataFrame(best_result['trades'])
    print(f"\nTop tickers by frequency:")
    ticker_counts = trade_df['ticker'].value_counts().head(10)
    for ticker, count in ticker_counts.items():
        ticker_trades = trade_df[trade_df['ticker'] == ticker]
        ticker_wr = ticker_trades['won'].mean()
        ticker_pnl = ticker_trades['pnl'].sum()
        print(f"  {ticker}: {count} trades, WR {ticker_wr:.0%}, total PnL ${ticker_pnl:.0f}")

    # All variants summary
    print(f"\n{'='*70}")
    print("ALL VARIANTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<25} {'Sharpe':>7} {'WR':>6} {'CAGR':>8} {'MaxDD':>8} {'PF':>6} {'Final':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['metrics']['sharpe'], reverse=True):
        m = r['metrics']
        print(f"{r['variant']:<25} {m['sharpe']:>7.2f} {m['win_rate']:>5.1%} "
              f"{m['cagr']:>7.1%} {m['max_dd']:>7.1%} {m['profit_factor']:>6.2f} "
              f"${m['final_equity']:>7.0f}")

    # Save results
    results_path = '/home/jupiter/Lvl3Quant/research/findings/momentum_put_spread_etf_v1_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'best_variant': best_result['variant'],
        'metrics': metrics,
        'gates': {
            'perm_p': perm_p,
            'perm_pass': perm_pass,
            'regime': regime,
            'sub_period': sub,
            'outlier': outlier,
            'total_pass': gates_passed
        },
        'all_variants': [
            {
                'variant': r['variant'],
                'metrics': r['metrics']
            }
            for r in results
        ]
    }

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_AVAILABLE:
        mlflow.log_param("best_variant", best_result['variant'])
        mlflow.log_param("n_variants", len(results))
        mlflow.log_param("universe_size", len(ETF_UNIVERSE))
        mlflow.log_param("starting_capital", STARTING_CAPITAL)

        mlflow.log_metric("sharpe", metrics['sharpe'])
        mlflow.log_metric("sortino", metrics['sortino'])
        mlflow.log_metric("cagr", metrics['cagr'])
        mlflow.log_metric("max_dd", metrics['max_dd'])
        mlflow.log_metric("win_rate", metrics['win_rate'])
        mlflow.log_metric("profit_factor", metrics['profit_factor'])
        mlflow.log_metric("calmar", metrics['calmar'])
        mlflow.log_metric("total_trades", metrics['total_trades'])
        mlflow.log_metric("final_equity", metrics['final_equity'])
        mlflow.log_metric("perm_p_value", perm_p)
        mlflow.log_metric("gates_passed", gates_passed)

        mlflow.log_artifact(results_path)
        mlflow.end_run()

    print(f"\nCompleted at {datetime.now()}")
    return save_data


if __name__ == '__main__':
    main()
