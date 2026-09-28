#!/usr/bin/env python3
"""
Momentum Debit Spreads v1
=========================
Bull call debit spreads on top-momentum growth stocks.

MOTIVATION:
- Single-leg calls on growth stocks bleed theta (killed sector single-leg, growth swing)
- Debit spreads reduce theta exposure and cost per trade
- $50-150 per spread fits $645 account with $200 max position
- Defined risk: max loss = premium paid

STRATEGY:
- Screen 50+ growth stocks by momentum (cross-sectional ranking)
- Buy ATM call, sell OTM call on top-ranked stocks
- Spread widths: $2-5 depending on stock price
- Hold 10-20 trading days
- TP: +50% of max profit, SL: -40% of premium
- Rotate monthly

6 VARIANTS:
  A. Simple momentum, $3 wide spread, 20d hold
  B. LGBM-ranked momentum, $3 wide, 20d hold
  C. Momentum + volume surge filter, $3 wide, 10d hold
  D. Quality-momentum (add ROE/margins), $5 wide, 20d hold
  E. Top-1 concentrated (best signal), $3 wide, 20d hold
  F. Bear-safe (VIX filter + half size in bear), $3 wide, 20d hold

UNIVERSE: 52 growth stocks
PERIOD: 2022-01-01 to 2026-07-25
ACCOUNT: $645, max $200/spread
"""

import sys, os, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.stats import norm

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'momentum_debit_spreads_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

STARTING_CAPITAL = 645.0
COMMISSION_RT = 1.30  # Per contract leg
MAX_POSITION = 200.0
RISK_FREE_RATE = 0.05
START_DATE = '2022-01-01'
END_DATE = '2026-07-25'
N_PERMUTATIONS = 100

# ============================================================
# BLACK-SCHOLES
# ============================================================

def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def option_price(S, K, T, r, sigma, opt_type='call'):
    return bs_call(S, K, T, r, sigma) if opt_type == 'call' else bs_put(S, K, T, r, sigma)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'growth_stocks_prices.parquet')

    if os.path.exists(cache_path):
        prices = pd.read_parquet(cache_path)
        cache_end = prices.index.get_level_values('date').max()
        if cache_end >= pd.Timestamp(END_DATE) - pd.Timedelta(days=5):
            print(f"Loaded cached prices: {len(prices)} rows (through {cache_end.date()})")
            return prices
        else:
            print(f"Cache stale (ends {cache_end.date()}), refreshing...")

    print("Downloading prices...", flush=True)
    tickers = STOCK_UNIVERSE + ['SPY', '^VIX']
    frames = []
    for t in tickers:
        try:
            df = yf.download(t, start='2021-01-01', end=END_DATE, progress=False, auto_adjust=True)
            if len(df) < 50: continue
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            df['ticker'] = t
            df.index.name = 'date'
            frames.append(df)
        except:
            pass
    prices = pd.concat(frames).reset_index().set_index(['ticker', 'date']).sort_index()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    prices.to_parquet(cache_path)
    print(f"Saved prices cache: {len(prices)} rows")
    return prices


# ============================================================
# MOMENTUM RANKING
# ============================================================

def compute_momentum_features(prices, date, lookback=63):
    """Compute cross-sectional momentum features for all stocks on a given date."""
    features = {}
    spy_data = None
    try:
        spy_all = prices.loc['SPY', 'close']
        spy_mask = spy_all.index <= date
        if spy_mask.sum() >= lookback:
            spy_data = spy_all[spy_mask].iloc[-lookback:]
    except:
        pass

    for ticker in STOCK_UNIVERSE:
        try:
            t_prices = prices.loc[ticker]
            mask = t_prices.index <= date
            if mask.sum() < lookback + 10:
                continue
            t_close = t_prices.loc[mask, 'close']
            t_vol = t_prices.loc[mask, 'volume'] if 'volume' in t_prices.columns else None

            close = t_close.iloc[-lookback:]
            if len(close) < lookback:
                continue

            # Momentum features
            mom_21d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0
            mom_63d = close.iloc[-1] / close.iloc[0] - 1

            # Volatility
            vol = close.pct_change().dropna().std() * np.sqrt(252)

            # RSI
            rets = close.pct_change().dropna().iloc[-14:]
            gains = rets.clip(lower=0).mean()
            losses = (-rets).clip(lower=0).mean()
            rsi = 100 - 100 / (1 + gains/losses) if losses > 0 else 100

            # Relative strength vs SPY
            rel_str = 0
            if spy_data is not None and len(spy_data) >= 21:
                try:
                    rel_str = mom_21d - (spy_data.iloc[-1] / spy_data.iloc[-21] - 1)
                except:
                    pass

            # Volume surge
            vol_surge = 1.0
            if t_vol is not None:
                recent_vol = t_vol.iloc[-5:].mean()
                avg_vol = t_vol.iloc[-22:].mean()
                vol_surge = recent_vol / max(avg_vol, 1) if avg_vol > 0 else 1.0

            # Skewness (quality measure)
            daily_rets = close.pct_change().dropna()
            skew = float(daily_rets.skew()) if len(daily_rets) >= 10 else 0

            features[ticker] = {
                'mom_21d': mom_21d,
                'mom_63d': mom_63d,
                'vol': vol,
                'rsi': rsi,
                'rel_str': rel_str,
                'vol_surge': vol_surge,
                'skew': skew,
                'close': float(close.iloc[-1]),
            }
        except Exception:
            continue

    return features


def rank_stocks_simple(features, top_n=3):
    """Rank stocks by simple momentum (21d returns)."""
    if not features:
        return []
    sorted_stocks = sorted(features.items(), key=lambda x: x[1]['mom_21d'], reverse=True)
    return [t for t, _ in sorted_stocks[:top_n]]


def rank_stocks_quality(features, top_n=3):
    """Rank stocks by quality-adjusted momentum (momentum + low vol + positive skew)."""
    if not features:
        return []
    scores = {}
    for ticker, f in features.items():
        # Quality-momentum score: high momentum, low volatility, positive relative strength
        score = (
            0.4 * f['mom_21d'] +
            0.2 * f['mom_63d'] +
            0.2 * f['rel_str'] -
            0.1 * f['vol'] +
            0.1 * (f['rsi'] / 100 - 0.5)  # Mild RSI preference
        )
        scores[ticker] = score
    sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in sorted_stocks[:top_n]]


# ============================================================
# DEBIT SPREAD SIMULATION
# ============================================================

def simulate_debit_spread(stock_price, direction, spread_width, dte, iv,
                          holding_days, future_prices):
    """
    Simulate a bull call or bear put debit spread.

    Returns (entry_cost, exit_value, pnl_per_spread, exit_reason)
    """
    if direction == 'bull':
        # Bull call spread: buy ATM call, sell OTM call
        K_long = round(stock_price)
        K_short = K_long + spread_width
        T = dte / 252.0

        long_premium = bs_call(stock_price, K_long, T, RISK_FREE_RATE, iv)
        short_premium = bs_call(stock_price, K_short, T, RISK_FREE_RATE, iv)
        entry_cost = long_premium - short_premium
    else:
        # Bear put spread: buy ATM put, sell OTM put
        K_long = round(stock_price)
        K_short = K_long - spread_width
        T = dte / 252.0

        long_premium = bs_put(stock_price, K_long, T, RISK_FREE_RATE, iv)
        short_premium = bs_put(stock_price, K_short, T, RISK_FREE_RATE, iv)
        entry_cost = long_premium - short_premium

    if entry_cost <= 0.05:
        return None  # Spread too narrow/cheap

    max_profit = spread_width - entry_cost
    tp_target = entry_cost + 0.50 * max_profit  # 50% of max profit
    sl_target = entry_cost * 0.60  # Lose 40% of premium = exit

    exit_value = entry_cost
    exit_reason = 'time_stop'

    for day_idx in range(min(holding_days, len(future_prices))):
        spot = future_prices[day_idx]
        remaining_dte = max(dte - day_idx - 1, 0)
        T_rem = remaining_dte / 252.0

        # IV tends to increase slightly over time (not crush, unlike earnings)
        iv_adj = iv * (1 + 0.01 * day_idx)

        if direction == 'bull':
            long_val = bs_call(spot, K_long, T_rem, RISK_FREE_RATE, iv_adj)
            short_val = bs_call(spot, K_short, T_rem, RISK_FREE_RATE, iv_adj)
        else:
            long_val = bs_put(spot, K_long, T_rem, RISK_FREE_RATE, iv_adj)
            short_val = bs_put(spot, K_short, T_rem, RISK_FREE_RATE, iv_adj)

        spread_val = long_val - short_val

        if spread_val >= tp_target:
            exit_value = spread_val
            exit_reason = 'take_profit'
            break
        elif spread_val <= sl_target:
            exit_value = spread_val
            exit_reason = 'stop_loss'
            break

        exit_value = spread_val

    pnl = (exit_value - entry_cost) * 100 - COMMISSION_RT * 2  # 2 legs
    return {
        'entry_cost': entry_cost,
        'exit_value': exit_value,
        'pnl': pnl,
        'exit_reason': exit_reason,
        'max_profit': max_profit * 100,
        'spread_cost': entry_cost * 100,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_variant(variant_key, cfg, prices):
    """Run backtest for one variant."""
    print(f"\n{'='*60}")
    print(f"  VARIANT {variant_key}: {cfg['name']}")
    print(f"{'='*60}", flush=True)

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    # Get trading days from SPY
    try:
        spy_dates = prices.loc['SPY'].index.sort_values()
        trading_days = spy_dates[(spy_dates >= start) & (spy_dates <= end)]
    except:
        print("  ERROR: Cannot get SPY dates")
        return None

    # VIX data
    try:
        vix_data = prices.loc['^VIX', 'close']
    except:
        vix_data = None

    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0
    trades = []
    positions = []  # Open positions
    rebalance_days = cfg.get('rebal_days', 20)

    # Rebalance dates
    rebal_idx = list(range(0, len(trading_days), rebalance_days))

    for ri in rebal_idx:
        date = trading_days[ri]

        # Close expired positions first
        new_positions = []
        for pos in positions:
            if ri >= pos['exit_idx']:
                # Position expired — already counted in pnl
                pass
            else:
                new_positions.append(pos)
        positions = new_positions

        # VIX filter (variant F)
        if cfg.get('vix_filter'):
            if vix_data is not None:
                vix_mask = vix_data.index <= date
                if vix_mask.any():
                    current_vix = float(vix_data[vix_mask].iloc[-1])
                    if current_vix > 30:
                        continue  # Skip high-VIX periods

        # Rank stocks
        features = compute_momentum_features(prices, date, lookback=63)
        if not features:
            continue

        if cfg.get('quality_momentum'):
            top_picks = rank_stocks_quality(features, cfg.get('top_n', 3))
        else:
            top_picks = rank_stocks_simple(features, cfg.get('top_n', 3))

        # Volume surge filter (variant C)
        if cfg.get('vol_surge_filter'):
            top_picks = [t for t in top_picks if features.get(t, {}).get('vol_surge', 1.0) > 1.2]
            if not top_picks:
                # Fall back to top 1
                top_picks = rank_stocks_simple(features, 1)

        # Don't enter if already at max positions
        max_concurrent = cfg.get('max_concurrent', 3)
        slots_open = max_concurrent - len(positions)
        if slots_open <= 0:
            continue

        top_picks = top_picks[:slots_open]

        spread_width = cfg.get('spread_width', 3)
        hold_days = cfg.get('hold_days', 20)
        dte = hold_days + 7  # Option DTE = hold period + 7 days buffer

        # VIX-based position sizing (variant F)
        size_multiplier = 1.0
        if cfg.get('bear_safe') and vix_data is not None:
            vix_mask = vix_data.index <= date
            if vix_mask.any():
                current_vix = float(vix_data[vix_mask].iloc[-1])
                if current_vix > 25:
                    size_multiplier = 0.5

        for ticker in top_picks:
            if ticker not in features:
                continue

            stock_price = features[ticker]['close']
            iv = features[ticker]['vol']
            iv = max(iv, 0.15)  # Floor IV

            # Get future prices for simulation
            try:
                t_prices = prices.loc[ticker]
                future_mask = t_prices.index > date
                future_dates = t_prices.index[future_mask][:hold_days + 1]
                if len(future_dates) < 2:
                    continue
                future_prices = [float(t_prices.loc[d, 'close']) for d in future_dates]
            except:
                continue

            # Simulate spread
            result = simulate_debit_spread(
                stock_price, 'bull', spread_width, dte, iv,
                hold_days, future_prices
            )

            if result is None:
                continue

            spread_cost = result['spread_cost'] * size_multiplier
            if spread_cost > MAX_POSITION or spread_cost + COMMISSION_RT * 2 > equity:
                continue

            pnl = result['pnl'] * size_multiplier
            equity += pnl

            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd

            trades.append({
                'ticker': ticker,
                'date': str(date.date()),
                'direction': 'bull',
                'spread_width': spread_width,
                'spread_cost': round(result['spread_cost'], 2),
                'pnl': round(pnl, 2),
                'exit_reason': result['exit_reason'],
                'equity': round(equity, 2),
                'mom_21d': round(features[ticker]['mom_21d'] * 100, 1),
            })

            positions.append({
                'ticker': ticker,
                'entry_idx': ri,
                'exit_idx': ri + hold_days // rebalance_days + 1,
            })

    if not trades:
        print("  No trades generated!")
        return None

    # Compute metrics
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n_trades = len(trades)
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

    # Build daily equity curve for Sharpe/Sortino
    equity_by_date = defaultdict(float)
    eq = STARTING_CAPITAL
    for t in sorted(trades, key=lambda x: x['date']):
        eq += t['pnl']
        equity_by_date[t['date']] = eq

    # Create daily returns from trade dates
    dates_sorted = sorted(equity_by_date.keys())
    if len(dates_sorted) >= 2:
        equities = [STARTING_CAPITAL] + [equity_by_date[d] for d in dates_sorted]
        returns = np.diff(equities) / equities[:-1]
        returns = returns[np.isfinite(returns)]

        # Pad with zero-return days for proper Sharpe calculation
        total_days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days
        n_zero_days = max(0, total_days - len(returns))
        all_returns = np.concatenate([returns, np.zeros(n_zero_days)])

        sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-10) * np.sqrt(252)
        downside = all_returns[all_returns < 0]
        sortino = np.mean(all_returns) / (np.std(downside) + 1e-10) * np.sqrt(252) if len(downside) > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    total_return = (equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    years = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 365.25
    cagr = ((equity / STARTING_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0

    # Regime breakdown
    regime_pnls = {'green': [], 'red': [], 'flat': []}
    try:
        spy_close = prices.loc['SPY', 'close']
        for t in trades:
            d = pd.Timestamp(t['date'])
            mask = spy_close.index <= d
            if mask.sum() >= 2:
                spy_prev = float(spy_close[mask].iloc[-2])
                spy_curr = float(spy_close[mask].iloc[-1])
                day_ret = spy_curr / spy_prev - 1
                if day_ret > 0.001:
                    regime_pnls['green'].append(t['pnl'])
                elif day_ret < -0.001:
                    regime_pnls['red'].append(t['pnl'])
                else:
                    regime_pnls['flat'].append(t['pnl'])
    except:
        pass

    green_sharpe = np.mean(regime_pnls['green']) / (np.std(regime_pnls['green']) + 1e-10) if regime_pnls['green'] else 0
    red_sharpe = np.mean(regime_pnls['red']) / (np.std(regime_pnls['red']) + 1e-10) if regime_pnls['red'] else 0
    regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 1e-10)

    # Ticker concentration
    ticker_pnls = defaultdict(float)
    for t in trades:
        ticker_pnls[t['ticker']] += t['pnl']
    total_pnl = sum(abs(v) for v in ticker_pnls.values())
    max_ticker_conc = max(abs(v) for v in ticker_pnls.values()) / max(total_pnl, 1e-10) * 100

    result = {
        'variant': variant_key,
        'name': cfg['name'],
        'n_trades': n_trades,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 2),
        'wr': round(win_rate, 1),
        'mdd': round(max_dd * 100, 1),
        'total_return': round(total_return, 1),
        'cagr': round(cagr, 1),
        'final_equity': round(equity, 0),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'regime_gap': round(regime_gap, 3),
        'max_ticker_conc': round(max_ticker_conc, 1),
        'green_n': len(regime_pnls['green']),
        'red_n': len(regime_pnls['red']),
    }

    print(f"  Trades: {n_trades}, WR: {win_rate:.1f}%")
    print(f"  Sharpe: {sharpe:.3f}, Sortino: {sortino:.3f}, PF: {pf:.2f}")
    print(f"  Equity: ${STARTING_CAPITAL} → ${equity:.0f} (MDD: {max_dd*100:.1f}%)")
    print(f"  Regime gap: {regime_gap:.3f} (green: {len(regime_pnls['green'])}, red: {len(regime_pnls['red'])})")
    print(f"  Top ticker concentration: {max_ticker_conc:.0f}%")
    if ticker_pnls:
        top3 = sorted(ticker_pnls.items(), key=lambda x: x[1], reverse=True)[:3]
        bot3 = sorted(ticker_pnls.items(), key=lambda x: x[1])[:3]
        print(f"  Top: {', '.join(f'{t}: ${p:.0f}' for t, p in top3)}")
        print(f"  Bot: {', '.join(f'{t}: ${p:.0f}' for t, p in bot3)}")

    return result, trades


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """Shuffle trade directions to test if momentum ranking adds value."""
    if not trades:
        return 1.0, 0

    pnls = [t['pnl'] for t in trades]
    real_sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10)

    rng = np.random.RandomState(42)
    count_better = 0
    random_sharpes = []

    for _ in range(n_perms):
        # Shuffle the pnls (random entry timing)
        shuffled = rng.permutation(pnls)
        s = np.mean(shuffled) / (np.std(shuffled) + 1e-10)
        random_sharpes.append(s)
        if s >= real_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    random_mean = np.mean(random_sharpes)
    return p_value, random_mean


# ============================================================
# MONTE CARLO CI
# ============================================================

def monte_carlo_ci(trades, n_sims=1000):
    """Bootstrap confidence interval on final equity."""
    if len(trades) < 5:
        return 0, 0

    pnls = [t['pnl'] for t in trades]
    rng = np.random.RandomState(42)
    finals = []

    for _ in range(n_sims):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        eq = STARTING_CAPITAL + sum(sample)
        finals.append(eq)

    ci_5 = np.percentile(finals, 5)
    ci_95 = np.percentile(finals, 95)
    return ci_5 - STARTING_CAPITAL, ci_95 - STARTING_CAPITAL


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("  MOMENTUM DEBIT SPREADS v1")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60, flush=True)

    prices = load_data()

    # MLflow setup
    mlflow_exp_id = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("http://jupiter:5000")
            exp = mlflow.set_experiment("momentum_debit_spreads_v1")
            mlflow_exp_id = exp.experiment_id
            print(f"MLflow experiment: {mlflow_exp_id}")
        except:
            pass

    variants = {
        'A': {
            'name': 'Simple momentum, $3 spread, 20d',
            'spread_width': 3, 'hold_days': 20, 'rebal_days': 20,
            'top_n': 3, 'max_concurrent': 3,
        },
        'B': {
            'name': 'Quality-momentum ranked, $3 spread, 20d',
            'spread_width': 3, 'hold_days': 20, 'rebal_days': 20,
            'top_n': 3, 'max_concurrent': 3,
            'quality_momentum': True,
        },
        'C': {
            'name': 'Momentum + volume surge, $3 spread, 10d',
            'spread_width': 3, 'hold_days': 10, 'rebal_days': 10,
            'top_n': 3, 'max_concurrent': 3,
            'vol_surge_filter': True,
        },
        'D': {
            'name': 'Quality-momentum, $5 wide spread, 20d',
            'spread_width': 5, 'hold_days': 20, 'rebal_days': 20,
            'top_n': 3, 'max_concurrent': 3,
            'quality_momentum': True,
        },
        'E': {
            'name': 'Top-1 concentrated, $3 spread, 20d',
            'spread_width': 3, 'hold_days': 20, 'rebal_days': 20,
            'top_n': 1, 'max_concurrent': 1,
        },
        'F': {
            'name': 'Bear-safe (VIX filter + half size), $3 spread, 20d',
            'spread_width': 3, 'hold_days': 20, 'rebal_days': 20,
            'top_n': 3, 'max_concurrent': 3,
            'vix_filter': True, 'bear_safe': True,
        },
    }

    all_results = {}
    best_sharpe = -999
    best_variant = None

    for vk, cfg in variants.items():
        out = run_variant(vk, cfg, prices)
        if out is None:
            all_results[vk] = {'variant': vk, 'error': 'No trades'}
            continue

        result, trades = out

        # Permutation test
        p_val, random_sharpe = permutation_test(trades)
        result['perm_p'] = round(p_val, 3)
        result['random_sharpe'] = round(random_sharpe, 3)

        # Monte Carlo CI
        mc_lo, mc_hi = monte_carlo_ci(trades)
        result['mc_ci_5'] = round(mc_lo, 1)
        result['mc_ci_95'] = round(mc_hi, 1)

        # 5-Gate validation
        gates = {
            'sharpe_gt_1': result['sharpe'] > 1.0,
            'perm_p<0.05': result['perm_p'] < 0.05,
            'wr>40': result['wr'] > 40,
            'regime_gap<0.50': result['regime_gap'] < 0.50,
            'mc_ci>0': result['mc_ci_5'] > 0,
        }
        result['gates'] = gates
        result['gates_passed'] = sum(gates.values())

        print(f"\n  5-Gate: {result['gates_passed']}/5 {'✅' if result['gates_passed'] >= 4 else '❌'}")
        for g, v in gates.items():
            val = result.get(g.split('>')[0].split('<')[0].replace('_gt_', '>').replace('_lt_', '<'), '')
            print(f"    {g}: {'PASS' if v else 'FAIL'} ({val})")

        all_results[vk] = result

        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_variant = vk

        # Log to MLflow
        if MLFLOW_AVAILABLE and mlflow_exp_id:
            try:
                with mlflow.start_run(experiment_id=mlflow_exp_id,
                                      run_name=f"spread_{vk}_{cfg['name'][:30]}"):
                    mlflow.log_params({
                        'variant': vk,
                        'spread_width': cfg['spread_width'],
                        'hold_days': cfg['hold_days'],
                        'top_n': cfg['top_n'],
                    })
                    mlflow.log_metrics({
                        'sharpe': result['sharpe'],
                        'sortino': result['sortino'],
                        'pf': result['pf'],
                        'wr': result['wr'],
                        'mdd': result['mdd'],
                        'final_equity': result['final_equity'],
                        'perm_p': result['perm_p'],
                        'regime_gap': result['regime_gap'],
                        'n_trades': result['n_trades'],
                        'gates_passed': result['gates_passed'],
                    })
            except:
                pass

        # Save trades
        trades_path = os.path.join(OUTPUT_DIR, f'trades_variant_{vk}.json')
        with open(trades_path, 'w') as f:
            json.dump(trades, f, indent=2, default=str)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'backtest_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'runtime_seconds': round(time.time() - t0, 1),
            'variants': all_results,
            'best_variant': best_variant,
            'random_baseline': all_results.get(best_variant, {}).get('random_sharpe', 0),
        }, f, indent=2, default=str)

    # Summary
    runtime = time.time() - t0
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY — Momentum Debit Spreads v1")
    print(f"  Runtime: {runtime:.1f}s")
    if best_variant and best_variant in all_results:
        r = all_results[best_variant]
        print(f"  RANDOM BASELINE: Sharpe {r.get('random_sharpe', 'N/A')}")
    print(f"{'='*60}")
    for vk in sorted(all_results.keys()):
        r = all_results[vk]
        if 'error' in r:
            print(f"     {vk}: ERROR — {r['error']}")
            continue
        best_flag = '🏆' if vk == best_variant else '  '
        print(f"  {best_flag} {vk}: Sh {r['sharpe']:.2f}, Sort {r['sortino']:.2f}, "
              f"PF {r['pf']:.2f}, WR {r['wr']:.0f}%, MDD {r['mdd']:.0f}%, "
              f"${STARTING_CAPITAL}→${r['final_equity']:.0f}, "
              f"{r['gates_passed']}/5, p={r['perm_p']:.3f}, R1={r['regime_gap']:.3f}, "
              f"conc={r['max_ticker_conc']:.0f}%")

    print(f"\n  Best: {best_variant} ({all_results.get(best_variant, {}).get('gates_passed', 0)}/5 gates, "
          f"Sharpe {best_sharpe:.3f})")
    print(f"\n  Results saved.")

    if MLFLOW_AVAILABLE and mlflow_exp_id:
        print(f"🧪 View experiment at: http://jupiter:5000/#/experiments/{mlflow_exp_id}")


if __name__ == '__main__':
    main()
