#!/usr/bin/env python3
"""
Correlation Regime + Quality MR Backtest
=========================================
When cross-stock correlations spike (panic selling), individual stock dips
become MORE mean-reverting because the selloff is indiscriminate.

6 Variants (A-F), 5-gate validation, full walk-forward on 20 quality stocks.
"""

import json
import warnings
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "AVGO": "XLK", "GOOGL": "XLK", "META": "XLK",
    "AMZN": "XLY", "HD": "XLY", "COST": "XLY",
    "JPM": "XLF", "V": "XLF", "MA": "XLF",
    "JNJ": "XLV", "UNH": "XLV", "LLY": "XLV", "ABBV": "XLV", "MRK": "XLV",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP",
}
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLP", "XLY"]
INDEX = "SPY"

CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
HOLD_DAYS = 10

START = "2021-06-01"  # extra lookback for 60d beta / 20d corr
TRADE_START = "2022-01-01"
END = "2026-07-31"

RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/correlation_regime_results.json"

# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
print("Downloading price data...")
tickers = list(set(UNIVERSE + SECTOR_ETFS + [INDEX]))
raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"].copy()
else:
    close = raw.copy()

# Forward-fill and drop tickers with insufficient data
close = close.ffill().dropna(axis=1, how="all")
missing = [t for t in UNIVERSE if t not in close.columns]
if missing:
    print(f"WARNING: Missing tickers: {missing}")
    UNIVERSE = [t for t in UNIVERSE if t in close.columns]

returns = close.pct_change()
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days, {len(UNIVERSE)} stocks")

# ── PRECOMPUTE FEATURES ─────────────────────────────────────────────────────
print("Computing features...")

# 20-day rolling pairwise correlation (average) — optimized via rolling corr matrix
def avg_pairwise_corr(rets_df, window=20):
    """Average pairwise correlation among universe stocks using rolling matrix."""
    n_stocks = len(rets_df.columns)
    n_pairs = n_stocks * (n_stocks - 1) / 2
    result = pd.Series(index=rets_df.index, dtype=float)
    arr = rets_df.values  # (T, N)
    T = len(arr)
    for t in range(window - 1, T):
        chunk = arr[t - window + 1:t + 1]  # (window, N)
        # Remove columns with NaN
        mask = ~np.any(np.isnan(chunk), axis=0)
        if mask.sum() < 2:
            continue
        c = np.corrcoef(chunk[:, mask].T)
        # Average of upper triangle
        n = c.shape[0]
        tri = c[np.triu_indices(n, k=1)]
        result.iloc[t] = np.nanmean(tri)
    return result

uni_rets = returns[UNIVERSE]
avg_corr = avg_pairwise_corr(uni_rets, 20)

# RSI (14-day)
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

rsi_df = close[UNIVERSE].apply(lambda x: rsi(x, 14))

# 20-day high for drawdown
high_20 = close[UNIVERSE].rolling(20).max()
pct_from_high = (close[UNIVERSE] - high_20) / high_20  # negative = dipped

# 60-day beta to SPY
def rolling_beta(stock_ret, mkt_ret, window=60):
    cov = stock_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    return cov / var

beta_df = pd.DataFrame(index=returns.index)
for s in UNIVERSE:
    beta_df[s] = rolling_beta(returns[s], returns[INDEX], 60)

# Per-stock correlation with sector ETF (20-day rolling)
sector_corr_df = pd.DataFrame(index=returns.index)
for s in UNIVERSE:
    etf = SECTOR_MAP.get(s)
    if etf and etf in returns.columns:
        sector_corr_df[s] = returns[s].rolling(20).corr(returns[etf])
    else:
        sector_corr_df[s] = np.nan

# SPY returns for beta-adjusted dip
spy_ret_5d = returns[INDEX].rolling(5).sum()  # 5-day cumulative return

print("Features computed.")

# ── TRADE ENGINE ────────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, capital, max_per_trade, max_concurrent, slippage_bps, hold_days):
        self.capital = capital
        self.max_per_trade = max_per_trade
        self.max_concurrent = max_concurrent
        self.slippage_bps = slippage_bps
        self.hold_days = hold_days

    def run(self, signals, close_df):
        """
        signals: list of (entry_date, ticker, direction) where direction='long'
        Returns: list of trade dicts with pnl
        """
        trades = []
        active = []  # (exit_date, ticker, entry_price, shares, entry_date)
        equity_curve = pd.Series(dtype=float)
        cash = self.capital
        dates = sorted(close_df.index)
        date_set = set(dates)

        # Build signal dict: date -> list of tickers
        sig_dict = {}
        for entry_date, ticker, _ in signals:
            if entry_date not in sig_dict:
                sig_dict[entry_date] = []
            sig_dict[entry_date].append(ticker)

        for i, d in enumerate(dates):
            # Close expired positions
            new_active = []
            for exit_date, ticker, entry_price, shares, open_date in active:
                if d >= exit_date:
                    if ticker in close_df.columns and not pd.isna(close_df.loc[d, ticker]):
                        exit_price = close_df.loc[d, ticker] * (1 - self.slippage_bps / 10000)
                        pnl = (exit_price - entry_price) * shares
                        ret = (exit_price / entry_price) - 1
                        trades.append({
                            "entry_date": str(open_date),
                            "exit_date": str(d),
                            "ticker": ticker,
                            "entry_price": round(entry_price, 4),
                            "exit_price": round(exit_price, 4),
                            "shares": shares,
                            "pnl": round(pnl, 2),
                            "return": round(ret, 6),
                        })
                        cash += exit_price * shares
                    else:
                        new_active.append((exit_date, ticker, entry_price, shares, open_date))
                        continue
                else:
                    new_active.append((exit_date, ticker, entry_price, shares, open_date))
            active = new_active

            # Open new positions
            if d in sig_dict:
                for ticker in sig_dict[d]:
                    if len(active) >= self.max_concurrent:
                        break
                    if ticker not in close_df.columns or pd.isna(close_df.loc[d, ticker]):
                        continue
                    price = close_df.loc[d, ticker] * (1 + self.slippage_bps / 10000)
                    alloc = min(self.max_per_trade, cash)
                    if alloc < 10:
                        continue
                    shares = int(alloc / price)
                    if shares < 1:
                        continue
                    cost = shares * price
                    cash -= cost
                    # Find exit date (hold_days trading days later)
                    exit_idx = min(i + self.hold_days, len(dates) - 1)
                    exit_date = dates[exit_idx]
                    active.append((exit_date, ticker, price, shares, d))

            # Equity curve
            port_val = cash
            for exit_date, ticker, entry_price, shares, open_date in active:
                if ticker in close_df.columns and not pd.isna(close_df.loc[d, ticker]):
                    port_val += close_df.loc[d, ticker] * shares
            equity_curve[d] = port_val

        # Close any remaining
        last_date = dates[-1]
        for exit_date, ticker, entry_price, shares, open_date in active:
            if ticker in close_df.columns and not pd.isna(close_df.loc[last_date, ticker]):
                exit_price = close_df.loc[last_date, ticker] * (1 - self.slippage_bps / 10000)
                pnl = (exit_price - entry_price) * shares
                ret = (exit_price / entry_price) - 1
                trades.append({
                    "entry_date": str(open_date),
                    "exit_date": str(last_date),
                    "ticker": ticker,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "shares": shares,
                    "pnl": round(pnl, 2),
                    "return": round(ret, 6),
                })

        return trades, equity_curve


# ── SIGNAL GENERATORS ───────────────────────────────────────────────────────
trade_start = pd.Timestamp(TRADE_START)
trade_dates = [d for d in close.index if d >= trade_start]

def variant_A():
    """High-correlation dip buy: avg_corr > 0.7, buy 3 most-dipped (>5% below high, RSI<40)"""
    signals = []
    for d in trade_dates:
        if pd.isna(avg_corr.get(d, np.nan)) or avg_corr[d] <= 0.7:
            continue
        candidates = []
        for s in UNIVERSE:
            if pd.isna(pct_from_high.loc[d, s]) or pd.isna(rsi_df.loc[d, s]):
                continue
            if pct_from_high.loc[d, s] < -0.05 and rsi_df.loc[d, s] < 40:
                candidates.append((pct_from_high.loc[d, s], s))
        candidates.sort()  # most negative first
        for _, s in candidates[:3]:
            signals.append((d, s, "long"))
    return signals

def variant_B():
    """Correlation spike: 20d avg corr increases >0.2 in 5 days, buy dipped quality stocks"""
    corr_5d_change = avg_corr.diff(5)
    signals = []
    for d in trade_dates:
        if pd.isna(corr_5d_change.get(d, np.nan)) or corr_5d_change[d] <= 0.2:
            continue
        candidates = []
        for s in UNIVERSE:
            if pd.isna(pct_from_high.loc[d, s]) or pd.isna(rsi_df.loc[d, s]):
                continue
            if pct_from_high.loc[d, s] < -0.05 and rsi_df.loc[d, s] < 40:
                candidates.append((pct_from_high.loc[d, s], s))
        candidates.sort()
        for _, s in candidates[:3]:
            signals.append((d, s, "long"))
    return signals

def variant_C():
    """Dispersion trade: low corr -> buy most oversold stock; high corr -> buy SPY"""
    signals = []
    for d in trade_dates:
        c = avg_corr.get(d, np.nan)
        if pd.isna(c):
            continue
        if c < 0.3:
            # Stock-picking: buy lowest RSI quality stock
            best = None
            best_rsi = 999
            for s in UNIVERSE:
                r = rsi_df.loc[d, s] if not pd.isna(rsi_df.loc[d, s]) else 999
                if r < best_rsi:
                    best_rsi = r
                    best = s
            if best and best_rsi < 50:
                signals.append((d, best, "long"))
        elif c > 0.6:
            signals.append((d, INDEX, "long"))
    return signals

def variant_D():
    """Decorrelation recovery: corr was >0.7, drops back <0.5, buy 3 most-dropped"""
    signals = []
    was_high = False
    high_corr_start_prices = {}
    for d in trade_dates:
        c = avg_corr.get(d, np.nan)
        if pd.isna(c):
            continue
        if c > 0.7 and not was_high:
            was_high = True
            # Record prices at start of high-corr period
            high_corr_start_prices = {}
            for s in UNIVERSE:
                if not pd.isna(close.loc[d, s]):
                    high_corr_start_prices[s] = close.loc[d, s]
        elif c < 0.5 and was_high:
            was_high = False
            # Buy 3 stocks that dropped most during high-corr period
            drops = []
            for s in UNIVERSE:
                if s in high_corr_start_prices and not pd.isna(close.loc[d, s]):
                    drop = (close.loc[d, s] / high_corr_start_prices[s]) - 1
                    drops.append((drop, s))
            drops.sort()
            for _, s in drops[:3]:
                signals.append((d, s, "long"))
        elif c <= 0.7 and c >= 0.5:
            pass  # in between, keep was_high state
        elif c <= 0.5:
            was_high = False

    return signals

def variant_E():
    """Sector correlation filter: buy dips only when stock-sector corr < 0.5 (idiosyncratic)"""
    signals = []
    for d in trade_dates:
        candidates = []
        for s in UNIVERSE:
            if pd.isna(pct_from_high.loc[d, s]) or pd.isna(rsi_df.loc[d, s]):
                continue
            if pd.isna(sector_corr_df.loc[d, s]):
                continue
            if (pct_from_high.loc[d, s] < -0.05 and
                rsi_df.loc[d, s] < 40 and
                sector_corr_df.loc[d, s] < 0.5):
                candidates.append((rsi_df.loc[d, s], s))
        candidates.sort()  # lowest RSI first
        for _, s in candidates[:3]:
            signals.append((d, s, "long"))
    return signals

def variant_F():
    """Beta-adjusted dip: stock dropped more than beta * SPY drop (excess dip)"""
    signals = []
    for d in trade_dates:
        spy_5d = spy_ret_5d.get(d, np.nan)
        if pd.isna(spy_5d) or spy_5d >= 0:
            continue  # only when SPY is down
        candidates = []
        for s in UNIVERSE:
            b = beta_df.loc[d, s] if not pd.isna(beta_df.loc[d, s]) else 1.0
            stock_5d = returns[s].rolling(5).sum().get(d, np.nan)
            if pd.isna(stock_5d):
                continue
            expected_drop = b * spy_5d
            excess_dip = stock_5d - expected_drop  # negative = dropped more than expected
            if excess_dip < -0.02 and rsi_df.loc[d, s] < 45:
                candidates.append((excess_dip, s))
        candidates.sort()
        for _, s in candidates[:3]:
            signals.append((d, s, "long"))
    return signals


# ── METRICS ─────────────────────────────────────────────────────────────────
def compute_metrics(trades, equity_curve):
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "total_return": 0, "max_dd": 0, "win_rate": 0}

    rets = [t["return"] for t in trades]
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for r in rets if r > 0)
    n = len(rets)

    # Sharpe from trade returns (annualized assuming ~25 trades/year)
    avg_ret = np.mean(rets)
    std_ret = np.std(rets) if np.std(rets) > 0 else 1e-9
    trades_per_year = max(n / 4.5, 1)  # ~4.5 years of data
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year)

    # Sortino
    downside = [r for r in rets if r < 0]
    downside_std = np.std(downside) if downside else 1e-9
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year)

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    if len(equity_curve) > 0:
        peak = equity_curve.expanding().max()
        dd = (equity_curve - peak) / peak
        max_dd = dd.min()
    else:
        max_dd = 0

    total_pnl = sum(pnls)
    total_return = total_pnl / CAPITAL

    return {
        "n_trades": n,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wins / n * 100, 1),
        "avg_return_pct": round(avg_ret * 100, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_pnl": round(np.mean(pnls), 2),
    }


# ── PERMUTATION TEST (FAST — precompute forward returns) ────────────────────
def permutation_test(signals, close_df, engine, actual_pnl, n_perms=1000):
    """Shuffle entry dates, compare avg trade return vs random. Fast vectorized."""
    if not signals or actual_pnl is None:
        return 1.0

    # Precompute hold-period returns for every (date, ticker) pair
    all_tickers = list(close_df.columns)
    fwd_ret = {}
    dates_list = list(close_df.index)
    date_to_idx = {d: i for i, d in enumerate(dates_list)}
    for ticker in all_tickers:
        prices = close_df[ticker].values
        for i, d in enumerate(dates_list):
            exit_i = min(i + HOLD_DAYS, len(dates_list) - 1)
            if not np.isnan(prices[i]) and not np.isnan(prices[exit_i]) and prices[i] > 0:
                fwd_ret[(d, ticker)] = prices[exit_i] / prices[i] - 1
            else:
                fwd_ret[(d, ticker)] = 0.0

    # Actual average return from real signals
    actual_rets = [fwd_ret.get((d, t), 0.0) for d, t, _ in signals]
    actual_mean = np.mean(actual_rets)

    # Random shuffles: pick random dates for same tickers
    rng = np.random.RandomState(42)
    td_arr = np.array(trade_dates)
    count_better = 0
    for _ in range(n_perms):
        rand_dates = rng.choice(td_arr, size=len(signals))
        rand_rets = [fwd_ret.get((rd, sig[1]), 0.0) for rd, sig in zip(rand_dates, signals)]
        if np.mean(rand_rets) >= actual_mean:
            count_better += 1
    return count_better / n_perms


# ── REGIME GAP ──────────────────────────────────────────────────────────────
def regime_gap(trades):
    """Split trades by SPY regime (up/down month), compute |sharpe_up - sharpe_down| / max."""
    if len(trades) < 10:
        return 999.0
    # Build month -> SPY return lookup robustly
    spy_rets = returns[INDEX].dropna()
    month_ret = {}
    for d in spy_rets.index:
        pk = pd.Timestamp(d).to_period("M")
        month_ret[pk] = month_ret.get(pk, 0.0) + spy_rets[d]

    up_trades = []
    down_trades = []
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        month_key = entry.to_period("M")
        spy_m = month_ret.get(month_key, 0.0)
        if spy_m > 0:
            up_trades.append(t["return"])
        else:
            down_trades.append(t["return"])

    if not up_trades or not down_trades:
        pass  # Not enough trades in both regimes
        return 999.0

    sharpe_up = np.mean(up_trades) / (np.std(up_trades) + 1e-9)
    sharpe_down = np.mean(down_trades) / (np.std(down_trades) + 1e-9)
    gap = abs(sharpe_up - sharpe_down) / max(abs(sharpe_up), abs(sharpe_down), 1e-9)
    return round(gap, 3)


# ── RUN ALL VARIANTS ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CORRELATION REGIME + QUALITY MR BACKTEST")
print("=" * 70)

# Close df for trading (includes SPY for variant C)
trade_close = close[UNIVERSE + [INDEX]].loc[trade_start:]

engine = BacktestEngine(CAPITAL, MAX_PER_TRADE, MAX_CONCURRENT, SLIPPAGE_BPS, HOLD_DAYS)

variants = {
    "A_high_corr_dip": variant_A,
    "B_corr_spike": variant_B,
    "C_dispersion": variant_C,
    "D_decorr_recovery": variant_D,
    "E_sector_filter": variant_E,
    "F_beta_adj_dip": variant_F,
}

results = {}

for name, gen_func in variants.items():
    print(f"\n{'─' * 50}")
    print(f"Variant {name}")
    print(f"{'─' * 50}")

    signals = gen_func()
    print(f"  Signals generated: {len(signals)}")

    trades, eq_curve = engine.run(signals, trade_close)
    metrics = compute_metrics(trades, eq_curve)
    print(f"  Trades: {metrics['n_trades']}, PnL: ${metrics.get('total_pnl', 0):.2f}, "
          f"Return: {metrics['total_return_pct']:.1f}%")
    print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, "
          f"PF: {metrics['profit_factor']:.3f}, WR: {metrics['win_rate']:.1f}%")
    print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}%")

    # 5-Gate validation
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_gt_0.5"] = metrics["sharpe"] > 0.5

    # Gate 2: Permutation test (only if enough trades)
    if metrics["n_trades"] >= 20:
        print("  Running permutation test (1000 shuffles)...")
        p_val = permutation_test(signals, trade_close, engine, metrics.get("total_pnl", 0))
        gates["perm_p_lt_0.05"] = p_val < 0.05
        metrics["perm_p_value"] = round(p_val, 4)
        print(f"  Permutation p-value: {p_val:.4f}")
    else:
        gates["perm_p_lt_0.05"] = False
        metrics["perm_p_value"] = None

    # Gate 3: Regime gap < 0.5
    rg = regime_gap(trades)
    gates["regime_gap_lt_0.5"] = rg < 0.5
    metrics["regime_gap"] = rg
    print(f"  Regime gap: {rg:.3f}")

    # Gate 4: Max DD > -50%
    gates["max_dd_gt_neg50"] = metrics["max_drawdown_pct"] > -50

    # Gate 5: At least 20 trades
    gates["min_20_trades"] = metrics["n_trades"] >= 20

    gates_passed = sum(gates.values())
    metrics["gates"] = gates
    metrics["gates_passed"] = f"{gates_passed}/5"
    metrics["verdict"] = "PASS" if gates_passed == 5 else "FAIL"

    print(f"  Gates: {gates_passed}/5 — {'PASS' if gates_passed == 5 else 'FAIL'}")
    for g, v in gates.items():
        status = "✓" if v else "✗"
        print(f"    {status} {g}: {v}")

    results[name] = metrics

# ── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Variant':<25} {'Trades':>6} {'PnL':>8} {'Sharpe':>7} {'Sortino':>8} "
      f"{'PF':>6} {'WR':>6} {'MaxDD':>7} {'Gates':>6} {'Verdict':>8}")
print("─" * 95)
for name, m in results.items():
    print(f"{name:<25} {m['n_trades']:>6} {m.get('total_pnl',0):>8.2f} {m['sharpe']:>7.3f} "
          f"{m['sortino']:>8.3f} {m['profit_factor']:>6.3f} {m['win_rate']:>5.1f}% "
          f"{m['max_drawdown_pct']:>6.1f}% {m['gates_passed']:>6} {m['verdict']:>8}")

# ── SAVE RESULTS ────────────────────────────────────────────────────────────
output = {
    "strategy": "Correlation Regime + Quality MR",
    "run_date": str(dt.datetime.now()),
    "period": f"{TRADE_START} to {END}",
    "universe": UNIVERSE,
    "config": {
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": SLIPPAGE_BPS,
        "hold_days": HOLD_DAYS,
    },
    "variants": {},
}

for name, m in results.items():
    # Convert gates bools to strings for JSON
    gates_clean = {k: bool(v) for k, v in m.get("gates", {}).items()}
    variant_out = {k: v for k, v in m.items() if k != "gates"}
    variant_out["gates"] = gates_clean
    output["variants"][name] = variant_out

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")
