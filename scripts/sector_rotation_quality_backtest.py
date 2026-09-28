#!/usr/bin/env python3
"""
Sector Rotation Within Quality Universe Backtest
Tests whether rotating between sectors within a quality stock universe
adds alpha beyond simple MR timing.

6 Variants (A-F), 5-gate validation.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Universe & Sectors ──────────────────────────────────────────────
SECTORS = {
    "Tech": ["AAPL", "MSFT", "AVGO", "AMZN", "GOOGL", "META"],
    "Healthcare": ["UNH", "LLY", "ABBV", "MRK", "JNJ"],
    "Finance": ["JPM", "V", "MA"],
    "Consumer": ["PG", "KO", "PEP", "HD", "COST", "WMT"],
}
ALL_TICKERS = sorted(set(t for v in SECTORS.values() for t in v))
TICKER_TO_SECTOR = {}
for sec, ticks in SECTORS.items():
    for t in ticks:
        TICKER_TO_SECTOR[t] = sec

# ── Parameters ──────────────────────────────────────────────────────
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
HOLD_DAYS = 10
START = "2022-01-01"
END = "2026-07-31"
DOWNLOAD_START = "2021-06-01"

# ── Data Download ───────────────────────────────────────────────────
print("Downloading data...")
tickers_to_dl = ALL_TICKERS + ["SPY", "^VIX"]
raw = yf.download(tickers_to_dl, start=DOWNLOAD_START, end=END, auto_adjust=True, progress=False)

# Handle MultiIndex columns from yfinance
close = raw["Close"].copy()
# Rename ^VIX column
if "^VIX" in close.columns:
    close.rename(columns={"^VIX": "VIX"}, inplace=True)

close = close.dropna(how="all")
print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")

# ── Precompute Indicators ──────────────────────────────────────────
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

rsi = pd.DataFrame({t: compute_rsi(close[t]) for t in ALL_TICKERS})
high_20 = close[ALL_TICKERS].rolling(20).max()
pct_below_high = (close[ALL_TICKERS] - high_20) / high_20  # negative = below high
sma_20 = close[ALL_TICKERS].rolling(20).mean()
sma_50 = close[ALL_TICKERS].rolling(50).mean()
ret_20 = close[ALL_TICKERS].pct_change(20)

# Sector avg 20d returns
sector_ret_20 = pd.DataFrame()
for sec, ticks in SECTORS.items():
    sector_ret_20[sec] = ret_20[ticks].mean(axis=1)

# Sector avg price relative to 50-day SMA
sector_vs_sma50 = pd.DataFrame()
for sec, ticks in SECTORS.items():
    ratios = (close[ticks] - sma_50[ticks]) / sma_50[ticks]
    sector_vs_sma50[sec] = ratios.mean(axis=1)

# Sector breadth: % of stocks above their 20-SMA
sector_breadth = pd.DataFrame()
for sec, ticks in SECTORS.items():
    above_sma = (close[ticks] > sma_20[ticks]).astype(float)
    sector_breadth[sec] = above_sma.mean(axis=1)

# SPY regime
spy_ret_20 = close["SPY"].pct_change(20)

# VIX
vix = close["VIX"] if "VIX" in close.columns else None

# Date mask
dates = close.index
date_mask = dates >= START

# ── Backtesting Engine ──────────────────────────────────────────────
class Position:
    def __init__(self, ticker, entry_date, entry_price, shares):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.exit_date = None
        self.exit_price = None

    def pnl(self):
        if self.exit_price is None:
            return 0
        gross = (self.exit_price - self.entry_price) * self.shares
        slip_cost = (self.entry_price + self.exit_price) * self.shares * SLIPPAGE_BPS / 10000
        return gross - slip_cost


def run_backtest(signal_func, variant_name):
    """
    signal_func(date_idx, date) -> list of candidate tickers to buy
    Returns trades list and equity curve.
    """
    trades = []
    open_positions = []
    equity_curve = []
    cash = CAPITAL

    trade_dates = dates[date_mask]

    for date in trade_dates:
        idx = dates.get_loc(date)
        if idx < 50:  # need lookback
            equity_curve.append(CAPITAL)
            continue

        # Close positions at hold expiry
        still_open = []
        for pos in open_positions:
            days_held = np.busday_count(
                pos.entry_date.date() if hasattr(pos.entry_date, 'date') else pos.entry_date,
                date.date() if hasattr(date, 'date') else date
            )
            if days_held >= HOLD_DAYS:
                price = close.loc[date, pos.ticker]
                if pd.notna(price):
                    pos.exit_date = date
                    pos.exit_price = price
                    # Return sale proceeds minus slippage
                    proceeds = price * pos.shares * (1 - SLIPPAGE_BPS / 10000)
                    cash += proceeds
                    trades.append(pos)
                else:
                    still_open.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # Get candidates
        if len(open_positions) < MAX_CONCURRENT:
            candidates = signal_func(idx, date)
            # Filter out tickers we already hold
            held = {p.ticker for p in open_positions}
            candidates = [t for t in candidates if t not in held]

            for ticker in candidates:
                if len(open_positions) >= MAX_CONCURRENT:
                    break
                price = close.loc[date, ticker]
                if pd.isna(price) or price <= 0:
                    continue
                shares = int(MAX_PER_TRADE / price)
                if shares < 1:
                    continue
                cost = price * shares * (1 + SLIPPAGE_BPS / 10000)
                if cost > cash:
                    continue
                cash -= cost
                pos = Position(ticker, date, price, shares)
                open_positions.append(pos)

        # Mark-to-market
        mtm = cash
        for pos in open_positions:
            cur_price = close.loc[date, pos.ticker]
            if pd.notna(cur_price):
                mtm += cur_price * pos.shares
        equity_curve.append(mtm)

    # Force close remaining
    last_date = trade_dates[-1]
    for pos in open_positions:
        price = close.loc[last_date, pos.ticker]
        if pd.notna(price):
            pos.exit_date = last_date
            pos.exit_price = price
            trades.append(pos)

    return trades, equity_curve, trade_dates


# ── Signal Functions ────────────────────────────────────────────────

def variant_a_signal(idx, date):
    """Relative Sector Strength + Dip: buy from WEAKEST sector, RSI<40, >5% below high."""
    sr = sector_ret_20.iloc[idx]
    if sr.isna().all():
        return []
    weakest = sr.idxmin()
    ticks = SECTORS[weakest]
    cands = []
    for t in ticks:
        r = rsi.iloc[idx].get(t, np.nan)
        pb = pct_below_high.iloc[idx].get(t, np.nan)
        if pd.notna(r) and pd.notna(pb) and r < 40 and pb < -0.05:
            cands.append((t, r))
    cands.sort(key=lambda x: x[1])  # lowest RSI first
    return [c[0] for c in cands]


def variant_b_signal(idx, date):
    """Sector Momentum + Stock Dip: buy from STRONGEST sector, RSI<40, >5% below high."""
    sr = sector_ret_20.iloc[idx]
    if sr.isna().all():
        return []
    strongest = sr.idxmax()
    ticks = SECTORS[strongest]
    cands = []
    for t in ticks:
        r = rsi.iloc[idx].get(t, np.nan)
        pb = pct_below_high.iloc[idx].get(t, np.nan)
        if pd.notna(r) and pd.notna(pb) and r < 40 and pb < -0.05:
            cands.append((t, r))
    cands.sort(key=lambda x: x[1])
    return [c[0] for c in cands]


def variant_c_signal(idx, date):
    """Cross-Sector Divergence: one sector >5% above 50d avg, another >5% below -> buy lagging."""
    sv = sector_vs_sma50.iloc[idx]
    if sv.isna().all():
        return []
    above = sv[sv > 0.05]
    below = sv[sv < -0.05]
    if above.empty or below.empty:
        return []
    # Buy from the most lagging sector
    worst_sec = below.idxmin()
    ticks = SECTORS[worst_sec]
    cands = []
    for t in ticks:
        r = rsi.iloc[idx].get(t, np.nan)
        pb = pct_below_high.iloc[idx].get(t, np.nan)
        if pd.notna(r) and pd.notna(pb) and r < 40 and pb < -0.05:
            cands.append((t, r))
    cands.sort(key=lambda x: x[1])
    return [c[0] for c in cands]


def variant_d_signal(idx, date):
    """Sector Breadth + Dip: buy from sectors with breadth < 30%, stocks >7% below high."""
    sb = sector_breadth.iloc[idx]
    if sb.isna().all():
        return []
    weak_sectors = sb[sb < 0.30].index.tolist()
    if not weak_sectors:
        return []
    cands = []
    for sec in weak_sectors:
        for t in SECTORS[sec]:
            r = rsi.iloc[idx].get(t, np.nan)
            pb = pct_below_high.iloc[idx].get(t, np.nan)
            if pd.notna(r) and pd.notna(pb) and pb < -0.07:
                cands.append((t, r))
    cands.sort(key=lambda x: x[1])
    return [c[0] for c in cands]


def variant_e_signal(idx, date):
    """Defensive Rotation MR: VIX>25 -> Healthcare+Consumer; VIX<20 -> Tech+Finance."""
    if vix is None:
        return []
    v = vix.iloc[idx]
    if pd.isna(v):
        return []
    if v > 25:
        allowed_sectors = ["Healthcare", "Consumer"]
    elif v < 20:
        allowed_sectors = ["Tech", "Finance"]
    else:
        return []  # VIX 20-25: no trades
    cands = []
    for sec in allowed_sectors:
        for t in SECTORS[sec]:
            r = rsi.iloc[idx].get(t, np.nan)
            pb = pct_below_high.iloc[idx].get(t, np.nan)
            if pd.notna(r) and pd.notna(pb) and r < 40 and pb < -0.05:
                cands.append((t, r))
    cands.sort(key=lambda x: x[1])
    return [c[0] for c in cands]


def variant_f_signal(idx, date):
    """Sector MR Pairs: when Tech 20d return >5% worse than Consumer, buy most oversold Tech. Vice versa."""
    tech_r = sector_ret_20.iloc[idx].get("Tech", np.nan)
    cons_r = sector_ret_20.iloc[idx].get("Consumer", np.nan)
    if pd.isna(tech_r) or pd.isna(cons_r):
        return []

    target_sector = None
    if tech_r - cons_r < -0.05:
        target_sector = "Tech"
    elif cons_r - tech_r < -0.05:
        target_sector = "Consumer"
    else:
        return []

    ticks = SECTORS[target_sector]
    cands = []
    for t in ticks:
        r = rsi.iloc[idx].get(t, np.nan)
        pb = pct_below_high.iloc[idx].get(t, np.nan)
        if pd.notna(r) and pd.notna(pb) and r < 40 and pb < -0.05:
            cands.append((t, r))
    cands.sort(key=lambda x: x[1])
    return [c[0] for c in cands[:1]]  # only most oversold


# ── Metrics ─────────────────────────────────────────────────────────

def compute_metrics(trades, equity_curve, trade_dates):
    if len(trades) < 2:
        return {"n_trades": len(trades), "sharpe": 0, "total_return_pct": 0,
                "max_dd_pct": 0, "win_rate": 0, "avg_pnl": 0}

    eq = np.array(equity_curve, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]

    sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Trade-level
    pnls = [t.pnl() for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(pnls) if pnls else 0
    total_ret = (eq[-1] - CAPITAL) / CAPITAL * 100

    return {
        "n_trades": len(trades),
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_ret, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate": round(wr, 3),
        "avg_pnl": round(np.mean(pnls), 3),
        "total_pnl": round(sum(pnls), 2),
        "final_equity": round(eq[-1], 2),
    }


def permutation_test(trades, equity_curve, trade_dates, n_perms=1000):
    """Random-entry permutation test: pick random dates + random tickers,
    compute 10-day returns, compare to actual strategy Sharpe."""
    if len(trades) < 5:
        return 1.0

    # Actual trade-level returns
    actual_trade_rets = np.array([t.pnl() / (t.entry_price * t.shares + 1e-10) for t in trades])
    actual_sharpe = np.mean(actual_trade_rets) / (np.std(actual_trade_rets) + 1e-10)
    n_trades = len(trades)

    # Collect tickers used
    tickers_used = list(set(t.ticker for t in trades))

    # Valid date indices for random entry (need HOLD_DAYS forward room)
    valid_dates = trade_dates[:-HOLD_DAYS - 1]
    if len(valid_dates) < 20:
        return 1.0

    rng = np.random.default_rng(42)
    count_ge = 0

    for _ in range(n_perms):
        # Pick random entry dates and tickers
        rand_date_indices = rng.choice(len(valid_dates), size=n_trades, replace=True)
        rand_tickers = rng.choice(tickers_used, size=n_trades, replace=True)

        perm_rets = []
        for di, ticker in zip(rand_date_indices, rand_tickers):
            entry_date = valid_dates[di]
            entry_idx = dates.get_loc(entry_date)
            exit_idx = min(entry_idx + HOLD_DAYS, len(dates) - 1)
            exit_date = dates[exit_idx]
            ep = close.loc[entry_date, ticker]
            xp = close.loc[exit_date, ticker]
            if pd.notna(ep) and pd.notna(xp) and ep > 0:
                gross_ret = (xp - ep) / ep
                slip = 2 * SLIPPAGE_BPS / 10000
                perm_rets.append(gross_ret - slip)

        if len(perm_rets) < 5:
            continue
        perm_rets = np.array(perm_rets)
        perm_sharpe = np.mean(perm_rets) / (np.std(perm_rets) + 1e-10)
        if perm_sharpe >= actual_sharpe:
            count_ge += 1

    return count_ge / n_perms


def regime_gap(trades, trade_dates):
    """Compute Sharpe in bull vs bear regimes (SPY 20d return > 0 = bull)."""
    if len(trades) < 5:
        return 999.0

    bull_pnl, bear_pnl = [], []
    for t in trades:
        idx = dates.get_loc(t.entry_date)
        spy_r = spy_ret_20.iloc[idx] if idx < len(spy_ret_20) else 0
        if pd.isna(spy_r):
            continue
        pnl = t.pnl()
        if spy_r > 0:
            bull_pnl.append(pnl)
        else:
            bear_pnl.append(pnl)

    def sharpe_from_pnls(pnls):
        if len(pnls) < 3:
            return 0
        arr = np.array(pnls)
        return np.mean(arr) / (np.std(arr) + 1e-10)

    s_bull = sharpe_from_pnls(bull_pnl)
    s_bear = sharpe_from_pnls(bear_pnl)
    denom = max(abs(s_bull), abs(s_bear), 1e-10)
    gap = abs(s_bull - s_bear) / denom
    return round(gap, 3)


# ── Run All Variants ────────────────────────────────────────────────

VARIANTS = {
    "A_WeakSectorDip": variant_a_signal,
    "B_StrongSectorDip": variant_b_signal,
    "C_CrossSectorDivergence": variant_c_signal,
    "D_SectorBreadthDip": variant_d_signal,
    "E_DefensiveRotation": variant_e_signal,
    "F_SectorMRPairs": variant_f_signal,
}

results = {}

for name, sig_func in VARIANTS.items():
    print(f"\n{'='*60}")
    print(f"Running Variant {name}...")
    trades, eq_curve, td = run_backtest(sig_func, name)
    metrics = compute_metrics(trades, eq_curve, td)

    # 5-gate validation
    g1_sharpe = metrics["sharpe"] > 0.5
    g2_perm_p = permutation_test(trades, eq_curve, td) if metrics["n_trades"] >= 5 else 1.0
    g2_pass = g2_perm_p < 0.05
    g3_rgap = regime_gap(trades, td)
    g3_pass = g3_rgap < 0.5
    g4_dd = metrics["max_dd_pct"] > -50
    g5_trades = metrics["n_trades"] >= 20

    gates = {
        "G1_Sharpe_gt_0.5": g1_sharpe,
        "G2_PermTest_p_lt_0.05": g2_pass,
        "G2_perm_p_value": round(g2_perm_p, 4) if isinstance(g2_perm_p, float) else g2_perm_p,
        "G3_RegimeGap_lt_0.5": g3_pass,
        "G3_regime_gap": g3_rgap,
        "G4_MaxDD_gt_neg50": g4_dd,
        "G5_Trades_gte_20": g5_trades,
    }
    gates_passed = sum([g1_sharpe, g2_pass, g3_pass, g4_dd, g5_trades])
    all_pass = gates_passed == 5

    results[name] = {
        "metrics": metrics,
        "gates": gates,
        "gates_passed": f"{gates_passed}/5",
        "PASS": all_pass,
    }

    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
          f"Return: {metrics['total_return_pct']}%, MaxDD: {metrics['max_dd_pct']}%, "
          f"WR: {metrics['win_rate']}")
    print(f"  Gates: {gates_passed}/5 {'PASS' if all_pass else 'FAIL'}")
    for gname, gval in gates.items():
        print(f"    {gname}: {gval}")

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, r in results.items():
    m = r["metrics"]
    status = "PASS" if r["PASS"] else "FAIL"
    print(f"{name:30s} | Sharpe {m['sharpe']:6.3f} | Ret {m['total_return_pct']:7.2f}% | "
          f"DD {m['max_dd_pct']:7.2f}% | WR {m['win_rate']:.1%} | "
          f"Trades {m['n_trades']:3d} | {r['gates_passed']} {status}")

# ── Save Results ────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/sector_rotation_quality_results.json")
# Convert for JSON serialization
out = {
    "generated": dt.datetime.now().isoformat(),
    "parameters": {
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": SLIPPAGE_BPS,
        "hold_days": HOLD_DAYS,
        "period": f"{START} to {END}",
        "universe_size": len(ALL_TICKERS),
    },
    "variants": results,
}
output_path.write_text(json.dumps(out, indent=2, default=str))
print(f"\nResults saved to {output_path}")
