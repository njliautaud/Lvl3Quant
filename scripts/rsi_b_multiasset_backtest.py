#!/usr/bin/env python3
"""
RSI(5) Mean Reversion (Variant B) — Multi-Asset Expansion Backtest
==================================================================

Tests the proven RSI B signal (RSI(5)<20 + price>200-SMA, exit RSI(5)>50 or 10d)
across multiple asset classes and universes.

Variants:
  A) Sector ETFs (11 SPDR sectors)
  B) Index ETFs (QQQ, SPY, IWM, DIA)
  C) Commodity ETFs (GLD, SLV, USO, UNG, DBA)
  D) Expanded Growth (original 24 + 6 mega-cap additions)
  E) Multi-Asset Combined (all of the above, max 3 concurrent)
  F) Cherry-Pick (top 5 assets by per-asset Sharpe)

5-gate validation per variant:
  G1: Sharpe > 0.5
  G2: Win Rate > 55%
  G3: Profit Factor > 1.3
  G4: Max DD < -25%
  G5: Perm test p-value < 0.10

OOT: Jan 2022 – Jul 2026. $669 account. 0.02% slippage each way.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 669.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
START = "2021-01-01"   # extra lookback for 200-SMA
END = "2026-07-30"
OOT_START = "2022-01-01"
MAX_CONCURRENT = 3
N_PERM = 1000

# ── Asset Universes ──────────────────────────────────────────────────────
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
INDEX_ETFS = ["QQQ", "SPY", "IWM", "DIA"]
COMMODITY_ETFS = ["GLD", "SLV", "USO", "UNG", "DBA"]
GROWTH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "COIN",
    "RBLX", "UBER", "LYFT", "DDOG", "TTD", "SHOP", "NET", "ROKU",
    "AVGO", "ORCL", "ADBE", "INTC", "MU", "QCOM",
]

ALL_TICKERS = sorted(set(SECTOR_ETFS + INDEX_ETFS + COMMODITY_ETFS + GROWTH_STOCKS + ["SPY"]))

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data for all tickers ...")
raw = yf.download(ALL_TICKERS, start=START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(ALL_TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_TICKERS}
spy_close = closes.get("SPY", pd.Series(dtype=float))

loaded = sum(1 for v in closes.values() if len(v) > 200)
print(f"  Tickers with sufficient data: {loaded}/{len(ALL_TICKERS)}")
print(f"  SPY rows: {len(spy_close)}")


# ── Indicator Helpers ─────────────────────────────────────────────────────
def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def sma(series, period):
    return series.rolling(period).mean()


# ── Pre-compute indicators for all tickers ──────────────────────────────
indicators = {}
for t in ALL_TICKERS:
    c = closes[t]
    if len(c) < 250:
        print(f"  Skipping {t}: only {len(c)} bars")
        continue
    ind = pd.DataFrame(index=c.index)
    ind["close"] = c
    ind["sma200"] = sma(c, 200)
    ind["rsi5"] = rsi(c, 5)
    ind["above_200sma"] = c > ind["sma200"]
    indicators[t] = ind.dropna(subset=["sma200"])

spy_sma200 = sma(spy_close, 200)
spy_regime = (spy_close > spy_sma200).reindex(spy_close.index).fillna(False)

print(f"  Tickers with indicators: {len(indicators)}")


# ── Core Backtest Engine (multi-position aware) ─────────────────────────
def gen_signals(universe):
    """Generate RSI B signals for a given universe."""
    signals = []
    for t in universe:
        ind = indicators.get(t)
        if ind is None:
            continue
        mask = (ind["rsi5"] < 20) & ind["above_200sma"]
        for dt in ind.index[mask]:
            if str(dt.date()) >= OOT_START:
                signals.append((dt, t, float(ind.loc[dt, "close"])))
    return sorted(signals, key=lambda x: x[0])


def run_backtest(signals, capital=CAPITAL, slippage=SLIPPAGE_PCT, max_concurrent=MAX_CONCURRENT):
    """
    Run backtest with max_concurrent positions.
    Equal-weight sizing across concurrent positions.
    Returns (per-trade returns array, equity_curve list, realized trades list).
    """
    if not signals:
        return np.array([]), [], []

    # Sort chronologically
    signals = sorted(signals, key=lambda x: x[0])

    equity = capital
    equity_curve = [(OOT_START, capital)]
    realized = []
    open_positions = []  # list of dicts with exit_date

    for date, ticker, entry_price in signals:
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
        if date_str < OOT_START:
            continue

        # Close any positions that have exited by this date
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= date_str:
                # Position already closed (accounted for at open time)
                pass
            else:
                still_open.append(pos)
        open_positions = still_open

        # Check concurrent limit
        if len(open_positions) >= max_concurrent:
            continue

        # Find exit
        ind = indicators.get(ticker)
        if ind is None or date not in ind.index:
            continue
        loc = ind.index.get_loc(date)

        exit_idx = None
        for i in range(loc + 1, min(loc + 11, len(ind))):
            if ind["rsi5"].iloc[i] > 50:
                exit_idx = i
                break
        if exit_idx is None:
            exit_idx = min(loc + 10, len(ind) - 1)
        if exit_idx <= loc:
            exit_idx = min(loc + 1, len(ind) - 1)

        exit_date_str = str(ind.index[exit_idx].date())

        # Position sizing: equal weight across max_concurrent slots
        alloc = equity / max_concurrent
        actual_entry = entry_price * (1 + slippage)
        exit_price = float(ind["close"].iloc[exit_idx]) * (1 - slippage)

        shares = int(alloc / actual_entry)
        if shares < 1:
            # For high-priced assets, allow fractional (simulate 1 share if affordable)
            if actual_entry <= equity:
                shares = 1
            else:
                continue

        pnl = shares * (exit_price - actual_entry)
        ret = pnl / (shares * actual_entry)
        hold_days = (ind.index[exit_idx] - ind.index[loc]).days

        # Determine regime
        regime = "Bull" if spy_regime.get(ind.index[loc], False) else "Bear"

        trade = {
            "ticker": ticker,
            "entry_date": date_str,
            "exit_date": exit_date_str,
            "entry_price": round(actual_entry, 4),
            "exit_price": round(exit_price, 4),
            "shares": shares,
            "pnl": round(pnl, 2),
            "return": round(ret, 6),
            "hold_days": hold_days,
            "regime": regime,
        }

        equity += pnl
        equity_curve.append((exit_date_str, round(equity, 2)))
        realized.append(trade)
        open_positions.append({"ticker": ticker, "exit_date": exit_date_str})

    returns = np.array([t["return"] for t in realized])
    return returns, equity_curve, realized


def calc_metrics(returns, equity_curve, realized, capital=CAPITAL):
    """Full metrics dict."""
    if len(returns) < 2:
        return {
            "n_trades": len(returns), "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0,
            "total_return_pct": 0.0, "final_equity": capital, "avg_hold_days": 0.0,
        }

    n = len(returns)
    wins = int((returns > 0).sum())
    wr = wins / n

    avg_hold = np.mean([t["hold_days"] for t in realized]) if realized else 8.0
    trades_per_year = max(1, 252 / max(avg_hold, 1))
    ann_factor = np.sqrt(trades_per_year)

    mean_r = returns.mean()
    std_r = returns.std() if returns.std() > 0 else 1e-9
    sharpe = (mean_r / std_r) * ann_factor

    downside = returns[returns < 0]
    down_std = downside.std() if len(downside) > 0 and downside.std() > 0 else 1e-9
    sortino = (mean_r / down_std) * ann_factor

    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    eq_vals = [e[1] for e in equity_curve]
    peak = eq_vals[0]
    max_dd = 0
    for v in eq_vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    final_eq = eq_vals[-1]
    total_return_pct = ((final_eq - capital) / capital) * 100

    # Regime breakdown
    bull_rets = np.array([t["return"] for t in realized if t["regime"] == "Bull"])
    bear_rets = np.array([t["return"] for t in realized if t["regime"] == "Bear"])

    def regime_sharpe(rets):
        if len(rets) < 2:
            return 0.0
        s = rets.std()
        if s < 1e-12:
            return 0.0
        return float((rets.mean() / s) * ann_factor)

    return {
        "n_trades": int(n),
        "win_rate": round(float(wr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "final_equity": round(float(final_eq), 2),
        "avg_hold_days": round(float(avg_hold), 1),
        "bull_trades": int(len(bull_rets)),
        "bear_trades": int(len(bear_rets)),
        "bull_sharpe": round(regime_sharpe(bull_rets), 3),
        "bear_sharpe": round(regime_sharpe(bear_rets), 3),
        "bull_wr": round(float((bull_rets > 0).mean()), 4) if len(bull_rets) > 0 else 0.0,
        "bear_wr": round(float((bear_rets > 0).mean()), 4) if len(bear_rets) > 0 else 0.0,
    }


def calc_sharpe_quick(returns):
    """Quick Sharpe for permutation tests."""
    if len(returns) < 2:
        return 0.0
    std_r = returns.std()
    if std_r < 1e-12:
        return 0.0
    avg_hold = 8.0
    ann_factor = np.sqrt(252 / avg_hold)
    return float((returns.mean() / std_r) * ann_factor)


# ── 5-Gate Validation ────────────────────────────────────────────────────
def five_gate_validation(returns, metrics, signals, universe_name):
    """Run 5-gate validation. Returns dict with pass/fail per gate."""
    gates = {}

    # G1: Sharpe > 0.5
    gates["G1_sharpe"] = {
        "criterion": "Sharpe > 0.5",
        "value": metrics["sharpe"],
        "passed": metrics["sharpe"] > 0.5,
    }

    # G2: Win Rate > 55%
    gates["G2_win_rate"] = {
        "criterion": "Win Rate > 55%",
        "value": round(metrics["win_rate"] * 100, 2),
        "passed": metrics["win_rate"] > 0.55,
    }

    # G3: Profit Factor > 1.3
    gates["G3_profit_factor"] = {
        "criterion": "Profit Factor > 1.3",
        "value": metrics["profit_factor"],
        "passed": metrics["profit_factor"] > 1.3,
    }

    # G4: Max DD < -25%
    gates["G4_max_dd"] = {
        "criterion": "Max DD > -25%",
        "value": metrics["max_dd_pct"],
        "passed": metrics["max_dd_pct"] > -25.0,
    }

    # G5: Permutation test p-value < 0.10
    if len(returns) >= 3 and len(signals) >= 3:
        real_sharpe = metrics["sharpe"]

        # Build oversold windows per ticker for permutation
        oot_dates_per_ticker = {}
        for t in set(s[1] for s in signals):
            ind = indicators.get(t)
            if ind is not None:
                valid = ind.index[ind.index >= OOT_START]
                if len(valid) > 15:
                    oot_dates_per_ticker[t] = valid[:-10]

        # Get the tickers from realized signals
        signal_tickers = [s[1] for s in signals if str(s[0].date()) >= OOT_START and s[1] in oot_dates_per_ticker]

        perm_sharpes = np.zeros(N_PERM)
        for i in range(N_PERM):
            shuffled = []
            for ticker in signal_tickers:
                valid_dates = oot_dates_per_ticker[ticker]
                rand_idx = np.random.randint(0, len(valid_dates))
                rand_date = valid_dates[rand_idx]
                ind = indicators[ticker]
                ep = ind.loc[rand_date, "close"]
                if isinstance(ep, pd.Series):
                    ep = ep.iloc[0]
                shuffled.append((rand_date, ticker, float(ep)))

            perm_ret, _, _ = run_backtest(shuffled)
            perm_sharpes[i] = calc_sharpe_quick(perm_ret)

        p_value = float(np.mean(perm_sharpes >= real_sharpe))
        percentile = float(np.mean(perm_sharpes < real_sharpe) * 100)
    else:
        p_value = 1.0
        percentile = 0.0

    gates["G5_perm_test"] = {
        "criterion": "Perm test p-value < 0.10",
        "p_value": round(p_value, 4),
        "percentile": round(percentile, 1),
        "passed": p_value < 0.10,
    }

    gates_passed = sum(1 for g in gates.values() if g["passed"])
    return {
        "gates": gates,
        "gates_passed": gates_passed,
        "total_gates": 5,
        "score": f"{gates_passed}/5",
        "verdict": "STRONG" if gates_passed >= 4 else "ACCEPTABLE" if gates_passed >= 3 else "WEAK" if gates_passed >= 2 else "REJECT",
    }


# ── Per-Asset Breakdown ──────────────────────────────────────────────────
def per_asset_breakdown(realized):
    """Compute per-asset Sharpe, WR, n_trades."""
    asset_stats = {}
    tickers = set(t["ticker"] for t in realized)
    for tk in tickers:
        tk_trades = [t for t in realized if t["ticker"] == tk]
        tk_rets = np.array([t["return"] for t in tk_trades])
        n = len(tk_rets)
        if n < 2:
            sharpe = 0.0
        else:
            s = tk_rets.std()
            sharpe = float((tk_rets.mean() / s) * np.sqrt(252 / 8)) if s > 1e-12 else 0.0
        wr = float((tk_rets > 0).mean()) if n > 0 else 0.0
        total_ret = float(tk_rets.sum())
        asset_stats[tk] = {
            "n_trades": n,
            "sharpe": round(sharpe, 3),
            "win_rate": round(wr, 4),
            "total_return": round(total_ret, 6),
            "avg_return": round(float(tk_rets.mean()), 6) if n > 0 else 0.0,
        }
    return asset_stats


# ══════════════════════════════════════════════════════════════════════════
#  VARIANT A: Sector ETFs
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VARIANT A: SECTOR ETF RSI B")
print("  Universe:", ", ".join(SECTOR_ETFS))
print("=" * 70)

signals_A = gen_signals(SECTOR_ETFS)
returns_A, eq_A, trades_A = run_backtest(signals_A)
metrics_A = calc_metrics(returns_A, eq_A, trades_A)
assets_A = per_asset_breakdown(trades_A)

print(f"  Signals generated: {len(signals_A)}")
print(f"  Trades taken: {metrics_A['n_trades']}")
print(f"  Sharpe: {metrics_A['sharpe']:.3f}  Sortino: {metrics_A['sortino']:.3f}")
print(f"  WR: {metrics_A['win_rate']:.1%}  PF: {metrics_A['profit_factor']:.2f}")
print(f"  Max DD: {metrics_A['max_dd_pct']:.1f}%  Return: {metrics_A['total_return_pct']:.1f}%")
print(f"  Bull Sharpe: {metrics_A['bull_sharpe']:.3f}  Bear Sharpe: {metrics_A['bear_sharpe']:.3f}")
print(f"  Per-asset breakdown:")
for tk, st in sorted(assets_A.items(), key=lambda x: -x[1]["sharpe"]):
    print(f"    {tk:5s}: {st['n_trades']:3d} trades, Sharpe {st['sharpe']:+.3f}, WR {st['win_rate']:.0%}")

print("\n  Running 5-gate validation ...")
validation_A = five_gate_validation(returns_A, metrics_A, signals_A, "Sector ETFs")
for gname, gdata in validation_A["gates"].items():
    status = "PASS" if gdata["passed"] else "FAIL"
    print(f"    [{status}] {gname}: {gdata['criterion']} -> {gdata.get('value', gdata.get('p_value', 'N/A'))}")
print(f"  Score: {validation_A['score']}  Verdict: {validation_A['verdict']}")


# ══════════════════════════════════════════════════════════════════════════
#  VARIANT B: Index ETFs
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VARIANT B: INDEX ETF RSI B")
print("  Universe:", ", ".join(INDEX_ETFS))
print("=" * 70)

signals_B = gen_signals(INDEX_ETFS)
returns_B, eq_B, trades_B = run_backtest(signals_B)
metrics_B = calc_metrics(returns_B, eq_B, trades_B)
assets_B = per_asset_breakdown(trades_B)

print(f"  Signals generated: {len(signals_B)}")
print(f"  Trades taken: {metrics_B['n_trades']}")
print(f"  Sharpe: {metrics_B['sharpe']:.3f}  Sortino: {metrics_B['sortino']:.3f}")
print(f"  WR: {metrics_B['win_rate']:.1%}  PF: {metrics_B['profit_factor']:.2f}")
print(f"  Max DD: {metrics_B['max_dd_pct']:.1f}%  Return: {metrics_B['total_return_pct']:.1f}%")
print(f"  Bull Sharpe: {metrics_B['bull_sharpe']:.3f}  Bear Sharpe: {metrics_B['bear_sharpe']:.3f}")
for tk, st in sorted(assets_B.items(), key=lambda x: -x[1]["sharpe"]):
    print(f"    {tk:5s}: {st['n_trades']:3d} trades, Sharpe {st['sharpe']:+.3f}, WR {st['win_rate']:.0%}")

print("\n  Running 5-gate validation ...")
validation_B = five_gate_validation(returns_B, metrics_B, signals_B, "Index ETFs")
for gname, gdata in validation_B["gates"].items():
    status = "PASS" if gdata["passed"] else "FAIL"
    print(f"    [{status}] {gname}: {gdata['criterion']} -> {gdata.get('value', gdata.get('p_value', 'N/A'))}")
print(f"  Score: {validation_B['score']}  Verdict: {validation_B['verdict']}")


# ══════════════════════════════════════════════════════════════════════════
#  VARIANT C: Commodity ETFs
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VARIANT C: COMMODITY ETF RSI B")
print("  Universe:", ", ".join(COMMODITY_ETFS))
print("=" * 70)

signals_C = gen_signals(COMMODITY_ETFS)
returns_C, eq_C, trades_C = run_backtest(signals_C)
metrics_C = calc_metrics(returns_C, eq_C, trades_C)
assets_C = per_asset_breakdown(trades_C)

print(f"  Signals generated: {len(signals_C)}")
print(f"  Trades taken: {metrics_C['n_trades']}")
print(f"  Sharpe: {metrics_C['sharpe']:.3f}  Sortino: {metrics_C['sortino']:.3f}")
print(f"  WR: {metrics_C['win_rate']:.1%}  PF: {metrics_C['profit_factor']:.2f}")
print(f"  Max DD: {metrics_C['max_dd_pct']:.1f}%  Return: {metrics_C['total_return_pct']:.1f}%")
print(f"  Bull Sharpe: {metrics_C['bull_sharpe']:.3f}  Bear Sharpe: {metrics_C['bear_sharpe']:.3f}")
for tk, st in sorted(assets_C.items(), key=lambda x: -x[1]["sharpe"]):
    print(f"    {tk:5s}: {st['n_trades']:3d} trades, Sharpe {st['sharpe']:+.3f}, WR {st['win_rate']:.0%}")

print("\n  Running 5-gate validation ...")
validation_C = five_gate_validation(returns_C, metrics_C, signals_C, "Commodity ETFs")
for gname, gdata in validation_C["gates"].items():
    status = "PASS" if gdata["passed"] else "FAIL"
    print(f"    [{status}] {gname}: {gdata['criterion']} -> {gdata.get('value', gdata.get('p_value', 'N/A'))}")
print(f"  Score: {validation_C['score']}  Verdict: {validation_C['verdict']}")


# ══════════════════════════════════════════════════════════════════════════
#  VARIANT D: Expanded Growth Stocks
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VARIANT D: EXPANDED GROWTH STOCKS RSI B")
print(f"  Universe: {len(GROWTH_STOCKS)} stocks")
print("=" * 70)

signals_D = gen_signals(GROWTH_STOCKS)
returns_D, eq_D, trades_D = run_backtest(signals_D)
metrics_D = calc_metrics(returns_D, eq_D, trades_D)
assets_D = per_asset_breakdown(trades_D)

print(f"  Signals generated: {len(signals_D)}")
print(f"  Trades taken: {metrics_D['n_trades']}")
print(f"  Sharpe: {metrics_D['sharpe']:.3f}  Sortino: {metrics_D['sortino']:.3f}")
print(f"  WR: {metrics_D['win_rate']:.1%}  PF: {metrics_D['profit_factor']:.2f}")
print(f"  Max DD: {metrics_D['max_dd_pct']:.1f}%  Return: {metrics_D['total_return_pct']:.1f}%")
print(f"  Bull Sharpe: {metrics_D['bull_sharpe']:.3f}  Bear Sharpe: {metrics_D['bear_sharpe']:.3f}")
print(f"  Top 10 assets by Sharpe:")
sorted_assets_D = sorted(assets_D.items(), key=lambda x: -x[1]["sharpe"])
for tk, st in sorted_assets_D[:10]:
    print(f"    {tk:5s}: {st['n_trades']:3d} trades, Sharpe {st['sharpe']:+.3f}, WR {st['win_rate']:.0%}")

print("\n  Running 5-gate validation ...")
validation_D = five_gate_validation(returns_D, metrics_D, signals_D, "Expanded Growth")
for gname, gdata in validation_D["gates"].items():
    status = "PASS" if gdata["passed"] else "FAIL"
    print(f"    [{status}] {gname}: {gdata['criterion']} -> {gdata.get('value', gdata.get('p_value', 'N/A'))}")
print(f"  Score: {validation_D['score']}  Verdict: {validation_D['verdict']}")


# ══════════════════════════════════════════════════════════════════════════
#  VARIANT E: Multi-Asset Combined (all universes, max 3 concurrent)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VARIANT E: MULTI-ASSET COMBINED RSI B")
all_universe = sorted(set(SECTOR_ETFS + INDEX_ETFS + COMMODITY_ETFS + GROWTH_STOCKS))
print(f"  Universe: {len(all_universe)} assets across all classes")
print("=" * 70)

signals_E = gen_signals(all_universe)
returns_E, eq_E, trades_E = run_backtest(signals_E, max_concurrent=MAX_CONCURRENT)
metrics_E = calc_metrics(returns_E, eq_E, trades_E)
assets_E = per_asset_breakdown(trades_E)

print(f"  Signals generated: {len(signals_E)}")
print(f"  Trades taken: {metrics_E['n_trades']}")
print(f"  Sharpe: {metrics_E['sharpe']:.3f}  Sortino: {metrics_E['sortino']:.3f}")
print(f"  WR: {metrics_E['win_rate']:.1%}  PF: {metrics_E['profit_factor']:.2f}")
print(f"  Max DD: {metrics_E['max_dd_pct']:.1f}%  Return: {metrics_E['total_return_pct']:.1f}%")
print(f"  Bull Sharpe: {metrics_E['bull_sharpe']:.3f}  Bear Sharpe: {metrics_E['bear_sharpe']:.3f}")
print(f"  Top 10 assets traded:")
sorted_assets_E = sorted(assets_E.items(), key=lambda x: -x[1]["n_trades"])
for tk, st in sorted_assets_E[:10]:
    print(f"    {tk:5s}: {st['n_trades']:3d} trades, Sharpe {st['sharpe']:+.3f}, WR {st['win_rate']:.0%}")

# Asset class breakdown
class_map = {}
for t in SECTOR_ETFS:
    class_map[t] = "Sector ETF"
for t in INDEX_ETFS:
    class_map[t] = "Index ETF"
for t in COMMODITY_ETFS:
    class_map[t] = "Commodity ETF"
for t in GROWTH_STOCKS:
    class_map[t] = "Growth Stock"

class_trades = {}
for t in trades_E:
    cls = class_map.get(t["ticker"], "Unknown")
    if cls not in class_trades:
        class_trades[cls] = []
    class_trades[cls].append(t["return"])

print(f"  By asset class:")
class_breakdown = {}
for cls, rets in sorted(class_trades.items()):
    rets_arr = np.array(rets)
    n = len(rets_arr)
    wr = float((rets_arr > 0).mean()) if n > 0 else 0.0
    avg_r = float(rets_arr.mean()) if n > 0 else 0.0
    class_breakdown[cls] = {"n_trades": n, "win_rate": round(wr, 4), "avg_return": round(avg_r, 6)}
    print(f"    {cls:15s}: {n:3d} trades, WR {wr:.0%}, avg ret {avg_r:+.4%}")

print("\n  Running 5-gate validation ...")
validation_E = five_gate_validation(returns_E, metrics_E, signals_E, "Multi-Asset Combined")
for gname, gdata in validation_E["gates"].items():
    status = "PASS" if gdata["passed"] else "FAIL"
    print(f"    [{status}] {gname}: {gdata['criterion']} -> {gdata.get('value', gdata.get('p_value', 'N/A'))}")
print(f"  Score: {validation_E['score']}  Verdict: {validation_E['verdict']}")


# ══════════════════════════════════════════════════════════════════════════
#  VARIANT F: Cherry-Pick (top 5 by per-asset Sharpe from combined)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VARIANT F: CHERRY-PICK RSI B (top 5 assets by Sharpe)")
print("=" * 70)

# Use per-asset stats from ALL variants to find best assets
all_asset_stats = {}
for variant_assets in [assets_A, assets_B, assets_C, assets_D]:
    for tk, st in variant_assets.items():
        if st["n_trades"] >= 3:  # Require minimum 3 trades for reliability
            if tk not in all_asset_stats or st["sharpe"] > all_asset_stats[tk]["sharpe"]:
                all_asset_stats[tk] = st

# Sort by Sharpe, take top 5
top5 = sorted(all_asset_stats.items(), key=lambda x: -x[1]["sharpe"])[:5]
top5_tickers = [t[0] for t in top5]

print(f"  Top 5 assets selected:")
for tk, st in top5:
    print(f"    {tk:5s}: Sharpe {st['sharpe']:+.3f}, WR {st['win_rate']:.0%}, {st['n_trades']} trades")

signals_F = gen_signals(top5_tickers)
returns_F, eq_F, trades_F = run_backtest(signals_F)
metrics_F = calc_metrics(returns_F, eq_F, trades_F)
assets_F = per_asset_breakdown(trades_F)

print(f"\n  Trades taken: {metrics_F['n_trades']}")
print(f"  Sharpe: {metrics_F['sharpe']:.3f}  Sortino: {metrics_F['sortino']:.3f}")
print(f"  WR: {metrics_F['win_rate']:.1%}  PF: {metrics_F['profit_factor']:.2f}")
print(f"  Max DD: {metrics_F['max_dd_pct']:.1f}%  Return: {metrics_F['total_return_pct']:.1f}%")
print(f"  Bull Sharpe: {metrics_F['bull_sharpe']:.3f}  Bear Sharpe: {metrics_F['bear_sharpe']:.3f}")

print("\n  Running 5-gate validation ...")
validation_F = five_gate_validation(returns_F, metrics_F, signals_F, "Cherry-Pick")
for gname, gdata in validation_F["gates"].items():
    status = "PASS" if gdata["passed"] else "FAIL"
    print(f"    [{status}] {gname}: {gdata['criterion']} -> {gdata.get('value', gdata.get('p_value', 'N/A'))}")
print(f"  Score: {validation_F['score']}  Verdict: {validation_F['verdict']}")

# NOTE: Cherry-pick has look-ahead bias (selecting best assets after seeing results).
# We flag this explicitly.
print("\n  WARNING: Cherry-pick variant has LOOK-AHEAD BIAS (selected after seeing results).")
print("  Use only as upper-bound estimate. Do NOT deploy without fresh OOT validation.")


# ══════════════════════════════════════════════════════════════════════════
#  OVERALL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OVERALL COMPARISON — RSI B MULTI-ASSET EXPANSION")
print("=" * 70)

variants = {
    "A_sector_etfs": {"metrics": metrics_A, "validation": validation_A, "assets": assets_A},
    "B_index_etfs": {"metrics": metrics_B, "validation": validation_B, "assets": assets_B},
    "C_commodity_etfs": {"metrics": metrics_C, "validation": validation_C, "assets": assets_C},
    "D_expanded_growth": {"metrics": metrics_D, "validation": validation_D, "assets": assets_D},
    "E_multi_asset": {"metrics": metrics_E, "validation": validation_E, "assets": assets_E,
                      "class_breakdown": class_breakdown},
    "F_cherry_pick": {"metrics": metrics_F, "validation": validation_F, "assets": assets_F,
                      "top5_tickers": top5_tickers, "look_ahead_bias": True},
}

print(f"  {'Variant':25s} {'Trades':>6s} {'Sharpe':>7s} {'Sortino':>8s} {'WR':>6s} {'PF':>6s} {'DD':>7s} {'Ret%':>7s} {'Gates':>6s} {'Verdict':>10s}")
print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*10}")

for vname, vdata in variants.items():
    m = vdata["metrics"]
    v = vdata["validation"]
    print(f"  {vname:25s} {m['n_trades']:6d} {m['sharpe']:7.3f} {m['sortino']:8.3f} "
          f"{m['win_rate']:5.0%} {m['profit_factor']:6.2f} {m['max_dd_pct']:6.1f}% "
          f"{m['total_return_pct']:6.1f}% {v['score']:>6s} {v['verdict']:>10s}")

# Determine best variant (excluding F due to look-ahead bias)
non_biased = {k: v for k, v in variants.items() if k != "F_cherry_pick"}
best_variant = max(non_biased.items(), key=lambda x: x[1]["metrics"]["sharpe"])
print(f"\n  Best variant (excl. cherry-pick): {best_variant[0]} — Sharpe {best_variant[1]['metrics']['sharpe']:.3f}")

# Regime analysis across all variants
print(f"\n  Regime Analysis (Bull vs Bear Sharpe):")
for vname, vdata in variants.items():
    m = vdata["metrics"]
    print(f"    {vname:25s}: Bull {m['bull_sharpe']:+.3f}  Bear {m['bear_sharpe']:+.3f}")


# ══════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════
results = {
    "metadata": {
        "strategy": "RSI(5) Mean Reversion Variant B — Multi-Asset Expansion",
        "entry_rule": "RSI(5) < 20 AND price > 200-SMA",
        "exit_rule": "RSI(5) > 50 OR 10 trading days max",
        "oot_period": f"{OOT_START} to {END}",
        "capital": CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "max_concurrent": MAX_CONCURRENT,
        "n_permutations": N_PERM,
        "run_timestamp": datetime.now().isoformat(),
    },
    "variants": {},
    "comparison": {
        "best_variant_excl_cherrypick": best_variant[0],
        "best_sharpe": best_variant[1]["metrics"]["sharpe"],
    },
}

for vname, vdata in variants.items():
    results["variants"][vname] = {
        "metrics": vdata["metrics"],
        "validation": vdata["validation"],
        "per_asset": vdata["assets"],
    }
    if "class_breakdown" in vdata:
        results["variants"][vname]["class_breakdown"] = vdata["class_breakdown"]
    if "top5_tickers" in vdata:
        results["variants"][vname]["top5_tickers"] = vdata["top5_tickers"]
        results["variants"][vname]["look_ahead_bias_warning"] = True

output_path = Path("/home/jupiter/Lvl3Quant/data/rsi_b_multiasset_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("DONE.")
