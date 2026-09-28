#!/usr/bin/env python3
"""
Volume-Confirmed Mean Reversion on Quality Stocks
6 variants testing whether volume analysis improves dip-buying timing alpha.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
SLIPPAGE_BPS = 2
N_PERMUTATIONS = 1000
SPY_TICKER = "SPY"
RESULT_PATH = "/home/jupiter/Lvl3Quant/data/volume_price_mr_results.json"


# ── DATA ────────────────────────────────────────────────────────────────
def download_data():
    tickers = UNIVERSE + [SPY_TICKER]
    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    # Normalize columns: yfinance may return MultiIndex (Price, Ticker) or flat
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", level=0, axis=1)
        high = raw["High"] if "High" in raw.columns.get_level_values(0) else raw.xs("High", level=0, axis=1)
        low = raw["Low"] if "Low" in raw.columns.get_level_values(0) else raw.xs("Low", level=0, axis=1)
        volume = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else raw.xs("Volume", level=0, axis=1)
        opn = raw["Open"] if "Open" in raw.columns.get_level_values(0) else raw.xs("Open", level=0, axis=1)
    else:
        close = raw["Close"].to_frame() if "Close" in raw.columns else None
        high = raw["High"].to_frame()
        low = raw["Low"].to_frame()
        volume = raw["Volume"].to_frame()
        opn = raw["Open"].to_frame()
    return close, high, low, volume, opn


def compute_indicators(close, high, low, volume, opn):
    """Pre-compute indicators used across strategies."""
    ind = {}
    for tk in UNIVERSE:
        if tk not in close.columns:
            continue
        c = close[tk].dropna()
        h = high[tk].reindex(c.index)
        l = low[tk].reindex(c.index)
        v = volume[tk].reindex(c.index)
        o = opn[tk].reindex(c.index)

        d = pd.DataFrame(index=c.index)
        d["close"] = c
        d["high"] = h
        d["low"] = l
        d["open"] = o
        d["volume"] = v
        d["ret"] = c.pct_change()
        d["vol_ma20"] = v.rolling(20).mean()
        d["high20"] = c.rolling(20).max()
        d["low20"] = c.rolling(20).min()

        # RSI(14)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        d["rsi14"] = 100 - 100 / (1 + rs)

        # OBV
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
        d["obv"] = obv
        d["obv_ma20"] = obv.rolling(20).mean()

        # VWAP-like: 20-day rolling VWAP
        typical = (c + h + l) / 3
        tp_vol = typical * v
        d["vwap20"] = tp_vol.rolling(20).sum() / v.rolling(20).sum()
        d["vwap20_std"] = (c - d["vwap20"]).rolling(20).std()

        # Pct off 20-day high
        d["pct_off_high20"] = (c - d["high20"]) / d["high20"]

        ind[tk] = d.dropna()
    return ind


# ── SIGNAL GENERATORS ──────────────────────────────────────────────────
def signals_A(ind):
    """Capitulation volume: drop >3% AND volume > 2x 20d avg."""
    trades = []
    for tk, d in ind.items():
        mask = (d["ret"] < -0.03) & (d["volume"] > 2 * d["vol_ma20"])
        for dt in d.index[mask]:
            trades.append((dt, tk))
    return trades


def signals_B(ind):
    """Volume dry-up: down >5% from 20d high AND volume < 0.5x 20d avg."""
    trades = []
    for tk, d in ind.items():
        mask = (d["pct_off_high20"] < -0.05) & (d["volume"] < 0.5 * d["vol_ma20"])
        for dt in d.index[mask]:
            trades.append((dt, tk))
    return trades


def signals_C(ind):
    """Climax reversal: >3% drop on 2x vol, then next day green on lower vol."""
    trades = []
    for tk, d in ind.items():
        panic = (d["ret"] < -0.03) & (d["volume"] > 2 * d["vol_ma20"])
        panic_idx = d.index[panic]
        for dt in panic_idx:
            loc = d.index.get_loc(dt)
            if loc + 1 >= len(d.index):
                continue
            nxt = d.index[loc + 1]
            if d.loc[nxt, "close"] > d.loc[nxt, "open"] and d.loc[nxt, "volume"] < d.loc[dt, "volume"]:
                trades.append((nxt, tk))
    return trades


def signals_D(ind):
    """OBV divergence: price at 20d low but OBV above its 20d avg."""
    trades = []
    for tk, d in ind.items():
        at_low = d["close"] <= d["low20"] * 1.005  # within 0.5% of 20d low
        obv_strong = d["obv"] > d["obv_ma20"]
        mask = at_low & obv_strong
        for dt in d.index[mask]:
            trades.append((dt, tk))
    return trades


def signals_E(ind):
    """Volume-weighted dip: down >5% from high, negative 5d vol flow, RSI<40."""
    trades = []
    for tk, d in ind.items():
        # 5-day volume flow: sum of signed volume
        signed_vol = np.sign(d["ret"]) * d["volume"]
        vol_flow_5 = signed_vol.rolling(5).sum()
        mask = (d["pct_off_high20"] < -0.05) & (vol_flow_5 < 0) & (d["rsi14"] < 40)
        for dt in d.index[mask]:
            trades.append((dt, tk))
    return trades


def signals_F(ind):
    """VWAP reversion: >2 std below 20d VWAP AND RSI < 35."""
    trades = []
    for tk, d in ind.items():
        vwap_dist = (d["close"] - d["vwap20"]) / d["vwap20_std"].replace(0, np.nan)
        mask = (vwap_dist < -2) & (d["rsi14"] < 35)
        mask = mask.fillna(False)
        for dt in d.index[mask]:
            trades.append((dt, tk))
    return trades


VARIANTS = {
    "A_capitulation_volume": signals_A,
    "B_volume_dryup": signals_B,
    "C_climax_reversal": signals_C,
    "D_obv_divergence": signals_D,
    "E_volume_weighted_dip": signals_E,
    "F_vwap_reversion": signals_F,
}


# ── BACKTEST ENGINE ────────────────────────────────────────────────────
def run_backtest(signal_dates, close, hold_days=HOLD_DAYS):
    """
    Execute trades with position limits and slippage.
    Returns daily equity curve and trade list.
    """
    # Sort signals by date
    signal_dates.sort(key=lambda x: x[0])

    all_dates = close.index.sort_values()
    equity = CAPITAL
    equity_curve = pd.Series(dtype=float)
    active_trades = []  # list of (ticker, entry_date, entry_price, shares, exit_date)
    completed = []
    trade_returns = []

    for i, today in enumerate(all_dates):
        # Close expired trades
        still_active = []
        for tr in active_trades:
            tk, entry_dt, entry_px, shares, exit_dt = tr
            if today >= exit_dt:
                if today in close.index and tk in close.columns and not np.isnan(close.loc[today, tk]):
                    exit_px = close.loc[today, tk] * (1 - SLIPPAGE_BPS / 10000)
                else:
                    # Use last available price
                    sub = close[tk].loc[:today].dropna()
                    exit_px = sub.iloc[-1] * (1 - SLIPPAGE_BPS / 10000) if len(sub) > 0 else entry_px
                pnl = (exit_px - entry_px) * shares
                ret = (exit_px / entry_px) - 1
                equity += pnl
                trade_returns.append(ret)
                completed.append({
                    "ticker": tk,
                    "entry": str(entry_dt.date()),
                    "exit": str(today.date()),
                    "ret": round(ret * 100, 2),
                })
            else:
                still_active.append(tr)
        active_trades = still_active

        # Open new trades
        todays_signals = [(dt, tk) for dt, tk in signal_dates if dt == today]
        for _, tk in todays_signals:
            if len(active_trades) >= MAX_CONCURRENT:
                break
            if any(t[0] == tk for t in active_trades):
                continue  # already have position in this ticker
            if tk not in close.columns:
                continue
            px = close.loc[today, tk]
            if np.isnan(px) or px <= 0:
                continue
            entry_px = px * (1 + SLIPPAGE_BPS / 10000)
            shares_possible = MAX_PER_TRADE / entry_px
            shares = int(shares_possible)  # whole shares
            if shares < 1:
                continue
            cost = shares * entry_px
            if cost > equity:
                continue
            # Find exit date
            future = all_dates[all_dates > today]
            if len(future) >= hold_days:
                exit_dt = future[hold_days - 1]
            elif len(future) > 0:
                exit_dt = future[-1]
            else:
                continue
            equity -= cost
            equity += shares * entry_px  # we hold shares, track via mark-to-market at exit
            active_trades.append((tk, today, entry_px, shares, exit_dt))

        # Mark-to-market equity
        mtm = equity
        for tr in active_trades:
            tk, entry_dt, entry_px, shares, exit_dt = tr
            if tk in close.columns and today in close.index:
                cur = close.loc[today, tk]
                if not np.isnan(cur):
                    # equity already has cash back; compute unrealized
                    pass
        equity_curve[today] = equity  # simplified: track cash equity (realized only)

    # Force-close remaining
    last_day = all_dates[-1]
    for tr in active_trades:
        tk, entry_dt, entry_px, shares, exit_dt = tr
        sub = close[tk].loc[:last_day].dropna()
        if len(sub) > 0:
            exit_px = sub.iloc[-1] * (1 - SLIPPAGE_BPS / 10000)
            ret = (exit_px / entry_px) - 1
            trade_returns.append(ret)
            completed.append({
                "ticker": tk,
                "entry": str(entry_dt.date()),
                "exit": str(last_day.date()),
                "ret": round(ret * 100, 2),
            })

    return trade_returns, completed, equity_curve


def run_backtest_returns_only(signal_dates, close, hold_days=HOLD_DAYS):
    """Lightweight version returning just trade returns (for permutation test).
    Optimized: pre-index signals by date, use numpy arrays for price lookup."""
    # Build signal dict: date -> list of tickers
    sig_by_date = {}
    for dt, tk in signal_dates:
        sig_by_date.setdefault(dt, []).append(tk)

    all_dates = close.index.sort_values()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    active_trades = []  # (tk, entry_px, exit_idx)
    trade_returns = []
    n_dates = len(all_dates)

    # Pre-convert close to dict of arrays for fast lookup
    close_arrays = {}
    for col in close.columns:
        close_arrays[col] = close[col].values

    for i, today in enumerate(all_dates):
        # Close expired trades
        still_active = []
        for tk, entry_px, exit_idx in active_trades:
            if i >= exit_idx:
                exit_px = close_arrays[tk][i] * (1 - SLIPPAGE_BPS / 10000)
                if not np.isnan(exit_px) and exit_px > 0:
                    trade_returns.append((exit_px / entry_px) - 1)
            else:
                still_active.append((tk, entry_px, exit_idx))
        active_trades = still_active

        # Open new trades
        if today in sig_by_date:
            active_tickers = set(t[0] for t in active_trades)
            for tk in sig_by_date[today]:
                if len(active_trades) >= MAX_CONCURRENT:
                    break
                if tk in active_tickers:
                    continue
                if tk not in close_arrays:
                    continue
                px = close_arrays[tk][i]
                if np.isnan(px) or px <= 0:
                    continue
                entry_px = px * (1 + SLIPPAGE_BPS / 10000)
                shares = int(MAX_PER_TRADE / entry_px)
                if shares < 1:
                    continue
                exit_idx = min(i + hold_days, n_dates - 1)
                active_trades.append((tk, entry_px, exit_idx))
                active_tickers.add(tk)

    # Force-close remaining
    for tk, entry_px, exit_idx in active_trades:
        exit_px = close_arrays[tk][-1] * (1 - SLIPPAGE_BPS / 10000)
        if not np.isnan(exit_px) and exit_px > 0:
            trade_returns.append((exit_px / entry_px) - 1)

    return trade_returns


# ── METRICS ─────────────────────────────────────────────────────────────
def compute_metrics(trade_returns):
    if len(trade_returns) == 0:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "profit_factor": 0,
                "win_rate": 0, "avg_ret_pct": 0, "max_dd_pct": 0, "total_ret_pct": 0}
    r = np.array(trade_returns)
    n = len(r)
    avg = np.mean(r)
    std = np.std(r, ddof=1) if n > 1 else 1e-9
    downside = np.std(r[r < 0], ddof=1) if np.sum(r < 0) > 1 else 1e-9

    # Annualize: avg ~36 trades/yr assumption, scale by sqrt
    trades_per_year = max(n / 4.5, 1)  # ~4.5 yr period
    sharpe = (avg / std) * np.sqrt(trades_per_year) if std > 0 else 0
    sortino = (avg / downside) * np.sqrt(trades_per_year) if downside > 0 else 0

    gains = r[r > 0].sum() if np.any(r > 0) else 0
    losses = abs(r[r < 0].sum()) if np.any(r < 0) else 1e-9
    pf = gains / losses if losses > 0 else float("inf")
    wr = np.mean(r > 0) * 100

    # Max drawdown from cumulative returns
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    total_ret = (cum[-1] - 1) * 100

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 1),
        "avg_ret_pct": round(avg * 100, 3),
        "max_dd_pct": round(max_dd, 2),
        "total_ret_pct": round(total_ret, 2),
    }


# ── VALIDATION GATES ────────────────────────────────────────────────────
def permutation_test(real_returns, signal_dates, close, n_perms=N_PERMUTATIONS):
    """Shuffle entry dates, re-run backtest, compute p-value."""
    if len(real_returns) < 5:
        return 1.0
    real_sharpe = compute_metrics(real_returns)["sharpe"]
    all_dates = close.index.tolist()
    valid_dates = [d for d in all_dates if d >= close.index[30]]  # skip warmup

    count_better = 0
    tickers = list(set(tk for _, tk in signal_dates))
    # Cap signal count for permutations to keep runtime reasonable
    n_signals = min(len(signal_dates), 500)

    for _ in range(n_perms):
        random_dates = np.random.choice(valid_dates, size=n_signals, replace=True)
        random_tickers = np.random.choice(tickers, size=n_signals, replace=True)
        shuffled = list(zip(random_dates, random_tickers))
        perm_returns = run_backtest_returns_only(shuffled, close)
        if len(perm_returns) > 0:
            perm_sharpe = compute_metrics(perm_returns)["sharpe"]
            if perm_sharpe >= real_sharpe:
                count_better += 1

    return round((count_better + 1) / (n_perms + 1), 4)


def regime_test(trade_returns, signal_dates, completed_trades, close):
    """Split trades into bull/bear regimes using SPY vs 200-SMA."""
    if SPY_TICKER not in close.columns or len(completed_trades) == 0:
        return 0.0, 0.0, 1.0

    spy = close[SPY_TICKER].dropna()
    spy_sma200 = spy.rolling(200).mean()

    bull_rets, bear_rets = [], []
    for i, tr in enumerate(completed_trades):
        entry_date = pd.Timestamp(tr["entry"])
        if entry_date in spy.index and entry_date in spy_sma200.index:
            if spy.loc[entry_date] > spy_sma200.loc[entry_date]:
                bull_rets.append(trade_returns[i] if i < len(trade_returns) else 0)
            else:
                bear_rets.append(trade_returns[i] if i < len(trade_returns) else 0)

    bull_sharpe = compute_metrics(bull_rets)["sharpe"] if len(bull_rets) > 2 else 0
    bear_sharpe = compute_metrics(bear_rets)["sharpe"] if len(bear_rets) > 2 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return round(bull_sharpe, 3), round(bear_sharpe, 3), round(gap, 3)


def validate(metrics, p_value, regime_gap, label):
    """5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "permutation_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    passed = sum(gates.values())
    return gates, passed


# ── MAIN ────────────────────────────────────────────────────────────────
def main():
    close, high, low, volume, opn = download_data()
    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days, {len(close.columns)} tickers")

    ind = compute_indicators(close, high, low, volume, opn)
    print(f"Indicators computed for {len(ind)} tickers\n")

    results = {}

    for name, sig_fn in VARIANTS.items():
        print(f"{'='*60}")
        print(f"  Variant: {name}")
        print(f"{'='*60}")

        raw_signals = sig_fn(ind)
        print(f"  Raw signals: {len(raw_signals)}")

        trade_returns, completed, eq_curve = run_backtest(raw_signals, close)
        metrics = compute_metrics(trade_returns)
        print(f"  Trades executed: {metrics['n_trades']}")
        print(f"  Sharpe: {metrics['sharpe']}  Sortino: {metrics['sortino']}")
        print(f"  PF: {metrics['profit_factor']}  WR: {metrics['win_rate']}%")
        print(f"  Avg ret: {metrics['avg_ret_pct']}%  Total: {metrics['total_ret_pct']}%")
        print(f"  Max DD: {metrics['max_dd_pct']}%")

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...", end=" ", flush=True)
        p_val = permutation_test(trade_returns, raw_signals, close)
        print(f"p={p_val}")

        # Regime test
        bull_s, bear_s, regime_gap = regime_test(trade_returns, raw_signals, completed, close)
        print(f"  Bull Sharpe: {bull_s}  Bear Sharpe: {bear_s}  Gap: {regime_gap}")

        gates, n_passed = validate(metrics, p_val, regime_gap, name)
        print(f"  Gates passed: {n_passed}/5  {'PASS' if n_passed == 5 else 'FAIL'}")
        for g, v in gates.items():
            status = "PASS" if v else "FAIL"
            print(f"    {status}: {g}")

        # Top tickers
        ticker_counts = {}
        for t in completed:
            ticker_counts[t["ticker"]] = ticker_counts.get(t["ticker"], 0) + 1
        top_tickers = sorted(ticker_counts.items(), key=lambda x: -x[1])[:5]

        results[name] = {
            "metrics": metrics,
            "permutation_p": p_val,
            "regime": {"bull_sharpe": bull_s, "bear_sharpe": bear_s, "gap": regime_gap},
            "gates": {k: v for k, v in gates.items()},
            "gates_passed": f"{n_passed}/5",
            "verdict": "PASS" if n_passed == 5 else "FAIL",
            "top_tickers": top_tickers,
            "sample_trades": completed[:10],
        }
        print()

    # ── SUMMARY ──
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<28} {'Sharpe':>7} {'Sort':>7} {'PF':>6} {'WR%':>6} {'#Tr':>5} {'p-val':>6} {'RGap':>5} {'Gates':>6} {'Verdict':>8}")
    print("-" * 98)
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<28} {m['sharpe']:>7.3f} {m['sortino']:>7.3f} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>5.1f}% {m['n_trades']:>5} {r['permutation_p']:>6.4f} "
              f"{r['regime']['gap']:>5.2f} {r['gates_passed']:>6} {r['verdict']:>8}")

    # Save
    output = {
        "strategy": "Volume-Confirmed Mean Reversion",
        "universe": UNIVERSE,
        "period": f"{START} to {END}",
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "hold_days": HOLD_DAYS,
        "slippage_bps": SLIPPAGE_BPS,
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "variants": {},
    }
    for name, r in results.items():
        # Convert numpy types for JSON
        entry = {}
        for k, v in r.items():
            if isinstance(v, dict):
                entry[k] = {kk: (bool(vv) if isinstance(vv, (np.bool_,)) else
                                  float(vv) if isinstance(vv, (np.floating,)) else
                                  int(vv) if isinstance(vv, (np.integer,)) else vv)
                             for kk, vv in v.items()}
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], tuple):
                entry[k] = [[str(x) if not isinstance(x, (int, float)) else x for x in t] for t in v]
            else:
                entry[k] = v
        output["variants"][name] = entry

    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
