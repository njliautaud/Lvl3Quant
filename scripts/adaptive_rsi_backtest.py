#!/usr/bin/env python3
"""
Adaptive RSI Thresholds Backtest
================================
Dynamically adjusts RSI entry/exit thresholds based on market volatility regime.

Baseline: Fixed RSI B (RSI(5)<20, exit RSI>50 or 10d, 200-SMA filter) — Sharpe 1.63, 65% WR.

Variants:
  A) Vol-Scaled RSI Entry
  B) Percentile RSI
  C) Bollinger-Band RSI
  D) VIX-Adjusted
  E) Vol-Regime Bucketed
  F) Combined Best (union of best adaptive + fixed RSI B)

5-Gate: Sharpe>0.5, Perm p<0.05, Regime gap<0.5, MaxDD>-50%, >=20 trades.
OOT: Jan 2022 - Jul 2026. $669 account. Robinhood ($0 commission, 0.02% slippage).
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
START = "2020-01-01"    # extra lookback for 252-day rolling + 200-SMA
END = "2026-07-30"
OOT_START = "2022-01-01"
N_PERM = 1000

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "AVGO", "ORCL", "ADBE", "MU", "QCOM", "PLTR",
    "SOFI", "HOOD", "COIN", "UBER", "LYFT", "SHOP", "NET", "TTD",
    "DDOG", "RBLX", "SNAP", "PINS", "ROKU", "INTC",
]

ALL_TICKERS = sorted(set(TICKERS + ["SPY", "^VIX"]))

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
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
vix_close = closes.get("^VIX", pd.Series(dtype=float))

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 252)
print(f"  Tickers with sufficient data: {loaded}/{len(TICKERS)}")
print(f"  SPY rows: {len(spy_close)}, VIX rows: {len(vix_close)}")


# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def sma(series, period):
    return series.rolling(period).mean()


def realized_vol(series, window=21):
    """Annualized realized volatility from daily log returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


# ── Pre-compute indicators ───────────────────────────────────────────────
indicators = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 300:
        print(f"  Skipping {t}: only {len(c)} bars")
        continue
    ind = pd.DataFrame(index=c.index)
    ind["close"] = c
    ind["sma200"] = sma(c, 200)
    ind["rsi5"] = calc_rsi(c, 5)
    ind["above_200sma"] = c > ind["sma200"]
    # Volatility measures
    ind["vol21"] = realized_vol(c, 21)
    ind["vol_median"] = ind["vol21"].rolling(252, min_periods=63).median()
    # RSI percentiles (rolling 252-day)
    ind["rsi5_pct10"] = ind["rsi5"].rolling(252, min_periods=63).quantile(0.10)
    ind["rsi5_pct50"] = ind["rsi5"].rolling(252, min_periods=63).quantile(0.50)
    # RSI Bollinger bands
    ind["rsi5_mean20"] = ind["rsi5"].rolling(20).mean()
    ind["rsi5_std20"] = ind["rsi5"].rolling(20).std()
    ind["rsi5_lower_bb"] = ind["rsi5_mean20"] - 2 * ind["rsi5_std20"]

    indicators[t] = ind.dropna(subset=["sma200", "vol_median"])

spy_sma200 = sma(spy_close, 200)
spy_regime = (spy_close > spy_sma200).reindex(spy_close.index).fillna(False)

print(f"  Tickers with indicators: {len(indicators)}")


# ── Signal Generators per Variant ─────────────────────────────────────────
def gen_signals_fixed_rsi_b(tickers):
    """Baseline: RSI(5)<20, above 200-SMA. Exit RSI>50 or 10d."""
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        mask = (ind["rsi5"] < 20) & ind["above_200sma"]
        for dt in ind.index[mask]:
            if str(dt.date()) >= OOT_START:
                signals.append((dt, t, float(ind.loc[dt, "close"]), "fixed", 50, 10))
    return sorted(signals, key=lambda x: x[0])


def gen_signals_A_vol_scaled(tickers):
    """Vol-Scaled RSI Entry: threshold = 20 * (current_vol / median_vol)."""
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        for i in range(len(ind)):
            dt = ind.index[i]
            if str(dt.date()) < OOT_START:
                continue
            if not ind["above_200sma"].iloc[i]:
                continue
            vol = ind["vol21"].iloc[i]
            med_vol = ind["vol_median"].iloc[i]
            if pd.isna(vol) or pd.isna(med_vol) or med_vol < 0.01:
                continue
            ratio = vol / med_vol
            entry_thresh = 20.0 * ratio
            entry_thresh = max(5.0, min(entry_thresh, 50.0))  # clamp
            rsi_val = ind["rsi5"].iloc[i]
            if rsi_val < entry_thresh:
                signals.append((dt, t, float(ind["close"].iloc[i]), "vol_scaled", 50, 10))
    return sorted(signals, key=lambda x: x[0])


def gen_signals_B_percentile(tickers):
    """Percentile RSI: enter when RSI(5) < its own 10th percentile (rolling 252d)."""
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        for i in range(len(ind)):
            dt = ind.index[i]
            if str(dt.date()) < OOT_START:
                continue
            if not ind["above_200sma"].iloc[i]:
                continue
            rsi_val = ind["rsi5"].iloc[i]
            pct10 = ind["rsi5_pct10"].iloc[i]
            pct50 = ind["rsi5_pct50"].iloc[i]
            if pd.isna(pct10) or pd.isna(pct50):
                continue
            if rsi_val < pct10:
                # Exit when RSI > 50th percentile or 10 days
                signals.append((dt, t, float(ind["close"].iloc[i]), "percentile",
                                float(pct50), 10))
    return sorted(signals, key=lambda x: x[0])


def gen_signals_C_bollinger(tickers):
    """Bollinger-Band RSI: entry when RSI < (RSI_20d_mean - 2*RSI_20d_std)."""
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        for i in range(len(ind)):
            dt = ind.index[i]
            if str(dt.date()) < OOT_START:
                continue
            if not ind["above_200sma"].iloc[i]:
                continue
            rsi_val = ind["rsi5"].iloc[i]
            lower_bb = ind["rsi5_lower_bb"].iloc[i]
            rsi_mean = ind["rsi5_mean20"].iloc[i]
            if pd.isna(lower_bb) or pd.isna(rsi_mean):
                continue
            if rsi_val < lower_bb:
                # Exit when RSI > RSI_20d_mean or 10 days
                signals.append((dt, t, float(ind["close"].iloc[i]), "bollinger",
                                float(rsi_mean), 10))
    return sorted(signals, key=lambda x: x[0])


def gen_signals_D_vix_adjusted(tickers):
    """VIX-Adjusted: VIX>25 -> RSI<30, VIX<15 -> RSI<15, else RSI<20."""
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        for i in range(len(ind)):
            dt = ind.index[i]
            if str(dt.date()) < OOT_START:
                continue
            if not ind["above_200sma"].iloc[i]:
                continue
            # Get VIX for this date
            vix_val = vix_close.get(dt, np.nan) if dt in vix_close.index else np.nan
            if pd.isna(vix_val):
                # Try nearest date
                try:
                    nearest = vix_close.index[vix_close.index.get_indexer([dt], method="ffill")[0]]
                    vix_val = float(vix_close.loc[nearest])
                except Exception:
                    continue

            if vix_val > 25:
                entry_thresh = 30.0
            elif vix_val < 15:
                entry_thresh = 15.0
            else:
                entry_thresh = 20.0

            rsi_val = ind["rsi5"].iloc[i]
            if rsi_val < entry_thresh:
                signals.append((dt, t, float(ind["close"].iloc[i]), "vix_adj", 50, 10))
    return sorted(signals, key=lambda x: x[0])


def gen_signals_E_vol_regime(tickers):
    """Vol-Regime Bucketed: Low(<20%)->RSI<15/hold15, Med(20-40%)->RSI<20/hold10, High(>40%)->RSI<30/hold5."""
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        for i in range(len(ind)):
            dt = ind.index[i]
            if str(dt.date()) < OOT_START:
                continue
            if not ind["above_200sma"].iloc[i]:
                continue
            vol = ind["vol21"].iloc[i]
            if pd.isna(vol):
                continue

            if vol < 0.20:
                entry_thresh, max_hold = 15.0, 15
            elif vol < 0.40:
                entry_thresh, max_hold = 20.0, 10
            else:
                entry_thresh, max_hold = 30.0, 5

            rsi_val = ind["rsi5"].iloc[i]
            if rsi_val < entry_thresh:
                signals.append((dt, t, float(ind["close"].iloc[i]), "vol_regime", 50, max_hold))
    return sorted(signals, key=lambda x: x[0])


# ── Backtest Engine ───────────────────────────────────────────────────────
def run_backtest(signals, capital=CAPITAL, slippage=SLIPPAGE_PCT, max_concurrent=1):
    """
    Run backtest. Signals are (date, ticker, entry_price, variant_tag, exit_rsi_thresh, max_hold_days).
    For max_concurrent=1, invest full capital per trade.
    """
    if not signals:
        return np.array([]), [], []

    signals = sorted(signals, key=lambda x: x[0])
    equity = capital
    equity_curve = [(OOT_START, capital)]
    realized = []
    open_positions = []

    for sig in signals:
        date, ticker, entry_price, tag, exit_rsi_thresh, max_hold = sig
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
        if date_str < OOT_START:
            continue

        # Close expired positions
        still_open = [p for p in open_positions if p["exit_date"] > date_str]
        open_positions = still_open

        if len(open_positions) >= max_concurrent:
            continue

        ind = indicators.get(ticker)
        if ind is None or date not in ind.index:
            continue
        loc = ind.index.get_loc(date)

        # Find exit: RSI > exit_rsi_thresh or max_hold days
        exit_idx = None
        for i in range(loc + 1, min(loc + max_hold + 1, len(ind))):
            if ind["rsi5"].iloc[i] > exit_rsi_thresh:
                exit_idx = i
                break
        if exit_idx is None:
            exit_idx = min(loc + max_hold, len(ind) - 1)
        if exit_idx <= loc:
            exit_idx = min(loc + 1, len(ind) - 1)

        exit_date_str = str(ind.index[exit_idx].date())

        # Position sizing
        alloc = equity / max_concurrent
        actual_entry = entry_price * (1 + slippage)
        exit_price = float(ind["close"].iloc[exit_idx]) * (1 - slippage)

        shares = int(alloc / actual_entry)
        if shares < 1:
            if actual_entry <= equity:
                shares = 1
            else:
                continue

        pnl = shares * (exit_price - actual_entry)
        ret = pnl / (shares * actual_entry)
        hold_days = (ind.index[exit_idx] - ind.index[loc]).days

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
            "tag": tag,
        }

        equity += pnl
        equity_curve.append((exit_date_str, round(equity, 2)))
        realized.append(trade)
        open_positions.append({"ticker": ticker, "exit_date": exit_date_str})

    returns = np.array([t["return"] for t in realized])
    return returns, equity_curve, realized


# ── Metrics ───────────────────────────────────────────────────────────────
def calc_metrics(returns, equity_curve, realized, capital=CAPITAL):
    if len(returns) < 2:
        return {
            "n_trades": len(returns), "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0,
            "total_return_pct": 0.0, "cagr_pct": 0.0, "final_equity": capital,
            "avg_hold_days": 0.0, "bull_sharpe": 0.0, "bear_sharpe": 0.0,
            "regime_gap": 0.0, "bull_trades": 0, "bear_trades": 0,
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

    # CAGR
    if realized:
        first_date = pd.Timestamp(realized[0]["entry_date"])
        last_date = pd.Timestamp(realized[-1]["exit_date"])
        years = max((last_date - first_date).days / 365.25, 0.5)
        cagr = ((final_eq / capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

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

    bull_s = regime_sharpe(bull_rets)
    bear_s = regime_sharpe(bear_rets)
    denom = max(abs(bull_s), abs(bear_s), 1e-9)
    regime_gap = abs(bull_s - bear_s) / denom

    return {
        "n_trades": int(n),
        "win_rate": round(float(wr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "cagr_pct": round(float(cagr), 2),
        "final_equity": round(float(final_eq), 2),
        "avg_hold_days": round(float(avg_hold), 1),
        "bull_trades": int(len(bull_rets)),
        "bear_trades": int(len(bear_rets)),
        "bull_sharpe": round(float(bull_s), 3),
        "bear_sharpe": round(float(bear_s), 3),
        "regime_gap": round(float(regime_gap), 3),
    }


# ── Permutation Test ──────────────────────────────────────────────────────
def permutation_test(returns, signals, realized, metrics):
    """Shuffle entry timing 1000 times, compute p-value."""
    if len(returns) < 3 or len(signals) < 3:
        return 1.0

    real_sharpe = metrics["sharpe"]

    # Build valid OOT dates per ticker
    oot_dates_per_ticker = {}
    for t in set(s[1] for s in signals):
        ind = indicators.get(t)
        if ind is not None:
            valid = ind.index[ind.index >= OOT_START]
            if len(valid) > 15:
                oot_dates_per_ticker[t] = valid[:-10]

    signal_tickers = [s[1] for s in signals
                      if str(s[0].date()) >= OOT_START and s[1] in oot_dates_per_ticker]

    if len(signal_tickers) < 3:
        return 1.0

    perm_sharpes = np.zeros(N_PERM)
    for i in range(N_PERM):
        shuffled = []
        for ticker in signal_tickers:
            valid_dates = oot_dates_per_ticker[ticker]
            rand_date = valid_dates[np.random.randint(0, len(valid_dates))]
            ind = indicators[ticker]
            ep = ind.loc[rand_date, "close"]
            if isinstance(ep, pd.Series):
                ep = ep.iloc[0]
            shuffled.append((rand_date, ticker, float(ep), "perm", 50, 10))
        perm_ret, _, _ = run_backtest(shuffled)
        if len(perm_ret) >= 2:
            avg_h = 8.0
            af = np.sqrt(252 / avg_h)
            s = perm_ret.std()
            perm_sharpes[i] = (perm_ret.mean() / s * af) if s > 1e-12 else 0.0
        else:
            perm_sharpes[i] = 0.0

    p_value = float(np.mean(perm_sharpes >= real_sharpe))
    return round(p_value, 4)


# ── 5-Gate Validation ─────────────────────────────────────────────────────
def five_gate_check(metrics, p_value):
    """Returns dict with gate results."""
    gates = {
        "G1_sharpe_gt_0.5": {"value": metrics["sharpe"], "passed": metrics["sharpe"] > 0.5},
        "G2_perm_p_lt_0.05": {"value": p_value, "passed": p_value < 0.05},
        "G3_regime_gap_lt_0.5": {"value": metrics["regime_gap"], "passed": metrics["regime_gap"] < 0.5},
        "G4_maxdd_gt_neg50": {"value": metrics["max_dd_pct"], "passed": metrics["max_dd_pct"] > -50.0},
        "G5_trades_gte_20": {"value": metrics["n_trades"], "passed": metrics["n_trades"] >= 20},
    }
    n_passed = sum(1 for g in gates.values() if g["passed"])
    verdict = ("PASS" if n_passed == 5 else
               "STRONG" if n_passed == 4 else
               "MARGINAL" if n_passed == 3 else "FAIL")
    return {"gates": gates, "passed": n_passed, "total": 5, "verdict": verdict}


# ══════════════════════════════════════════════════════════════════════════
#  RUN ALL VARIANTS
# ══════════════════════════════════════════════════════════════════════════

results = {}

variant_configs = {
    "A_vol_scaled": {
        "name": "A) Vol-Scaled RSI Entry",
        "gen_fn": gen_signals_A_vol_scaled,
        "max_concurrent": 1,
    },
    "B_percentile": {
        "name": "B) Percentile RSI",
        "gen_fn": gen_signals_B_percentile,
        "max_concurrent": 1,
    },
    "C_bollinger": {
        "name": "C) Bollinger-Band RSI",
        "gen_fn": gen_signals_C_bollinger,
        "max_concurrent": 1,
    },
    "D_vix_adjusted": {
        "name": "D) VIX-Adjusted",
        "gen_fn": gen_signals_D_vix_adjusted,
        "max_concurrent": 1,
    },
    "E_vol_regime": {
        "name": "E) Vol-Regime Bucketed",
        "gen_fn": gen_signals_E_vol_regime,
        "max_concurrent": 1,
    },
}

# Run A-E
for key, cfg in variant_configs.items():
    print(f"\n{'='*70}")
    print(f"  {cfg['name']}")
    print(f"{'='*70}")

    signals = cfg["gen_fn"](TICKERS)
    returns, eq_curve, trades = run_backtest(signals, max_concurrent=cfg["max_concurrent"])
    metrics = calc_metrics(returns, eq_curve, trades)

    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
          f"Sortino: {metrics['sortino']}, WR: {metrics['win_rate']:.1%}")
    print(f"  PF: {metrics['profit_factor']}, MaxDD: {metrics['max_dd_pct']:.1f}%, "
          f"Total Return: {metrics['total_return_pct']:.1f}%, CAGR: {metrics['cagr_pct']:.1f}%")
    print(f"  Bull Sharpe: {metrics['bull_sharpe']}, Bear Sharpe: {metrics['bear_sharpe']}, "
          f"Regime Gap: {metrics['regime_gap']:.3f}")

    print(f"  Running permutation test ({N_PERM} shuffles) ...")
    p_val = permutation_test(returns, signals, trades, metrics)
    print(f"  Perm p-value: {p_val}")

    gate_result = five_gate_check(metrics, p_val)
    print(f"  5-Gate: {gate_result['passed']}/5 — {gate_result['verdict']}")
    for gname, ginfo in gate_result["gates"].items():
        status = "PASS" if ginfo["passed"] else "FAIL"
        print(f"    {gname}: {ginfo['value']} [{status}]")

    results[key] = {
        "name": cfg["name"],
        "metrics": metrics,
        "p_value": p_val,
        "five_gate": gate_result,
        "n_signals_generated": len(signals),
    }

# ── Variant F: Combined Best ─────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  F) Combined Best (best adaptive + fixed RSI B, max 3 concurrent)")
print(f"{'='*70}")

# Find best adaptive variant by Sharpe
best_key = max(results.keys(), key=lambda k: results[k]["metrics"]["sharpe"])
best_name = results[best_key]["name"]
print(f"  Best adaptive: {best_name} (Sharpe {results[best_key]['metrics']['sharpe']})")

# Generate signals from both best adaptive and fixed RSI B
best_gen_fn = variant_configs[best_key]["gen_fn"]
signals_best = best_gen_fn(TICKERS)
signals_fixed = gen_signals_fixed_rsi_b(TICKERS)

# Union: merge and deduplicate (same ticker+date = keep one)
all_signals_f = signals_best + signals_fixed
seen = set()
deduped = []
for sig in sorted(all_signals_f, key=lambda x: x[0]):
    key_id = (str(sig[0].date()), sig[1])
    if key_id not in seen:
        seen.add(key_id)
        deduped.append(sig)

returns_f, eq_f, trades_f = run_backtest(deduped, max_concurrent=3)
metrics_f = calc_metrics(returns_f, eq_f, trades_f)

print(f"  Trades: {metrics_f['n_trades']}, Sharpe: {metrics_f['sharpe']}, "
      f"Sortino: {metrics_f['sortino']}, WR: {metrics_f['win_rate']:.1%}")
print(f"  PF: {metrics_f['profit_factor']}, MaxDD: {metrics_f['max_dd_pct']:.1f}%, "
      f"Total Return: {metrics_f['total_return_pct']:.1f}%, CAGR: {metrics_f['cagr_pct']:.1f}%")
print(f"  Bull Sharpe: {metrics_f['bull_sharpe']}, Bear Sharpe: {metrics_f['bear_sharpe']}, "
      f"Regime Gap: {metrics_f['regime_gap']:.3f}")

print(f"  Running permutation test ({N_PERM} shuffles) ...")
p_val_f = permutation_test(returns_f, deduped, trades_f, metrics_f)
print(f"  Perm p-value: {p_val_f}")

gate_f = five_gate_check(metrics_f, p_val_f)
print(f"  5-Gate: {gate_f['passed']}/5 — {gate_f['verdict']}")
for gname, ginfo in gate_f["gates"].items():
    status = "PASS" if ginfo["passed"] else "FAIL"
    print(f"    {gname}: {ginfo['value']} [{status}]")

results["F_combined"] = {
    "name": f"F) Combined Best ({best_name} + Fixed RSI B, max 3 concurrent)",
    "metrics": metrics_f,
    "p_value": p_val_f,
    "five_gate": gate_f,
    "best_adaptive_used": best_name,
    "n_signals_generated": len(deduped),
}

# ── Save Results ──────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/adaptive_rsi_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  ADAPTIVE RSI THRESHOLDS — SUMMARY")
print("=" * 90)
print(f"{'Variant':<45} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} "
      f"{'MaxDD':>7} {'CAGR':>7} {'p-val':>6} {'Gate':>5}")
print("-" * 90)

for key in ["A_vol_scaled", "B_percentile", "C_bollinger", "D_vix_adjusted", "E_vol_regime", "F_combined"]:
    r = results[key]
    m = r["metrics"]
    g = r["five_gate"]
    print(f"{r['name']:<45} {m['n_trades']:>6} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
          f"{m['win_rate']:>5.0%} {m['profit_factor']:>6.2f} {m['max_dd_pct']:>6.1f}% "
          f"{m['cagr_pct']:>6.1f}% {r['p_value']:>6.3f} {g['verdict']:>5}")

print("-" * 90)

# Highlight best
all_keys = list(results.keys())
passing = [k for k in all_keys if results[k]["five_gate"]["passed"] == 5]
if passing:
    best = max(passing, key=lambda k: results[k]["metrics"]["sharpe"])
    print(f"\nBEST (all 5 gates passed): {results[best]['name']}")
    m = results[best]["metrics"]
    print(f"  Sharpe {m['sharpe']}, Sortino {m['sortino']}, WR {m['win_rate']:.0%}, "
          f"PF {m['profit_factor']}, MaxDD {m['max_dd_pct']:.1f}%, CAGR {m['cagr_pct']:.1f}%")
else:
    best = max(all_keys, key=lambda k: results[k]["five_gate"]["passed"])
    print(f"\nNo variant passed all 5 gates. Best: {results[best]['name']} "
          f"({results[best]['five_gate']['passed']}/5)")

# Compare to baseline
print(f"\nBaseline Fixed RSI B reference: Sharpe 1.63, WR 65%")
best_adaptive_sharpe = max(results[k]["metrics"]["sharpe"] for k in all_keys)
print(f"Best adaptive Sharpe: {best_adaptive_sharpe:.2f}")
if best_adaptive_sharpe > 1.63:
    print("RESULT: Adaptive thresholds IMPROVED over fixed RSI B baseline.")
else:
    print("RESULT: Fixed RSI B baseline remains superior. "
          "Adaptive thresholds did not improve signal quality.")

print("\nDone.")
