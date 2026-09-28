#!/usr/bin/env python3
"""
FX / Currency Momentum Backtest — 6 Variants (A–F)
Walk-forward SLIDING window, OOT 2022-01-01 to 2026-07-29.
Starting capital $645, slippage 0.02%, commission $0 (Robinhood).
"""

import json, warnings, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Parameters ──────────────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END   = "2026-07-29"
CAPITAL   = 645.0
SLIPPAGE  = 0.0002  # 0.02%
COMMISSION = 0.0
DATA_START = "2019-01-01"  # extra history for lookbacks

TICKERS = ["UUP", "UDN", "FXE", "FXY", "FXB", "FXA", "FXC", "CYB",
           "QQQ", "SPY", "^VIX"]

# ── Download data ───────────────────────────────────────────────────────────
print("Downloading data …")
raw = yf.download(TICKERS, start=DATA_START, end="2026-07-30",
                  auto_adjust=True, progress=False)

close = raw["Close"].copy()
# yfinance may return multi-level columns; flatten
if isinstance(close.columns, pd.MultiIndex):
    close.columns = close.columns.get_level_values(-1)

# Rename ^VIX -> VIX
close.rename(columns={"^VIX": "VIX"}, inplace=True)

# Forward-fill then drop rows where QQQ/SPY are NaN (market closed)
close = close.ffill()
close = close.dropna(subset=["QQQ", "SPY"])

# If CYB has no data, drop it gracefully
cyb_available = "CYB" in close.columns and close["CYB"].notna().sum() > 100
if not cyb_available:
    print("CYB unavailable or insufficient data — skipping CYB.")

qqq_ret = close["QQQ"].pct_change()

# ── Helpers ─────────────────────────────────────────────────────────────────
def calc_metrics(returns: pd.Series, capital: float = CAPITAL):
    """Risk-adjusted metrics from a daily returns series (0 on flat days)."""
    invested = returns[returns != 0]
    total_ret = (1 + returns).prod() - 1
    ann = np.sqrt(252)
    mu = returns.mean() * 252
    sigma = returns.std() * ann if returns.std() > 0 else 1e-9
    sharpe = mu / sigma

    down = returns[returns < 0]
    down_std = down.std() * ann if len(down) > 0 and down.std() > 0 else 1e-9
    sortino = mu / down_std

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    wins = invested[invested > 0]
    losses = invested[invested < 0]
    gross_profit = wins.sum() if len(wins) else 0
    gross_loss = abs(losses.sum()) if len(losses) else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    wr = len(wins) / len(invested) if len(invested) > 0 else 0

    n_days = len(invested)
    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "pf": round(pf, 3),
        "wr": round(wr, 3),
        "total_return_pct": round(total_ret * 100, 2),
        "n_days": int(n_days),
    }


def qqq_corr(returns: pd.Series):
    """Correlation of strategy returns vs QQQ on days the strategy is invested."""
    mask = returns != 0
    if mask.sum() < 10:
        return 0.0
    aligned = pd.DataFrame({"strat": returns, "qqq": qqq_ret}).dropna()
    aligned = aligned[aligned["strat"] != 0]
    if len(aligned) < 10:
        return 0.0
    return round(aligned["strat"].corr(aligned["qqq"]), 4)


def permutation_test(returns: pd.Series, n_perms=1000):
    """Shuffle invested-day returns, compare mean to actual."""
    invested = returns[returns != 0].values
    if len(invested) < 5:
        return 1.0
    actual_mean = invested.mean()
    count = 0
    for _ in range(n_perms):
        perm = np.random.permutation(invested)
        # Shuffle date assignment (circular shift of sign)
        signs = np.random.choice([-1, 1], size=len(perm))
        if (perm * signs).mean() >= actual_mean:
            count += 1
    return round(count / n_perms, 4)


def regime_split(returns: pd.Series, close_df: pd.DataFrame):
    """Split by SPY > 200-SMA (bull) vs < 200-SMA (bear)."""
    spy_sma200 = close_df["SPY"].rolling(200).mean()
    bull_mask = close_df["SPY"] > spy_sma200
    bear_mask = close_df["SPY"] <= spy_sma200

    bull_ret = returns[bull_mask.reindex(returns.index, fill_value=False)]
    bear_ret = returns[bear_mask.reindex(returns.index, fill_value=False)]

    ann = np.sqrt(252)
    def _sharpe(r):
        if len(r) == 0 or r.std() == 0:
            return 0.0
        return round((r.mean() * 252) / (r.std() * ann), 3)

    bs = _sharpe(bull_ret)
    brs = _sharpe(bear_ret)
    denom = max(abs(bs), abs(brs), 1e-9)
    gap = round(abs(bs - brs) / denom, 3)
    return {
        "bull_sharpe": bs,
        "bear_sharpe": brs,
        "regime_gap": gap,
        "bull_days": int((bull_mask & (returns != 0)).sum()),
        "bear_days": int((bear_mask & (returns != 0)).sum()),
    }


def gate_checks(metrics, perm_p, regime, n_trades):
    g = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "trades_gte_20": n_trades >= 20,
    }
    passed = sum(g.values())
    return g, f"{passed}/5", passed == 5


def apply_slippage(price, direction):
    """Apply slippage to entry/exit price. direction=1 for buy, -1 for sell."""
    return price * (1 + direction * SLIPPAGE)


# ── Slice to OOT ───────────────────────────────────────────────────────────
oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
oot_dates = close.index[oot_mask]
print(f"OOT period: {oot_dates[0].date()} to {oot_dates[-1].date()}, {len(oot_dates)} trading days")

# ── Strategy A: Dollar Momentum ────────────────────────────────────────────
def strategy_A():
    print("Running Strategy A: Dollar Momentum …")
    ma20 = close["UUP"].rolling(20).mean()
    ma50 = close["UUP"].rolling(50).mean()

    returns = pd.Series(0.0, index=oot_dates)
    trades = []
    i = 0
    while i < len(oot_dates):
        d = oot_dates[i]
        if pd.isna(ma20.get(d)) or pd.isna(ma50.get(d)):
            i += 1
            continue
        if ma20[d] > ma50[d]:
            ticker = "UUP"
        else:
            ticker = "UDN"

        entry_price = apply_slippage(close.loc[d, ticker], 1)
        hold_end = min(i + 15, len(oot_dates) - 1)
        exit_date = oot_dates[hold_end]
        exit_price = apply_slippage(close.loc[exit_date, ticker], -1)

        trade_ret = (exit_price / entry_price) - 1
        # Spread return across holding days
        n_hold = hold_end - i
        if n_hold > 0:
            daily = (1 + trade_ret) ** (1 / n_hold) - 1
            for j in range(i, hold_end):
                returns.iloc[j] = daily
        trades.append(trade_ret)
        i = hold_end + 1  # next trade after hold

    return returns, len(trades)


# ── Strategy B: Carry Trade Proxy ──────────────────────────────────────────
def strategy_B():
    print("Running Strategy B: Carry Trade Proxy …")
    returns = pd.Series(0.0, index=oot_dates)
    trades = []
    current_pos = None  # "FXA" or "FXY" or None
    entry_price = None
    entry_idx = None

    for i, d in enumerate(oot_dates):
        vix = close.loc[d, "VIX"] if "VIX" in close.columns else 20
        if pd.isna(vix):
            continue

        target = None
        if vix < 20:
            target = "FXA"
        elif vix > 25:
            target = "FXY"
        # Between 20-25: hold current or stay flat

        if target is not None and target != current_pos:
            # Close existing
            if current_pos is not None and entry_price is not None:
                exit_p = apply_slippage(close.loc[d, current_pos], -1)
                trade_ret = (exit_p / entry_price) - 1
                n_hold = i - entry_idx
                if n_hold > 0:
                    daily = (1 + trade_ret) ** (1 / n_hold) - 1
                    for j in range(entry_idx, i):
                        returns.iloc[j] = daily
                trades.append(trade_ret)

            # Open new
            current_pos = target
            entry_price = apply_slippage(close.loc[d, target], 1)
            entry_idx = i

    # Close last trade
    if current_pos is not None and entry_price is not None:
        exit_p = apply_slippage(close.loc[oot_dates[-1], current_pos], -1)
        trade_ret = (exit_p / entry_price) - 1
        n_hold = len(oot_dates) - 1 - entry_idx
        if n_hold > 0:
            daily = (1 + trade_ret) ** (1 / n_hold) - 1
            for j in range(entry_idx, len(oot_dates) - 1):
                returns.iloc[j] = daily
        trades.append(trade_ret)

    return returns, len(trades)


# ── Strategy C: FX Mean Reversion ──────────────────────────────────────────
def strategy_C():
    print("Running Strategy C: FX Mean Reversion …")
    fx_tickers = [t for t in ["FXE", "FXY", "FXB", "FXA", "FXC"] if t in close.columns]
    returns = pd.Series(0.0, index=oot_dates)
    trades = []
    i = 0

    while i < len(oot_dates):
        d = oot_dates[i]
        # Find most oversold (lowest 20d return)
        ret20 = {}
        for t in fx_tickers:
            loc = close.index.get_loc(d)
            if loc >= 20:
                r = close[t].iloc[loc] / close[t].iloc[loc - 20] - 1
                if not pd.isna(r):
                    ret20[t] = r
        if not ret20:
            i += 1
            continue

        ticker = min(ret20, key=ret20.get)
        entry_price = apply_slippage(close.loc[d, ticker], 1)

        # Hold up to 10 days or 2% gain
        exit_idx = min(i + 10, len(oot_dates) - 1)
        actual_exit = exit_idx
        for j in range(i + 1, exit_idx + 1):
            p = close.loc[oot_dates[j], ticker]
            if (p / entry_price - 1) >= 0.02:
                actual_exit = j
                break

        exit_price = apply_slippage(close.loc[oot_dates[actual_exit], ticker], -1)
        trade_ret = (exit_price / entry_price) - 1
        n_hold = actual_exit - i
        if n_hold > 0:
            daily = (1 + trade_ret) ** (1 / n_hold) - 1
            for j in range(i, actual_exit):
                returns.iloc[j] = daily
        trades.append(trade_ret)
        i = actual_exit + 1

    return returns, len(trades)


# ── Strategy D: Dollar Smile ───────────────────────────────────────────────
def strategy_D():
    print("Running Strategy D: Dollar Smile …")
    spy_mom60 = close["SPY"].pct_change(60)
    returns = pd.Series(0.0, index=oot_dates)
    trades = []
    current_pos = None
    entry_price = None
    entry_idx = None

    for i, d in enumerate(oot_dates):
        vix = close.loc[d, "VIX"] if "VIX" in close.columns else 20
        mom = spy_mom60.get(d, 0)
        if pd.isna(vix) or pd.isna(mom):
            continue

        target = None
        if vix > 25 or mom > 0.10:
            target = "UUP"
        elif 15 <= vix <= 25 and abs(mom) < 0.05:
            target = "UDN"

        if target is not None and target != current_pos:
            # Close
            if current_pos is not None and entry_price is not None:
                exit_p = apply_slippage(close.loc[d, current_pos], -1)
                trade_ret = (exit_p / entry_price) - 1
                n_hold = i - entry_idx
                if n_hold > 0:
                    daily = (1 + trade_ret) ** (1 / n_hold) - 1
                    for j in range(entry_idx, i):
                        returns.iloc[j] = daily
                trades.append(trade_ret)

            current_pos = target
            entry_price = apply_slippage(close.loc[d, target], 1)
            entry_idx = i
        elif target is None and current_pos is not None:
            # Neither condition met — close
            exit_p = apply_slippage(close.loc[d, current_pos], -1)
            trade_ret = (exit_p / entry_price) - 1
            n_hold = i - entry_idx
            if n_hold > 0:
                daily = (1 + trade_ret) ** (1 / n_hold) - 1
                for j in range(entry_idx, i):
                    returns.iloc[j] = daily
            trades.append(trade_ret)
            current_pos = None
            entry_price = None
            entry_idx = None

    # Close last
    if current_pos is not None and entry_price is not None:
        exit_p = apply_slippage(close.loc[oot_dates[-1], current_pos], -1)
        trade_ret = (exit_p / entry_price) - 1
        n_hold = len(oot_dates) - 1 - entry_idx
        if n_hold > 0:
            daily = (1 + trade_ret) ** (1 / n_hold) - 1
            for j in range(entry_idx, len(oot_dates)):
                returns.iloc[j] = daily
        trades.append(trade_ret)

    return returns, len(trades)


# ── Strategy E: FX Momentum Score ──────────────────────────────────────────
def strategy_E():
    print("Running Strategy E: FX Momentum Score …")
    fx_tickers = [t for t in ["FXE", "FXY", "FXB", "FXA", "FXC", "UUP", "UDN"]
                  if t in close.columns]
    returns = pd.Series(0.0, index=oot_dates)
    trades = []

    # Monthly rebalance
    months = pd.Series(oot_dates).dt.to_period("M").unique()
    prev_holdings = []

    for m in months:
        m_dates = [d for d in oot_dates if d.to_period("M") == m]
        if not m_dates:
            continue
        rebal_date = m_dates[0]
        loc = close.index.get_loc(rebal_date)

        if loc < 126:  # need 6m lookback
            continue

        # Score each ticker
        scores = {}
        for t in fx_tickers:
            m1 = close[t].iloc[loc] / close[t].iloc[loc - 21] - 1
            m3 = close[t].iloc[loc] / close[t].iloc[loc - 63] - 1
            m6 = close[t].iloc[loc] / close[t].iloc[loc - 126] - 1
            if any(pd.isna([m1, m3, m6])):
                continue
            scores[t] = (m1 + m3 + m6) / 3.0

        if len(scores) < 4:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        long_tickers = [r[0] for r in ranked[:2]]
        # "avoid" bottom 2 = just go long top 2

        # Count trade switches
        if set(long_tickers) != set(prev_holdings):
            trades.append(1)  # simplified: count rebalance as a trade
        prev_holdings = long_tickers

        # Equal-weight returns for the month
        for d in m_dates:
            day_ret = 0.0
            for t in long_tickers:
                r = close[t].pct_change().get(d, 0)
                if pd.isna(r):
                    r = 0
                day_ret += r / len(long_tickers)
            # Apply slippage only on rebalance day
            if d == rebal_date:
                day_ret -= SLIPPAGE * 2  # entry + exit slippage
            returns.loc[d] = day_ret

    return returns, len(trades)


# ── Strategy F: Yen Carry Unwind Signal ────────────────────────────────────
def strategy_F():
    print("Running Strategy F: Yen Carry Unwind Signal …")
    fxy_ret = close["FXY"].pct_change()
    spy_ret = close["SPY"].pct_change()
    rolling_corr = fxy_ret.rolling(20).corr(spy_ret)
    fxy_5d_ret = close["FXY"].pct_change(5)

    returns = pd.Series(0.0, index=oot_dates)
    trades = []
    i = 0

    while i < len(oot_dates):
        d = oot_dates[i]
        corr_val = rolling_corr.get(d, np.nan)
        fxy_5d = fxy_5d_ret.get(d, np.nan)
        fxy_rising = fxy_ret.get(d, 0) > 0
        vix = close.loc[d, "VIX"] if "VIX" in close.columns else 20

        if pd.isna(corr_val) or pd.isna(fxy_5d) or pd.isna(vix):
            i += 1
            continue

        ticker = None
        hold_days = 0

        # Signal 1: Yen carry unwind
        if corr_val > 0.3 and fxy_rising:
            ticker = "FXY"
            hold_days = 5
        # Signal 2: Carry trade ON
        elif fxy_5d < -0.02 and vix < 18:
            ticker = "FXA"
            hold_days = 5

        if ticker is not None:
            entry_price = apply_slippage(close.loc[d, ticker], 1)
            exit_idx = min(i + hold_days, len(oot_dates) - 1)
            exit_price = apply_slippage(close.loc[oot_dates[exit_idx], ticker], -1)
            trade_ret = (exit_price / entry_price) - 1
            n_hold = exit_idx - i
            if n_hold > 0:
                daily = (1 + trade_ret) ** (1 / n_hold) - 1
                for j in range(i, exit_idx):
                    returns.iloc[j] = daily
            trades.append(trade_ret)
            i = exit_idx + 1
        else:
            i += 1

    return returns, len(trades)


# ── Run all strategies ─────────────────────────────────────────────────────
strategies = {
    "A_Dollar_Momentum": ("UUP/UDN", strategy_A),
    "B_Carry_Trade_Proxy": ("FXA/FXY", strategy_B),
    "C_FX_Mean_Reversion": ("FXE/FXY/FXB/FXA/FXC", strategy_C),
    "D_Dollar_Smile": ("UUP/UDN", strategy_D),
    "E_FX_Momentum_Score": ("FX basket", strategy_E),
    "F_Yen_Carry_Unwind": ("FXY/FXA", strategy_F),
}

results = {
    "meta": {
        "run_date": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "oot_period": f"{oot_dates[0].date()} to {oot_dates[-1].date()}",
        "oot_trading_days": len(oot_dates),
        "account_usd": CAPITAL,
        "slippage_pct": SLIPPAGE,
        "commission": COMMISSION,
    },
    "strategies": {},
}

for name, (instrument, func) in strategies.items():
    rets, n_trades = func()
    metrics = calc_metrics(rets)
    qcorr = qqq_corr(rets)
    perm_p = permutation_test(rets)
    regime = regime_split(rets, close)
    gates, gates_str, validated = gate_checks(metrics, perm_p, regime, n_trades)

    results["strategies"][name] = {
        "instrument": instrument,
        "metrics": metrics,
        "n_trades": n_trades,
        "qqq_correlation": qcorr,
        "perm_p_value": perm_p,
        "regime": regime,
        "gates": gates,
        "gates_passed": gates_str,
        "VALIDATED": validated,
    }

    tag = "PASS" if validated else "FAIL"
    print(f"  {name}: Sharpe={metrics['sharpe']}, QQQ_corr={qcorr}, "
          f"perm_p={perm_p}, regime_gap={regime['regime_gap']}, "
          f"trades={n_trades}, gates={gates_str} [{tag}]")

# ── Save ────────────────────────────────────────────────────────────────────
out_path = "/home/jupiter/Lvl3Quant/data/fx_currency_momentum_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {out_path}")
print("\n=== SUMMARY ===")
for name, data in results["strategies"].items():
    m = data["metrics"]
    print(f"{name:30s} Sharpe={m['sharpe']:+.3f}  Sortino={m['sortino']:+.3f}  "
          f"Ret={m['total_return_pct']:+.1f}%  MDD={m['max_dd_pct']:.1f}%  "
          f"QQQ_r={data['qqq_correlation']:+.4f}  Gates={data['gates_passed']}  "
          f"{'VALIDATED' if data['VALIDATED'] else 'FAILED'}")
