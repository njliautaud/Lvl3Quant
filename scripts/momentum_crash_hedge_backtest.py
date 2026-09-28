#!/usr/bin/env python3
"""
Momentum Crash HEDGE Backtest
==============================
Detects momentum crash conditions and rotates between aggressive/defensive.
6 variants, walk-forward OOT: Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
QQQ correlation target: <0.3
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0

START_DATE = "2021-06-01"  # buffer for 200-SMA + lookbacks
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "UUP", "IWM", "EEM"]
VIX_TICKER = "^VIX"

N_PERM = 1000
RANDOM_SEED = 42

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/momentum_crash_hedge_results.json")


# ── Data Download ─────────────────────────────────────────────────────────────
def download_data():
    print(f"Downloading {len(TICKERS)} tickers + VIX...")
    data = yf.download(TICKERS, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    vix_data = yf.download(VIX_TICKER, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_close = vix_data["Close"].squeeze()
    elif isinstance(vix_data["Close"], pd.DataFrame):
        vix_close = vix_data["Close"].squeeze()
    else:
        vix_close = vix_data["Close"]

    close = close.ffill().dropna(how="all")
    vix_close = vix_close.ffill()

    # Make sure index is tz-naive DatetimeIndex
    if hasattr(close.index, 'tz') and close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    if hasattr(vix_close.index, 'tz') and vix_close.index.tz is not None:
        vix_close.index = vix_close.index.tz_localize(None)

    print(f"  Data range: {close.index[0].date()} to {close.index[-1].date()}")
    return close, vix_close


# ── Helpers ───────────────────────────────────────────────────────────────────
def apply_slippage(price, direction="buy"):
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def spy_regime(close_df, date):
    loc = close_df.index.get_loc(date)
    if loc < 200:
        return "bull"
    sma200 = close_df["SPY"].iloc[max(0, loc - 199):loc + 1].mean()
    return "bull" if close_df["SPY"].iloc[loc] > sma200 else "bear"


def get_val(series, date):
    """Safely get value from series at date."""
    if date in series.index:
        v = series.loc[date]
        return float(v.iloc[-1]) if isinstance(v, pd.Series) else float(v)
    prior = series.index[series.index <= date]
    if len(prior) == 0:
        return np.nan
    v = series.loc[prior[-1]]
    return float(v.iloc[-1]) if isinstance(v, pd.Series) else float(v)


def roc(close_df, ticker, date, days):
    """Rate of change over N days (percentage)."""
    loc = close_df.index.get_loc(date)
    if loc < days:
        return 0.0
    cur = float(close_df[ticker].iloc[loc])
    prev = float(close_df[ticker].iloc[loc - days])
    return (cur / prev - 1) * 100 if prev > 0 else 0.0


def realized_vol(close_df, ticker, date, days):
    """Realized volatility (annualized std of log returns) over N days."""
    loc = close_df.index.get_loc(date)
    if loc < days:
        return 0.0
    seg = close_df[ticker].iloc[max(0, loc - days + 1):loc + 1]
    lr = np.log(seg / seg.shift(1)).dropna()
    return float(lr.std() * np.sqrt(252)) if len(lr) > 2 else 0.0


def sma_val(close_df, ticker, date, period):
    """Simple moving average at date."""
    loc = close_df.index.get_loc(date)
    if loc < period - 1:
        return float(close_df[ticker].iloc[loc])
    return float(close_df[ticker].iloc[max(0, loc - period + 1):loc + 1].mean())


def n_day_return(close_df, ticker, date, days):
    """Return over last N days (fractional)."""
    loc = close_df.index.get_loc(date)
    if loc < days:
        return 0.0
    cur = float(close_df[ticker].iloc[loc])
    prev = float(close_df[ticker].iloc[loc - days])
    return (cur / prev - 1) if prev > 0 else 0.0


def count_red_days(close_df, ticker, date, lookback):
    """Count consecutive red days ending at date (inclusive)."""
    loc = close_df.index.get_loc(date)
    count = 0
    for i in range(loc, max(loc - lookback, 0), -1):
        if i < 1:
            break
        if float(close_df[ticker].iloc[i]) < float(close_df[ticker].iloc[i - 1]):
            count += 1
        else:
            break
    return count


# ── Strategy Signal Logic ────────────────────────────────────────────────────
def _variant_signal(close_df, vix_series, variant, date, loc, current_holding, hold_counter):
    """
    Determine target allocation for each variant.
    Returns: ticker string, "CASH", or tuple of (ticker, weight) pairs for splits.
    """

    if variant == "A":
        # Breadth-Based: SPY 10d ROC < -3% AND VIX 5d ROC > 30% -> TLT, else QQQ
        spy_roc_10 = roc(close_df, "SPY", date, 10)
        vix_now = get_val(vix_series, date)
        vix_5d_ago_date = close_df.index[max(0, loc - 5)]
        vix_5d = get_val(vix_series, vix_5d_ago_date)
        vix_roc_5 = ((vix_now / vix_5d) - 1) * 100 if vix_5d > 0 else 0

        if spy_roc_10 < -3.0 and vix_roc_5 > 30.0:
            return "TLT"
        # If already in TLT, stay until VIX normalizes (VIX < 20)
        if current_holding == "TLT":
            if vix_now < 20:
                return "QQQ"
            return "TLT"
        return "QQQ"

    elif variant == "B":
        # Cross-Momentum Divergence: SPY, QQQ, IWM 20d momentum
        spy_mom = n_day_return(close_df, "SPY", date, 20)
        qqq_mom = n_day_return(close_df, "QQQ", date, 20)
        iwm_mom = n_day_return(close_df, "IWM", date, 20)
        moms = [spy_mom, qqq_mom, iwm_mom]
        pos_count = sum(1 for m in moms if m > 0)
        neg_count = sum(1 for m in moms if m < 0)

        # Divergence: at least one positive AND at least one negative
        if pos_count >= 1 and neg_count >= 1:
            # Defensive: 50% cash, 50% GLD
            return (("GLD", 0.5),)  # 50% GLD, 50% stays cash
        else:
            return "QQQ"

    elif variant == "C":
        # Velocity of Decline: QQQ drops >2% in single day AND follows 2+ prior red days -> TLT for 10 days
        if current_holding == "TLT" and hold_counter < 10:
            return "TLT"

        day_ret = n_day_return(close_df, "QQQ", date, 1)
        red_days = count_red_days(close_df, "QQQ", date, 5)

        if day_ret < -0.02 and red_days >= 3:
            return "TLT"
        if current_holding == "TLT" and hold_counter >= 10:
            return "QQQ"
        if current_holding is None:
            return "QQQ"
        return current_holding if current_holding != "CASH" else "QQQ"

    elif variant == "D":
        # Vol Expansion: 5d realized vol > 2x 20d realized vol -> GLD, else QQQ
        vol_5 = realized_vol(close_df, "QQQ", date, 5)
        vol_20 = realized_vol(close_df, "QQQ", date, 20)

        if vol_20 > 0 and vol_5 > 2 * vol_20:
            return "GLD"
        if current_holding == "GLD":
            if vol_20 > 0 and vol_5 < 1.2 * vol_20:
                return "QQQ"
            return "GLD"
        return "QQQ"

    elif variant == "E":
        # Dollar Strength Hedge: UUP 10d momentum > +2% -> UUP, else QQQ
        uup_mom = roc(close_df, "UUP", date, 10)
        if uup_mom > 2.0:
            return "UUP"
        return "QQQ"

    elif variant == "F":
        # Combined Stress Score
        vix_val = get_val(vix_series, date)
        spy_sma50 = sma_val(close_df, "SPY", date, 50)
        spy_price = float(close_df["SPY"].iloc[loc])
        qqq_5d_ret = roc(close_df, "QQQ", date, 5)
        tlt_5d_ret = roc(close_df, "TLT", date, 5)

        score = 0
        if vix_val > 20:
            score += 1
        if spy_price < spy_sma50:
            score += 1
        if qqq_5d_ret < -3.0:
            score += 1
        if tlt_5d_ret > 2.0:
            score += 1

        if score >= 3:
            return "GLD"
        elif score <= 1:
            return "QQQ"
        else:
            return (("QQQ", 0.5), ("GLD", 0.5))

    return "QQQ"


# ── Strategy Runner ───────────────────────────────────────────────────────────
def run_variant(close_df, vix_series, variant):
    """Run a single variant. Returns daily equity curve and trade log."""
    oot_mask = (close_df.index >= pd.Timestamp(OOT_START)) & (close_df.index <= pd.Timestamp(OOT_END))
    oot_dates = close_df.index[oot_mask]

    equity = ACCOUNT_SIZE
    current_holding = None
    current_shares = 0.0
    current_entry = 0.0
    current_pieces = None  # for tuple holdings
    trades = []
    equity_curve = []
    regime_returns = {"bull": [], "bear": []}
    prev_equity = equity
    hold_counter = 0

    for date in oot_dates:
        regime = spy_regime(close_df, date)
        loc = close_df.index.get_loc(date)

        target = _variant_signal(close_df, vix_series, variant, date, loc, current_holding, hold_counter)

        # Normalize target for comparison
        target_key = str(target)
        holding_key = str(current_holding)

        if target_key != holding_key:
            # SELL current position
            if current_holding and current_holding != "CASH":
                if current_pieces is not None:
                    for (tk, sh, ep) in current_pieces:
                        sp = apply_slippage(float(close_df[tk].iloc[loc]), "sell")
                        pnl = (sp - ep) * sh
                        equity += pnl
                        trades.append({"date": str(date.date()), "ticker": tk, "side": "sell",
                                       "price": round(sp, 2), "pnl": round(pnl, 2)})
                elif current_shares > 0:
                    sp = apply_slippage(float(close_df[current_holding].iloc[loc]), "sell")
                    pnl = (sp - current_entry) * current_shares
                    equity += pnl
                    trades.append({"date": str(date.date()), "ticker": current_holding, "side": "sell",
                                   "price": round(sp, 2), "pnl": round(pnl, 2)})

                current_holding = None
                current_shares = 0
                current_entry = 0
                current_pieces = None

            # BUY new position
            if target and target != "CASH":
                if isinstance(target, tuple):
                    pieces = []
                    for item in target:
                        if isinstance(item, tuple) and len(item) == 2:
                            tk, wt = item
                            bp = apply_slippage(float(close_df[tk].iloc[loc]), "buy")
                            sh = (equity * wt) / bp if bp > 0 else 0
                            pieces.append((tk, sh, bp))
                            trades.append({"date": str(date.date()), "ticker": tk, "side": "buy",
                                           "price": round(bp, 2), "shares": round(sh, 4)})
                    current_holding = target
                    current_pieces = pieces
                    current_shares = 0
                    current_entry = 0
                else:
                    bp = apply_slippage(float(close_df[target].iloc[loc]), "buy")
                    current_shares = equity / bp if bp > 0 else 0
                    current_entry = bp
                    current_holding = target
                    current_pieces = None
                    trades.append({"date": str(date.date()), "ticker": target, "side": "buy",
                                   "price": round(bp, 2), "shares": round(current_shares, 4)})
            else:
                current_holding = "CASH"
                current_shares = 0
                current_entry = 0
                current_pieces = None

        # Mark to market
        if current_pieces is not None:
            port_val = equity
            for (tk, sh, ep) in current_pieces:
                cp = float(close_df[tk].iloc[loc])
                port_val += (cp - ep) * sh
        elif current_holding and current_holding != "CASH" and current_shares > 0:
            cp = float(close_df[current_holding].iloc[loc])
            port_val = equity + (cp - current_entry) * current_shares
        else:
            port_val = equity

        daily_ret = (port_val - prev_equity) / prev_equity if prev_equity > 0 else 0
        regime_returns[regime].append(daily_ret)
        equity_curve.append({"date": str(date.date()), "equity": round(port_val, 2), "regime": regime})
        prev_equity = port_val

        # Update hold counter for variant C
        if variant == "C":
            if isinstance(current_holding, str) and current_holding == "TLT":
                hold_counter += 1
            else:
                hold_counter = 0

    return equity_curve, trades, regime_returns, prev_equity


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(equity_curve, trades, regime_returns):
    eq = pd.DataFrame(equity_curve)
    eq["equity"] = eq["equity"].astype(float)
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")

    daily_returns = eq["equity"].pct_change().dropna()
    if len(daily_returns) < 10:
        return None

    total_return = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) - 1
    n_days = len(daily_returns)
    n_years = n_days / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 and n_years > 0 else -1

    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

    downside = daily_returns[daily_returns < 0]
    ds_std = downside.std() if len(downside) > 0 else 1e-6
    sortino = (mean_ret / ds_std) * np.sqrt(252) if ds_std > 0 else 0

    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    closed = [t for t in trades if t["side"] == "sell"]
    n_trades = len(closed)
    winners = [t for t in closed if t.get("pnl", 0) > 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0

    gross_profit = sum(t["pnl"] for t in closed if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in closed if t.get("pnl", 0) < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0)

    bull_rets = regime_returns.get("bull", [])
    bear_rets = regime_returns.get("bear", [])
    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets)) * np.sqrt(252) if len(bull_rets) > 20 and np.std(bull_rets) > 0 else 0
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets)) * np.sqrt(252) if len(bear_rets) > 20 and np.std(bear_rets) > 0 else 0
    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)

    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "profit_factor": round(pf, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "n_trades": n_trades,
        "final_equity": round(eq["equity"].iloc[-1], 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_days": len(bull_rets),
        "bear_days": len(bear_rets),
        "daily_returns": daily_returns,
    }


def qqq_correlation(strat_daily_returns, close_df):
    """Correlation between strategy daily returns and QQQ buy-and-hold."""
    qqq_rets = close_df["QQQ"].pct_change().dropna()
    qqq_rets.index = pd.to_datetime(qqq_rets.index)
    common = pd.concat([strat_daily_returns.rename("strat"), qqq_rets.rename("qqq")], axis=1).dropna()
    if len(common) < 20:
        return 0.0
    return float(common["strat"].corr(common["qqq"]))


# ── Permutation Test ──────────────────────────────────────────────────────────
def permutation_test(close_df, vix_series, variant, actual_sharpe, n_perm=N_PERM):
    """Shuffle signal dates to test significance."""
    rng = np.random.RandomState(RANDOM_SEED)

    oot_mask = (close_df.index >= pd.Timestamp(OOT_START)) & (close_df.index <= pd.Timestamp(OOT_END))
    oot_dates = close_df.index[oot_mask]

    # Build actual signal sequence
    signals = []
    hold_counter = 0
    current = None
    for date in oot_dates:
        loc = close_df.index.get_loc(date)
        sig = _variant_signal(close_df, vix_series, variant, date, loc, current, hold_counter)
        signals.append(sig)
        if variant == "C" and isinstance(sig, str) and sig == "TLT":
            hold_counter += 1
        elif variant == "C":
            hold_counter = 0
        current = sig

    perm_sharpes = []
    for _ in range(n_perm):
        shuffled = signals.copy()
        rng.shuffle(shuffled)

        equity = ACCOUNT_SIZE
        prev_eq = equity
        c_holding = None
        c_shares = 0.0
        c_entry = 0.0
        c_pieces = None
        daily_rets = []

        for i, date in enumerate(oot_dates):
            loc = close_df.index.get_loc(date)
            target = shuffled[i]
            target_key = str(target)
            holding_key = str(c_holding)

            if target_key != holding_key:
                # Sell
                if c_holding and c_holding != "CASH":
                    if c_pieces is not None:
                        for (tk, sh, ep) in c_pieces:
                            sp = apply_slippage(float(close_df[tk].iloc[loc]), "sell")
                            equity += (sp - ep) * sh
                    elif c_shares > 0:
                        sp = apply_slippage(float(close_df[c_holding].iloc[loc]), "sell")
                        equity += (sp - c_entry) * c_shares
                    c_holding = None
                    c_shares = 0
                    c_entry = 0
                    c_pieces = None

                # Buy
                if target and target != "CASH":
                    if isinstance(target, tuple):
                        pieces = []
                        for item in target:
                            if isinstance(item, tuple) and len(item) == 2:
                                tk, wt = item
                                bp = apply_slippage(float(close_df[tk].iloc[loc]), "buy")
                                sh = (equity * wt) / bp if bp > 0 else 0
                                pieces.append((tk, sh, bp))
                        c_holding = target
                        c_pieces = pieces
                    else:
                        bp = apply_slippage(float(close_df[target].iloc[loc]), "buy")
                        c_shares = equity / bp if bp > 0 else 0
                        c_entry = bp
                        c_holding = target
                        c_pieces = None
                else:
                    c_holding = "CASH"
                    c_shares = 0
                    c_entry = 0
                    c_pieces = None

            # MTM
            if c_pieces is not None:
                port_val = equity
                for (tk, sh, ep) in c_pieces:
                    cp = float(close_df[tk].iloc[loc])
                    port_val += (cp - ep) * sh
            elif c_holding and c_holding != "CASH" and c_shares > 0:
                cp = float(close_df[c_holding].iloc[loc])
                port_val = equity + (cp - c_entry) * c_shares
            else:
                port_val = equity

            daily_ret = (port_val - prev_eq) / prev_eq if prev_eq > 0 else 0
            daily_rets.append(daily_ret)
            prev_eq = port_val

        rets = np.array(daily_rets)
        s = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= actual_sharpe)
    return round(float(p_value), 4), round(float(np.mean(perm_sharpes)), 3), round(float(np.std(perm_sharpes)), 3)


# ── 5-Gate Validation ─────────────────────────────────────────────────────────
def validate_5gate(metrics, p_value):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MOMENTUM CRASH HEDGE BACKTEST")
    print("Walk-Forward OOT: Jan 2022 - Jul 2026 | Account: $645")
    print("=" * 70)

    close_df, vix = download_data()

    variants = {
        "A": "Breadth-Based (SPY ROC + VIX spike -> TLT)",
        "B": "Cross-Momentum Divergence (SPY/QQQ/IWM -> GLD)",
        "C": "Velocity of Decline (QQQ crash -> TLT 10d)",
        "D": "Vol Expansion (5d vol > 2x 20d vol -> GLD)",
        "E": "Dollar Strength Hedge (UUP momentum -> UUP)",
        "F": "Combined Stress Score (4-factor -> GLD/QQQ/blend)",
    }

    results = {
        "strategy": "Momentum Crash Hedge",
        "account_size": ACCOUNT_SIZE,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "slippage_pct": SLIPPAGE_PCT,
        "commission": COMMISSION,
        "n_permutations": N_PERM,
        "timestamp": dt.datetime.now().isoformat(),
        "variants": {},
    }

    for var_key, var_name in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {var_key}: {var_name}")
        print(f"{'─' * 60}")

        eq_curve, trade_log, regime_rets, final_val = run_variant(close_df, vix, var_key)
        metrics = compute_metrics(eq_curve, trade_log, regime_rets)

        if metrics is None:
            print("  !! Insufficient data for metrics")
            results["variants"][var_key] = {"name": var_name, "error": "insufficient data"}
            continue

        qqq_corr = qqq_correlation(metrics["daily_returns"], close_df)

        print(f"  Final Equity: ${metrics['final_equity']:.2f}  |  Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}  |  MaxDD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Win Rate: {metrics['win_rate_pct']:.1f}%  |  PF: {metrics['profit_factor']:.2f}  |  Trades: {metrics['n_trades']}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f}  |  Bear Sharpe: {metrics['bear_sharpe']:.3f}  |  Regime Gap: {metrics['regime_gap']:.3f}")
        print(f"  QQQ Correlation: {qqq_corr:.3f}  {'<0.3 TARGET' if abs(qqq_corr) < 0.3 else '>0.3 HIGH'}")

        # Permutation test
        print(f"  Running {N_PERM} permutations...")
        p_val, perm_mean, perm_std = permutation_test(close_df, vix, var_key, metrics["sharpe"])
        print(f"  Perm p-value: {p_val}  |  Perm Sharpe mean: {perm_mean} +/- {perm_std}")

        # 5-gate
        gates = validate_5gate(metrics, p_val)
        gate_status = "PASS" if gates["all_passed"] else "FAIL"
        failed = [k for k, v in gates.items() if not v and k != "all_passed"]
        print(f"  5-Gate: {gate_status}" + (f"  (failed: {', '.join(failed)})" if failed else ""))

        metrics_save = {k: v for k, v in metrics.items() if k != "daily_returns"}

        results["variants"][var_key] = {
            "name": var_name,
            "metrics": metrics_save,
            "qqq_correlation": round(qqq_corr, 4),
            "permutation": {"p_value": p_val, "perm_sharpe_mean": perm_mean, "perm_sharpe_std": perm_std},
            "five_gate": gates,
            "gate_result": gate_status,
            "sample_trades": trade_log[:10],
            "equity_curve_endpoints": {
                "start": eq_curve[0] if eq_curve else None,
                "end": eq_curve[-1] if eq_curve else None,
            }
        }

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Var':<4} {'Name':<50} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'QQQ r':>6} {'Gate':>5}")
    print(f"{'─' * 87}")
    for var_key in variants:
        v = results["variants"][var_key]
        if "error" in v:
            print(f"  {var_key:<3} {v['name']:<50} {'ERROR':>7}")
            continue
        m = v["metrics"]
        print(f"  {var_key:<3} {v['name']:<50} {m['sharpe']:>7.3f} {m['total_return_pct']:>7.1f}% {m['max_drawdown_pct']:>6.1f}% {v['qqq_correlation']:>5.2f} {v['gate_result']:>5}")

    # Recommendation
    passing = [(k, v) for k, v in results["variants"].items()
               if "metrics" in v and v["five_gate"].get("all_passed")]
    if passing:
        best_key, best = max(passing, key=lambda x: x[1]["metrics"]["sharpe"])
        results["recommendation"] = {
            "best_variant": best_key,
            "name": best["name"],
            "sharpe": best["metrics"]["sharpe"],
            "qqq_correlation": best["qqq_correlation"],
            "reason": "Highest Sharpe among 5-gate passing variants"
        }
        print(f"\n  RECOMMENDED: Variant {best_key} - {best['name']}")
        print(f"    Sharpe {best['metrics']['sharpe']:.3f} | Sortino {best['metrics']['sortino']:.3f} | "
              f"Return {best['metrics']['total_return_pct']:.1f}% | MaxDD {best['metrics']['max_drawdown_pct']:.1f}% | "
              f"QQQ corr {best['qqq_correlation']:.3f}")
    else:
        all_with = [(k, v) for k, v in results["variants"].items() if "metrics" in v]
        if all_with:
            best_key, best = max(all_with, key=lambda x: x[1]["metrics"]["sharpe"])
            results["recommendation"] = {
                "best_variant": best_key,
                "name": best["name"],
                "sharpe": best["metrics"]["sharpe"],
                "qqq_correlation": best["qqq_correlation"],
                "reason": "Highest Sharpe (none passed all 5 gates)"
            }
            print(f"\n  NO VARIANT PASSED ALL 5 GATES.")
            print(f"  Best available: Variant {best_key} - {best['name']}")
            print(f"    Sharpe {best['metrics']['sharpe']:.3f} | MaxDD {best['metrics']['max_drawdown_pct']:.1f}% | QQQ corr {best['qqq_correlation']:.3f}")
            failed_gates = [k for k, v in best["five_gate"].items() if not v and k != "all_passed"]
            print(f"    Failed gates: {', '.join(failed_gates)}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
