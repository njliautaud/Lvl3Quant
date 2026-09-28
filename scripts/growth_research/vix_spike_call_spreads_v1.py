#!/usr/bin/env python3
"""
VIX Spike Recovery Call Spreads v1
===================================
Prior research: VIX>30 → SPY +7% 1mo (71% WR), +20.4% 3mo (75% WR).
VIX>40 → +49% avg 3mo UPRO (91% WR). 28 events since 2010.

But with $645 capital, can't buy UPRO shares effectively.
This tests: buy bull call spreads on SPY (and sector ETFs) after VIX spikes.

6 Variants:
  A. VIX>30, SPY call spreads, 45 DTE, hold to expiry
  B. VIX>30, TOP 3 sector ETF call spreads (LGBM ranking), 45 DTE
  C. VIX>25, SPY call spreads, 45 DTE (more trades)
  D. VIX>30, SPY call spreads, 90 DTE (longer for bigger recovery)
  E. VIX>30 AND VIX 5d change > +5pts (confirmed spike), 45 DTE
  F. VIX>30, deep OTM (5% OTM) SPY call spreads, 45 DTE

MLflow experiment: vix_spike_call_spreads_v1
"""
import sys
import warnings
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
    exit_spread_value,
    spread_pnl,
)
from research.tools.adversarial_validator import validate_trades

def fprint(*a, **kw):
    print(*a, **kw, flush=True)

OUT_DIR = Path(__file__).resolve().parents[2] / "research" / "findings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
SPREAD_WIDTH_PCT = 0.03  # 3% spread width
OTM_OFFSET_PCT = 0.05    # 5% OTM for variant F
COMMISSION_RT = COMMISSION_RT_SPREAD  # $2.60
ENTRY_HAIRCUT = DEFAULT_HAIRCUT  # 15%
# Cooldown = DTE so positions don't overlap; set per-variant in run_variant
DEFAULT_COOLDOWN_DAYS = 45  # overridden by DTE in each variant

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
EXTRA_TICKERS = ["TLT", "HYG", "GLD", "SHY"]


# ═══════════════════════════════════════════════════════════════════════
# Data Download
# ═══════════════════════════════════════════════════════════════════════

def download_data():
    """Download SPY, VIX, sector ETFs, and cross-asset data."""
    import yfinance as yf

    all_tickers = ["SPY", "^VIX"] + SECTOR_ETFS + EXTRA_TICKERS
    fprint("Downloading data...")
    frames = {}
    for tk in all_tickers:
        name = tk.replace("^", "")
        fprint(f"  {name}...", end=" ")
        df = yf.download(tk, start="2006-01-01", end="2026-07-27", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames[name] = df[["Open", "High", "Low", "Close"]].rename(
            columns=lambda c: f"{name}_{c}"
        )
        fprint(f"{len(df)} rows")

    merged = pd.concat(frames.values(), axis=1)
    # Forward-fill gaps (sector ETFs start later)
    merged = merged.ffill().dropna(subset=["SPY_Close", "VIX_Close"])
    fprint(f"Merged: {len(merged)} days, {merged.index[0].date()} to {merged.index[-1].date()}")
    return merged


# ═══════════════════════════════════════════════════════════════════════
# Signal Detection
# ═══════════════════════════════════════════════════════════════════════

def find_vix_spike_dates(data, vix_threshold=30, cooldown_days=45,
                         require_spike_speed=False, spike_pts=5,
                         start_year=2009):
    """Find FIRST CROSS above threshold (not every day above it).

    Key logic: VIX must have been BELOW threshold within the lookback
    window before we count a new spike. This prevents repeated signals
    during prolonged high-VIX periods (e.g. 2008 crisis).

    Cooldown = DTE so positions don't overlap.
    """
    vix = data["VIX_Close"]
    signals = []
    last_signal = None
    was_below = True  # track if VIX was below threshold recently

    for i in range(20, len(vix)):
        dt = vix.index[i]

        # Only start from specified year
        if dt.year < start_year:
            # But track was_below state
            if vix.iloc[i] < vix_threshold:
                was_below = True
            continue

        v = vix.iloc[i]

        # Track when VIX drops below threshold
        if v < vix_threshold:
            was_below = True
            continue

        # VIX is above threshold — but is this a NEW cross?
        if not was_below:
            continue  # still in same elevated period

        # Cooldown from last signal
        if last_signal is not None and (dt - last_signal).days < cooldown_days:
            continue

        # Speed filter
        if require_spike_speed:
            v_5d_ago = vix.iloc[i - 5]
            if (v - v_5d_ago) < spike_pts:
                continue

        signals.append(dt)
        last_signal = dt
        was_below = False  # don't trigger again until VIX drops below

    return signals


# ═══════════════════════════════════════════════════════════════════════
# LGBM Sector Ranker (for Variant B)
# ═══════════════════════════════════════════════════════════════════════

def build_sector_features(data, sector, lookback_idx):
    """Build features for sector ranking at a specific point in time."""
    end = lookback_idx
    if end < 60:
        return None

    close_col = f"{sector}_Close"
    if close_col not in data.columns:
        return None

    close = data[close_col].iloc[:end + 1]
    spy_close = data["SPY_Close"].iloc[:end + 1]
    vix = data["VIX_Close"].iloc[:end + 1]

    if len(close) < 60 or close.iloc[-1] <= 0:
        return None

    ret = close.pct_change()
    spy_ret = spy_close.pct_change()

    feats = {}
    # Momentum features
    for w in [5, 10, 20, 60]:
        if len(close) > w:
            feats[f"ret_{w}d"] = float(close.iloc[-1] / close.iloc[-w] - 1)
        else:
            feats[f"ret_{w}d"] = 0.0
    # Relative strength vs SPY
    for w in [10, 20]:
        if len(close) > w:
            feats[f"rs_spy_{w}d"] = float(
                (close.iloc[-1] / close.iloc[-w]) / (spy_close.iloc[-1] / spy_close.iloc[-w]) - 1
            )
        else:
            feats[f"rs_spy_{w}d"] = 0.0
    # Volatility
    if len(ret) > 20:
        feats["vol_20d"] = float(ret.iloc[-20:].std() * np.sqrt(252))
    else:
        feats["vol_20d"] = 0.2
    # Drawdown from 20d high
    if len(close) > 20:
        feats["dd_20d"] = float(close.iloc[-1] / close.iloc[-20:].max() - 1)
    else:
        feats["dd_20d"] = 0.0
    # VIX interaction
    feats["vix_level"] = float(vix.iloc[-1])
    feats["vix_chg_5d"] = float(vix.iloc[-1] - vix.iloc[-5]) if len(vix) > 5 else 0.0
    # Correlation with SPY
    if len(ret) > 60:
        feats["spy_corr_60d"] = float(ret.iloc[-60:].corr(spy_ret.iloc[-60:]))
    else:
        feats["spy_corr_60d"] = 0.5
    # Beta
    if len(ret) > 60:
        cov = ret.iloc[-60:].cov(spy_ret.iloc[-60:])
        var = spy_ret.iloc[-60:].var()
        feats["beta_60d"] = float(cov / var) if var > 0 else 1.0
    else:
        feats["beta_60d"] = 1.0
    # RSI-like
    if len(ret) > 14:
        gains = ret.iloc[-14:].clip(lower=0).mean()
        losses = (-ret.iloc[-14:].clip(upper=0)).mean()
        feats["rsi_14"] = float(100 - 100 / (1 + gains / losses)) if losses > 0 else 50.0
    else:
        feats["rsi_14"] = 50.0
    # Mean reversion potential (distance from 20d SMA)
    if len(close) > 20:
        feats["dist_sma20"] = float(close.iloc[-1] / close.iloc[-20:].mean() - 1)
    else:
        feats["dist_sma20"] = 0.0
    # Breadth: how many sectors are down > X%
    feats["sectors_down_5pct"] = 0.0  # placeholder, filled at caller level
    # Volume proxy: ATR ratio
    if len(close) > 28:
        high = data[f"{sector}_High"].iloc[:end + 1] if f"{sector}_High" in data.columns else close
        low = data[f"{sector}_Low"].iloc[:end + 1] if f"{sector}_Low" in data.columns else close
        recent_atr = compute_atr(high.iloc[-28:], low.iloc[-28:], close.iloc[-28:], 14)
        feats["atr_pct"] = recent_atr / close.iloc[-1] if close.iloc[-1] > 0 else 0.0
    else:
        feats["atr_pct"] = 0.02

    return feats


def lgbm_rank_sectors(data, signal_dates, train_window=504):
    """Walk-forward LGBM to rank sectors by expected forward return.

    For each signal date, train on history before it, predict forward 45d return,
    return top-3 sectors.
    """
    from lightgbm import LGBMRegressor

    fprint("  Building LGBM sector ranker (walk-forward)...")

    # Pre-compute forward returns for all sectors at all dates
    fwd_returns = {}
    for sector in SECTOR_ETFS:
        col = f"{sector}_Close"
        if col in data.columns:
            fwd_returns[sector] = data[col].pct_change(45).shift(-45)

    results = {}  # signal_date -> list of (sector, predicted_return) top 3

    for sig_date in signal_dates:
        sig_idx = data.index.get_loc(sig_date)

        # Build training data: all dates from max(0, sig_idx-train_window) to sig_idx-45
        # (we need 45d forward return to be known)
        train_end = sig_idx - 45
        train_start = max(60, train_end - train_window)

        if train_end - train_start < 100:
            # Not enough training data — fallback to equal weight
            results[sig_date] = SECTOR_ETFS[:3]
            continue

        X_train, y_train = [], []
        for idx in range(train_start, train_end, 10):  # every 10th day for speed
            for sector in SECTOR_ETFS:
                feats = build_sector_features(data, sector, idx)
                if feats is None:
                    continue
                fwd_col = f"{sector}_Close"
                if fwd_col not in data.columns:
                    continue
                fwd_ret = fwd_returns[sector].iloc[idx]
                if pd.isna(fwd_ret):
                    continue
                X_train.append(feats)
                y_train.append(fwd_ret)

        if len(X_train) < 50:
            results[sig_date] = SECTOR_ETFS[:3]
            continue

        X_df = pd.DataFrame(X_train)
        y_arr = np.array(y_train)

        model = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            verbose=-1,
            random_state=42,
        )
        model.fit(X_df, y_arr)

        # Predict current sectors
        preds = {}
        for sector in SECTOR_ETFS:
            feats = build_sector_features(data, sector, sig_idx)
            if feats is None:
                continue
            feat_df = pd.DataFrame([feats])
            preds[sector] = model.predict(feat_df)[0]

        # Top 3 by predicted return
        sorted_sectors = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        top3 = [s[0] for s in sorted_sectors[:3]]
        results[sig_date] = top3

    return results


# ═══════════════════════════════════════════════════════════════════════
# Trade Execution
# ═══════════════════════════════════════════════════════════════════════

def execute_call_spread_trade(
    data,
    entry_date,
    ticker,
    dte,
    capital_available,
    otm_offset=0.0,
):
    """Execute a bull call spread trade and return trade dict.

    Args:
        otm_offset: 0.0 for ATM, 0.05 for 5% OTM
    Returns:
        trade dict or None if can't execute
    """
    entry_idx = data.index.get_loc(entry_date)
    close_col = f"{ticker}_Close"
    high_col = f"{ticker}_High"
    low_col = f"{ticker}_Low"

    if close_col not in data.columns:
        return None

    S = float(data[close_col].iloc[entry_idx])
    if S <= 0:
        return None

    # Compute ATR for IV estimation
    lookback = min(entry_idx, 28)
    if lookback < 14:
        return None

    high = data[high_col].iloc[entry_idx - lookback : entry_idx + 1] if high_col in data.columns else None
    low = data[low_col].iloc[entry_idx - lookback : entry_idx + 1] if low_col in data.columns else None
    close = data[close_col].iloc[entry_idx - lookback : entry_idx + 1]

    if high is not None and low is not None:
        atr = compute_atr(high, low, close, 14)
    else:
        atr = S * 0.015  # fallback: 1.5% of price

    vix = float(data["VIX_Close"].iloc[entry_idx])

    # Strike selection
    K1 = S * (1.0 + otm_offset)  # lower strike
    K2 = K1 * (1.0 + SPREAD_WIDTH_PCT)  # upper strike

    # Price the spread
    entry_cost_ps, max_profit_ps = price_bull_call_spread(
        S=S, K1=K1, K2=K2, dte=dte, atr=atr, vix=vix, haircut=ENTRY_HAIRCUT
    )

    entry_cost_contract = entry_cost_ps * 100  # per contract
    if entry_cost_contract <= 0:
        return None

    # Position sizing: max contracts within budget
    max_contracts = int(min(capital_available, MAX_PER_TRADE) / (entry_cost_contract + COMMISSION_RT))
    if max_contracts < 1:
        return None

    total_cost = max_contracts * entry_cost_contract + max_contracts * COMMISSION_RT

    # Find expiry date
    expiry_idx = min(entry_idx + dte, len(data) - 1)
    expiry_date = data.index[expiry_idx]
    S_expiry = float(data[close_col].iloc[expiry_idx])

    # Expiry value (intrinsic, no haircut at expiry)
    exit_value_ps = exit_spread_value(
        S=S_expiry, K1=K1, K2=K2, remaining_dte=0, original_dte=dte, atr=atr, vix=vix
    )

    # PnL
    pnl = spread_pnl(entry_cost_ps, exit_value_ps, contracts=max_contracts, commission_rt=COMMISSION_RT)

    # SPY return during holding period (for regime classification)
    spy_entry = float(data["SPY_Close"].iloc[entry_idx])
    spy_exit = float(data["SPY_Close"].iloc[expiry_idx])
    spy_return = (spy_exit / spy_entry - 1) if spy_entry > 0 else 0.0

    return {
        "entry_date": str(entry_date.date()),
        "exit_date": str(expiry_date.date()),
        "ticker": ticker,
        "pnl": float(pnl),
        "entry_price": S,
        "expiry_price": S_expiry,
        "K1": round(K1, 2),
        "K2": round(K2, 2),
        "contracts": max_contracts,
        "entry_cost_total": round(total_cost, 2),
        "exit_value_total": round(exit_value_ps * max_contracts * 100, 2),
        "vix_at_entry": round(vix, 1),
        "spy_return_during": round(spy_return * 100, 2),
        "dte": dte,
        "otm_offset": otm_offset,
    }


# ═══════════════════════════════════════════════════════════════════════
# Backtest Variants
# ═══════════════════════════════════════════════════════════════════════

def run_variant(data, variant_name, signal_dates, ticker_list, dte, otm_offset=0.0,
                sector_rankings=None):
    """Run backtest for one variant."""
    fprint(f"\n{'='*60}")
    fprint(f"  Variant {variant_name}")
    fprint(f"  Signals: {len(signal_dates)} | DTE: {dte} | OTM: {otm_offset*100:.0f}%")
    fprint(f"{'='*60}")

    trades = []
    equity = INITIAL_CAPITAL

    for sig_date in signal_dates:
        if equity <= 50:
            fprint(f"  Equity depleted at {sig_date.date()}, stopping")
            break

        # Determine tickers for this signal
        if sector_rankings is not None and sig_date in sector_rankings:
            tickers = sector_rankings[sig_date]
        else:
            tickers = ticker_list

        per_ticker_budget = min(equity, MAX_PER_TRADE * len(tickers)) / len(tickers)

        for ticker in tickers:
            trade = execute_call_spread_trade(
                data, sig_date, ticker, dte,
                capital_available=per_ticker_budget,
                otm_offset=otm_offset,
            )
            if trade is not None:
                trades.append(trade)
                equity += trade["pnl"]

    fprint(f"  Total trades: {len(trades)}")
    if trades:
        pnls = [t["pnl"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        fprint(f"  Win rate: {wins/len(pnls)*100:.1f}%")
        fprint(f"  Avg PnL: ${np.mean(pnls):.2f}")
        fprint(f"  Total PnL: ${sum(pnls):.2f}")
        fprint(f"  Final equity: ${equity:.2f}")

    return trades, equity


def run_all_variants(data):
    """Run all 6 variants and collect results."""
    results = {}

    # === Variant A: VIX>30, SPY, 45 DTE ===
    signals_30_45 = find_vix_spike_dates(data, vix_threshold=30, cooldown_days=45)
    fprint(f"\nVIX>30 signal dates (45d cooldown): {len(signals_30_45)}")
    for d in signals_30_45[:8]:
        fprint(f"  {d.date()} VIX={data['VIX_Close'].loc[d]:.1f}")
    if len(signals_30_45) > 8:
        fprint(f"  ... and {len(signals_30_45)-8} more")

    trades_a, eq_a = run_variant(data, "A: VIX>30 SPY 45DTE", signals_30_45, ["SPY"], 45)
    results["A"] = {"trades": trades_a, "equity": eq_a, "label": "VIX>30 SPY 45DTE ATM"}

    # === Variant B: VIX>30, Top 3 Sectors (LGBM), 45 DTE ===
    sector_rankings = lgbm_rank_sectors(data, signals_30_45)
    trades_b, eq_b = run_variant(
        data, "B: VIX>30 Top3 Sectors 45DTE", signals_30_45, SECTOR_ETFS[:3],
        45, sector_rankings=sector_rankings
    )
    results["B"] = {"trades": trades_b, "equity": eq_b, "label": "VIX>30 Top3Sector 45DTE"}

    # === Variant C: VIX>25, SPY, 45 DTE ===
    signals_25 = find_vix_spike_dates(data, vix_threshold=25, cooldown_days=45)
    fprint(f"\nVIX>25 signal dates (45d cooldown): {len(signals_25)}")
    trades_c, eq_c = run_variant(data, "C: VIX>25 SPY 45DTE", signals_25, ["SPY"], 45)
    results["C"] = {"trades": trades_c, "equity": eq_c, "label": "VIX>25 SPY 45DTE ATM"}

    # === Variant D: VIX>30, SPY, 90 DTE ===
    signals_30_90 = find_vix_spike_dates(data, vix_threshold=30, cooldown_days=90)
    fprint(f"\nVIX>30 signal dates (90d cooldown): {len(signals_30_90)}")
    trades_d, eq_d = run_variant(data, "D: VIX>30 SPY 90DTE", signals_30_90, ["SPY"], 90)
    results["D"] = {"trades": trades_d, "equity": eq_d, "label": "VIX>30 SPY 90DTE ATM"}

    # === Variant E: VIX>30 + 5pt spike, SPY, 45 DTE ===
    signals_spike = find_vix_spike_dates(
        data, vix_threshold=30, cooldown_days=45,
        require_spike_speed=True, spike_pts=5
    )
    fprint(f"\nVIX>30 + 5pt spike dates: {len(signals_spike)}")
    trades_e, eq_e = run_variant(data, "E: VIX>30+spike SPY 45DTE", signals_spike, ["SPY"], 45)
    results["E"] = {"trades": trades_e, "equity": eq_e, "label": "VIX>30+spike SPY 45DTE"}

    # === Variant F: VIX>30, SPY 5% OTM, 45 DTE ===
    trades_f, eq_f = run_variant(
        data, "F: VIX>30 SPY 5%OTM 45DTE", signals_30_45, ["SPY"], 45, otm_offset=OTM_OFFSET_PCT
    )
    results["F"] = {"trades": trades_f, "equity": eq_f, "label": "VIX>30 SPY 5%OTM 45DTE"}

    return results


# ═══════════════════════════════════════════════════════════════════════
# Adversarial Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_all(results, spy_prices):
    """Run 5-gate adversarial validation on each variant."""
    fprint("\n" + "=" * 65)
    fprint("  ADVERSARIAL VALIDATION — ALL VARIANTS")
    fprint("=" * 65)

    validations = {}
    for key, res in results.items():
        trades = res["trades"]
        label = res["label"]
        if len(trades) < 3:
            fprint(f"\n  Variant {key} ({label}): only {len(trades)} trades, skipping validation")
            validations[key] = None
            continue

        vr = validate_trades(
            trades=trades,
            initial_capital=INITIAL_CAPITAL,
            spy_prices=spy_prices,
            strategy_name=f"Variant {key}: {label}",
        )
        vr.print_summary()
        validations[key] = vr

    return validations


# ═══════════════════════════════════════════════════════════════════════
# MLflow Logging
# ═══════════════════════════════════════════════════════════════════════

def log_to_mlflow(results, validations):
    """Log all results to MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("vix_spike_call_spreads_v1")

        for key, res in results.items():
            label = res["label"]
            trades = res["trades"]
            vr = validations.get(key)

            with mlflow.start_run(run_name=f"Variant_{key}_{label.replace(' ', '_')}"):
                mlflow.log_param("variant", key)
                mlflow.log_param("label", label)
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("max_per_trade", MAX_PER_TRADE)
                mlflow.log_param("spread_width_pct", SPREAD_WIDTH_PCT)
                mlflow.log_param("entry_haircut", ENTRY_HAIRCUT)
                mlflow.log_param("commission_rt", COMMISSION_RT)

                mlflow.log_metric("n_trades", len(trades))
                mlflow.log_metric("final_equity", res["equity"])
                mlflow.log_metric("total_pnl", res["equity"] - INITIAL_CAPITAL)
                mlflow.log_metric("total_return_pct", (res["equity"] / INITIAL_CAPITAL - 1) * 100)

                if trades:
                    pnls = [t["pnl"] for t in trades]
                    mlflow.log_metric("win_rate", sum(1 for p in pnls if p > 0) / len(pnls))
                    mlflow.log_metric("avg_pnl", np.mean(pnls))
                    mlflow.log_metric("avg_win", np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0)
                    mlflow.log_metric("avg_loss", np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0)

                if vr is not None:
                    mlflow.log_metric("sharpe", vr.sharpe)
                    mlflow.log_metric("sortino", vr.sortino)
                    mlflow.log_metric("cagr", vr.cagr)
                    mlflow.log_metric("max_dd", vr.max_dd)
                    mlflow.log_metric("profit_factor", min(vr.profit_factor, 99))
                    mlflow.log_metric("gates_passed", vr.gates_passed)
                    mlflow.log_metric("gates_total", vr.gates_total)

        fprint("\nMLflow logging complete.")
    except Exception as e:
        fprint(f"\nMLflow logging failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Summary Report
# ═══════════════════════════════════════════════════════════════════════

def print_summary(results, validations):
    """Print final comparison table."""
    fprint("\n" + "=" * 90)
    fprint("  FINAL COMPARISON — VIX SPIKE CALL SPREADS")
    fprint("=" * 90)
    fprint(f"{'Variant':<8} {'Label':<30} {'Trades':>6} {'WR':>6} {'AvgPnL':>8} "
           f"{'TotalPnL':>9} {'Final$':>8} {'Sharpe':>7} {'Gates':>6}")
    fprint("-" * 90)

    for key in ["A", "B", "C", "D", "E", "F"]:
        res = results[key]
        trades = res["trades"]
        label = res["label"][:29]
        vr = validations.get(key)

        n = len(trades)
        if n > 0:
            pnls = [t["pnl"] for t in trades]
            wr = sum(1 for p in pnls if p > 0) / n * 100
            avg_pnl = np.mean(pnls)
            total_pnl = sum(pnls)
        else:
            wr = avg_pnl = total_pnl = 0

        sharpe = vr.sharpe if vr else 0.0
        gates = f"{vr.gates_passed}/{vr.gates_total}" if vr else "N/A"

        fprint(f"{key:<8} {label:<30} {n:>6} {wr:>5.1f}% ${avg_pnl:>7.2f} "
               f"${total_pnl:>8.2f} ${res['equity']:>7.2f} {sharpe:>7.2f} {gates:>6}")

    fprint("=" * 90)

    # Key insights
    fprint("\nKEY INSIGHTS:")

    # Best variant by total PnL
    best_key = max(results, key=lambda k: results[k]["equity"])
    best = results[best_key]
    fprint(f"  Best variant: {best_key} ({best['label']}) — Final equity ${best['equity']:.2f}")

    # Trade detail for best variant
    if best["trades"]:
        pnls = [t["pnl"] for t in best["trades"]]
        fprint(f"  Best variant stats: {len(pnls)} trades, "
               f"WR={sum(1 for p in pnls if p > 0)/len(pnls)*100:.0f}%, "
               f"avg=${np.mean(pnls):.2f}, total=${sum(pnls):.2f}")

    # DTE comparison
    if results["A"]["trades"] and results["D"]["trades"]:
        pnl_a = sum(t["pnl"] for t in results["A"]["trades"])
        pnl_d = sum(t["pnl"] for t in results["D"]["trades"])
        fprint(f"  45 DTE vs 90 DTE: ${pnl_a:.2f} vs ${pnl_d:.2f} "
               f"({'90 DTE better' if pnl_d > pnl_a else '45 DTE better'})")

    # ATM vs OTM
    if results["A"]["trades"] and results["F"]["trades"]:
        pnl_a = sum(t["pnl"] for t in results["A"]["trades"])
        pnl_f = sum(t["pnl"] for t in results["F"]["trades"])
        fprint(f"  ATM vs 5% OTM: ${pnl_a:.2f} vs ${pnl_f:.2f} "
               f"({'OTM better' if pnl_f > pnl_a else 'ATM better'})")

    # VIX>25 vs VIX>30
    if results["A"]["trades"] and results["C"]["trades"]:
        n_a = len(results["A"]["trades"])
        n_c = len(results["C"]["trades"])
        wr_a = sum(1 for t in results["A"]["trades"] if t["pnl"] > 0) / max(n_a, 1) * 100
        wr_c = sum(1 for t in results["C"]["trades"] if t["pnl"] > 0) / max(n_c, 1) * 100
        fprint(f"  VIX>30 ({n_a} trades, {wr_a:.0f}% WR) vs VIX>25 ({n_c} trades, {wr_c:.0f}% WR)")

    # Year-by-year for best variant
    if best["trades"]:
        df_trades = pd.DataFrame(best["trades"])
        df_trades["year"] = pd.to_datetime(df_trades["exit_date"]).dt.year
        yearly = df_trades.groupby("year")["pnl"].agg(["sum", "count"])
        fprint(f"\n  Year-by-year for Variant {best_key}:")
        for yr, row in yearly.iterrows():
            marker = "+" if row["sum"] > 0 else ""
            fprint(f"    {yr}: {marker}${row['sum']:.2f} ({int(row['count'])} trades)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    fprint("=" * 65)
    fprint("  VIX SPIKE RECOVERY — BULL CALL SPREADS v1")
    fprint(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"  Capital: ${INITIAL_CAPITAL:.0f} | Max/trade: ${MAX_PER_TRADE:.0f}")
    fprint("=" * 65)

    t0 = time.time()

    # Download data
    data = download_data()

    # SPY prices for regime classification
    spy_prices = data["SPY_Close"].copy()
    spy_prices.index = pd.to_datetime(spy_prices.index)

    # Run all 6 variants
    results = run_all_variants(data)

    # Adversarial validation
    validations = validate_all(results, spy_prices)

    # MLflow logging
    log_to_mlflow(results, validations)

    # Final summary
    print_summary(results, validations)

    elapsed = time.time() - t0
    fprint(f"\nCompleted in {elapsed:.0f}s")

    # Save results
    out_file = OUT_DIR / "vix_spike_call_spreads_v1_results.json"
    import json
    save_data = {}
    for key, res in results.items():
        vr = validations.get(key)
        save_data[key] = {
            "label": res["label"],
            "n_trades": len(res["trades"]),
            "final_equity": round(res["equity"], 2),
            "total_pnl": round(res["equity"] - INITIAL_CAPITAL, 2),
            "trades": res["trades"],
        }
        if vr:
            save_data[key]["validation"] = vr.to_dict()

    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nResults saved.")


if __name__ == "__main__":
    main()
