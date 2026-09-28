#!/usr/bin/env python3
"""
Leveraged Variants of Strategy Rotation Variant A
==================================================
Variant A (Simple Regime Switch) is the validated winner:
  - Bull + no VIX spike → QQQ (buy and hold Nasdaq-100)
  - VIX > 25 and declining → SPY (crisis recovery)
  - Bear, no trigger → Cash

Validated metrics (OOT 2022-2026): 37.4% CAGR, Sharpe 2.13, MDD -10.8%, WR 57.1%

This script builds leveraged variants of EXACTLY that logic:
  L-A1: TQQQ instead of QQQ (3x Nasdaq-100) + UPRO instead of SPY
  L-A2: 2x margin on QQQ/SPY positions with de-lever on drawdown
  L-A3: ATM call options on QQQ/SPY (realistic theta model)
  L-A4: TQQQ with vol-targeting (target 30% portfolio vol, scale down in high-vol)
  L-A5: 1.5x TQQQ (partial exposure to control drawdown vs full 3x)

Costs:
  - Leveraged ETFs: 10 bps slippage per side + annual vol-decay (empirical ~5% CAGR)
  - Margin: 6.5% annual rate on borrowed capital
  - Options: ~4% IV premium per position entry + daily theta
  - All: same timing signals as original Variant A

Walk-forward OOT: 2022-01-01 → 2026-07-25 (matches original)
"""

import warnings, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ───────────────────────────────────────────────────────────────────
STARTING_CAPITAL   = 100_000.0
OOT_START          = "2022-01-01"
OOT_END            = "2026-07-25"

# Cost constants
SLIPPAGE_1X        = 0.0005    # 5 bps per side for SPY/QQQ
SLIPPAGE_3X        = 0.001     # 10 bps per side for TQQQ/UPRO
MARGIN_ANNUAL      = 0.065     # 6.5% annual margin interest
LEV3X_VOL_DECAY    = 0.05      # 5% annual vol-decay penalty for 3x ETFs
OPTIONS_IV_PREMIUM = 0.04      # 4% of notional paid as option premium on entry

OUT_DIR = Path("/home/jupiter/Lvl3Quant/research/high_growth")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── DATA ─────────────────────────────────────────────────────────────────────
def download_data():
    print("[1/5] Downloading market data...")
    tickers = ["SPY", "QQQ", "UPRO", "TQQQ", "^VIX"]
    raw = yf.download(tickers, start="2021-01-01", end=OOT_END,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    prices = prices.ffill()
    print(f"  Data: {prices.index[0].date()} → {prices.index[-1].date()}, {len(prices)} rows")
    for t in tickers:
        col = t if t in prices.columns else None
        if col:
            print(f"  {t}: {prices[col].notna().sum()} valid rows")
    return prices


def compute_signals(prices):
    """Replicate exactly the Variant A signals from the original backtest."""
    spy = prices["SPY"]
    vix = prices["^VIX"]

    df = pd.DataFrame(index=prices.index)
    df["SPY"]      = spy
    df["QQQ"]      = prices["QQQ"]
    df["UPRO"]     = prices["UPRO"]
    df["TQQQ"]     = prices["TQQQ"]
    df["VIX"]      = vix
    df["SMA200"]   = spy.rolling(200).mean()

    # RSI-14 for SPY
    delta = spy.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    df["VIX_chg5"] = vix.pct_change(5)
    df["SPY_ret5"] = spy.pct_change(5)
    df["SPY_ret20"] = spy.pct_change(20)
    df["bull"] = (spy > df["SMA200"]).astype(int)

    return df.dropna()


def get_signal(row):
    """
    Exact replica of Variant A from strategy_rotation_backtest.py:
      VIX > 25 → vix_fade (buy SPY on VIX retreat)
      bull → earnings_momentum (QQQ)
      bear → contrarian (buy SPY on -3% dip)
      else → cash
    """
    if row["VIX"] > 25:
        return "vix_fade"
    elif row["bull"]:
        return "earnings_momentum"
    else:
        return "contrarian"


# ── ORIGINAL BASELINE (exact replication) ────────────────────────────────────
def simulate_baseline(df):
    """Exact replication of Variant A: QQQ in bull, SPY on VIX/dip."""
    spy_ret = df["SPY"].pct_change()
    qqq_ret = df["QQQ"].pct_change()

    daily_rets = []
    holdings   = None
    hold_rem   = 0

    for i in range(len(df)):
        if i == 0:
            daily_rets.append(0.0)
            continue

        sig = get_signal(df.iloc[i])
        prev_sig = get_signal(df.iloc[i-1])

        # Contrarian: only enter on -3% dip, hold 5 days
        if sig == "contrarian":
            if hold_rem > 0:
                r = spy_ret.iloc[i]
                hold_rem -= 1
            elif df["SPY_ret5"].iloc[i] <= -0.03:
                r = spy_ret.iloc[i]
                hold_rem = 4
                if sig != prev_sig:
                    r -= SLIPPAGE_1X * 2
            else:
                r = 0.0  # cash
        elif sig == "vix_fade":
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                r = spy_ret.iloc[i]
                if sig != prev_sig:
                    r -= SLIPPAGE_1X * 2
            else:
                r = 0.0
        elif sig == "earnings_momentum":
            r = qqq_ret.iloc[i]
            if sig != prev_sig:
                r -= SLIPPAGE_1X * 2
        else:
            r = 0.0

        daily_rets.append(r)

    return pd.Series(daily_rets, index=df.index, name="Baseline_V_A")


# ── VARIANT L-A1: 3x ETF SUBSTITUTION ────────────────────────────────────────
def simulate_lA1_3x_etf(df):
    """
    Replace QQQ with TQQQ (3x Nasdaq), SPY with UPRO (3x S&P).
    Same rotation signals. Pay vol-decay penalty daily while holding 3x ETF.
    Slippage 10 bps per side for leveraged ETFs.
    """
    upro_ret = df["UPRO"].pct_change()
    tqqq_ret = df["TQQQ"].pct_change()
    decay_daily = (1 + LEV3X_VOL_DECAY) ** (1/252) - 1

    daily_rets = []
    holdings   = None
    hold_rem   = 0

    for i in range(len(df)):
        if i == 0:
            daily_rets.append(0.0)
            continue

        sig = get_signal(df.iloc[i])

        if sig == "contrarian":
            if hold_rem > 0:
                r = upro_ret.iloc[i] - decay_daily
                hold_rem -= 1
            elif df["SPY_ret5"].iloc[i] <= -0.03:
                r = upro_ret.iloc[i] - decay_daily - SLIPPAGE_3X * 2
                hold_rem = 4
            else:
                r = 0.0
        elif sig == "vix_fade":
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                r = upro_ret.iloc[i] - decay_daily
                if holdings not in ["vix_fade"]:
                    r -= SLIPPAGE_3X * 2
            else:
                r = 0.0
        elif sig == "earnings_momentum":
            r = tqqq_ret.iloc[i] - decay_daily
            if holdings != "earnings_momentum":
                r -= SLIPPAGE_3X * 2
        else:
            r = 0.0

        holdings = sig
        daily_rets.append(r)

    return pd.Series(daily_rets, index=df.index, name="LA1_3x_ETF_Sub")


# ── VARIANT L-A2: 2x MARGIN ──────────────────────────────────────────────────
def simulate_lA2_2x_margin(df):
    """
    2x margin on 1x ETFs (QQQ/SPY).
    Leverage = 2x while equity drawdown < 20%.
    Auto-delever to 1x if drawdown hits 20%.
    Pay 6.5% annual margin interest on borrowed portion.
    """
    spy_ret  = df["SPY"].pct_change()
    qqq_ret  = df["QQQ"].pct_change()
    margin_d = MARGIN_ANNUAL / 252

    daily_rets  = []
    holdings    = None
    hold_rem    = 0
    equity      = STARTING_CAPITAL
    peak_equity = STARTING_CAPITAL

    for i in range(len(df)):
        if i == 0:
            daily_rets.append(0.0)
            continue

        sig  = get_signal(df.iloc[i])
        dd   = (equity - peak_equity) / peak_equity
        lev  = 2.0 if dd > -0.20 else 1.0   # Auto-delever at 20% DD

        if sig == "contrarian":
            if hold_rem > 0:
                r = spy_ret.iloc[i] * lev - margin_d * (lev - 1)
                hold_rem -= 1
            elif df["SPY_ret5"].iloc[i] <= -0.03:
                r = spy_ret.iloc[i] * lev - margin_d * (lev-1) - SLIPPAGE_1X * 2 * lev
                hold_rem = 4
            else:
                r = 0.0
        elif sig == "vix_fade":
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                r = spy_ret.iloc[i] * lev - margin_d * (lev - 1)
                if holdings != "vix_fade":
                    r -= SLIPPAGE_1X * 2 * lev
            else:
                r = 0.0
        elif sig == "earnings_momentum":
            r = qqq_ret.iloc[i] * lev - margin_d * (lev - 1)
            if holdings != "earnings_momentum":
                r -= SLIPPAGE_1X * 2 * lev
        else:
            r = 0.0

        holdings     = sig
        equity      *= (1 + r)
        peak_equity  = max(peak_equity, equity)
        daily_rets.append(r)

    return pd.Series(daily_rets, index=df.index, name="LA2_2x_Margin")


# ── VARIANT L-A3: ATM CALL OPTIONS ───────────────────────────────────────────
def simulate_lA3_options(df):
    """
    ATM call options instead of shares on long signals.
    - Delta 0.50 (ATM), adjusts with underlying moves
    - Theta: OPTIONS_IV_PREMIUM / 40 per day
    - Roll at 15 DTE (every ~25 calendar days → ~18 trading days)
    - Full capital allocated to buying calls
    - On vix_fade/contrarian: use SPY calls; on earnings_momentum: QQQ calls

    P&L model per dollar of premium:
      call_ret = delta * underlying_return - theta_daily
    """
    spy_ret  = df["SPY"].pct_change()
    qqq_ret  = df["QQQ"].pct_change()

    DTE_ENTRY     = 40
    DTE_ROLL      = 18  # Trading days (~25 calendar days for 40-45 DTE roll)
    THETA_DAILY   = OPTIONS_IV_PREMIUM / DTE_ENTRY
    DELTA_INIT    = 0.50

    daily_rets = []
    holdings   = None
    hold_rem   = 0
    dte        = DTE_ENTRY
    days_held  = 0

    for i in range(len(df)):
        if i == 0:
            daily_rets.append(0.0)
            dte = DTE_ENTRY
            continue

        sig = get_signal(df.iloc[i])

        # Determine underlying return
        if sig == "contrarian":
            if hold_rem > 0:
                underlying = spy_ret.iloc[i]
                hold_rem -= 1
                active = True
            elif df["SPY_ret5"].iloc[i] <= -0.03:
                underlying = spy_ret.iloc[i]
                hold_rem  = 4
                active    = True
                dte       = DTE_ENTRY  # New position
                days_held = 0
            else:
                underlying = 0.0
                active     = False
        elif sig == "vix_fade":
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                underlying = spy_ret.iloc[i]
                active     = True
                if holdings != "vix_fade":
                    dte       = DTE_ENTRY
                    days_held = 0
            else:
                underlying = 0.0
                active     = False
        elif sig == "earnings_momentum":
            underlying = qqq_ret.iloc[i]
            active     = True
            if holdings != "earnings_momentum":
                dte       = DTE_ENTRY
                days_held = 0
        else:
            underlying = 0.0
            active     = False

        if active:
            # Dynamic delta: moves with underlying (ATM delta drifts)
            delta = np.clip(DELTA_INIT + underlying * 3, 0.20, 0.90)
            r = delta * underlying - THETA_DAILY

            # Roll cost: pay premium again when DTE gets low
            days_held += 1
            if days_held >= DTE_ROLL:
                r -= SLIPPAGE_1X * 2  # Close + reopen spread
                dte       = DTE_ENTRY
                days_held = 0
        else:
            r = 0.0

        holdings = sig
        daily_rets.append(r)

    return pd.Series(daily_rets, index=df.index, name="LA3_Options")


# ── VARIANT L-A4: TQQQ + VOL TARGETING ───────────────────────────────────────
def simulate_lA4_tqqq_voltarget(df):
    """
    Use TQQQ but with a vol-targeting overlay.
    Target portfolio vol = 30% annualized.
    If realized 20-day vol of TQQQ > target, scale down position.
    If vol < target, scale up (but cap at 100% — no additional margin).
    Cash = T-bill proxy (0% since account earns ~5% HYSA but we model conservatively).

    This is the academically validated way to reduce leveraged ETF drawdowns
    while keeping substantial upside.
    """
    tqqq_ret   = df["TQQQ"].pct_change()
    upro_ret   = df["UPRO"].pct_change()
    spy_ret    = df["SPY"].pct_change()
    decay_d    = (1 + LEV3X_VOL_DECAY) ** (1/252) - 1

    TARGET_VOL = 0.30  # 30% annualized

    daily_rets = []
    holdings   = None
    hold_rem   = 0

    for i in range(len(df)):
        if i == 0:
            daily_rets.append(0.0)
            continue

        sig = get_signal(df.iloc[i])

        # Compute vol-scaling weight using last 20 days of TQQQ
        if i >= 20:
            realized_vol = tqqq_ret.iloc[i-20:i].std() * np.sqrt(252)
            weight = min(TARGET_VOL / (realized_vol + 1e-6), 1.0)
        else:
            weight = 0.5  # Conservative start

        if sig == "contrarian":
            if hold_rem > 0:
                r = upro_ret.iloc[i] * weight - decay_d * weight
                hold_rem -= 1
            elif df["SPY_ret5"].iloc[i] <= -0.03:
                r = upro_ret.iloc[i] * weight - decay_d * weight - SLIPPAGE_3X * 2 * weight
                hold_rem = 4
            else:
                r = 0.0
        elif sig == "vix_fade":
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                r = upro_ret.iloc[i] * weight - decay_d * weight
                if holdings != "vix_fade":
                    r -= SLIPPAGE_3X * 2 * weight
            else:
                r = 0.0
        elif sig == "earnings_momentum":
            r = tqqq_ret.iloc[i] * weight - decay_d * weight
            if holdings != "earnings_momentum":
                r -= SLIPPAGE_3X * 2 * weight
        else:
            r = 0.0

        holdings = sig
        daily_rets.append(r)

    return pd.Series(daily_rets, index=df.index, name="LA4_TQQQ_VolTarget30")


# ── VARIANT L-A5: 1.5x TQQQ BLEND ────────────────────────────────────────────
def simulate_lA5_partial_tqqq(df):
    """
    50% TQQQ + 50% QQQ = effectively ~2x Nasdaq-100 exposure.
    Lower vol-decay than pure TQQQ. Rational middle ground.
    Same signals as Variant A.
    """
    qqq_ret  = df["QQQ"].pct_change()
    tqqq_ret = df["TQQQ"].pct_change()
    spy_ret  = df["SPY"].pct_change()
    upro_ret = df["UPRO"].pct_change()
    decay_d  = (1 + LEV3X_VOL_DECAY) ** (1/252) - 1

    daily_rets = []
    holdings   = None
    hold_rem   = 0

    for i in range(len(df)):
        if i == 0:
            daily_rets.append(0.0)
            continue

        sig = get_signal(df.iloc[i])

        if sig == "contrarian":
            if hold_rem > 0:
                # 50% UPRO, 50% SPY
                r = 0.5 * upro_ret.iloc[i] + 0.5 * spy_ret.iloc[i]
                r -= decay_d * 0.5
                hold_rem -= 1
            elif df["SPY_ret5"].iloc[i] <= -0.03:
                r  = 0.5 * upro_ret.iloc[i] + 0.5 * spy_ret.iloc[i]
                r -= decay_d * 0.5 + SLIPPAGE_3X
                hold_rem = 4
            else:
                r = 0.0
        elif sig == "vix_fade":
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                r  = 0.5 * upro_ret.iloc[i] + 0.5 * spy_ret.iloc[i]
                r -= decay_d * 0.5
                if holdings != "vix_fade":
                    r -= SLIPPAGE_3X
            else:
                r = 0.0
        elif sig == "earnings_momentum":
            # 50% TQQQ + 50% QQQ ≈ 2x Nasdaq
            r  = 0.5 * tqqq_ret.iloc[i] + 0.5 * qqq_ret.iloc[i]
            r -= decay_d * 0.5
            if holdings != "earnings_momentum":
                r -= SLIPPAGE_3X * 0.5 + SLIPPAGE_1X * 0.5
        else:
            r = 0.0

        holdings = sig
        daily_rets.append(r)

    return pd.Series(daily_rets, index=df.index, name="LA5_1p5x_Blend")


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_metrics(daily_ret, name="Strategy"):
    dr     = daily_ret.fillna(0)
    equity = STARTING_CAPITAL * (1 + dr).cumprod()

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n_years   = len(dr) / 252
    cagr      = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0

    vol    = dr.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0

    neg    = dr[dr < 0]
    down_v = neg.std() * np.sqrt(252) if len(neg) > 0 else 1e-9
    sortino = cagr / down_v

    peak = equity.cummax()
    dd   = (equity - peak) / peak
    mdd  = dd.min()

    active = dr[dr != 0]
    wins   = (active > 0).sum()
    losses = (active < 0).sum()
    wr     = wins / (wins + losses) if (wins + losses) > 0 else 0
    avg_w  = active[active > 0].mean() if wins > 0 else 0
    avg_l  = abs(active[active < 0].mean()) if losses > 0 else 1e-9
    pf     = (avg_w * wins) / (avg_l * losses) if (avg_l * losses) > 0 else 999

    calmar = cagr / abs(mdd) if mdd < 0 else 0

    per_yr = {}
    for yr in range(2022, 2027):
        mask = dr.index.year == yr
        if mask.sum() > 50:
            per_yr[str(yr)] = round((1 + dr[mask]).prod() - 1, 4)

    # Worst monthly drawdown
    monthly = dr.resample("ME").apply(lambda x: (1+x).prod() - 1)
    worst_mo = monthly.min()

    return {
        "name":          name,
        "cagr":          round(cagr, 4),
        "total_return":  round(total_ret, 4),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "max_drawdown":  round(mdd, 4),
        "calmar":        round(calmar, 3),
        "win_rate":      round(wr, 3),
        "profit_factor": round(pf, 3),
        "annual_vol":    round(vol, 4),
        "worst_month":   round(worst_mo, 4),
        "final_equity":  round(equity.iloc[-1], 2),
        "per_year":      per_yr,
    }


def spy_metrics(df):
    r = df["SPY"].pct_change().fillna(0)
    return compute_metrics(r, name="SPY_BuyHold")

def qqq_metrics(df):
    r = df["QQQ"].pct_change().fillna(0)
    return compute_metrics(r, name="QQQ_BuyHold")

def upro_metrics(df):
    r = df["UPRO"].pct_change().fillna(0)
    return compute_metrics(r, name="UPRO_BuyHold_naive")

def tqqq_metrics(df):
    r = df["TQQQ"].pct_change().fillna(0)
    return compute_metrics(r, name="TQQQ_BuyHold_naive")


# ── MONTE CARLO ───────────────────────────────────────────────────────────────
def mc_stress(daily_ret, n_sims=1000, horizon_years=4):
    dr_arr = daily_ret.fillna(0).values
    n_days = int(horizon_years * 252)
    cagrs, mdds = [], []
    rng = np.random.RandomState(42)

    for _ in range(n_sims):
        sim  = rng.choice(dr_arr, size=n_days, replace=True)
        eq   = np.cumprod(1 + sim)
        tot  = eq[-1] - 1
        cagr = (1 + tot) ** (1/horizon_years) - 1
        peak = np.maximum.accumulate(eq)
        mdd  = ((eq - peak) / peak).min()
        cagrs.append(cagr)
        mdds.append(mdd)

    return {
        "cagr_p10":   round(np.percentile(cagrs, 10), 4),
        "cagr_p25":   round(np.percentile(cagrs, 25), 4),
        "cagr_p50":   round(np.percentile(cagrs, 50), 4),
        "cagr_p75":   round(np.percentile(cagrs, 75), 4),
        "cagr_p90":   round(np.percentile(cagrs, 90), 4),
        "mdd_p25":    round(np.percentile(mdds, 25), 4),
        "mdd_p50":    round(np.percentile(mdds, 50), 4),
        "mdd_p75":    round(np.percentile(mdds, 75), 4),
        "prob_loss":  round((np.array(cagrs) < 0).mean(), 4),
        "prob_beat_spy": round((np.array(cagrs) > 0.116).mean(), 4),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("LEVERAGED VARIANTS — STRATEGY ROTATION VARIANT A")
    print(f"OOT: {OOT_START} → {OOT_END}  |  Capital: ${STARTING_CAPITAL:,.0f}")
    print("=" * 72)

    # 1. Data
    prices = download_data()

    # 2. Signals
    print("\n[2/5] Computing signals...")
    df_full = compute_signals(prices)
    df_oot  = df_full.loc[OOT_START:OOT_END].copy()
    print(f"  OOT: {len(df_oot)} trading days ({df_oot.index[0].date()} → {df_oot.index[-1].date()})")

    # 3. Run variants
    print("\n[3/5] Running variants...")
    v0  = simulate_baseline(df_oot)
    vA1 = simulate_lA1_3x_etf(df_oot)
    vA2 = simulate_lA2_2x_margin(df_oot)
    vA3 = simulate_lA3_options(df_oot)
    vA4 = simulate_lA4_tqqq_voltarget(df_oot)
    vA5 = simulate_lA5_partial_tqqq(df_oot)
    print("  All variants complete.")

    # Benchmarks
    bm_spy  = spy_metrics(df_oot)
    bm_qqq  = qqq_metrics(df_oot)
    bm_upro = upro_metrics(df_oot)
    bm_tqqq = tqqq_metrics(df_oot)

    # 4. Metrics
    print("\n[4/5] Computing metrics...")
    metrics_list = [
        compute_metrics(v0,  "V_A_Baseline_Unlev"),
        compute_metrics(vA1, "LA1_3x_ETF_Substitution"),
        compute_metrics(vA2, "LA2_2x_Margin"),
        compute_metrics(vA3, "LA3_ATM_Calls"),
        compute_metrics(vA4, "LA4_TQQQ_VolTarget30pct"),
        compute_metrics(vA5, "LA5_1p5x_TQQQ_QQQ_Blend"),
        bm_spy, bm_qqq, bm_upro, bm_tqqq,
    ]

    # 5. Print table
    print("\n" + "=" * 85)
    print(f"{'Strategy':<35} {'CAGR':>7} {'MDD':>8} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'Calmar':>7}")
    print("-" * 85)
    for m in metrics_list:
        print(f"{m['name']:<35} {m['cagr']:>7.1%} {m['max_drawdown']:>8.1%} "
              f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['win_rate']:>6.1%} "
              f"{m['profit_factor']:>6.2f} {m['calmar']:>7.2f}")
    print("-" * 85)

    print("\nPer-year returns:")
    for m in metrics_list:
        yr_s = "  ".join(f"{y}:{r:+.0%}" for y, r in m.get("per_year", {}).items())
        print(f"  {m['name']:<35} {yr_s}")

    print("\nWorst month:")
    for m in metrics_list:
        print(f"  {m['name']:<35} {m['worst_month']:.1%}")

    # Monte Carlo
    print("\n[5/5] Monte Carlo stress (1000 sims, 4-year horizon)...")
    mc_out = {}
    for v, name in [(v0,"Baseline"), (vA1,"3x_ETF"), (vA2,"2x_Margin"),
                    (vA3,"Options"), (vA4,"TQQQ_Vol"), (vA5,"1p5x_Blend")]:
        mc = mc_stress(v)
        mc_out[name] = mc
        print(f"  {name:<20} CAGR: p10={mc['cagr_p10']:.0%} / p50={mc['cagr_p50']:.0%} / p90={mc['cagr_p90']:.0%} | "
              f"MDD p50={mc['mdd_p50']:.0%} | P(beat SPY)={mc['prob_beat_spy']:.0%}")

    # Save
    results = {
        "meta": {
            "strategy":         "Strategy Rotation Variant A + Leveraged",
            "run_timestamp":    datetime.now().isoformat(),
            "oot_start":        OOT_START,
            "oot_end":          OOT_END,
            "starting_capital": STARTING_CAPITAL,
            "costs": {
                "slippage_1x_pct":      SLIPPAGE_1X,
                "slippage_3x_pct":      SLIPPAGE_3X,
                "margin_annual":        MARGIN_ANNUAL,
                "lev3x_vol_decay_annual": LEV3X_VOL_DECAY,
                "options_iv_premium":   OPTIONS_IV_PREMIUM,
            },
        },
        "variants": {m["name"]: m for m in metrics_list},
        "monte_carlo": mc_out,
    }

    out_path = OUT_DIR / "leveraged_variant_A_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Save equity curves CSV
    curves = pd.DataFrame({
        "V0_Baseline":        STARTING_CAPITAL * (1 + v0.fillna(0)).cumprod(),
        "LA1_3x_ETF":         STARTING_CAPITAL * (1 + vA1.fillna(0)).cumprod(),
        "LA2_2x_Margin":      STARTING_CAPITAL * (1 + vA2.fillna(0)).cumprod(),
        "LA3_Options":        STARTING_CAPITAL * (1 + vA3.fillna(0)).cumprod(),
        "LA4_TQQQ_Vol":       STARTING_CAPITAL * (1 + vA4.fillna(0)).cumprod(),
        "LA5_1p5x_Blend":     STARTING_CAPITAL * (1 + vA5.fillna(0)).cumprod(),
        "SPY_BuyHold":        STARTING_CAPITAL * (1 + df_oot["SPY"].pct_change().fillna(0)).cumprod(),
        "QQQ_BuyHold":        STARTING_CAPITAL * (1 + df_oot["QQQ"].pct_change().fillna(0)).cumprod(),
        "TQQQ_BuyHold_naive": STARTING_CAPITAL * (1 + df_oot["TQQQ"].pct_change().fillna(0)).cumprod(),
    })
    curves.index.name = "date"
    curves_path = OUT_DIR / "equity_curves_variant_A_leveraged.csv"
    curves.to_csv(curves_path)
    print(f"Saved: {curves_path}")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    return results


if __name__ == "__main__":
    main()
