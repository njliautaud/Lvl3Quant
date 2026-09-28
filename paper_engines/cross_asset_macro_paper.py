#!/usr/bin/env python3
"""
Cross-Asset Macro Sector Rotation Paper Engine (v36 -- AVO-evolved unified composite ranking)
==============================================================================================
Full cross-asset macro sector rotation with VIX regime gates, composite momentum
ranking (60/40 blend), sector-macro coupling (oil/dollar/curve filters),
SPY-GLD divergence gate, calm-grind skip, and dead-money exits.

AVO lockbox result: Sharpe 2.49 on unseen 2026H1.

Strategy source: runs/cross_asset_macro-20260902-101136/work/strategy.py (AVO v36)
Evolved from v25 (defensive-only dip-buying) via AVO evolutionary search.

Cron: 15 16 * * 1-5  (4:15 PM ET, after market close)
State: /home/jupiter/Lvl3Quant/paper_engines/state/cross_asset_macro_state.json
Trades: /home/jupiter/Lvl3Quant/paper_engines/logs/cross_asset_macro_trades.csv
"""
import csv
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings("ignore")

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "paper_engines" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "cross_asset_macro_state.json"
LOG_DIR = BASE / "paper_engines" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "cross_asset_macro.log"
TRADE_CSV = LOG_DIR / "cross_asset_macro_trades.csv"

# ── Strategy Constants (v36) ──
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
BENCHMARK = 'SPY'
MACRO_TICKERS = ['TLT', 'IEF', 'GLD', 'USO', 'UUP']
VIX_TICKER = '^VIX'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK] + MACRO_TICKERS + [VIX_TICKER]

DEFENSIVE_SECTORS = ['XLU', 'XLP', 'XLV']
ALL_ELIGIBLE = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU']
CURVE_SENSITIVE_CYCLICALS = ['XLF', 'XLI', 'XLY', 'XLRE']

MOMENTUM_SHORT = 5
MOMENTUM_MED = 10
TOP_N = 2

VIX_HIGH = 25.0
VIX_EXTREME = 40.0
VIX_DECLINE_THRESHOLD = -0.02
VIX_STABLE_BAND = 0.025

GLD_RISKOFF_LOOKBACK = 10
GLD_RISKOFF_THRESHOLD = 0.02

CURVE_LOOKBACK = 20
CURVE_FLATTENING_THRESHOLD = 0.01

UUP_LOOKBACK = 10
UUP_STRONG_THRESHOLD = 0.01
DOLLAR_SENSITIVE = ['XLK', 'XLI', 'XLB', 'XLE']

USO_LOOKBACK = 10
USO_DECLINE_THRESHOLD = -0.03
OIL_SENSITIVE_CYCLICALS = ['XLE', 'XLI', 'XLB', 'XLY']

SPY_GLD_DIVERGE_LOOKBACK = 5
SPY_GLD_DIVERGE_THRESHOLD = 0.01
SPY_GLD_DIVERGE_GLD_THRESHOLD = 0.007

CALM_GRIND_VIX_MAX = 16.0
CALM_GRIND_SPY_20D_MIN = 0.05
CALM_GRIND_DISP_MAX = 0.015
CALM_GRIND_VIX_5D_FLAT = 0.01

DEAD_MONEY_DAYS = 1
DEAD_MONEY_THRESHOLD = 0.001
DEAD_MONEY_DAY2_THRESHOLD = 0.005
DEAD_MONEY_DAY2_CALM_THRESHOLD = 0.009
CALM_UPTREND_VIX = 18.0
CALM_UPTREND_SPY_MOM = 0.03
CALM_MIN_SECTOR_MOM = 0.015
LOW_VIX_DECLINE_THRESHOLD = -0.035

DISPERSION_LOOKBACK = 5
MIN_DISPERSION_CALM = 0.012

CALM_MIN_CONVICTION = 5
CALM_XS_BYPASS_SPREAD = 0.035

MAX_HOLD_DAYS = 3
MAX_PER_TRADE = 1500.0
MAX_CONCURRENT = 2
SLIPPAGE_PCT = 0.0001

TRAILING_STOP_PCT = -0.015
TRAILING_STOP_CALM = -0.020
CALM_TRAILING_VIX = 18.0
TAKE_PROFIT_PCT = 0.040
HARD_STOP_PCT = -0.02
HARD_STOP_MID_VIX = -0.015
MID_VIX_LOW = 18.0
MID_VIX_HIGH = 25.0

VIX_ELEVATED = 25.0
MAX_HOLD_ELEVATED = 2

# Paper trading
INITIAL_CAPITAL = 10_000.0
DATA_LOOKBACK_DAYS = 90  # need 30+ trading days for curve/trend lookbacks


# ── VIX cache for exit logic ──
_vix_cache = pd.Series(dtype=float)


# ── Signal Generation (v36 logic) ──

def generate_signals_today(prices, spy, vix, macro_data):
    """
    Run v36 signal generation on the full price history and return
    the signal for the most recent date.
    Returns list of sector tickers that have entry signals today, or [].
    """
    global _vix_cache
    _vix_cache = vix.copy()

    today_idx = prices.index[-1]

    # Pre-compute all the momentum series
    spy_short_ret = spy.pct_change(MOMENTUM_SHORT)
    spy_ret_20d = spy.pct_change(20)

    sector_short_mom = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS)
    sector_rel_strength = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS)
    sector_3d_mom = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS)
    sector_med_mom = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS)

    for etf in SECTOR_ETFS:
        if etf in prices.columns:
            sector_short_mom[etf] = prices[etf].pct_change(MOMENTUM_SHORT)
            sector_rel_strength[etf] = prices[etf].pct_change(MOMENTUM_SHORT) - spy_short_ret
            sector_3d_mom[etf] = prices[etf].pct_change(3)
            sector_med_mom[etf] = prices[etf].pct_change(MOMENTUM_MED)

    vix_change_1d = vix.pct_change(1)
    vix_change_5d = vix.pct_change(5)

    tlt = macro_data['TLT'] if 'TLT' in macro_data.columns else spy * 0
    ief = macro_data['IEF'] if 'IEF' in macro_data.columns else spy * 0
    tlt_ret = tlt.pct_change(20)
    bond_equity_spread = tlt_ret - spy_ret_20d

    curve_ratio = tlt / ief.replace(0, np.nan)
    curve_mom = curve_ratio.pct_change(CURVE_LOOKBACK)

    gld = macro_data['GLD'] if 'GLD' in macro_data.columns else spy * 0
    gld_mom = gld.pct_change(GLD_RISKOFF_LOOKBACK)
    gld_short_mom = gld.pct_change(SPY_GLD_DIVERGE_LOOKBACK)

    uup = macro_data['UUP'] if 'UUP' in macro_data.columns else None
    uup_mom = uup.pct_change(UUP_LOOKBACK) if uup is not None else pd.Series(0.0, index=prices.index)

    uso = macro_data['USO'] if 'USO' in macro_data.columns else None
    uso_mom = uso.pct_change(USO_LOOKBACK) if uso is not None else pd.Series(0.0, index=prices.index)

    sector_5d_returns = pd.DataFrame(index=prices.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf in prices.columns:
            sector_5d_returns[etf] = prices[etf].pct_change(DISPERSION_LOOKBACK)
    sector_dispersion = sector_5d_returns.std(axis=1)

    # Only evaluate the most recent date
    date = today_idx
    signals = []

    vix_val = vix.get(date, np.nan) if hasattr(vix, 'get') else np.nan
    if pd.isna(vix_val) or vix_val > VIX_EXTREME:
        return signals

    vix_chg = vix_change_1d.get(date, np.nan) if date in vix_change_1d.index else np.nan
    if pd.isna(vix_chg):
        return signals

    # --- HIGH VIX REGIME (>25) ---
    if vix_val > VIX_HIGH:
        if vix_chg < VIX_DECLINE_THRESHOLD:
            best_s, best_mom = None, -999
            for s in DEFENSIVE_SECTORS:
                if s not in prices.columns:
                    continue
                smom = sector_short_mom.loc[date, s] if s in sector_short_mom.columns else np.nan
                if not pd.isna(smom) and smom > best_mom:
                    best_mom = smom
                    best_s = s
            if best_s is not None:
                signals.append(best_s)
                return signals

        if abs(vix_chg) < VIX_STABLE_BAND:
            uso_val_hv = uso_mom.get(date, np.nan) if date in uso_mom.index else np.nan
            curve_val_hv = curve_mom.get(date, np.nan) if date in curve_mom.index else np.nan
            oil_neg_hv = not pd.isna(uso_val_hv) and uso_val_hv < 0
            curve_flat_hv = not pd.isna(curve_val_hv) and curve_val_hv > CURVE_FLATTENING_THRESHOLD
            hvix_eligible = list(ALL_ELIGIBLE)
            if oil_neg_hv:
                hvix_eligible = [s for s in hvix_eligible if s not in ('XLE', 'XLB')]
            if curve_flat_hv:
                hvix_eligible = [s for s in hvix_eligible if s != 'XLF']
            rs_row = sector_rel_strength.loc[date]
            eligible_rs = rs_row[[s for s in hvix_eligible if s in rs_row.index]]
            rs_valid = eligible_rs.dropna().sort_values(ascending=False)

            if len(rs_valid) > 0:
                count = 0
                for s_name in rs_valid.head(3).index:
                    if count >= 2:
                        break
                    rs_val = rs_valid[s_name]
                    if rs_val <= 0.003:
                        break
                    abs_mom = sector_short_mom.loc[date, s_name] if s_name in sector_short_mom.columns else np.nan
                    if not pd.isna(abs_mom) and abs_mom > 0:
                        signals.append(s_name)
                        count += 1
        return signals

    # --- NORMAL / MID / LOW VIX REGIME ---

    # VIX decline gate (regime-dependent threshold)
    if vix_val < 16:
        decline_thresh = LOW_VIX_DECLINE_THRESHOLD
    elif vix_val >= MID_VIX_LOW:
        decline_thresh = -0.025
    else:
        decline_thresh = VIX_DECLINE_THRESHOLD
    if vix_chg >= decline_thresh:
        return signals

    # Calm-grind skip
    if vix_val < CALM_GRIND_VIX_MAX:
        spy_20d_cg = spy_ret_20d.get(date, np.nan) if date in spy_ret_20d.index else np.nan
        vix_5d_cg = vix_change_5d.get(date, np.nan) if date in vix_change_5d.index else np.nan
        disp_cg = sector_dispersion.get(date, np.nan) if date in sector_dispersion.index else np.nan
        if (not pd.isna(spy_20d_cg) and spy_20d_cg > CALM_GRIND_SPY_20D_MIN
                and not pd.isna(vix_5d_cg) and abs(vix_5d_cg) < CALM_GRIND_VIX_5D_FLAT
                and not pd.isna(disp_cg) and disp_cg < CALM_GRIND_DISP_MAX):
            return signals

    # SPY-GLD divergence gate
    spy_5d_ret = spy_short_ret.get(date, np.nan) if date in spy_short_ret.index else np.nan
    gld_5d_ret = gld_short_mom.get(date, np.nan) if date in gld_short_mom.index else np.nan
    if (not pd.isna(spy_5d_ret) and not pd.isna(gld_5d_ret)
            and spy_5d_ret > SPY_GLD_DIVERGE_THRESHOLD
            and gld_5d_ret > SPY_GLD_DIVERGE_GLD_THRESHOLD):
        return signals

    # Macro signals
    be_val = bond_equity_spread.get(date, np.nan) if date in bond_equity_spread.index else np.nan
    is_risk_off = not pd.isna(be_val) and be_val > 0

    gld_val = gld_mom.get(date, np.nan) if date in gld_mom.index else np.nan
    gld_riskoff = not pd.isna(gld_val) and gld_val > GLD_RISKOFF_THRESHOLD

    curve_val = curve_mom.get(date, np.nan) if date in curve_mom.index else np.nan
    curve_flattening = not pd.isna(curve_val) and curve_val > CURVE_FLATTENING_THRESHOLD

    uup_val = uup_mom.get(date, np.nan) if date in uup_mom.index else np.nan
    dollar_strong = not pd.isna(uup_val) and uup_val > UUP_STRONG_THRESHOLD

    uso_val = uso_mom.get(date, np.nan) if date in uso_mom.index else np.nan
    oil_declining = not pd.isna(uso_val) and uso_val < USO_DECLINE_THRESHOLD

    # Candidate selection based on macro
    if is_risk_off or gld_riskoff:
        candidates = DEFENSIVE_SECTORS
    elif curve_flattening:
        candidates = [s for s in ALL_ELIGIBLE if s not in CURVE_SENSITIVE_CYCLICALS]
    else:
        candidates = ALL_ELIGIBLE

    if oil_declining and vix_val < CALM_UPTREND_VIX and not (is_risk_off or gld_riskoff):
        candidates = [s for s in candidates if s not in OIL_SENSITIVE_CYCLICALS]
        if len(candidates) == 0:
            candidates = DEFENSIVE_SECTORS

    if dollar_strong and not (is_risk_off or gld_riskoff):
        candidates = [s for s in candidates if s not in DOLLAR_SENSITIVE]
        if len(candidates) == 0:
            candidates = DEFENSIVE_SECTORS

    # Calm uptrend detection
    spy_20d = spy_ret_20d.get(date, np.nan) if date in spy_ret_20d.index else np.nan
    calm_uptrend = (vix_val < CALM_UPTREND_VIX
                    and not pd.isna(spy_20d)
                    and spy_20d > CALM_UPTREND_SPY_MOM)

    spy_5d = spy_short_ret.get(date, np.nan) if date in spy_short_ret.index else np.nan
    if calm_uptrend and (pd.isna(spy_5d) or spy_5d <= 0):
        calm_uptrend = False

    # Double headwind skip
    if oil_declining and dollar_strong and calm_uptrend:
        return signals

    # Calm uptrend conviction/dispersion gates
    if calm_uptrend:
        disp = sector_dispersion.get(date, np.nan) if date in sector_dispersion.index else np.nan
        mom_row_xs = sector_short_mom.loc[date]
        valid_moms_xs = mom_row_xs[[s for s in ALL_ELIGIBLE if s in mom_row_xs.index]].dropna()
        xs_spread = (valid_moms_xs.max() - valid_moms_xs.min()) if len(valid_moms_xs) >= 3 else 0.0
        xs_bypass = xs_spread > CALM_XS_BYPASS_SPREAD

        if not pd.isna(disp) and disp < MIN_DISPERSION_CALM and not xs_bypass:
            return signals

        conviction = 0
        conviction += 1
        if not pd.isna(be_val) and be_val < 0:
            conviction += 1
        if not curve_flattening:
            conviction += 1
        if not dollar_strong:
            conviction += 1
        if not gld_riskoff:
            conviction += 1
        if not oil_declining:
            conviction += 1

        if conviction < CALM_MIN_CONVICTION and not xs_bypass:
            return signals

    # Momentum ranking (regime-dependent composite)
    mom_row = sector_short_mom.loc[date]
    is_mid_vix = MID_VIX_LOW <= vix_val <= MID_VIX_HIGH
    is_low_vix_nonclam = vix_val < MID_VIX_LOW and not calm_uptrend

    if is_mid_vix:
        mom3_row = sector_3d_mom.loc[date]
        composite = pd.Series(dtype=float)
        for s in candidates:
            if s in mom_row.index and s in mom3_row.index:
                m5 = mom_row[s]
                m3 = mom3_row[s]
                if not pd.isna(m5) and not pd.isna(m3):
                    composite[s] = 0.6 * m5 + 0.4 * m3
        mom_valid = composite.sort_values(ascending=False)
    elif is_low_vix_nonclam:
        mom10_row = sector_med_mom.loc[date]
        composite = pd.Series(dtype=float)
        for s in candidates:
            if s in mom_row.index and s in mom10_row.index:
                m5 = mom_row[s]
                m10 = mom10_row[s]
                if not pd.isna(m5) and not pd.isna(m10):
                    composite[s] = 0.65 * m5 + 0.35 * m10
        mom_valid = composite.sort_values(ascending=False)
    else:
        cand_mom = mom_row[[s for s in candidates if s in mom_row.index]]
        mom_valid = cand_mom.dropna().sort_values(ascending=False)

    if len(mom_valid) == 0:
        return signals

    pick_n = 1 if calm_uptrend else TOP_N
    min_mom = CALM_MIN_SECTOR_MOM if calm_uptrend else -999
    med_mom_floor = 0.01 if calm_uptrend else -0.08

    top = mom_valid.head(pick_n).index.tolist()
    for idx, s in enumerate(top):
        smom = mom_row[s] if s in mom_row.index else np.nan
        if pd.isna(smom) or smom < min_mom:
            continue
        if idx >= 1 and not calm_uptrend:
            rs_val = sector_rel_strength.loc[date, s] if s in sector_rel_strength.columns else np.nan
            if pd.isna(rs_val) or rs_val <= 0:
                continue
        med_m = sector_med_mom.loc[date, s] if s in sector_med_mom.columns else np.nan
        if not pd.isna(med_m) and med_m > med_mom_floor:
            signals.append(s)

    return signals


# ── Exit Logic (v36) ──

def check_exit(pos, current_price, today_str):
    """
    Check v36 exit conditions. Returns (should_exit: bool, reason: str).
    Uses VIX-regime-dependent trailing stops and hard stops, plus dead-money exits.
    """
    entry_price = pos["entry_price"]
    days_held = np.busday_count(
        np.datetime64(pos["entry_date"], "D"),
        np.datetime64(today_str, "D"),
    )
    pnl_pct = (current_price - entry_price) / entry_price

    # Update high water mark
    hwm = pos.get("hwm", entry_price)
    if current_price > hwm:
        hwm = current_price
        pos["hwm"] = hwm
    dd_from_hwm = (current_price - hwm) / hwm

    # Get current VIX for regime-dependent exits
    current_vix = np.nan
    if len(_vix_cache) > 0:
        try:
            dt = pd.Timestamp(today_str)
            if dt in _vix_cache.index:
                current_vix = _vix_cache[dt]
            else:
                valid_dates = _vix_cache.index[_vix_cache.index <= dt]
                if len(valid_dates) > 0:
                    current_vix = _vix_cache[valid_dates[-1]]
        except (KeyError, TypeError):
            pass

    elevated_vix = not pd.isna(current_vix) and current_vix > VIX_ELEVATED
    calm_vix = not pd.isna(current_vix) and current_vix < CALM_TRAILING_VIX
    mid_vix = not pd.isna(current_vix) and MID_VIX_LOW <= current_vix <= MID_VIX_HIGH
    max_hold = MAX_HOLD_ELEVATED if elevated_vix else MAX_HOLD_DAYS

    trailing = TRAILING_STOP_CALM if calm_vix else TRAILING_STOP_PCT
    hard_stop = HARD_STOP_MID_VIX if mid_vix else HARD_STOP_PCT

    # 1. Max hold (shorter in elevated VIX)
    if days_held >= max_hold:
        return True, "max_hold"

    # 2. Take profit
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True, "take_profit"

    # 3. Trailing stop (wider in calm VIX)
    if dd_from_hwm <= trailing:
        return True, "trailing_stop"

    # 4. Hard stop (tighter in mid-VIX)
    if pnl_pct <= hard_stop:
        return True, "hard_stop"

    # 5. Dead money exit day 1
    if days_held >= DEAD_MONEY_DAYS and pnl_pct < DEAD_MONEY_THRESHOLD:
        return True, "dead_money"

    # 6. Dead money exit day 2 (regime-dependent threshold)
    day2_thresh = DEAD_MONEY_DAY2_CALM_THRESHOLD if calm_vix else DEAD_MONEY_DAY2_THRESHOLD
    if days_held >= 2 and pnl_pct < day2_thresh:
        return True, "dead_money_d2"

    return False, ""


# ── Engine Infrastructure ──

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [CROSS-ASSET-MACRO-v36] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "strategy_version": "v36",
        "positions": [],
        "closed_trades": [],
        "equity_curve": [],
        "capital": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "peak_equity": INITIAL_CAPITAL,
        "max_drawdown": 0.0,
        "total_pnl": 0.0,
        "wins": 0,
        "losses": 0,
        "n_trades": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "created": datetime.now(ET).isoformat(),
        "last_run_date": None,
        "last_updated": None,
    }


def save_state(state: dict):
    state["last_updated"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    state["strategy_version"] = "v36"
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_FILE)


def log_trade_csv(trade: dict):
    """Append a closed trade to the CSV log."""
    fieldnames = [
        "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
        "shares", "pnl", "pnl_pct", "days_held", "exit_reason",
    ]
    write_header = not TRADE_CSV.exists()
    try:
        with open(TRADE_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(trade)
    except Exception as e:
        log(f"WARN: CSV log write failed: {e}")


def fetch_data():
    """Fetch all required tickers via yfinance. Returns (sector_prices, spy, vix, macro_data)."""
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK_DAYS)

    try:
        raw = yf.download(
            ALL_TICKERS,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=30,
        )
    except Exception as e:
        log(f"ERROR: yfinance download failed: {e}")
        return None, None, None, None

    if raw.empty:
        log("ERROR: yfinance returned empty data")
        return None, None, None, None

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    # Split into components
    sector_cols = [c for c in SECTOR_ETFS if c in close.columns]
    sector_prices = close[sector_cols].dropna(how="all") if sector_cols else pd.DataFrame()

    spy = close[BENCHMARK] if BENCHMARK in close.columns else pd.Series(dtype=float)

    vix = close[VIX_TICKER] if VIX_TICKER in close.columns else pd.Series(dtype=float)
    # yfinance sometimes returns ^VIX under different column names
    if vix.empty:
        for col in close.columns:
            if 'VIX' in str(col).upper():
                vix = close[col]
                break

    macro_cols = [c for c in MACRO_TICKERS if c in close.columns]
    macro_data = close[macro_cols].dropna(how="all") if macro_cols else pd.DataFrame()

    return sector_prices, spy, vix, macro_data


def run(dry_run=False):
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    # Skip weekends (unless dry-run)
    if today.weekday() >= 5 and not dry_run:
        log("Weekend -- skipping.")
        return

    log(f"=== Cross-Asset Macro Paper Engine v36 -- {today_str} ===")
    state = load_state()

    # Check for strategy version upgrade
    if state.get("strategy_version") != "v36":
        log(f"Strategy upgrade detected: {state.get('strategy_version', 'unknown')} -> v36")
        # Keep existing positions/state but mark the upgrade
        state["strategy_version"] = "v36"
        state["upgrade_date"] = today_str

    # Skip if already ran today (unless dry-run)
    if state.get("last_run_date") == today_str and not dry_run:
        log("Already ran today -- skipping.")
        print_summary(state)
        return

    # Fetch data
    sector_prices, spy, vix, macro_data = fetch_data()
    if sector_prices is None or sector_prices.empty:
        log("ERROR: No sector price data. Aborting.")
        save_state(state)
        return

    if len(sector_prices) < 30:
        log(f"ERROR: Insufficient data ({len(sector_prices)} rows, need 30+). Aborting.")
        save_state(state)
        return

    if vix is None or vix.empty or len(vix.dropna()) < 10:
        log("ERROR: Insufficient VIX data. Aborting.")
        save_state(state)
        return

    # Data freshness check
    last_data_date = sector_prices.index[-1]
    if hasattr(last_data_date, "date"):
        last_data_date = last_data_date.date()
    data_age = (today - last_data_date).days
    if data_age > 3:
        log(f"WARNING: Latest data is {data_age} days old ({last_data_date}). Possible holiday.")

    # Get current VIX for logging
    current_vix = float(vix.dropna().iloc[-1]) if len(vix.dropna()) > 0 else np.nan
    log(f"VIX: {current_vix:.1f}" + (" [HIGH]" if current_vix > VIX_HIGH else ""))

    # Get current prices for all sectors
    current_prices = {}
    for etf in SECTOR_ETFS:
        if etf in sector_prices.columns:
            val = sector_prices[etf].dropna()
            if len(val) > 0:
                current_prices[etf] = float(val.iloc[-1])

    top5 = sorted(current_prices.items(), key=lambda x: x[0])[:5]
    log(f"Prices (sample): " + ", ".join(f"{k}=${v:.2f}" for k, v in top5) + " ...")

    # --- 1. Check exits on open positions ---
    still_open = []
    exits_today = []
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker not in current_prices:
            log(f"  WARN: No price for {ticker}, keeping position open")
            still_open.append(pos)
            continue

        price = current_prices[ticker]
        exit_flag, exit_reason = check_exit(pos, price, today_str)

        if exit_flag:
            exit_price = price * (1 - SLIPPAGE_PCT)
            shares = pos["shares"]
            pnl = (exit_price - pos["entry_price"]) * shares
            pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
            days_held = int(np.busday_count(
                np.datetime64(pos["entry_date"], "D"),
                np.datetime64(today_str, "D"),
            ))

            state["capital"] += shares * exit_price
            state["total_pnl"] += pnl
            state["n_trades"] += 1
            if pnl >= 0:
                state["wins"] += 1
                state["gross_profit"] += pnl
            else:
                state["losses"] += 1
                state["gross_loss"] += abs(pnl)

            trade_record = {
                "ticker": ticker,
                "entry_date": pos["entry_date"],
                "exit_date": today_str,
                "entry_price": pos["entry_price"],
                "exit_price": round(exit_price, 4),
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": days_held,
                "exit_reason": exit_reason,
            }
            state["closed_trades"].append(trade_record)
            state["closed_trades"] = state["closed_trades"][-200:]
            exits_today.append(trade_record)

            log_trade_csv(trade_record)
            log(f"  EXIT {ticker}: ${pnl:+.2f} ({pnl_pct:+.2f}%) after {days_held}d [{exit_reason}]")

            if dry_run:
                log(f"  [DRY RUN] Would have exited {ticker}")
        else:
            still_open.append(pos)

    state["positions"] = still_open

    # --- 2. Check for new entries ---
    entries_today = []
    n_open = len(state["positions"])

    if n_open < MAX_CONCURRENT:
        signal_tickers = generate_signals_today(sector_prices, spy, vix, macro_data)

        if signal_tickers:
            log(f"  Signals: {signal_tickers}")
        else:
            log("  No entry signals today.")

        held_tickers = {p["ticker"] for p in state["positions"]}
        slots_available = MAX_CONCURRENT - n_open

        for signal_ticker in signal_tickers[:slots_available]:
            if signal_ticker in held_tickers:
                log(f"  SKIP {signal_ticker}: already holding")
                continue
            if signal_ticker not in current_prices:
                log(f"  SKIP {signal_ticker}: no price available")
                continue

            price = current_prices[signal_ticker]
            entry_price = price * (1 + SLIPPAGE_PCT)
            position_size = min(MAX_PER_TRADE, state["capital"] * 0.95)

            if position_size < 50:
                log(f"  SKIP {signal_ticker}: insufficient capital (${state['capital']:.0f})")
                continue

            shares = position_size / entry_price
            cost = shares * entry_price

            if dry_run:
                log(f"  [DRY RUN] Would enter {signal_ticker}: {shares:.4f} shares @ ${entry_price:.2f}")
            else:
                state["capital"] -= cost
                pos = {
                    "ticker": signal_ticker,
                    "shares": round(shares, 6),
                    "entry_price": round(entry_price, 4),
                    "entry_date": today_str,
                    "hwm": round(entry_price, 4),
                }
                state["positions"].append(pos)
                entries_today.append(pos)
                held_tickers.add(signal_ticker)
                log(f"  ENTRY {signal_ticker}: {shares:.4f} shares @ ${entry_price:.2f}")
    else:
        log(f"  Max concurrent positions ({MAX_CONCURRENT}) -- skipping signal scan")

    # --- 3. Mark to market ---
    portfolio_value = state["capital"]
    for pos in state["positions"]:
        ticker = pos["ticker"]
        if ticker in current_prices:
            portfolio_value += pos["shares"] * current_prices[ticker]
        else:
            portfolio_value += pos["shares"] * pos["entry_price"]

    state["equity"] = round(portfolio_value, 2)

    if state["equity"] > state["peak_equity"]:
        state["peak_equity"] = state["equity"]
    current_dd = 0.0
    if state["peak_equity"] > 0:
        current_dd = (state["equity"] - state["peak_equity"]) / state["peak_equity"]
    if current_dd < state["max_drawdown"]:
        state["max_drawdown"] = round(current_dd, 6)

    if not dry_run:
        state["equity_curve"].append({
            "date": today_str,
            "equity": state["equity"],
            "positions": len(state["positions"]),
            "vix": round(current_vix, 1) if not pd.isna(current_vix) else None,
        })
        state["equity_curve"] = state["equity_curve"][-500:]

        state["last_run_date"] = today_str
        save_state(state)

    # --- 4. Print summary ---
    print_summary(state, entries_today, exits_today)
    log("Done.")


def print_summary(state, entries_today=None, exits_today=None):
    """Print clean status summary to stdout."""
    entries_today = entries_today or []
    exits_today = exits_today or []

    total_trades = state["n_trades"]
    wr = state["wins"] / total_trades * 100 if total_trades > 0 else 0
    pf = state["gross_profit"] / state["gross_loss"] if state["gross_loss"] > 0 else float("inf")
    avg_win = state["gross_profit"] / state["wins"] if state["wins"] > 0 else 0
    avg_loss = state["gross_loss"] / state["losses"] if state["losses"] > 0 else 0
    return_pct = (state["equity"] / INITIAL_CAPITAL - 1) * 100
    dd_pct = state["max_drawdown"] * 100

    print("\n" + "=" * 60)
    print("  CROSS-ASSET MACRO SECTOR ROTATION v36 -- Paper Trading")
    print("  (AVO-evolved | lockbox Sharpe 2.49 on 2026H1)")
    print("=" * 60)
    print(f"  Equity:     ${state['equity']:,.2f}  ({return_pct:+.2f}%)")
    print(f"  Cash:       ${state['capital']:,.2f}")
    print(f"  Max DD:     {dd_pct:.2f}%")
    print(f"  Total PnL:  ${state['total_pnl']:+,.2f}")
    print("-" * 60)
    print(f"  Trades:     {total_trades}  |  W/L: {state['wins']}/{state['losses']}  |  WR: {wr:.1f}%")
    print(f"  PF: {pf:.2f}  |  Avg Win: ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")
    print("-" * 60)

    if state["positions"]:
        print("  Open Positions:")
        for pos in state["positions"]:
            print(f"    {pos['ticker']:5s}  {pos['shares']:.2f} sh @ ${pos['entry_price']:.2f}  "
                  f"(entered {pos['entry_date']})")
    else:
        print("  No open positions.")

    if entries_today:
        print(f"\n  Today's Entries: {', '.join(p['ticker'] for p in entries_today)}")
    if exits_today:
        exit_strs = [f"{t['ticker']} ${t['pnl']:+.2f}" for t in exits_today]
        print(f"  Today's Exits:  {', '.join(exit_strs)}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    if dry:
        print("[DRY RUN MODE -- no state changes will be saved]\n")
    run(dry_run=dry)
