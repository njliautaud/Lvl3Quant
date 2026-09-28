#!/usr/bin/env python3
"""
AVO Options Live Scanner — drives real RH entries
====================================================
Runs during market hours. Uses the AVO-evolved v32 strategy logic
(RSI, Bollinger Bands, MACD, volume dispersion, relative strength)
to generate actionable sector ETF option signals.

v33 upgrades (2026-09-22):
  - PUT signal generation (put_bb_rsi, put_rsi_overbought, put_macd_turn)
  - Macro regime integration (Fed, yields, dollar, sector rotation, credit)
  - Backtest edge quality loading from signal_backtest_results.json
  - Fixed scoring math (anti-correlation issue between RSI + MACD)

Outputs signals to state/avo_live_signals.json so the spread_execution
prompt picks them up as PRIMARY entry source.

Also implements the AVO FEEDBACK LOOP: after every 5 closed trades,
compares real outcomes vs AVO predictions and flags parameter drift.

Cron: */30 9-15 * * 1-5  (every 30 min during RTH)
"""
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed")
    sys.exit(1)

warnings.filterwarnings("ignore")

ET = pytz.timezone("US/Eastern")
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "state"
SIGNAL_FILE = STATE_DIR / "avo_live_signals.json"
FEEDBACK_FILE = STATE_DIR / "avo_feedback_log.json"
TRADE_LOG = STATE_DIR / "agentic_trade_log.json"
PAPER_STATE = BASE / "paper_engines" / "state" / "options_execution_avo_state.json"
LOG_FILE = BASE / "logs" / "avo_options_scanner.log"
BACKTEST_FILE = BASE / "data" / "signal_backtest_results.json"

# ── AVO-Evolved Parameters (v33, macro-aware + puts) ──

# Account sizing (updated 2026-09-25 — actual RH balance)
ACCOUNT_SIZE = 391.0
MAX_CONCURRENT = 1  # conservative for real money — one at a time
MAX_PER_TRADE_PCT = 0.45  # up to 45% of account per trade (tiny account, need meaningful position)

# Entry — multi-signal with priority scoring
RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 35
RSI_DEEP_OVERSOLD = 25
BB_PERIOD = 20
BB_STD = 2.0
MIN_VOLUME_RATIO = 0.8

# Ticker universe — sector ETFs
CALL_TICKERS = ['XLE', 'XLU', 'XLI', 'XLK', 'XLF', 'XLB', 'XLRE']
AVOID_CALL_TICKERS = ['XLV', 'XLP']  # negative edge historically

# PUT ticker universe — all sector ETFs are eligible for puts
PUT_TICKERS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLC', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLY', 'XLP']

# Macro proxy tickers
MACRO_TICKERS = ['TLT', 'HYG', 'UUP']

# Options targeting
OPTION_DTE_TARGET = 21  # 3 weeks
DELTA_TARGET = 0.40

# Exit params (HC #808 — AVO-validated)
LOW_VOL_TP = 0.43
LOW_VOL_SL = -0.17
HIGH_VOL_TP = 0.25
HIGH_VOL_SL = -0.12
TRAILING_ACTIVATE = 0.08
TRAILING_GIVEBACK = 0.20

# Regime
VIX_HIGH = 25
VIX_EXTREME = 35

# Sector-specific hold days
SECTOR_HOLD = {
    'XLK': 4, 'XLF': 4, 'XLC': 4, 'XLY': 4,
    'XLE': 5, 'XLI': 5,
    'XLU': 10, 'XLB': 9, 'XLRE': 11,
}

DATA_LOOKBACK = 120  # calendar days

# Anti-re-entry cooldown (HC #807 R2 adapted for real account)
COOLDOWN_DAYS = 5       # business days to wait after a LOSS on a ticker
COOLDOWN_PENALTY = 30   # score reduction for tickers in cooldown (blocks all but highest conviction)

# ── Real Trade Performance Scoring (2026-09-25 data-driven update) ──
# Based on analysis of 23 actual RH options trades (Aug-Sep 2026):
#   August: 73% WR (+$38), mostly puts (64%). September: 17% WR (-$312), mostly calls (83%).
#   Root cause: calls in RISK_OFF regime lost -$312. Puts were flat across both months.
#
# IMPORTANT (XLU backtest 2026-09-25): XLU "put wins" (86% WR, PF 5.62) came from the
# SECTOR MOMENTUM SPREADS engine (relative ranking), NOT from AVO scanner RSI/BB signals.
# The AVO scanner's put_bb_rsi for XLU has NO edge (XLU rises after overbought signals).
# XLU CALL signals at extreme oversold (RSI<20) DO have backtest support (94% WR in-sample).
# Boost structure: call_boost for oversold bounce, put_boost = 0 for scanner puts (edge is
# in the momentum spreads engine, not here).
TICKER_PERFORMANCE_CALL = {
    # Call-specific boosts
    'XLU': 10,   # extreme oversold bounce has strong in-sample edge
    'XLE': 5,    # 80% WR on real call trades
    'XLP': 0,    # breakeven
    'XLF': -20,  # BANNED (HC #815)
    'XLC': -10,  # bad track record
    'XLV': -5,   # avoided
}
TICKER_PERFORMANCE_PUT = {
    # Put-specific boosts — scanner put signals have weak/no edge for most tickers.
    # Actual put edge comes from sector momentum spreads engine, not this scanner.
    'XLU': 0,    # AVO scanner put signals for XLU have NO edge (backtest verified)
    'XLE': 3,    # modest edge
    'XLP': 0,    # no data
    'XLF': -20,  # BANNED
    'XLC': -5,   # weak
    'XLV': -5,   # weak
    'XLK': 3,    # put_bb_rsi potential (RSI 68+, BB>0.9)
}
# Direction bias: put signals get a small boost in bearish macro (data-driven).
# But only +3 not +5, since scanner put signals are weaker than momentum-based ones.
DIRECTION_BIAS_PUT_BOOST = 3

# ── Fed regime (UPDATE THIS WHEN POLICY CHANGES) ──
# Set to "HIKING" as of 2026-09-22: Fed hiked rates for first time in 3 years.
# Options: "HIKING", "CUTTING", "HOLD"
FED_REGIME = "HIKING"


def log(msg):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_bbands(series, period=20, num_std=2.0):
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle
    return upper, middle, lower, bandwidth


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Backtest Edge Loading ──

_backtest_cache = None


def load_backtest_results():
    """Load backtested signal edge data. Returns per_etf_results dict or None."""
    global _backtest_cache
    if _backtest_cache is not None:
        return _backtest_cache
    if not BACKTEST_FILE.exists():
        log("WARN: Backtest results file not found, edge quality adjustment disabled")
        _backtest_cache = {}
        return _backtest_cache
    try:
        with open(BACKTEST_FILE) as f:
            data = json.load(f)
        _backtest_cache = data.get("per_etf_results", {})
        log(f"Loaded backtest edge data for {len(_backtest_cache)} ETFs")
        return _backtest_cache
    except Exception as e:
        log(f"WARN: Failed to load backtest results: {e}")
        _backtest_cache = {}
        return _backtest_cache


def get_backtest_edge(ticker, signal_type, horizon="5d"):
    """
    Look up the backtested edge for a ticker/signal combo.
    Returns dict with win_rate, p_value, mean_return, edge_adjustment.
    Uses test set data (out-of-sample) when available, falls back to train.
    """
    bt = load_backtest_results()
    if not bt or ticker not in bt:
        return None

    etf_data = bt[ticker]
    if signal_type not in etf_data:
        return None

    # Prefer test (out-of-sample), fall back to train
    for split in ["test", "train"]:
        split_data = etf_data[signal_type].get(split, {})
        if split_data.get("insufficient_data", True):
            continue
        h_data = split_data.get(horizon)
        if h_data is None or h_data.get("insufficient_data", True):
            continue

        p_value = h_data.get("p_value", 1.0)
        win_rate = h_data.get("win_rate", 0.5)
        mean_ret = h_data.get("mean_return_pct", 0.0)
        n = h_data.get("n", 0)

        # Determine edge adjustment
        # For PUT signals: negative mean_return = the underlying dropped = GOOD for puts
        is_put = signal_type.startswith("put_")
        if is_put:
            # For puts, a negative return on the underlying means the put wins
            effective_wr = 1.0 - win_rate  # the "win_rate" in backtest is for going LONG
            effective_edge = -mean_ret  # flip sign: underlying dropped = put profit
        else:
            effective_wr = win_rate
            effective_edge = mean_ret

        # Score adjustment based on statistical significance + edge direction
        edge_adjustment = 0
        if p_value < 0.10 and effective_edge > 0:
            edge_adjustment = +10  # statistically significant positive edge
        elif p_value < 0.10 and effective_edge < 0:
            edge_adjustment = -15  # statistically significant NEGATIVE edge — penalize
        elif effective_edge < 0 and n >= 10:
            edge_adjustment = -5  # not significant but negative trend with decent sample

        return {
            "win_rate": round(effective_wr, 3),
            "p_value": round(p_value, 4),
            "mean_return_pct": round(mean_ret, 4),
            "effective_edge_pct": round(effective_edge, 4),
            "n_signals": n,
            "split": split,
            "horizon": horizon,
            "edge_adjustment": edge_adjustment,
        }

    return None


# ── Macro Regime ──

def compute_macro_regime(spy, vix, prices, macro_prices):
    """
    Compute macro regime context. Returns dict with fed_regime, yield_direction,
    dollar_trend, sector_rotation, credit_health, and macro_score.

    macro_score ranges from -20 (very bearish, favor puts) to +20 (very bullish, favor calls).
    """
    regime = {
        "fed_regime": FED_REGIME,  # hardcoded — update FED_REGIME constant when policy changes
        "yield_direction": "UNKNOWN",
        "dollar_trend": "UNKNOWN",
        "sector_rotation": "UNKNOWN",
        "credit_health": "UNKNOWN",
        "macro_score": 0,
        "macro_bias": "NEUTRAL",
    }

    score = 0

    # ── Fed regime component ──
    if FED_REGIME == "HIKING":
        score -= 6  # hawkish = bearish bias
    elif FED_REGIME == "CUTTING":
        score += 6  # dovish = bullish bias
    # HOLD = 0

    # ── Yield direction (TLT proxy) ──
    # TLT declining = yields rising = hawkish/bearish
    if 'TLT' in macro_prices.columns:
        tlt = macro_prices['TLT'].dropna()
        if len(tlt) >= 20:
            tlt_sma20 = tlt.rolling(20).mean()
            tlt_ret_20d = float((tlt.iloc[-1] / tlt.iloc[-20] - 1) * 100) if len(tlt) >= 20 else 0

            if tlt.iloc[-1] < tlt_sma20.iloc[-1] and tlt_ret_20d < -1.0:
                regime["yield_direction"] = "RISING"  # TLT falling = yields rising
                score -= 4
            elif tlt.iloc[-1] > tlt_sma20.iloc[-1] and tlt_ret_20d > 1.0:
                regime["yield_direction"] = "FALLING"  # TLT rising = yields falling
                score += 4
            elif tlt.iloc[-1] < tlt_sma20.iloc[-1]:
                regime["yield_direction"] = "RISING_MILD"
                score -= 2
            elif tlt.iloc[-1] > tlt_sma20.iloc[-1]:
                regime["yield_direction"] = "FALLING_MILD"
                score += 2
            else:
                regime["yield_direction"] = "STABLE"

    # ── Dollar trend (UUP proxy) ──
    # Dollar rising = tighter financial conditions = bearish
    if 'UUP' in macro_prices.columns:
        uup = macro_prices['UUP'].dropna()
        if len(uup) >= 20:
            uup_sma20 = uup.rolling(20).mean()
            uup_ret_20d = float((uup.iloc[-1] / uup.iloc[-20] - 1) * 100) if len(uup) >= 20 else 0

            if uup.iloc[-1] > uup_sma20.iloc[-1] and uup_ret_20d > 0.5:
                regime["dollar_trend"] = "STRONG"
                score -= 3
            elif uup.iloc[-1] < uup_sma20.iloc[-1] and uup_ret_20d < -0.5:
                regime["dollar_trend"] = "WEAK"
                score += 3
            else:
                regime["dollar_trend"] = "STABLE"

    # ── Sector rotation ──
    # Cyclicals (XLI, XLB, XLE) vs Defensives (XLU, XLP, XLV)
    cyclicals = ['XLI', 'XLB', 'XLE']
    defensives = ['XLU', 'XLP', 'XLV']

    all_tickers_in_data = set(prices.columns)
    cyc_available = [t for t in cyclicals if t in all_tickers_in_data]
    def_available = [t for t in defensives if t in all_tickers_in_data]

    if len(cyc_available) >= 2 and len(def_available) >= 2:
        cyc_ret = prices[cyc_available].pct_change(20).iloc[-1].mean()
        def_ret = prices[def_available].pct_change(20).iloc[-1].mean()

        if def_ret > cyc_ret + 0.005:
            regime["sector_rotation"] = "DEFENSIVE"
            score -= 3
        elif cyc_ret > def_ret + 0.005:
            regime["sector_rotation"] = "RISK_ON"
            score += 3
        else:
            regime["sector_rotation"] = "BALANCED"

    # ── Credit health (HYG proxy) ──
    # HYG declining = credit stress = bearish
    if 'HYG' in macro_prices.columns:
        hyg = macro_prices['HYG'].dropna()
        if len(hyg) >= 20:
            hyg_sma20 = hyg.rolling(20).mean()
            hyg_ret_20d = float((hyg.iloc[-1] / hyg.iloc[-20] - 1) * 100) if len(hyg) >= 20 else 0

            if hyg.iloc[-1] < hyg_sma20.iloc[-1] and hyg_ret_20d < -1.0:
                regime["credit_health"] = "STRESSED"
                score -= 4
            elif hyg.iloc[-1] > hyg_sma20.iloc[-1] and hyg_ret_20d > 0.5:
                regime["credit_health"] = "HEALTHY"
                score += 4
            else:
                regime["credit_health"] = "STABLE"

    # Clamp score to [-20, +20]
    score = max(-20, min(20, score))
    regime["macro_score"] = score

    # Determine bias label
    if score <= -8:
        regime["macro_bias"] = "BEARISH (strongly favoring puts over calls)"
    elif score <= -3:
        regime["macro_bias"] = "BEARISH (favoring puts over calls)"
    elif score >= 8:
        regime["macro_bias"] = "BULLISH (strongly favoring calls over puts)"
    elif score >= 3:
        regime["macro_bias"] = "BULLISH (favoring calls over puts)"
    else:
        regime["macro_bias"] = "NEUTRAL"

    return regime


def fetch_data():
    """Fetch current market data for all sector ETFs + SPY + VIX + macro proxies."""
    # Combine all tickers, dedup
    all_tickers = list(set(CALL_TICKERS + PUT_TICKERS + MACRO_TICKERS + ['SPY', '^VIX']))
    end = datetime.now(ET)
    start = end - timedelta(days=DATA_LOOKBACK)

    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 20:
                data[t] = df['Close'].squeeze()
        except Exception as e:
            log(f"  WARN: Failed to fetch {t}: {e}")

    if 'SPY' not in data or '^VIX' not in data:
        log("ERROR: Missing SPY or VIX data")
        return None, None, None, None

    all_prices = pd.DataFrame(data)
    spy = all_prices['SPY']
    vix = all_prices['^VIX']

    # Separate macro tickers from sector prices
    macro_cols = [t for t in MACRO_TICKERS if t in all_prices.columns]
    macro_prices = all_prices[macro_cols] if macro_cols else pd.DataFrame()

    # Sector prices = everything except SPY, VIX, and macro tickers
    drop_cols = ['SPY', '^VIX'] + macro_cols
    prices = all_prices.drop(columns=[c for c in drop_cols if c in all_prices.columns], errors='ignore')

    return prices, spy, vix, macro_prices


def score_call_signal(ticker, rsi_val, bb_pctb, macd_hist, vol_ratio, rel_strength,
                      vix_val, dispersion_high, macro_score, backtest_edge_info):
    """
    Score a CALL entry signal 0-100 based on AVO-evolved factors.
    v33 fix: RSI and MACD are scored with awareness of their anti-correlation.
    The 'momentum_turning' component replaces rigid MACD > 0 requirement
    so that deeply oversold + MACD improving (not yet positive) still scores well.
    """
    score = 0

    # RSI component (0-30 pts) — oversold = bullish for calls
    if rsi_val < RSI_DEEP_OVERSOLD:
        score += 30  # deep oversold — strongest signal
    elif rsi_val < RSI_ENTRY_THRESHOLD:
        score += 20  # standard oversold
    elif rsi_val < 40:
        score += 5   # mildly oversold (relaxed)

    # Bollinger Band component (0-20 pts) — below lower band = bullish for calls
    if bb_pctb is not None:
        if bb_pctb < 0:
            score += 20  # below lower band
        elif bb_pctb < 0.1:
            score += 12  # near lower band

    # MACD component (0-15 pts)
    # v33 fix: when RSI is deeply oversold, MACD is almost always negative.
    # Score MACD on "improving" rather than "positive" when RSI < 30.
    if macd_hist is not None:
        if rsi_val < 30:
            # Deep oversold: reward MACD improving (less negative) rather than positive
            if macd_hist > 0:
                score += 15  # rare but very strong — both oversold AND turning
            elif macd_hist > -0.05:
                score += 12  # nearly flat histogram = momentum about to turn
            elif macd_hist > -0.2:
                score += 8   # histogram improving from deep negative
            else:
                score += 3   # still deep negative but RSI is extremely oversold
        else:
            # Normal RSI: standard MACD scoring
            if macd_hist > 0:
                score += 15  # bullish histogram
            elif macd_hist > -0.1:
                score += 5   # histogram improving

    # Volume component (0-10 pts)
    if vol_ratio > 1.5:
        score += 10  # high volume conviction
    elif vol_ratio > 1.0:
        score += 5

    # Relative strength component (0-10 pts)
    if rel_strength is not None:
        if rel_strength > 0.01:
            score += 10  # outperforming SPY
        elif rel_strength > -0.02:
            score += 5   # near SPY

    # Dispersion boost (0-10 pts)
    if dispersion_high:
        score += 10  # high cross-sectional dispersion = better dip-buy

    # VIX penalty (0 to -10 pts)
    if vix_val > VIX_EXTREME:
        score -= 10
    elif vix_val > VIX_HIGH:
        score -= 5

    # ── Macro adjustment (-20 to +20 pts) ──
    # Positive macro_score = bullish = boost calls
    score += macro_score

    # ── Backtest edge adjustment (-15 to +10 pts) ──
    if backtest_edge_info is not None:
        score += backtest_edge_info.get("edge_adjustment", 0)

    # Cap at [0, 100]
    return min(max(score, 0), 100)


def score_put_signal(ticker, rsi_val, bb_pctb, macd_hist, vol_ratio, rel_strength,
                     vix_val, dispersion_high, macro_score, backtest_edge_info):
    """
    Score a PUT entry signal 0-100. Mirrors call scoring but inverted:
    overbought RSI, above upper BB, bearish MACD = strong put signal.
    """
    score = 0

    # RSI component (0-30 pts) — overbought = bearish for puts
    if rsi_val > 75:
        score += 30  # deeply overbought — strongest put signal
    elif rsi_val > 70:
        score += 20  # standard overbought
    elif rsi_val > 60:
        score += 5   # mildly overbought

    # Bollinger Band component (0-20 pts) — above upper band = bearish
    if bb_pctb is not None:
        if bb_pctb > 1.0:
            score += 20  # above upper band
        elif bb_pctb > 0.9:
            score += 12  # near upper band

    # MACD component (0-15 pts) — bearish histogram
    if macd_hist is not None:
        if rsi_val > 70:
            # Deep overbought: reward MACD deteriorating
            if macd_hist < 0:
                score += 15  # already bearish + overbought RSI = very strong
            elif macd_hist < 0.05:
                score += 12  # nearly flat, about to cross bearish
            elif macd_hist < 0.2:
                score += 8   # histogram declining from positive
            else:
                score += 3   # still positive but RSI is overbought
        else:
            # Normal RSI: standard bearish MACD scoring
            if macd_hist < 0:
                score += 15  # bearish histogram
            elif macd_hist < 0.1:
                score += 5   # histogram declining

    # Volume component (0-10 pts) — same as calls, high volume = conviction
    if vol_ratio > 1.5:
        score += 10
    elif vol_ratio > 1.0:
        score += 5

    # Relative strength component (0-10 pts) — INVERTED for puts
    # Underperforming SPY = better put candidate
    if rel_strength is not None:
        if rel_strength < -0.02:
            score += 10  # lagging SPY = weak sector
        elif rel_strength < 0.01:
            score += 5   # near SPY

    # Dispersion boost (0-10 pts)
    if dispersion_high:
        score += 10  # high dispersion = more extreme moves

    # VIX boost for puts (opposite of calls) — high VIX favors puts
    if vix_val > VIX_EXTREME:
        score += 5   # extreme fear benefits put buyers (if already overbought before crash)
    elif vix_val > VIX_HIGH:
        score += 3

    # ── Macro adjustment (-20 to +20 pts) ──
    # Negative macro_score = bearish = boost puts (subtract negative = add)
    score -= macro_score

    # ── Backtest edge adjustment (-15 to +10 pts) ──
    if backtest_edge_info is not None:
        score += backtest_edge_info.get("edge_adjustment", 0)

    # Cap at [0, 100]
    return min(max(score, 0), 100)


def generate_signals(prices, spy, vix, macro_regime):
    """
    Generate scored entry signals for today using AVO-evolved logic.
    Returns list of signal dicts sorted by score (highest first).
    Generates both CALL (dip-buy oversold) and PUT (overbought reversal) signals.
    """
    signals = []
    today = prices.index[-1]
    vix_today = float(vix.iloc[-1])
    macro_score = macro_regime.get("macro_score", 0)

    # Anti-re-entry cooldown check
    cooldown_tickers = get_cooldown_tickers()
    if cooldown_tickers:
        cd_parts = [f"{t} ({v['bdays_ago']}d ago)" for t, v in cooldown_tickers.items()]
        log(f"  Cooldown active for {len(cooldown_tickers)} ticker(s): {', '.join(cd_parts)}")

    # Cross-sectional dispersion
    sector_rets = prices.pct_change()
    dispersion = sector_rets.std(axis=1).rolling(15).mean()
    disp_avg = dispersion.rolling(60).mean()
    dispersion_high = bool(dispersion.iloc[-1] > disp_avg.iloc[-1] * 1.35) if len(disp_avg.dropna()) > 0 else False

    # SPY trend
    spy_sma50 = spy.rolling(50).mean()
    spy_sma200 = spy.rolling(200).mean()
    spy_trend_up = bool(spy.iloc[-1] > spy_sma200.iloc[-1]) if len(spy_sma200.dropna()) > 0 else True

    # ── Helper to compute indicators for a ticker ──
    def _compute_indicators(ticker):
        if ticker not in prices.columns:
            return None
        px = prices[ticker].dropna()
        if len(px) < max(RSI_PERIOD + 5, 50):
            return None

        rsi = compute_rsi(px, RSI_PERIOD)
        rsi_val = float(rsi.iloc[-1])

        bb_upper, bb_mid, bb_lower, bb_bw = compute_bbands(px, BB_PERIOD, BB_STD)
        bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        bb_pctb = float((px.iloc[-1] - bb_lower.iloc[-1]) / bb_range) if bb_range > 0 else 0.5

        macd_line, macd_signal, macd_hist = compute_macd(px)
        macd_hist_val = float(macd_hist.iloc[-1])
        # Check if MACD histogram is turning (for put_macd_turn detection)
        macd_hist_prev = float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else macd_hist_val

        vol = px.pct_change().abs()
        avg_vol = vol.rolling(20).mean()
        vol_ratio = float(vol.iloc[-1] / avg_vol.iloc[-1]) if avg_vol.iloc[-1] > 0 else 1.0

        # Relative strength vs SPY (10-day)
        sector_ret = float(px.pct_change(10).iloc[-1]) if len(px) > 10 else 0
        spy_ret = float(spy.pct_change(10).iloc[-1]) if len(spy) > 10 else 0
        rel_strength = sector_ret - spy_ret

        price_now = float(px.iloc[-1])

        return {
            "rsi_val": rsi_val,
            "bb_pctb": bb_pctb,
            "macd_hist_val": macd_hist_val,
            "macd_hist_prev": macd_hist_prev,
            "vol_ratio": vol_ratio,
            "rel_strength": rel_strength,
            "price_now": price_now,
        }

    # ── Helper to build signal dict ──
    def _build_signal(ticker, direction, score, ind, signal_type, backtest_info):
        in_cooldown = ticker in cooldown_tickers

        # Apply cooldown penalty
        if in_cooldown:
            original_score = score
            score -= COOLDOWN_PENALTY
            score = max(score, 0)
            log(f"  {ticker}: {direction.upper()} score {original_score} -> {score} "
                f"(cooldown penalty, lost {cooldown_tickers[ticker]['bdays_ago']}d ago)")

        # Determine regime-appropriate exit params
        if vix_today >= VIX_HIGH:
            tp_pct = HIGH_VOL_TP
            sl_pct = HIGH_VOL_SL
            hold_days = max(SECTOR_HOLD.get(ticker, 5) - 3, 2)
        else:
            tp_pct = LOW_VOL_TP
            sl_pct = LOW_VOL_SL
            hold_days = SECTOR_HOLD.get(ticker, 5)

        sig = {
            "ticker": ticker,
            "direction": direction,
            "signal_type": signal_type,
            "score": score,
            "rsi": round(ind["rsi_val"], 1),
            "bb_pctb": round(ind["bb_pctb"], 3),
            "macd_hist": round(ind["macd_hist_val"], 4),
            "volume_ratio": round(ind["vol_ratio"], 2),
            "rel_strength_10d": round(ind["rel_strength"], 4),
            "underlying_price": round(ind["price_now"], 2),
            "vix": round(vix_today, 2),
            "regime": "HIGH-VOL" if vix_today >= VIX_HIGH else "LOW-VOL",
            "macro_score": macro_score,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "hold_days": hold_days,
            "trailing_activate": TRAILING_ACTIVATE,
            "trailing_giveback": TRAILING_GIVEBACK,
            "dte_target": OPTION_DTE_TARGET,
            "delta_target": DELTA_TARGET,
            "dispersion_boost": dispersion_high,
            "spy_trend_up": spy_trend_up,
            "cooldown_active": in_cooldown,
            "cooldown_detail": (
                f"Lost {cooldown_tickers[ticker]['bdays_ago']}d ago, "
                f"penalty -{COOLDOWN_PENALTY}pts"
            ) if in_cooldown else None,
            "backtest_edge": backtest_info,
            "timestamp": datetime.now(ET).isoformat(),
        }
        return sig

    # ══════════════════════════════════════════════════
    #  CALL signals — oversold dip-buy
    # ══════════════════════════════════════════════════
    for ticker in CALL_TICKERS:
        if ticker in AVOID_CALL_TICKERS:
            continue

        ind = _compute_indicators(ticker)
        if ind is None:
            continue

        rsi_val = ind["rsi_val"]
        bb_pctb = ind["bb_pctb"]
        macd_hist_val = ind["macd_hist_val"]

        # Determine which call signal type(s) fired
        fired_signals = []

        # call_bb_rsi: RSI < 40 AND below/near lower BB
        if rsi_val < 40 and bb_pctb < 0.15:
            fired_signals.append("call_bb_rsi")

        # call_rsi_oversold: RSI < 35 (standard oversold)
        if rsi_val < RSI_ENTRY_THRESHOLD:
            fired_signals.append("call_rsi_oversold")

        # call_macd_turn: MACD histogram turns positive from negative
        if ind["macd_hist_prev"] < 0 and macd_hist_val >= 0:
            fired_signals.append("call_macd_turn")

        if not fired_signals:
            continue

        # Use the best signal type for scoring
        best_signal = fired_signals[0]
        backtest_info = get_backtest_edge(ticker, best_signal)

        score = score_call_signal(
            ticker, rsi_val, bb_pctb, macd_hist_val,
            ind["vol_ratio"], ind["rel_strength"],
            vix_today, dispersion_high,
            macro_score, backtest_info
        )

        # Apply real trade performance boost/penalty (data-driven 2026-09-25)
        call_boost = TICKER_PERFORMANCE_CALL.get(ticker, 0)
        score += call_boost

        # Minimum threshold for calls
        if score < 25:
            continue

        sig = _build_signal(ticker, "call", score, ind, best_signal, backtest_info)
        signals.append(sig)

    # ══════════════════════════════════════════════════
    #  PUT signals — overbought reversal
    # ══════════════════════════════════════════════════
    for ticker in PUT_TICKERS:
        ind = _compute_indicators(ticker)
        if ind is None:
            continue

        rsi_val = ind["rsi_val"]
        bb_pctb = ind["bb_pctb"]
        macd_hist_val = ind["macd_hist_val"]

        # Determine which put signal type(s) fired
        fired_signals = []

        # put_bb_rsi: RSI > 65 AND at/above upper BB
        if rsi_val > 65 and bb_pctb > 0.9:
            fired_signals.append("put_bb_rsi")

        # put_rsi_overbought: RSI > 70
        if rsi_val > 70:
            fired_signals.append("put_rsi_overbought")

        # put_macd_turn: MACD histogram turns negative from positive
        if ind["macd_hist_prev"] > 0 and macd_hist_val <= 0:
            fired_signals.append("put_macd_turn")

        if not fired_signals:
            continue

        # Use the best signal type for scoring
        best_signal = fired_signals[0]
        backtest_info = get_backtest_edge(ticker, best_signal)

        score = score_put_signal(
            ticker, rsi_val, bb_pctb, macd_hist_val,
            ind["vol_ratio"], ind["rel_strength"],
            vix_today, dispersion_high,
            macro_score, backtest_info
        )

        # Apply real trade performance boost/penalty (data-driven 2026-09-25)
        put_boost = TICKER_PERFORMANCE_PUT.get(ticker, 0)
        score += put_boost

        # Direction bias: modest put boost in bearish macro
        score += DIRECTION_BIAS_PUT_BOOST

        # Minimum threshold for puts — same as calls
        if score < 25:
            continue

        sig = _build_signal(ticker, "put", score, ind, best_signal, backtest_info)
        signals.append(sig)

    # Sort by score descending
    signals.sort(key=lambda x: x['score'], reverse=True)
    return signals


def check_active_positions():
    """Check if we already have an active RH option position."""
    active_file = BASE / "state" / "active_options.json"
    if not active_file.exists():
        return 0
    try:
        with open(active_file) as f:
            data = json.load(f)
        positions = data.get("positions", [])
        active = [p for p in positions
                  if p.get("status") not in {"closed", "expired", "cancelled"}]
        return len(active)
    except Exception:
        return 0


def get_cooldown_tickers():
    """
    Anti-re-entry cooldown: find tickers that hit stop-loss (LOSS) within
    the last COOLDOWN_DAYS business days. Returns dict of ticker -> info.
    Prevents the XLI-style death spiral of 3+ consecutive SL hits.
    """
    cooldown = {}
    today = datetime.now(ET).date()

    # Load agentic trade log
    if TRADE_LOG.exists():
        try:
            with open(TRADE_LOG) as f:
                data = json.load(f)
            trades = data.get("trades", [])
        except Exception:
            trades = []
    else:
        trades = []

    for t in trades:
        if t.get("status") != "LOSS":
            continue

        exit_date_str = t.get("exit_date", "")
        if not exit_date_str:
            continue

        try:
            exit_date = datetime.strptime(exit_date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                exit_date = datetime.strptime(exit_date_str[:10], "%Y-%m-%d").date()
            except ValueError:
                continue

        # Count business days since exit
        bdays = int(np.busday_count(exit_date, today))
        if bdays <= COOLDOWN_DAYS:
            ticker = t.get("ticker", "")
            if ticker:
                # Track the most recent loss for each ticker
                if ticker not in cooldown or exit_date > cooldown[ticker]["exit_date"]:
                    cooldown[ticker] = {
                        "exit_date": exit_date,
                        "bdays_ago": bdays,
                        "exit_pnl": t.get("exit_pnl", 0),
                        "reason": t.get("outcome_detail", {}).get("reason", "stop-loss"),
                    }

    return cooldown


def load_feedback():
    """Load the feedback log for AVO self-improvement tracking."""
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"trades": [], "last_avo_reeval": None, "version": "v33", "drift_alerts": []}


def save_feedback(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def check_feedback_loop():
    """
    AVO SELF-IMPROVEMENT LOOP:
    After every 5 closed real trades, compare outcomes vs what
    the AVO strategy would have done. Flag parameter drift.
    """
    feedback = load_feedback()

    # Load real trade outcomes
    if TRADE_LOG.exists():
        try:
            with open(TRADE_LOG) as f:
                trade_data = json.load(f)
            closed = [t for t in trade_data.get("trades", [])
                      if t.get("status") in ("WIN", "LOSS") and t.get("exit_pnl") is not None]
        except Exception:
            closed = []
    else:
        closed = []

    # Load paper trade outcomes
    if PAPER_STATE.exists():
        try:
            with open(PAPER_STATE) as f:
                paper_data = json.load(f)
            paper_trades = paper_data.get("closed_trades", [])
        except Exception:
            paper_trades = []
    else:
        paper_trades = []

    # Calculate drift metrics
    n_real = len(closed)
    n_paper = len(paper_trades)

    real_wr = sum(1 for t in closed if t.get("status") == "WIN") / max(n_real, 1)
    paper_wr = sum(1 for t in paper_trades if t.get("pnl", 0) > 0) / max(n_paper, 1)

    paper_avg_pnl = np.mean([t.get("pnl", 0) for t in paper_trades]) if paper_trades else 0

    drift = {
        "real_trades": n_real,
        "real_wr": round(real_wr, 3),
        "paper_trades": n_paper,
        "paper_wr": round(paper_wr, 3),
        "paper_avg_pnl": round(paper_avg_pnl, 2),
        "wr_gap": round(paper_wr - real_wr, 3),
        "checked_at": datetime.now(ET).isoformat(),
    }

    # Flag if real performance is significantly worse than paper
    if n_real >= 5 and drift["wr_gap"] > 0.15:
        drift["DRIFT_ALERT"] = (
            f"Real WR ({real_wr:.0%}) is {drift['wr_gap']:.0%} below paper WR ({paper_wr:.0%}). "
            f"The ad-hoc entry process is underperforming the AVO strategy. "
            f"RECOMMENDATION: Switch to AVO-driven entries."
        )
        feedback["drift_alerts"].append(drift)
        log(f"DRIFT ALERT: {drift['DRIFT_ALERT']}")

    feedback["latest_drift_check"] = drift
    save_feedback(feedback)
    return drift


def run():
    log("=" * 60)
    log("AVO Options Scanner v33 — scanning for live signals (calls + puts, macro-aware)")

    # Step 1: Check if we have room for a new position
    active = check_active_positions()
    if active >= MAX_CONCURRENT:
        log(f"Already have {active} active position(s), max is {MAX_CONCURRENT}. No new entries.")
        # Still save empty signals file so the prompt knows scanner ran
        output = {
            "signals": [],
            "meta": {
                "timestamp": datetime.now(ET).isoformat(),
                "reason": f"at_capacity ({active}/{MAX_CONCURRENT})",
                "scanner_version": "v33",
            }
        }
        with open(SIGNAL_FILE, "w") as f:
            json.dump(output, f, indent=2)
        return

    # Step 2: Fetch market data (now includes macro proxies)
    log("Fetching market data (sectors + macro proxies: TLT, HYG, UUP)...")
    prices, spy, vix, macro_prices = fetch_data()
    if prices is None:
        log("ERROR: Could not fetch market data")
        return

    log(f"Data fetched: {len(prices)} days, {len(prices.columns)} sectors, VIX={float(vix.iloc[-1]):.1f}")

    # Step 3: Compute macro regime
    log("Computing macro regime...")
    macro_regime = compute_macro_regime(spy, vix, prices, macro_prices)
    log(f"  Fed regime: {macro_regime['fed_regime']}")
    log(f"  Yield direction: {macro_regime['yield_direction']}")
    log(f"  Dollar trend: {macro_regime['dollar_trend']}")
    log(f"  Sector rotation: {macro_regime['sector_rotation']}")
    log(f"  Credit health: {macro_regime['credit_health']}")
    log(f"  Macro score: {macro_regime['macro_score']} -> bias: {macro_regime['macro_bias']}")

    # Step 4: Load backtest edge data
    load_backtest_results()

    # Step 5: Generate scored signals (calls + puts)
    signals = generate_signals(prices, spy, vix, macro_regime)
    log(f"Generated {len(signals)} signals above threshold")

    n_calls = sum(1 for s in signals if s['direction'] == 'call')
    n_puts = sum(1 for s in signals if s['direction'] == 'put')
    log(f"  Breakdown: {n_calls} call(s), {n_puts} put(s)")

    for s in signals:
        bt_tag = ""
        if s.get("backtest_edge"):
            bt_tag = f" bt_adj={s['backtest_edge'].get('edge_adjustment', 0):+d}"
        log(f"  {s['ticker']} {s['direction'].upper()} [{s['signal_type']}] score={s['score']} "
            f"RSI={s['rsi']} BB%B={s['bb_pctb']:.2f} MACD_hist={s['macd_hist']:.4f} "
            f"regime={s['regime']} macro={macro_regime['macro_score']:+d}{bt_tag}")

    # Step 6: Save signals for the execution prompt
    output = {
        "signals": signals,
        "meta": {
            "timestamp": datetime.now(ET).isoformat(),
            "scanner_version": "v33",
            "vix": round(float(vix.iloc[-1]), 2),
            "spy_price": round(float(spy.iloc[-1]), 2),
            "spy_trend": "UP" if signals and signals[0].get("spy_trend_up") else "DOWN/FLAT",
            "active_positions": active,
            "max_concurrent": MAX_CONCURRENT,
            "macro": {
                "fed_regime": macro_regime["fed_regime"],
                "yield_direction": macro_regime["yield_direction"],
                "dollar_trend": macro_regime["dollar_trend"],
                "sector_rotation": macro_regime["sector_rotation"],
                "credit_health": macro_regime["credit_health"],
                "macro_score": macro_regime["macro_score"],
                "macro_bias": macro_regime["macro_bias"],
            },
        }
    }
    with open(SIGNAL_FILE, "w") as f:
        json.dump(output, f, indent=2)

    if signals:
        top = signals[0]
        log(f"TOP SIGNAL: {top['ticker']} {top['direction'].upper()} [{top.get('signal_type', '?')}] "
            f"score={top['score']}, price={top['underlying_price']}, "
            f"TP={top['tp_pct']:.0%}, SL={top['sl_pct']:.0%}, "
            f"hold={top['hold_days']}d, macro_bias={macro_regime['macro_bias']}")
    else:
        log("No actionable signals right now. Market conditions don't match AVO entry criteria.")

    # Step 7: Run feedback loop check
    log("Running AVO feedback loop check...")
    drift = check_feedback_loop()
    log(f"Feedback: real={drift['real_trades']} trades ({drift['real_wr']:.0%} WR), "
        f"paper={drift['paper_trades']} trades ({drift['paper_wr']:.0%} WR), "
        f"gap={drift['wr_gap']:.0%}")

    log("Scanner complete.")


if __name__ == "__main__":
    run()
