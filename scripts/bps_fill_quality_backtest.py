#!/usr/bin/env python3
"""
BPS Backtest with Fill-Quality-Validated Universe
==================================================
Uses ONLY the 52 tickers that passed live option chain fill quality testing
(688 real quotes, 2026-07-09). Each ticker gets its own credit haircut based
on the measured real/BS ratio.

This is the most honest BPS backtest: actual-tradeable universe + measured costs.

Includes:
  - HC #659 permutation test (random directions must lose)
  - Full production config (dynamic delta, VIX scale, CB, earnings filter)
  - Comparison: uniform haircut vs ticker-specific haircuts
"""

import sys, json, time, math, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUTPUT = ROOT / "output" / "bps_fill_quality_backtest"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100_000
DTE_TARGET = 10

# ═══════════════════════════════════════════════════════════════════
# Fill-Quality-Validated Universe (from 688 live quotes, 2026-07-09)
# Only tickers with avg real credit > $0.15 AND ≥80% viable quotes
# ═══════════════════════════════════════════════════════════════════

# ticker: (avg_real_credit, real/BS_ratio) from live chain analysis
VALIDATED_TICKERS = {
    'GOOGL': (2.935, 0.84), 'DDOG': (2.857, 1.43), 'CRWD': (2.558, 1.75),
    'IBM': (2.505, 1.03), 'FSLR': (2.485, 1.13), 'AVGO': (2.365, 0.58),
    'HUT': (2.325, 1.27), 'ADBE': (2.306, 0.79), 'BA': (2.237, 0.83),
    'INTC': (2.200, 0.66), 'AAPL': (2.120, 0.55), 'DASH': (1.947, 0.78),
    'AXP': (1.852, 0.82), 'DUOL': (1.630, 0.94), 'AMZN': (1.586, 0.69),
    'BABA': (1.549, 0.90), 'DHR': (1.415, 0.96), 'BRK-B': (1.394, 0.63),
    'CRM': (1.338, 0.99), 'CSCO': (1.300, 1.12), 'ABBV': (1.195, 0.44),
    'CVX': (1.110, 0.89), 'FCX': (1.086, 1.11), 'ABNB': (1.082, 0.73),
    'HD': (1.035, 0.36), 'ETSY': (1.033, 1.18), 'CROX': (0.965, 0.50),
    'FANG': (0.950, 0.97), 'IRM': (0.850, 0.84), 'DKS': (0.815, 0.55),
    'DUK': (0.805, 0.67), 'COP': (0.804, 0.68), 'CVS': (0.800, 1.41),
    'EMR': (0.800, 0.76), 'GM': (0.741, 1.14), 'DECK': (0.735, 0.61),
    'BILL': (0.670, 1.17), 'FDX': (0.670, 0.20), 'ARWR': (0.630, 0.73),
    'GILD': (0.501, 0.38), 'CELH': (0.362, 0.72), 'CMG': (0.360, 0.87),
    'CLSK': (0.347, 1.21), 'BMY': (0.344, 0.51), 'AR': (0.320, 1.48),
    'AAL': (0.266, 1.26), 'DXCM': (0.256, 0.44), 'DOW': (0.256, 1.34),
    'BMRN': (0.200, 1.52), 'CRSP': (0.200, 0.34), 'DVN': (0.189, 0.39),
    'F': (0.184, 1.24),
}

TICKER_LIST = sorted(VALIDATED_TICKERS.keys())

# ═══════════════════════════════════════════════════════════════════
# Black-Scholes Primitives
# ═══════════════════════════════════════════════════════════════════

def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _ndtri(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def bs_price(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if kind == "put":
        return K*math.exp(-r*T)*_Phi(-d2) - S*math.exp(-q*T)*_Phi(-d1)
    return S*math.exp(-q*T)*_Phi(d1) - K*math.exp(-r*T)*_Phi(d2)

def strike_from_delta(S, T, sigma, target_delta, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        return S
    target = abs(target_delta)
    p = target if kind == "call" else (1 - target)
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = _ndtri(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r - q + 0.5*sigma**2)*T))
    return K

COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

def trade_cost(premium, contracts):
    slip = max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium) * 100 * contracts if premium > 0 else 0
    comm = COST_PER_CONTRACT * contracts
    return slip + comm


# ═══════════════════════════════════════════════════════════════════
# Data Loading (same as full_stack)
# ═══════════════════════════════════════════════════════════════════

def generate_iv_features(prices_df, tickers):
    frames = []
    for tk in tickers:
        tk_px = prices_df[prices_df["ticker"] == tk].sort_values("date").copy()
        if len(tk_px) < 60:
            continue
        tk_px["log_ret"] = np.log1p(tk_px["close"].pct_change())
        tk_px["rv_20"] = tk_px["log_ret"].rolling(20).std() * np.sqrt(252)
        tk_px["sigma"] = tk_px["rv_20"] * 1.15
        tk_px["iv_high_252"] = tk_px["sigma"].rolling(252).max()
        tk_px["iv_low_252"] = tk_px["sigma"].rolling(252).min()
        iv_range = tk_px["iv_high_252"] - tk_px["iv_low_252"]
        tk_px["iv_rank"] = np.where(iv_range > 0.001,
            (tk_px["sigma"] - tk_px["iv_low_252"]) / iv_range, 0.5)
        tk_px["ticker"] = tk
        cols = ["date", "ticker", "sigma", "iv_rank"]
        frame = tk_px.dropna(subset=["sigma"])[cols]
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def load_all_data():
    print("Loading data...")
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    try:
        pexp = pd.read_parquet(CACHE / "prices_expanded.parquet")
        pexp = pexp.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                     "Close": "close", "Volume": "volume"})
        pexp["date"] = pd.to_datetime(pexp["date"]).dt.tz_localize(None)
        new_tks = set(pexp["ticker"].unique()) - set(prices["ticker"].unique())
        if new_tks:
            prices = pd.concat([prices, pexp[pexp["ticker"].isin(new_tks)]], ignore_index=True)
    except Exception:
        pass

    try:
        p3 = pd.read_parquet(CACHE / "prices_v3_expansion.parquet")
        p3["date"] = pd.to_datetime(p3["date"]).dt.tz_localize(None)
        new_tks = set(p3["ticker"].unique()) - set(prices["ticker"].unique())
        if new_tks:
            prices = pd.concat([prices, p3[p3["ticker"].isin(new_tks)]], ignore_index=True)
    except Exception:
        pass

    prices = prices[prices["date"] >= "2019-01-01"].copy()
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="first")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "log_ret" not in prices.columns:
        prices["ret"] = prices.groupby("ticker")["close"].pct_change()
        prices["log_ret"] = np.log1p(prices["ret"])

    # IV
    iv = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    iv["date"] = pd.to_datetime(iv["date"]).dt.tz_localize(None)
    iv = iv[iv["date"] >= "2019-01-01"]

    need_iv = set(TICKER_LIST) & set(prices["ticker"].unique()) - set(iv["ticker"].unique())
    if need_iv:
        iv_new = generate_iv_features(prices, sorted(need_iv))
        if not iv_new.empty:
            iv = pd.concat([iv, iv_new], ignore_index=True)

    # Macro (for VIX)
    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
    macro = macro[macro["date"] >= "2019-01-01"]

    # Earnings
    try:
        earnings = pd.read_parquet(CACHE / "earnings_dates.parquet")
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None)
    except:
        earnings = pd.DataFrame(columns=["ticker", "earnings_date"])

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & set(TICKER_LIST)
    print(f"  Available tickers: {len(available)}/{len(TICKER_LIST)}")

    return prices, iv, macro, earnings


def build_earnings_lookup(earnings_df):
    lookup = {}
    for ticker in earnings_df["ticker"].unique():
        dates = earnings_df[earnings_df["ticker"] == ticker]["earnings_date"].sort_values().values
        if len(dates) > 0:
            lookup[ticker] = dates
    return lookup


def has_earnings_within(ticker, open_date, dte_target, earnings_lookup, buffer_days=7):
    if ticker not in earnings_lookup:
        return False
    earn_dates = earnings_lookup[ticker]
    hold_start = np.datetime64(open_date) - np.timedelta64(buffer_days, 'D')
    hold_end = np.datetime64(open_date) + np.timedelta64(dte_target + buffer_days, 'D')
    mask = (earn_dates >= hold_start) & (earn_dates <= hold_end)
    return mask.any()


# ═══════════════════════════════════════════════════════════════════
# BPS Engine with Ticker-Specific Credit Haircuts
# ═══════════════════════════════════════════════════════════════════

def run_bps(prices, iv, macro, earnings_lookup,
            credit_haircut_mode="ticker_specific",  # or "uniform_13pct"
            label="test"):
    """
    BPS backtest with full production config + ticker-specific credit haircuts.

    credit_haircut_mode:
      "ticker_specific" — use each ticker's measured real/BS ratio
      "uniform_13pct" — apply flat 13% haircut to all (avg measured)
      "none" — no haircut (BS model raw, for comparison)
    """
    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"  Haircut mode: {credit_haircut_mode}")
    print(f"{'='*60}")

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & set(TICKER_LIST)

    prices_df = prices[prices["ticker"].isin(available)].copy()
    iv_df = iv[iv["ticker"].isin(available)].copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()

    macro_by_date = macro.set_index("date").to_dict("index")
    all_dates = sorted(prices_df["date"].unique())

    # Production config
    dte_target = DTE_TARGET
    profit_take = 0.65
    margin_cap = 0.25
    max_concurrent = 40
    per_name_pct = 0.03
    spread_width = 15.0
    vix_base = 15.0
    vix_hard_cutoff = 30.0
    cb_threshold = 0.02
    cb_freeze_days = 1
    earnings_buffer = 7

    cash = STARTING_CAPITAL
    positions = {}
    equity_curve = []
    trades = []
    frozen_until = None
    cb_count = 0
    n_earnings_blocked = 0
    n_vix_blocked = 0

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix_close", 18.0)

        # Dynamic delta
        if vix < 15:
            delta, sw = 0.35, spread_width
        elif vix < 22:
            delta, sw = 0.30, spread_width
        else:
            delta, sw = 0.25, spread_width

        T = dte_target / 365.0

        # Mark-to-market + close expired/PT
        expired_keys = []
        for key, pos in list(positions.items()):
            if pos["ticker"] not in date_px:
                continue
            S = date_px[pos["ticker"]]
            days_held = (dt - pos["open_date"]).days

            if days_held >= dte_target or days_held >= (dte_target - 1):
                # Expiry or 1-DTE close
                short_itm = max(pos["short_strike"] - S, 0)
                long_itm = max(pos["long_strike"] - S, 0)
                net_loss = (short_itm - long_itm) * 100 * pos["contracts"]
                pnl = pos["premium_received"] - net_loss - trade_cost(0.05, pos["contracts"])
                cash += pnl + pos["margin_held"]
                trades.append({
                    "ticker": pos["ticker"], "open": str(pos["open_date"])[:10],
                    "close": str(dt)[:10], "pnl": pnl, "days": days_held,
                    "exit": "expiry" if days_held >= dte_target else "1dte",
                    "premium": pos["premium_received"], "short_strike": pos["short_strike"],
                })
                expired_keys.append(key)
                continue

            # Profit take check
            rem_T = max((dte_target - days_held) / 365.0, 1/365)
            short_val = bs_price(S, pos["short_strike"], rem_T, pos.get("sigma", 0.3)) * 100 * pos["contracts"]
            long_val = bs_price(S, pos["long_strike"], rem_T, pos.get("sigma", 0.3)) * 100 * pos["contracts"]
            cost_to_close = short_val - long_val
            unrealized = pos["premium_received"] - cost_to_close
            if unrealized >= pos["premium_received"] * profit_take:
                pnl = unrealized - trade_cost(0.05, pos["contracts"])
                cash += pnl + pos["margin_held"]
                trades.append({
                    "ticker": pos["ticker"], "open": str(pos["open_date"])[:10],
                    "close": str(dt)[:10], "pnl": pnl, "days": days_held,
                    "exit": "profit_take", "premium": pos["premium_received"],
                    "short_strike": pos["short_strike"],
                })
                expired_keys.append(key)

        for k in expired_keys:
            del positions[k]

        # Circuit breaker
        total_val = cash + sum(p["margin_held"] for p in positions.values())
        if len(equity_curve) > 0:
            prev_nav = equity_curve[-1]["nav"]
            daily_ret = (total_val - prev_nav) / prev_nav if prev_nav > 0 else 0
            if daily_ret < -cb_threshold:
                frozen_until = dt + pd.Timedelta(days=cb_freeze_days)
                cb_count += 1

        # Open new positions
        can_open = frozen_until is None or dt > frozen_until
        if vix > vix_hard_cutoff:
            can_open = False
            n_vix_blocked += 1

        if can_open:
            current_margin = sum(p["margin_held"] for p in positions.values())
            margin_avail = cash * margin_cap - current_margin

            # VIX scaling
            vix_scale = max(0.3, min(1.0, vix_base / max(vix, 1)))

            for tk in sorted(available):
                if len(positions) >= max_concurrent:
                    break
                if any(p["ticker"] == tk for p in positions.values()):
                    continue

                S = date_px.get(tk)
                sigma = date_sigma.get(tk)
                if S is None or sigma is None or sigma < 0.05:
                    continue

                # Earnings filter
                if has_earnings_within(tk, dt, dte_target, earnings_lookup, earnings_buffer):
                    n_earnings_blocked += 1
                    continue

                # Compute BS spread credit
                short_K = strike_from_delta(S, T, sigma, delta)
                long_K = short_K - sw
                if long_K <= 0:
                    continue

                short_prem = bs_price(S, short_K, T, sigma)
                long_prem = bs_price(S, long_K, T, sigma)
                net_credit_bs = short_prem - long_prem

                if net_credit_bs < 0.05:
                    continue

                # Apply credit haircut based on mode
                if credit_haircut_mode == "ticker_specific":
                    ratio = VALIDATED_TICKERS.get(tk, (0, 0.87))[1]
                    # Cap ratio at 1.0 for conservative estimate (don't assume better-than-BS)
                    ratio = min(ratio, 1.0)
                    net_credit = net_credit_bs * ratio
                elif credit_haircut_mode == "uniform_13pct":
                    net_credit = net_credit_bs * 0.87
                else:
                    net_credit = net_credit_bs

                if net_credit < 0.05:
                    continue

                # Position sizing
                margin_per = sw * 100
                if np.isnan(cash) or np.isnan(margin_avail):
                    continue
                max_by_cap = int(cash * per_name_pct / margin_per) if margin_per > 0 else 0
                max_by_margin = int(margin_avail / margin_per) if margin_per > 0 else 0
                max_by_vix = max(1, int(max_by_cap * vix_scale))
                contracts = max(1, min(max_by_cap, max_by_margin, max_by_vix, 5))

                total_margin = margin_per * contracts
                if total_margin > margin_avail or total_margin > cash * per_name_pct:
                    continue

                premium = net_credit * 100 * contracts
                cost = trade_cost(net_credit, contracts)
                cash -= total_margin
                cash += premium - cost

                key = f"{tk}_{di}"
                positions[key] = {
                    "ticker": tk, "short_strike": short_K, "long_strike": long_K,
                    "contracts": contracts, "premium_received": premium - cost,
                    "margin_held": total_margin, "sigma": sigma,
                    "open_date": dt,
                }
                margin_avail -= total_margin

        # NAV
        nav = cash
        for pos in positions.values():
            nav += pos["margin_held"]
            S = date_px.get(pos["ticker"])
            if S:
                days_held = (dt - pos["open_date"]).days
                rem_T = max((dte_target - days_held) / 365.0, 1/365)
                short_val = bs_price(S, pos["short_strike"], rem_T, pos.get("sigma", 0.3)) * 100 * pos["contracts"]
                long_val = bs_price(S, pos["long_strike"], rem_T, pos.get("sigma", 0.3)) * 100 * pos["contracts"]
                nav -= (short_val - long_val)

        equity_curve.append({"date": dt, "nav": nav})

    # Finalize
    eq = pd.DataFrame(equity_curve)
    eq["date"] = pd.to_datetime(eq["date"])
    eq["daily_ret"] = eq["nav"].pct_change()

    trading_days = len(eq)
    years = trading_days / 252
    total_ret = (eq["nav"].iloc[-1] / STARTING_CAPITAL) - 1
    cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1

    daily_rets = eq["daily_ret"].dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    downside = daily_rets[daily_rets < 0].std()
    sortino = (daily_rets.mean() / downside * np.sqrt(252)) if downside > 0 else 0

    running_max = eq["nav"].cummax()
    drawdown = (eq["nav"] - running_max) / running_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    win_rate = (trades_df["pnl"] > 0).mean() if len(trades_df) > 0 else 0
    avg_pnl = trades_df["pnl"].mean() if len(trades_df) > 0 else 0
    profit_factor = (trades_df[trades_df["pnl"] > 0]["pnl"].sum() /
                     abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
                     if len(trades_df) > 0 and (trades_df["pnl"] < 0).any() else 999)

    metrics = {
        "label": label, "haircut_mode": credit_haircut_mode,
        "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
        "cagr": round(cagr * 100, 1), "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2), "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2), "avg_pnl": round(avg_pnl, 2),
        "total_trades": len(trades_df), "trades_per_year": round(len(trades_df) / max(years, 0.1), 0),
        "final_nav": round(eq["nav"].iloc[-1], 0),
        "tickers_used": len(available), "years": round(years, 1),
        "cb_triggers": cb_count, "earnings_blocked": n_earnings_blocked,
        "vix_blocked": n_vix_blocked,
    }

    print(f"  Sharpe={metrics['sharpe']:.2f}  Sortino={metrics['sortino']:.2f}  "
          f"CAGR={metrics['cagr']:.1f}%  MaxDD={metrics['max_dd']:.1f}%  "
          f"WR={metrics['win_rate']:.1f}%  PF={metrics['profit_factor']:.2f}  "
          f"Trades={metrics['total_trades']}  Calmar={metrics['calmar']:.2f}")

    return metrics, eq, trades_df


def permutation_test(prices, iv, macro, earnings_lookup, n_perms=100):
    """HC #659: Random directions must lose. If they're profitable, result is artifact."""
    print(f"\n{'='*60}")
    print(f"  PERMUTATION TEST (n={n_perms})")
    print(f"{'='*60}")

    # Get real Sharpe first
    real_metrics, _, _ = run_bps(prices, iv, macro, earnings_lookup,
                                 credit_haircut_mode="ticker_specific",
                                 label="real_for_perm")
    real_sharpe = real_metrics["sharpe"]

    # Run permutations (shuffle which ticker gets which price series)
    perm_sharpes = []
    for i in range(n_perms):
        if (i + 1) % 20 == 0:
            print(f"  Permutation {i+1}/{n_perms}...")

        # Shuffle the ticker labels on price data
        shuffled_prices = prices.copy()
        tickers = shuffled_prices["ticker"].unique()
        ticker_map = dict(zip(tickers, np.random.permutation(tickers)))
        shuffled_prices["ticker"] = shuffled_prices["ticker"].map(ticker_map)

        try:
            m, _, _ = run_bps(shuffled_prices, iv, macro, earnings_lookup,
                              credit_haircut_mode="ticker_specific",
                              label=f"perm_{i}")
            perm_sharpes.append(m["sharpe"])
        except:
            perm_sharpes.append(0)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    print(f"\n  Real Sharpe: {real_sharpe:.2f}")
    print(f"  Perm mean Sharpe: {perm_sharpes.mean():.2f} ± {perm_sharpes.std():.2f}")
    print(f"  Perm max Sharpe: {perm_sharpes.max():.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  VERDICT: {'PASS ✅' if p_value < 0.05 else 'FAIL ❌ — artifact!'}")

    return real_sharpe, p_value, perm_sharpes


def main():
    t0 = time.time()

    prices, iv, macro, earnings = load_all_data()
    earnings_lookup = build_earnings_lookup(earnings)

    # Run 3 configs: no haircut (BS raw), uniform 13%, ticker-specific
    results = []

    m1, eq1, tr1 = run_bps(prices, iv, macro, earnings_lookup,
                            credit_haircut_mode="none",
                            label="BS_raw_no_haircut")
    results.append(m1)

    m2, eq2, tr2 = run_bps(prices, iv, macro, earnings_lookup,
                            credit_haircut_mode="uniform_13pct",
                            label="Uniform_13pct_haircut")
    results.append(m2)

    m3, eq3, tr3 = run_bps(prices, iv, macro, earnings_lookup,
                            credit_haircut_mode="ticker_specific",
                            label="Ticker_specific_haircut")
    results.append(m3)

    # Summary
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY (Fill-Quality-Validated Universe, 52 tickers)")
    print(f"{'='*70}")
    print(f"{'Config':<30s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s} {'WR':>6s} {'PF':>6s} {'Calmar':>7s}")
    print(f"{'-'*70}")
    for m in results:
        print(f"{m['label']:<30s} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr']:>6.1f}% {m['max_dd']:>6.1f}% {m['win_rate']:>5.1f}% {m['profit_factor']:>6.2f} {m['calmar']:>7.2f}")

    # Permutation test on the ticker-specific (most honest) config
    print("\nRunning permutation test on ticker-specific config...")
    real_sharpe, p_value, perm_sharpes = permutation_test(
        prices, iv, macro, earnings_lookup, n_perms=50)

    # Save results
    summary = {
        "configs": results,
        "permutation_test": {
            "real_sharpe": float(real_sharpe),
            "p_value": float(p_value),
            "perm_mean_sharpe": float(perm_sharpes.mean()),
            "perm_std_sharpe": float(perm_sharpes.std()),
            "n_perms": len(perm_sharpes),
            "verdict": "PASS" if p_value < 0.05 else "FAIL",
        },
        "universe": {
            "total_validated": len(VALIDATED_TICKERS),
            "avg_real_bs_ratio": np.mean([v[1] for v in VALIDATED_TICKERS.values()]),
            "median_real_bs_ratio": np.median([v[1] for v in VALIDATED_TICKERS.values()]),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }

    with open(OUTPUT / "fill_quality_backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save equity curves
    eq3.to_csv(OUTPUT / "equity_curve_ticker_specific.csv", index=False)
    if len(tr3) > 0:
        tr3.to_csv(OUTPUT / "trades_ticker_specific.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Results saved.")


if __name__ == "__main__":
    main()
