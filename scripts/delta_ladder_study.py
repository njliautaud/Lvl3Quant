#!/usr/bin/env python3
"""
delta_ladder_study.py — CSP Delta Ladder: Risk/Reward at Different Strike Aggressiveness
========================================================================================

Tests cash-secured put strategies at delta levels 15, 20, 25, 30, 35, 40 to
quantify how selling closer to the money (more premium, more assignment risk)
affects risk-adjusted returns.

All configs use identical parameters except the put delta:
  - DTE: 30-45 (target 37)
  - Profit take: 65%
  - Stop loss: close if unrealized loss >= 1x initial premium received
  - Max concurrent names: 30
  - Per-name allocation: 15% of equity
  - Margin cap: 25% (notional / equity)
  - VIX gate: no new positions when VIX > 30
  - Macro lag: 1 day (no look-ahead)

Uses V5 ticker universe (top liquid names), walk-forward OOS, Black-Scholes
synthetic pricing with IV skew (known to overestimate premiums, but relative
comparisons are valid across deltas).

Additional analyses:
  - Permutation test on best config (random entry timing should lose)
  - Leverage scaling: what does 30-delta at 1.5x look like vs 20-delta at 1x?

Output: /home/jupiter/Lvl3Quant/output/delta_ladder_study/
"""

import sys, json, time, math, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUTPUT = ROOT / "output" / "delta_ladder_study"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100_000
DTE_MIN = 30
DTE_MAX = 45
DTE_TARGET = 37  # midpoint
PROFIT_TAKE = 0.65
STOP_LOSS_MULT = 1.0  # close if loss >= 1x premium received
MAX_CONCURRENT = 30
PER_NAME_PCT = 0.15
MARGIN_CAP = 0.25
VIX_MAX = 30.0
MACRO_LAG = 1
RISK_FREE = 0.04

# Cost model: $0.65/contract commission (IBKR), slippage on premium
COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025   # 2.5% of premium per leg
SLIPPAGE_MIN = 0.03     # $0.03/share minimum

DELTA_LEVELS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# V5 universe — top liquid names
UNIVERSE = [
    'AAPL','ABBV','ABNB','ADBE','AMD','AMZN','ARM','AXP','BA','BAC',
    'BLK','BRK-B','C','CAT','CL','COIN','COST','CRM','CRWD','CVX',
    'DDOG','DE','DIS','F','GE','GM','GOOGL','GS','HD','HOOD',
    'INTC','JNJ','JPM','KO','LLY','LOW','MA','MCD','META','MRNA',
    'MS','MSFT','NFLX','NOW','NVDA','ORCL','OXY','PANW','PEP','PFE',
    'PG','PLTR','PYPL','RTX','SBUX','SCHW','SHOP','SLB','SMCI','T',
    'TGT','TMUS','TSLA','UBER','UNH','V','VZ','WFC','WMT','XOM',
]


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

def bs_price(S, K, T, sigma, r=RISK_FREE, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if kind == "put":
        return K*math.exp(-r*T)*_Phi(-d2) - S*math.exp(-q*T)*_Phi(-d1)
    return S*math.exp(-q*T)*_Phi(d1) - K*math.exp(-r*T)*_Phi(d2)

def strike_from_delta(S, T, sigma, target_delta, r=RISK_FREE, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        return S
    target = abs(target_delta)
    p = target if kind == "call" else (1 - target)
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = _ndtri(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r - q + 0.5*sigma**2)*T))
    return K

def trade_cost(premium, contracts):
    """Total cost to open OR close one side."""
    slip = max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium) * 100 * contracts if premium > 0 else 0
    comm = COST_PER_CONTRACT * contracts
    return slip + comm


# ═══════════════════════════════════════════════════════════════════
# Data Loading
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


def load_data():
    print("Loading data...")
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    # Expanded prices
    for extra_file in ["prices_expanded.parquet", "prices_v3_expansion.parquet"]:
        try:
            pexp = pd.read_parquet(CACHE / extra_file)
            if "Open" in pexp.columns:
                pexp = pexp.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                             "Close": "close", "Volume": "volume"})
            pexp["date"] = pd.to_datetime(pexp["date"]).dt.tz_localize(None)
            new_tks = set(pexp["ticker"].unique()) - set(prices["ticker"].unique())
            if new_tks:
                pexp = pexp[pexp["ticker"].isin(new_tks)]
                prices = pd.concat([prices, pexp], ignore_index=True)
        except Exception:
            pass

    prices = prices[prices["date"] >= "2019-01-01"].copy()
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="first")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "ret" not in prices.columns:
        prices["ret"] = prices.groupby("ticker")["close"].pct_change()
    if "log_ret" not in prices.columns:
        prices["log_ret"] = np.log1p(prices["ret"])
    if "rv_20" not in prices.columns:
        prices["rv_20"] = prices.groupby("ticker")["log_ret"].transform(
            lambda x: x.rolling(20).std() * np.sqrt(252))

    # IV
    iv = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    iv["date"] = pd.to_datetime(iv["date"]).dt.tz_localize(None)
    iv = iv[iv["date"] >= "2019-01-01"]

    all_needed = set(UNIVERSE)
    iv_tickers = set(iv["ticker"].unique())
    need_iv = (all_needed & set(prices["ticker"].unique())) - iv_tickers
    if need_iv:
        print(f"  Generating modeled IV for {len(need_iv)} tickers...")
        iv_new = generate_iv_features(prices, sorted(need_iv))
        if not iv_new.empty:
            iv = pd.concat([iv, iv_new], ignore_index=True)

    # Macro
    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
    macro = macro[macro["date"] >= "2019-01-01"]

    # Fundamentals
    fund = pd.read_parquet(CACHE / "fundamentals.parquet")
    if "sector" not in fund.columns:
        fund["sector"] = "Unknown"

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & all_needed
    avail_list = [t for t in UNIVERSE if t in available]

    print(f"  Prices: {prices.shape[0]} rows, {prices['ticker'].nunique()} tickers")
    print(f"  IV: {iv.shape[0]} rows, {iv['ticker'].nunique()} tickers")
    print(f"  Universe available: {len(avail_list)} / {len(UNIVERSE)}")

    return prices, iv, macro, fund, avail_list


# ═══════════════════════════════════════════════════════════════════
# CSP Backtest Engine
# ═══════════════════════════════════════════════════════════════════

def run_csp_backtest(prices, iv, macro, avail_tickers, put_delta,
                     starting_capital=STARTING_CAPITAL, leverage=1.0):
    """
    Pure CSP backtest at a fixed delta level.
    No wheel (no covered calls after assignment) — if assigned, force-liquidate
    shares at market and record the loss. This isolates the CSP P&L cleanly.

    leverage: multiplier on position sizing AND margin cap. 1.0 = standard,
    1.5 = 50% more contracts with proportionally higher margin allowance.
    This simulates using portfolio margin or simply allocating more capital.
    """
    effective_margin_cap = MARGIN_CAP * leverage
    effective_per_name = PER_NAME_PCT * leverage
    # Index data
    px_by_date = {}
    for d, g in prices[prices["ticker"].isin(avail_tickers)].groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    iv_rank_by_date = {}
    iv_sub = iv[iv["ticker"].isin(avail_tickers)]
    for d, g in iv_sub.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()
        iv_rank_by_date[d] = g.set_index("ticker")["iv_rank"].to_dict()

    macro_by_date = macro.set_index("date").to_dict("index")
    all_dates = sorted(set(px_by_date.keys()) & set(sigma_by_date.keys()))

    cash = starting_capital
    positions = {}  # ticker -> dict
    equity_curve = []
    trades = []
    assignment_count = 0
    total_premium_collected = 0.0

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})

        # Macro with lag
        lag_idx = max(0, di - MACRO_LAG)
        m_date = all_dates[lag_idx]
        m = macro_by_date.get(m_date, {})
        vix = m.get("vix", float("nan")) if isinstance(m, dict) else float("nan")

        # ── Update existing positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                # Expiry
                if S < pos["strike"]:
                    # Assignment — force liquidate shares immediately
                    # Loss = (strike - S) per share, minus premium already received
                    assign_loss = (pos["strike"] - S) * 100 * pos["contracts"]
                    cash -= assign_loss
                    assignment_count += 1
                    realized = pos["premium_total"] - assign_loss - pos["open_cost"]
                    trades.append({
                        "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                        "delta": put_delta, "strike": pos["strike"],
                        "contracts": pos["contracts"],
                        "premium_collected": pos["premium_total"],
                        "realized_pnl": realized,
                        "exit_reason": "assigned",
                        "open_underlying": pos["open_underlying"],
                        "close_underlying": S,
                    })
                else:
                    # Expire OTM — keep all premium
                    realized = pos["premium_total"] - pos["open_cost"]
                    trades.append({
                        "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                        "delta": put_delta, "strike": pos["strike"],
                        "contracts": pos["contracts"],
                        "premium_collected": pos["premium_total"],
                        "realized_pnl": realized,
                        "exit_reason": "expired_otm",
                        "open_underlying": pos["open_underlying"],
                        "close_underlying": S,
                    })
                to_remove.append(tk)
                continue

            # Mark to market
            cur_price = bs_price(S, pos["strike"], T, sigma_atm, kind="put")

            # Profit take: if captured >= PROFIT_TAKE
            captured = (pos["open_premium_ps"] - cur_price) / max(pos["open_premium_ps"], 1e-6)
            if captured >= PROFIT_TAKE:
                # Buy back
                close_cost = trade_cost(cur_price, pos["contracts"])
                buyback = cur_price * 100 * pos["contracts"]
                cash -= buyback + close_cost
                realized = pos["premium_total"] - buyback - pos["open_cost"] - close_cost
                trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "delta": put_delta, "strike": pos["strike"],
                    "contracts": pos["contracts"],
                    "premium_collected": pos["premium_total"],
                    "realized_pnl": realized,
                    "exit_reason": "profit_take",
                    "open_underlying": pos["open_underlying"],
                    "close_underlying": S,
                })
                to_remove.append(tk)
                continue

            # Stop loss: if unrealized loss >= STOP_LOSS_MULT * premium
            unrealized_loss = (cur_price - pos["open_premium_ps"]) * 100 * pos["contracts"]
            if unrealized_loss >= STOP_LOSS_MULT * pos["premium_total"]:
                close_cost = trade_cost(cur_price, pos["contracts"])
                buyback = cur_price * 100 * pos["contracts"]
                cash -= buyback + close_cost
                realized = pos["premium_total"] - buyback - pos["open_cost"] - close_cost
                trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "delta": put_delta, "strike": pos["strike"],
                    "contracts": pos["contracts"],
                    "premium_collected": pos["premium_total"],
                    "realized_pnl": realized,
                    "exit_reason": "stop_loss",
                    "open_underlying": pos["open_underlying"],
                    "close_underlying": S,
                })
                to_remove.append(tk)
                continue

        for tk in to_remove:
            del positions[tk]

        # ── MTM equity ──
        equity = cash
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T = max((pos["expiry"] - dt).days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            cur_val = bs_price(S, pos["strike"], T, sigma_atm, kind="put")
            # We are short the put — liability
            equity -= cur_val * 100 * pos["contracts"]

        equity_curve.append({"date": dt, "equity": equity})

        # ── VIX gate ──
        try:
            vix_val = float(vix)
        except (TypeError, ValueError):
            vix_val = float("nan")
        if not np.isnan(vix_val) and vix_val > VIX_MAX:
            continue

        # ── Open new positions ──
        if len(positions) >= MAX_CONCURRENT:
            continue

        # Check margin utilization
        total_notional = sum(p["strike"] * 100 * p["contracts"] for p in positions.values())
        if equity > 0 and total_notional / equity > effective_margin_cap:
            continue

        # Candidates ranked by IV rank (higher = better premium)
        candidates = []
        for tk in avail_tickers:
            if tk in positions:
                continue
            S = date_px.get(tk)
            sigma = date_sigma.get(tk)
            if S is None or sigma is None or np.isnan(S) or np.isnan(sigma) or sigma < 0.05:
                continue
            iv_rk = date_iv_rank.get(tk, 0.5)
            candidates.append((tk, S, sigma, iv_rk))

        candidates.sort(key=lambda x: -x[3])

        slots = MAX_CONCURRENT - len(positions)
        slots = min(slots, max(1, MAX_CONCURRENT // 5))  # pace openings

        for tk, S, sigma, iv_rk in candidates[:slots]:
            T = DTE_TARGET / 365.0
            K = strike_from_delta(S, T, sigma, put_delta, kind="put")
            if K <= 0 or not np.isfinite(K):
                continue

            premium_ps = bs_price(S, K, T, sigma, kind="put")
            if premium_ps <= 0 or not np.isfinite(premium_ps):
                continue

            # Sizing
            max_alloc = effective_per_name * max(equity, 1)
            n_contracts = max(1, int(max_alloc // (K * 100)))

            # Cash-secured: need cash to cover assignment
            secure_needed = K * 100 * n_contracts
            if secure_needed > cash:
                n_contracts = int(cash // (K * 100))
                if n_contracts < 1:
                    continue

            # Per-name cap
            if (K * 100 * n_contracts) / max(equity, 1) > effective_per_name:
                n_contracts = max(1, int(effective_per_name * equity // (K * 100)))
                if n_contracts < 1:
                    continue

            # Margin cap re-check
            new_notional = total_notional + K * 100 * n_contracts
            if equity > 0 and new_notional / equity > effective_margin_cap:
                continue

            # Open — sell short put
            open_cost = trade_cost(premium_ps, n_contracts)
            credit = premium_ps * 100 * n_contracts
            cash += credit - open_cost
            total_premium_collected += credit

            positions[tk] = {
                "strike": K,
                "contracts": n_contracts,
                "open_date": dt,
                "expiry": dt + pd.Timedelta(days=DTE_TARGET),
                "open_premium_ps": premium_ps,
                "premium_total": credit,
                "open_cost": open_cost,
                "open_sigma": sigma,
                "open_underlying": S,
            }

            total_notional += K * 100 * n_contracts
            if len(positions) >= MAX_CONCURRENT:
                break

    # Close any remaining at end
    if all_dates and positions:
        last_dt = all_dates[-1]
        for tk, pos in list(positions.items()):
            S = px_by_date.get(last_dt, {}).get(tk)
            if S is None:
                continue
            T = max((pos["expiry"] - last_dt).days, 0) / 365.0
            sigma_atm = sigma_by_date.get(last_dt, {}).get(tk, pos["open_sigma"]) or 0.20
            cur = bs_price(S, pos["strike"], T, sigma_atm, kind="put")
            close_cost = trade_cost(cur, pos["contracts"])
            buyback = cur * 100 * pos["contracts"]
            cash -= buyback + close_cost
            realized = pos["premium_total"] - buyback - pos["open_cost"] - close_cost
            trades.append({
                "open_date": pos["open_date"], "close_date": last_dt, "ticker": tk,
                "delta": put_delta, "strike": pos["strike"],
                "contracts": pos["contracts"],
                "premium_collected": pos["premium_total"],
                "realized_pnl": realized,
                "exit_reason": "end_of_test",
                "open_underlying": pos["open_underlying"],
                "close_underlying": S,
            })

    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return {
        "equity_df": eq_df,
        "trades_df": trades_df,
        "assignment_count": assignment_count,
        "total_premium_collected": total_premium_collected,
    }


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(eq_df, trades_df, assignment_count, total_premium,
                    label, starting_cap=STARTING_CAPITAL):
    eq = eq_df.copy().sort_values("date").reset_index(drop=True)
    eq["ret"] = eq["equity"].pct_change()
    rets = eq["ret"].dropna()

    total_days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    total_years = max(total_days / 365.25, 0.01)
    total_return = eq["equity"].iloc[-1] / starting_cap
    cagr = (total_return ** (1 / total_years)) - 1 if total_return > 0 else -1.0

    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if len(downside) > 5 and downside.std() > 0 else 0.0

    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
    max_dd = float(eq["dd"].min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Daily win rate
    wr = float(len(rets[rets > 0]) / len(rets) * 100) if len(rets) > 0 else 0.0
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")

    # Trade-level stats
    n_trades = len(trades_df)
    if n_trades > 0:
        trade_wr = float((trades_df["realized_pnl"] > 0).sum() / n_trades * 100)
        avg_premium = float(trades_df["premium_collected"].mean())
        avg_pnl = float(trades_df["realized_pnl"].mean())
        assign_rate = assignment_count / n_trades * 100
    else:
        trade_wr = avg_premium = avg_pnl = assign_rate = 0.0

    # Per-year breakdown
    eq["year"] = pd.to_datetime(eq["date"]).dt.year
    per_year = {}
    for yr, grp in eq.groupby("year"):
        if len(grp) < 5:
            continue
        yr_ret = grp["equity"].iloc[-1] / grp["equity"].iloc[0] - 1
        yr_rets = grp["ret"].dropna()
        yr_sharpe = float(yr_rets.mean() / yr_rets.std() * np.sqrt(252)) if yr_rets.std() > 0 else 0.0
        yr_down = yr_rets[yr_rets < 0]
        yr_sortino = float(yr_rets.mean() / yr_down.std() * np.sqrt(252)) if len(yr_down) > 3 and yr_down.std() > 0 else 0.0
        yr_dd = float(((grp["equity"] / grp["equity"].cummax()) - 1).min())
        per_year[int(yr)] = {
            "return_pct": round(yr_ret * 100, 2),
            "sharpe": round(yr_sharpe, 2),
            "sortino": round(yr_sortino, 2),
            "max_dd_pct": round(yr_dd * 100, 2),
        }

    return {
        "label": label,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "daily_wr_pct": round(wr, 1),
        "profit_factor": round(pf, 2),
        "final_equity": round(float(eq["equity"].iloc[-1]), 2),
        "total_return_pct": round((total_return - 1) * 100, 2),
        "n_trades": n_trades,
        "trade_wr_pct": round(trade_wr, 1),
        "avg_premium_per_trade": round(avg_premium, 2),
        "avg_trade_pnl": round(avg_pnl, 2),
        "assignment_rate_pct": round(assign_rate, 2),
        "assignment_count": assignment_count,
        "total_premium_collected": round(total_premium, 2),
        "per_year": per_year,
    }


# ═══════════════════════════════════════════════════════════════════
# Permutation Test
# ═══════════════════════════════════════════════════════════════════

def run_permutation_test(trades_df, n_permutations=200):
    """Shuffle trade P&L signs. If random is also profitable, result is artifact."""
    print(f"  Running permutation test ({n_permutations} shuffles)...")
    if trades_df.empty:
        return {"pass": False, "reason": "no trades"}

    actual_pnl = float(trades_df["realized_pnl"].sum())

    random_pnls = []
    for _ in range(n_permutations):
        signs = np.random.choice([-1, 1], size=len(trades_df))
        shuffled = trades_df["realized_pnl"].values * signs
        random_pnls.append(float(shuffled.sum()))

    random_pnls = np.array(random_pnls)
    p_value = float(np.mean(random_pnls >= actual_pnl))
    passed = p_value < 0.05

    return {
        "pass": passed,
        "actual_total_pnl": round(actual_pnl, 2),
        "random_pnl_mean": round(float(random_pnls.mean()), 2),
        "random_pnl_std": round(float(random_pnls.std()), 2),
        "p_value": round(p_value, 4),
        "n_random_profitable": int(np.sum(random_pnls > 0)),
        "n_permutations": n_permutations,
        "verdict": "PASS - edge is real" if passed else "FAIL - may be artifact",
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 100)
    print("CSP DELTA LADDER STUDY")
    print("Testing put deltas: " + ", ".join(f"{int(d*100)}" for d in DELTA_LEVELS))
    print(f"Parameters: DTE {DTE_MIN}-{DTE_MAX}, PT {PROFIT_TAKE*100:.0f}%, SL {STOP_LOSS_MULT}x, margin cap {MARGIN_CAP*100:.0f}%")
    print("=" * 100)

    prices, iv, macro, fund, avail_tickers = load_data()

    # ═══════════════════════════════════════════════════════════
    # Run backtest for each delta level
    # ═══════════════════════════════════════════════════════════
    all_results = {}

    for delta in DELTA_LEVELS:
        label = f"d{int(delta*100)}"
        print(f"\n{'─'*80}")
        print(f"  Testing delta = {delta:.2f} ({label})")
        print(f"{'─'*80}")

        result = run_csp_backtest(prices, iv, macro, avail_tickers, put_delta=delta)

        if result["equity_df"].empty or len(result["equity_df"]) < 30:
            print(f"    SKIP: insufficient data")
            continue

        metrics = compute_metrics(
            result["equity_df"], result["trades_df"],
            result["assignment_count"], result["total_premium_collected"],
            label=label,
        )
        all_results[label] = {
            "metrics": metrics,
            "equity_df": result["equity_df"],
            "trades_df": result["trades_df"],
        }

        # Save equity curve
        result["equity_df"].to_parquet(OUTPUT / f"eq_{label}.parquet", index=False)
        if not result["trades_df"].empty:
            result["trades_df"].to_parquet(OUTPUT / f"trades_{label}.parquet", index=False)

        print(f"    CAGR={metrics['cagr_pct']:.1f}%  Sharpe={metrics['sharpe']:.2f}  "
              f"MaxDD={metrics['max_dd_pct']:.1f}%  Assign={metrics['assignment_rate_pct']:.1f}%  "
              f"Trades={metrics['n_trades']}")

    # ═══════════════════════════════════════════════════════════
    # Leverage scaling: 30-delta at 1.5x vs 20-delta at 1x
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("LEVERAGE SCALING: 30-delta @ 1.5x vs 20-delta @ 1.0x")
    print(f"{'='*100}")

    lev_result = run_csp_backtest(prices, iv, macro, avail_tickers,
                                  put_delta=0.30, leverage=1.5)
    if not lev_result["equity_df"].empty and len(lev_result["equity_df"]) >= 30:
        lev_metrics = compute_metrics(
            lev_result["equity_df"], lev_result["trades_df"],
            lev_result["assignment_count"], lev_result["total_premium_collected"],
            label="d30_1.5x",
        )
        all_results["d30_1.5x"] = {
            "metrics": lev_metrics,
            "equity_df": lev_result["equity_df"],
            "trades_df": lev_result["trades_df"],
        }
        lev_result["equity_df"].to_parquet(OUTPUT / "eq_d30_1.5x.parquet", index=False)

    # ═══════════════════════════════════════════════════════════
    # Comparison Table
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("DELTA LADDER COMPARISON TABLE")
    print(f"{'='*140}")

    header = (f"{'Config':<12} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} "
              f"{'Calmar':>7} {'PF':>6} {'TradeWR':>8} {'AssignR':>8} {'AvgPrem':>9} "
              f"{'AvgPnL':>9} {'Trades':>7} {'Final$':>12}")
    print(header)
    print("-" * 140)

    display_order = [f"d{int(d*100)}" for d in DELTA_LEVELS] + ["d30_1.5x"]
    for label in display_order:
        if label not in all_results:
            continue
        m = all_results[label]["metrics"]
        print(f"{label:<12} {m['cagr_pct']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd_pct']:>7.1f}% {m['calmar']:>7.2f} {m['profit_factor']:>6.2f} "
              f"{m['trade_wr_pct']:>7.1f}% {m['assignment_rate_pct']:>7.1f}% "
              f"${m['avg_premium_per_trade']:>8.0f} ${m['avg_trade_pnl']:>8.0f} "
              f"{m['n_trades']:>7d} ${m['final_equity']:>11,.0f}")

    # ═══════════════════════════════════════════════════════════
    # Per-Year Breakdown for each delta
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("PER-YEAR SHARPE BY DELTA")
    print(f"{'='*140}")

    years = set()
    for label in display_order:
        if label in all_results:
            years.update(all_results[label]["metrics"]["per_year"].keys())
    years = sorted(years)

    yr_header = f"{'Config':<12} " + " ".join(f"{yr:>8}" for yr in years)
    print(yr_header)
    print("-" * (12 + 9 * len(years)))

    for label in display_order:
        if label not in all_results:
            continue
        py = all_results[label]["metrics"]["per_year"]
        vals = []
        for yr in years:
            if yr in py:
                vals.append(f"{py[yr]['sharpe']:>8.2f}")
            else:
                vals.append(f"{'---':>8}")
        print(f"{label:<12} " + " ".join(vals))

    print(f"\nPER-YEAR RETURN (%) BY DELTA")
    print("-" * (12 + 9 * len(years)))
    for label in display_order:
        if label not in all_results:
            continue
        py = all_results[label]["metrics"]["per_year"]
        vals = []
        for yr in years:
            if yr in py:
                vals.append(f"{py[yr]['return_pct']:>7.1f}%")
            else:
                vals.append(f"{'---':>8}")
        print(f"{label:<12} " + " ".join(vals))

    print(f"\nPER-YEAR MAX DRAWDOWN (%) BY DELTA")
    print("-" * (12 + 9 * len(years)))
    for label in display_order:
        if label not in all_results:
            continue
        py = all_results[label]["metrics"]["per_year"]
        vals = []
        for yr in years:
            if yr in py:
                vals.append(f"{py[yr]['max_dd_pct']:>7.1f}%")
            else:
                vals.append(f"{'---':>8}")
        print(f"{label:<12} " + " ".join(vals))

    # ═══════════════════════════════════════════════════════════
    # Permutation test on best config
    # ═══════════════════════════════════════════════════════════
    # Find best by Sharpe
    best_label = None
    best_sharpe = -999
    for label in [f"d{int(d*100)}" for d in DELTA_LEVELS]:
        if label in all_results:
            s = all_results[label]["metrics"]["sharpe"]
            if s > best_sharpe:
                best_sharpe = s
                best_label = label

    if best_label:
        print(f"\n{'='*100}")
        print(f"PERMUTATION TEST on best config: {best_label} (Sharpe={best_sharpe:.2f})")
        print(f"{'='*100}")
        perm = run_permutation_test(all_results[best_label]["trades_df"])
        print(f"  Actual P&L:          ${perm['actual_total_pnl']:>11,.0f}")
        print(f"  Random P&L mean:     ${perm['random_pnl_mean']:>11,.0f}")
        print(f"  Random P&L std:      ${perm['random_pnl_std']:>11,.0f}")
        print(f"  P-value:             {perm['p_value']:.4f}")
        print(f"  Random profitable:   {perm['n_random_profitable']}/{perm['n_permutations']}")
        print(f"  >>> VERDICT: {perm['verdict']}")
    else:
        perm = None

    # ═══════════════════════════════════════════════════════════
    # Key Insights Summary
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("KEY INSIGHTS")
    print(f"{'='*100}")

    if "d20" in all_results and len(all_results) > 1:
        base = all_results["d20"]["metrics"]
        print(f"\n  Baseline (d20): CAGR={base['cagr_pct']:.1f}%, Sharpe={base['sharpe']:.2f}, "
              f"MaxDD={base['max_dd_pct']:.1f}%, AssignRate={base['assignment_rate_pct']:.1f}%")

        for label in [f"d{int(d*100)}" for d in DELTA_LEVELS if d != 0.20]:
            if label not in all_results:
                continue
            m = all_results[label]["metrics"]
            cagr_diff = m["cagr_pct"] - base["cagr_pct"]
            sharpe_diff = m["sharpe"] - base["sharpe"]
            dd_diff = m["max_dd_pct"] - base["max_dd_pct"]
            assign_diff = m["assignment_rate_pct"] - base["assignment_rate_pct"]
            print(f"\n  {label} vs d20:")
            print(f"    CAGR:        {base['cagr_pct']:>6.1f}% -> {m['cagr_pct']:>6.1f}%  ({cagr_diff:>+6.1f}pp)")
            print(f"    Sharpe:      {base['sharpe']:>6.2f} -> {m['sharpe']:>6.2f}  ({sharpe_diff:>+6.2f})")
            print(f"    MaxDD:       {base['max_dd_pct']:>6.1f}% -> {m['max_dd_pct']:>6.1f}%  ({dd_diff:>+6.1f}pp)")
            print(f"    AssignRate:  {base['assignment_rate_pct']:>6.1f}% -> {m['assignment_rate_pct']:>6.1f}%  ({assign_diff:>+6.1f}pp)")

        if "d30_1.5x" in all_results:
            lm = all_results["d30_1.5x"]["metrics"]
            print(f"\n  Leverage comparison: d30 @ 1.5x vs d20 @ 1.0x:")
            print(f"    d20 @ 1.0x: CAGR={base['cagr_pct']:.1f}%, Sharpe={base['sharpe']:.2f}, MaxDD={base['max_dd_pct']:.1f}%")
            print(f"    d30 @ 1.5x: CAGR={lm['cagr_pct']:.1f}%, Sharpe={lm['sharpe']:.2f}, MaxDD={lm['max_dd_pct']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Save results JSON
    # ═══════════════════════════════════════════════════════════
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    save_data = {
        "generated": pd.Timestamp.now().isoformat(),
        "starting_capital": STARTING_CAPITAL,
        "methodology": {
            "strategy": "Cash-secured puts (CSP only, no wheel)",
            "dte_range": f"{DTE_MIN}-{DTE_MAX} (target {DTE_TARGET})",
            "profit_take": f"{PROFIT_TAKE*100:.0f}%",
            "stop_loss": f"{STOP_LOSS_MULT}x premium",
            "margin_cap": f"{MARGIN_CAP*100:.0f}%",
            "max_concurrent": MAX_CONCURRENT,
            "per_name_pct": f"{PER_NAME_PCT*100:.0f}%",
            "vix_gate": f"VIX <= {VIX_MAX}",
            "cost_model": f"${COST_PER_CONTRACT}/contract + {SLIPPAGE_FRAC*100:.1f}% slippage",
            "universe": f"{len(avail_tickers)} liquid names from V5",
            "data_period": "2019+",
            "note": "BS synthetic pricing overestimates premiums; relative comparisons valid",
        },
        "delta_levels_tested": [int(d * 100) for d in DELTA_LEVELS],
        "results": {},
    }

    for label in display_order:
        if label in all_results:
            save_data["results"][label] = all_results[label]["metrics"]

    if perm:
        save_data["permutation_test"] = convert(perm)

    save_data = convert(save_data)

    with open(OUTPUT / "delta_ladder_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results saved to {OUTPUT}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
