#!/usr/bin/env python3
"""
Earnings Week Planner v1 — Specific Trade Ideas for $645 Account
================================================================
Generates concrete iron condor trade ideas for upcoming earnings using:
- Historical earnings move analysis (avg move, implied vs realized)
- Account-appropriate sizing ($645 capital)
- Risk management rules (max 40% of capital per position)
- Expected value calculation with realistic commission costs

For week of 2026-07-28: PG (Mon 7/29) and AAPL (Wed 7/30)
"""
import json, os, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_PATH = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_PATH / "output" / "growth_research" / "earnings_week_planner_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAPITAL = 645.0
MAX_POSITION_PCT = 0.40  # max 40% of capital per position
COMMISSION_PER_LEG = 0.65  # $0.65 per option leg
IC_LEGS = 4  # iron condor has 4 legs
IC_COMMISSION = IC_LEGS * COMMISSION_PER_LEG  # $2.60 per IC

# Target earnings tickers for this week
TARGETS = {
    "PG": {"earnings_date": "2026-07-29", "name": "Procter & Gamble", "day": "Monday"},
    "AAPL": {"earnings_date": "2026-07-30", "name": "Apple", "day": "Wednesday"},
}

# Also analyze recent mega-cap earnings for validation
MEGA_CAPS = ["MSFT", "AMZN", "GOOGL", "META", "V", "JPM", "JNJ", "UNH", "PG", "AAPL"]

def fprint(*a, **kw): print(*a, **kw, flush=True)

def analyze_earnings_moves(ticker, n_quarters=20):
    """Analyze historical earnings day moves."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        if len(hist) < 252:
            return None

        close = hist['Close']
        # Find big gap days (likely earnings) — gaps > 1.5% either direction
        daily_ret = close.pct_change()
        gap_days = daily_ret[abs(daily_ret) > 0.015].tail(n_quarters)

        if len(gap_days) < 4:
            return None

        moves = gap_days.values * 100  # convert to percentage
        abs_moves = np.abs(moves)

        # Current price and volatility
        current_price = float(close.iloc[-1])
        vol_20d = float(daily_ret.tail(20).std() * np.sqrt(252) * 100)

        return {
            "ticker": ticker,
            "current_price": current_price,
            "n_earnings": len(moves),
            "avg_abs_move_pct": float(np.mean(abs_moves)),
            "median_abs_move_pct": float(np.median(abs_moves)),
            "max_move_pct": float(np.max(abs_moves)),
            "pct_within_2pct": float(np.mean(abs_moves < 2.0) * 100),
            "pct_within_3pct": float(np.mean(abs_moves < 3.0) * 100),
            "pct_within_5pct": float(np.mean(abs_moves < 5.0) * 100),
            "up_pct": float(np.mean(moves > 0) * 100),
            "vol_20d": vol_20d,
            "moves": moves.tolist(),
        }
    except Exception as e:
        fprint(f"  Error analyzing {ticker}: {e}")
        return None

def design_iron_condor(ticker_data, capital=CAPITAL):
    """Design an iron condor trade for the given ticker."""
    if not ticker_data:
        return None

    price = ticker_data["current_price"]
    avg_move = ticker_data["avg_abs_move_pct"]
    med_move = ticker_data["median_abs_move_pct"]

    # IC wing placement: outside expected move
    # Short strikes at 1.5x median move, long strikes 2-3% wider
    short_call_pct = med_move * 1.5
    short_put_pct = med_move * 1.5
    wing_width_pct = max(2.0, med_move * 0.5)  # width between short and long

    short_call = round(price * (1 + short_call_pct/100), 2)
    long_call = round(price * (1 + (short_call_pct + wing_width_pct)/100), 2)
    short_put = round(price * (1 - short_put_pct/100), 2)
    long_put = round(price * (1 - (short_put_pct + wing_width_pct)/100), 2)

    # Estimate credit received (rough: ~30% of wing width for ATM-adjacent IC)
    call_width = long_call - short_call
    put_width = short_put - long_put
    avg_width = (call_width + put_width) / 2

    # Credit estimate: higher for more volatile stocks, lower for wider wings
    vol_factor = min(1.5, ticker_data["vol_20d"] / 20)
    credit_per_contract = avg_width * 0.30 * vol_factor * 100  # in dollars per contract

    # Max loss per contract
    max_loss_per_contract = (avg_width * 100) - credit_per_contract + IC_COMMISSION

    # How many contracts can we afford?
    max_position = capital * MAX_POSITION_PCT
    n_contracts = max(1, int(max_position / max_loss_per_contract))

    # Expected value
    # P(profit) ≈ P(price stays within short strikes)
    p_profit = ticker_data["pct_within_3pct"] / 100  # rough approximation
    ev_per_contract = p_profit * (credit_per_contract - IC_COMMISSION) - (1 - p_profit) * max_loss_per_contract

    total_credit = n_contracts * credit_per_contract
    total_max_loss = n_contracts * max_loss_per_contract
    total_commission = n_contracts * IC_COMMISSION

    return {
        "ticker": ticker_data["ticker"],
        "current_price": price,
        "short_put": short_put,
        "long_put": long_put,
        "short_call": short_call,
        "long_call": long_call,
        "wing_width": round(avg_width, 2),
        "credit_per_contract": round(credit_per_contract, 2),
        "max_loss_per_contract": round(max_loss_per_contract, 2),
        "n_contracts": n_contracts,
        "total_credit": round(total_credit, 2),
        "total_max_loss": round(total_max_loss, 2),
        "total_commission": round(total_commission, 2),
        "p_profit": round(p_profit * 100, 1),
        "ev_per_contract": round(ev_per_contract, 2),
        "risk_pct_capital": round(total_max_loss / capital * 100, 1),
        "reward_risk": round(total_credit / max(1, total_max_loss), 2),
    }

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("EARNINGS WEEK PLANNER v1 — Week of 2026-07-28")
    fprint(f"Capital: ${CAPITAL:.0f} | Max position: {MAX_POSITION_PCT*100:.0f}%")
    fprint("=" * 70)

    results = {}

    # ─── Section 1: Historical Move Analysis ───
    fprint("\n" + "=" * 50)
    fprint("SECTION 1: Historical Earnings Move Analysis")
    fprint("=" * 50)

    all_analysis = {}
    for ticker in MEGA_CAPS:
        fprint(f"\n  Analyzing {ticker}...")
        data = analyze_earnings_moves(ticker)
        if data:
            all_analysis[ticker] = data
            fprint(f"    Price: ${data['current_price']:.2f}")
            fprint(f"    Avg earnings move: {data['avg_abs_move_pct']:.1f}% | "
                   f"Median: {data['median_abs_move_pct']:.1f}%")
            fprint(f"    Max move: {data['max_move_pct']:.1f}%")
            fprint(f"    Within 2%: {data['pct_within_2pct']:.0f}% | "
                   f"Within 3%: {data['pct_within_3pct']:.0f}% | "
                   f"Within 5%: {data['pct_within_5pct']:.0f}%")
            fprint(f"    Up %: {data['up_pct']:.0f}% | 20d IV: {data['vol_20d']:.1f}%")
    results["move_analysis"] = all_analysis

    # ─── Section 2: Trade Recommendations ───
    fprint("\n" + "=" * 50)
    fprint("SECTION 2: Iron Condor Trade Recommendations")
    fprint("=" * 50)

    trades = {}
    for ticker, info in TARGETS.items():
        fprint(f"\n  === {ticker} ({info['name']}) — {info['day']} {info['earnings_date']} ===")
        if ticker not in all_analysis:
            fprint(f"    SKIP — no data")
            continue

        data = all_analysis[ticker]
        trade = design_iron_condor(data)
        if trade:
            trades[ticker] = trade
            fprint(f"    IRON CONDOR:")
            fprint(f"    Buy  {trade['long_put']:.2f}P / "
                   f"Sell {trade['short_put']:.2f}P / "
                   f"Sell {trade['short_call']:.2f}C / "
                   f"Buy  {trade['long_call']:.2f}C")
            fprint(f"    Wing width: ${trade['wing_width']:.2f}")
            fprint(f"    Contracts: {trade['n_contracts']}")
            fprint(f"    Credit: ${trade['total_credit']:.2f} | "
                   f"Max loss: ${trade['total_max_loss']:.2f}")
            fprint(f"    Commission: ${trade['total_commission']:.2f}")
            fprint(f"    P(profit): {trade['p_profit']:.0f}% | "
                   f"EV/contract: ${trade['ev_per_contract']:.2f}")
            fprint(f"    Risk: {trade['risk_pct_capital']:.1f}% of capital")
            fprint(f"    Reward/Risk: {trade['reward_risk']:.2f}")

            # Go/No-Go
            go = trade["ev_per_contract"] > 0 and trade["risk_pct_capital"] < 40
            fprint(f"\n    VERDICT: {'✅ GO' if go else '❌ NO-GO'}")
            if go:
                fprint(f"    → Enter {info['day']} morning, close Friday or at 50% profit")
            else:
                reasons = []
                if trade["ev_per_contract"] <= 0:
                    reasons.append("negative expected value")
                if trade["risk_pct_capital"] >= 40:
                    reasons.append("risk too high for $645 account")
                fprint(f"    → Reasons: {', '.join(reasons)}")
            trade["go"] = go
    results["trades"] = trades

    # ─── Section 3: Portfolio-Level Risk ───
    fprint("\n" + "=" * 50)
    fprint("SECTION 3: Portfolio Risk Check")
    fprint("=" * 50)

    go_trades = {k: v for k, v in trades.items() if v.get("go")}
    if go_trades:
        total_risk = sum(v["total_max_loss"] for v in go_trades.values())
        total_credit = sum(v["total_credit"] for v in go_trades.values())
        fprint(f"\n  GO trades: {len(go_trades)}")
        fprint(f"  Total max risk: ${total_risk:.2f} ({total_risk/CAPITAL*100:.1f}% of capital)")
        fprint(f"  Total credit: ${total_credit:.2f}")
        if total_risk > CAPITAL * 0.60:
            fprint(f"  ⚠️ WARNING: Combined risk exceeds 60% of capital!")
            fprint(f"  → Recommendation: Run only the higher-EV trade, skip the other")

        results["portfolio_risk"] = {
            "n_trades": len(go_trades),
            "total_risk": total_risk,
            "total_credit": total_credit,
            "risk_pct": total_risk / CAPITAL * 100,
        }
    else:
        fprint("\n  No GO trades this week.")

    # ─── Section 4: Best Mega-Cap IC Opportunities ───
    fprint("\n" + "=" * 50)
    fprint("SECTION 4: Best IC Setups by Historical Edge")
    fprint("=" * 50)

    # Rank by P(within 3%) × reward/risk
    ic_scores = []
    for ticker, data in all_analysis.items():
        trade = design_iron_condor(data)
        if trade and trade["ev_per_contract"] > 0:
            score = (data["pct_within_3pct"] / 100) * trade["reward_risk"]
            ic_scores.append({
                "ticker": ticker,
                "score": score,
                "p_within_3pct": data["pct_within_3pct"],
                "avg_move": data["avg_abs_move_pct"],
                "ev": trade["ev_per_contract"],
                "reward_risk": trade["reward_risk"],
            })

    ic_scores.sort(key=lambda x: x["score"], reverse=True)
    fprint(f"\n  {'Ticker':<8} {'Score':>6} {'P(<3%)':>7} {'Avg Move':>9} {'EV':>8} {'R/R':>5}")
    fprint(f"  {'-'*45}")
    for s in ic_scores:
        fprint(f"  {s['ticker']:<8} {s['score']:>6.2f} {s['p_within_3pct']:>6.0f}% "
               f"{s['avg_move']:>8.1f}% ${s['ev']:>7.2f} {s['reward_risk']:>5.2f}")

    results["ic_rankings"] = ic_scores

    fprint("\n" + "=" * 70)
    elapsed = time.time() - t0
    fprint(f"\nCompleted in {elapsed:.1f}s")

    # Save
    out = OUTPUT_DIR / "earnings_week_plan.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"Saved: {out}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("earnings_week_planner_v1")
        with mlflow.start_run(run_name=f"earnings_wk_{datetime.now():%Y%m%d_%H%M}"):
            mlflow.log_params({
                "capital": CAPITAL,
                "targets": ",".join(TARGETS.keys()),
                "n_mega_caps": len(MEGA_CAPS),
            })
            if go_trades:
                mlflow.log_metrics({
                    "n_go_trades": len(go_trades),
                    "total_credit": total_credit,
                    "total_risk": total_risk,
                })
            mlflow.log_artifact(str(out))
            fprint("Logged to MLflow")
    except Exception as e:
        fprint(f"MLflow skip: {e}")

    fprint("Done.")

if __name__ == "__main__":
    main()
