#!/usr/bin/env python3
"""
Live Readiness Scorecard — HC #678 R4
Evaluates all paper strategies for real-money deployment readiness.

Criteria:
  1. Backtest quality (Sharpe, Sortino, MaxDD, WR, PF, Calmar)
  2. Regime robustness (R1 pass — |Sharpe_green - Sharpe_red| / max < 0.50)
  3. Permutation significance (p < 0.05)
  4. Paper track record length (days)
  5. Paper P&L (sanity check)
  6. Operational complexity (1=simple, 5=complex)
  7. Capital requirement
  8. Real pricing validation status (BS gap %)
  9. Overall readiness score (0-100)
"""

import json
import os
import sys
from datetime import datetime, timedelta

ROOT = "/home/jupiter/Lvl3Quant"

# ── Strategy backtest profiles (from completed studies) ──
# Using honest/permutation-corrected numbers where available

STRATEGIES = {
    "V5_CSP_d25": {
        "backtest": {
            "sharpe": 2.20, "sortino": 2.80, "maxdd_pct": -2.0,
            "calmar": 5.38, "wr_pct": 65.7, "pf": 2.15,
            "cagr_pct": 10.75, "years": 7.2,
        },
        "regime": {"gap": 0.19, "r1_pass": True},  # with VIX TS hedge
        "permutation": {"p": 0.000, "passes": True},
        "complexity": 2,  # CSP is straightforward
        "min_capital": 25000,  # need margin for puts
        "bs_gap_pct": 5,  # tier-1 names ~3-17% overpriced by BS
        "hedge": "VIX TS + dynamic SPY beta",
        "paper_engine": "wheel-v5-paper",
        "state_path": "live_trading_linux/wheel_v5_state/state.json",
        "notes": "Simplest strategy. Delta 25 optimal (ladder study). Low assignment rate (0.6%). "
                 "Weekly CSP on 10 core tickers (NVDA, AMZN, TSLA, GOOGL, etc). "
                 "CAGR modest but extremely consistent — zero losing years at 1x.",
    },
    "IC_Condors": {
        "backtest": {
            "sharpe": 2.05, "sortino": 2.33, "maxdd_pct": -14.67,  # permutation-corrected
            "calmar": 1.34, "wr_pct": 57.9, "pf": 1.52,
            "cagr_pct": 19.62, "years": 7.2,
        },
        "regime": {"gap": 0.145, "r1_pass": True},  # with hedge
        "permutation": {"p": 0.000, "passes": True},
        "complexity": 4,  # 4-leg spreads, more Greeks to manage
        "min_capital": 50000,  # wider spreads need more margin
        "bs_gap_pct": 10,  # index options better priced
        "hedge": "VIX TS + dynamic SPY beta (mandatory)",
        "paper_engine": "wheel-ic-paper",
        "state_path": "live_trading_linux/wheel_ic_state/state.json",
        "notes": "Highest return strategy after corrections. Uses permutation-based Sharpe (2.05), "
                 "not inflated backtest (5.68). MaxDD -14.7% concerning. Hedge is MANDATORY (without: "
                 "regime gap 1.25 FAILS R1). Operationally complex — 4 legs per trade. "
                 "Best as second strategy to add, not first.",
    },
    "ETF_Rotation_v3": {
        "backtest": {
            "sharpe": 2.39, "sortino": 3.19, "maxdd_pct": -9.98,
            "calmar": 2.28, "wr_pct": 55.0, "pf": 1.75,
            "cagr_pct": 22.78, "years": 7.2,
        },
        "regime": {"gap": 0.007, "r1_pass": True},  # with beta hedge
        "permutation": {"p": 0.000, "passes": True},
        "complexity": 2,  # weekly rebalance across ETFs
        "min_capital": 10000,  # ETFs, no options margin
        "bs_gap_pct": 0,  # no options, no BS pricing needed
        "hedge": "Beta-scaled SPY hedge",
        "paper_engine": "etf-rotation-v3",
        "state_path": "live_trading_linux/etf_rotation_v3_state/state.json",
        "notes": "Best regime gap (0.007 — nearly identical performance green/red days). "
                 "No options → no BS pricing risk. Anti-concentration validated (Herfindahl 0.093). "
                 "Highest CAGR (22.8%) of any honest strategy. Good diversifier — near-zero "
                 "correlation with V5 (-0.05) and IC (-0.02). Could deploy on Alpaca immediately.",
    },
    "BPS_GA": {
        "backtest": {
            "sharpe": 2.30, "sortino": 2.80, "maxdd_pct": -19.2,
            "calmar": 1.20, "wr_pct": 58.0, "pf": 1.65,
            "cagr_pct": 31.6, "years": 7.2,
        },
        "regime": {"gap": 0.439, "r1_pass": True},  # barely passes
        "permutation": {"p": 0.085, "passes": False},  # FAILS
        "complexity": 3,  # bull put spreads
        "min_capital": 25000,
        "bs_gap_pct": 15,  # tier-2 names gap higher
        "hedge": "None proven",
        "paper_engine": "wheel-bps-ga",
        "state_path": "live_trading_linux/wheel_bps_ga_state/state.json",
        "notes": "FAILS permutation test (p=0.085 — random achieves Sharpe 2.15). "
                 "Technically passes R1 but barely (gap 0.439 vs 0.50 limit). "
                 "MaxDD -19.2% is worst of all strategies. NOT standalone-viable. "
                 "Keep as diversifier only.",
    },
    "Wheel_Balanced": {
        "backtest": {
            "sharpe": 1.80, "sortino": 2.10, "maxdd_pct": -8.0,
            "calmar": 1.50, "wr_pct": 62.0, "pf": 1.85,
            "cagr_pct": 8.5, "years": 7.2,
        },
        "regime": {"gap": 0.25, "r1_pass": True},
        "permutation": {"p": 0.001, "passes": True},
        "complexity": 1,  # simplest — SPY-only wheel
        "min_capital": 15000,
        "bs_gap_pct": 3,  # SPY options very liquid
        "hedge": "None needed (SPY inherently diversified)",
        "paper_engine": "wheel-paper-balanced",
        "state_path": "live_trading_linux/wheel_paper_balanced_state/state.json",
        "notes": "Simplest possible strategy — SPY-only wheel. Longest paper track (32 days). "
                 "Lowest returns but also lowest operational risk. Good starter strategy. "
                 "User has expressed preference for more complex/higher-return strategies.",
    },
}


def score_strategy(name, s):
    """Compute readiness score 0-100."""
    score = 0
    bt = s["backtest"]

    # Sharpe (0-20 pts): 2.0+ = 20, 1.5 = 15, 1.0 = 10
    score += min(20, max(0, bt["sharpe"] * 10))

    # MaxDD (0-15 pts): < -5% = 15, < -10% = 10, < -20% = 5
    dd = abs(bt["maxdd_pct"])
    if dd < 3: score += 15
    elif dd < 8: score += 12
    elif dd < 15: score += 8
    elif dd < 20: score += 5

    # Regime robustness R1 (0-20 pts)
    if s["regime"]["r1_pass"]:
        gap = s["regime"]["gap"]
        if gap < 0.10: score += 20
        elif gap < 0.20: score += 18
        elif gap < 0.30: score += 15
        elif gap < 0.50: score += 10

    # Permutation test (0-15 pts)
    if s["permutation"]["passes"]:
        if s["permutation"]["p"] < 0.001: score += 15
        elif s["permutation"]["p"] < 0.01: score += 12
        elif s["permutation"]["p"] < 0.05: score += 8

    # Operational simplicity (0-10 pts): lower complexity = higher score
    score += max(0, (6 - s["complexity"]) * 2)

    # Capital efficiency (0-10 pts)
    if s["min_capital"] <= 10000: score += 10
    elif s["min_capital"] <= 25000: score += 7
    elif s["min_capital"] <= 50000: score += 4

    # BS pricing risk (0-10 pts): lower gap = better
    gap = s["bs_gap_pct"]
    if gap == 0: score += 10
    elif gap <= 5: score += 8
    elif gap <= 10: score += 6
    elif gap <= 20: score += 3

    return min(100, score)


def get_paper_stats(state_path):
    """Get paper engine stats from state file."""
    full_path = os.path.join(ROOT, state_path)
    if not os.path.exists(full_path):
        return None
    try:
        with open(full_path) as f:
            st = json.load(f)
        nav = st.get("nav", st.get("NAV", st.get("cash")))
        inception = st.get("inception", st.get("start_date"))
        pnl = st.get("total_pnl", st.get("realized_pnl", 0))
        positions = len(st.get("positions", st.get("open_positions", [])))
        if inception:
            try:
                inc_dt = datetime.fromisoformat(inception.replace("Z", "+00:00").split("+")[0])
                days = (datetime.now() - inc_dt).days
            except:
                days = 0
        else:
            days = 0
        return {"nav": nav, "pnl": pnl, "days": days, "positions": positions}
    except:
        return None


def main():
    print("=" * 80)
    print("LIVE READINESS SCORECARD — HC #678 R4")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 80)

    results = []
    for name, s in STRATEGIES.items():
        score = score_strategy(name, s)
        paper = get_paper_stats(s["state_path"])
        results.append((score, name, s, paper))

    results.sort(key=lambda x: -x[0])

    print(f"\n{'Rank':>4} | {'Strategy':20s} | {'Score':>5} | {'Sharpe':>6} | {'MaxDD':>6} | {'CAGR':>6} | {'R1':>4} | {'Perm':>4} | {'Paper':>8}")
    print("-" * 92)

    for rank, (score, name, s, paper) in enumerate(results, 1):
        bt = s["backtest"]
        r1 = "✅" if s["regime"]["r1_pass"] else "❌"
        perm = "✅" if s["permutation"]["passes"] else "❌"
        paper_str = f"{paper['days']}d" if paper else "N/A"

        print(f"{rank:>4} | {name:20s} | {score:>5} | {bt['sharpe']:>6.2f} | {bt['maxdd_pct']:>5.1f}% | {bt['cagr_pct']:>5.1f}% | {r1:>4} | {perm:>4} | {paper_str:>8}")

    print("\n" + "=" * 80)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 80)

    top = results[0]
    print(f"\n🥇 FIRST TO DEPLOY: {top[1]} (Score: {top[0]}/100)")
    print(f"   {STRATEGIES[top[1]]['notes'][:200]}")

    if len(results) > 1:
        second = results[1]
        print(f"\n🥈 SECOND TO DEPLOY: {second[1]} (Score: {second[0]}/100)")
        print(f"   {STRATEGIES[second[1]]['notes'][:200]}")

    print(f"\n📊 PORTFOLIO (per optimizer): 68% V5 + 14% IC + 18% ETF = Sharpe 3.18, CAGR 14.2%, MaxDD -1.7%")
    print(f"   Near-zero cross-strategy correlations → diversification is real")

    # Minimum viable deployment
    print(f"\n💰 MINIMUM VIABLE DEPLOYMENT:")
    print(f"   Option A: ETF Rotation v3 only — $10K min, no options, Sharpe 2.39, CAGR 22.8%")
    print(f"   Option B: V5 CSP d25 only — $25K min, weekly CSPs, Sharpe 2.20, CAGR 10.8%")
    print(f"   Option C: Combined — $50K min, portfolio Sharpe 3.18, CAGR 14.2%")

    print(f"\n⏳ BLOCKERS:")
    paper_min_days = 60  # from SESSION_STATE item #113
    for _, name, s, paper in results:
        if paper and paper["days"] < paper_min_days:
            remaining = paper_min_days - paper["days"]
            print(f"   {name}: {paper['days']}/{paper_min_days} paper days — {remaining} more needed (target: ~{(datetime.now() + timedelta(days=remaining)).strftime('%b %d')})")
        elif not paper:
            print(f"   {name}: No paper engine data")

    # Save results
    out_path = os.path.join(ROOT, "output/live_readiness_scorecard.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = {
        "generated": datetime.now().isoformat(),
        "rankings": [
            {"rank": i+1, "strategy": name, "score": score,
             "backtest": STRATEGIES[name]["backtest"],
             "regime_pass": STRATEGIES[name]["regime"]["r1_pass"],
             "permutation_pass": STRATEGIES[name]["permutation"]["passes"],
             "paper_days": paper["days"] if paper else 0,
             "notes": STRATEGIES[name]["notes"]}
            for i, (score, name, _, paper) in enumerate(results)
        ]
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
