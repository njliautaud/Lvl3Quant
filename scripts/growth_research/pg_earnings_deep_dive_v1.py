#!/usr/bin/env python3
"""PG Earnings Deep Dive — Pre-trade analysis for Monday 7/29 iron condor.

Our earnings week planner (item 949) recommended:
  PG IC: sell 142P/152C, buy 139P/155C, credit ~$104, max loss ~$194

This script validates the trade setup with:
1. Historical PG earnings move distribution (last 20 quarters)
2. Optimal strike placement based on realized moves
3. Expected value calculation with realistic fill assumptions
4. Risk scenarios (what if PG moves more than expected?)
5. Alternative setups (different widths, tighter/wider wings)
"""
import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'pg_earnings_deep_dive_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

def main():
    import yfinance as yf

    t0 = datetime.now()
    fprint(f"PG Earnings Deep Dive v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*70}")
    fprint(f"Pre-trade analysis for PG earnings IC on Monday 7/29")
    fprint(f"{'='*70}")

    # Download PG data
    fprint("\nDownloading PG data...")
    pg = yf.download('PG', start='2018-01-01', end='2026-07-26', progress=False)
    if isinstance(pg.columns, pd.MultiIndex):
        pg.columns = pg.columns.get_level_values(0)
    close = pg['Close'].dropna()
    high = pg['High'].dropna()
    low = pg['Low'].dropna()

    current_price = float(close.iloc[-1])
    fprint(f"PG current price: ${current_price:.2f}")

    # 1. Find earnings dates (look for overnight gaps > 1.5%)
    fprint(f"\n{'='*70}")
    fprint("1. HISTORICAL EARNINGS MOVES")
    fprint(f"{'='*70}")
    rets = close.pct_change()
    gaps = rets[abs(rets) > 0.015].dropna()
    # Filter to quarterly pattern (roughly every 90 days)
    earnings_dates = []
    last_ed = None
    for dt, r in gaps.items():
        if last_ed is None or (dt - last_ed).days > 60:
            earnings_dates.append((dt, float(r)))
            last_ed = dt

    # Get last 20 earnings
    recent = earnings_dates[-20:]
    moves = [abs(r)*100 for _, r in recent]
    signed_moves = [r*100 for _, r in recent]

    fprint(f"\nLast {len(recent)} earnings moves:")
    fprint(f"{'Date':<12} {'Move':>7} {'Direction':>10}")
    fprint("-"*32)
    for dt, r in recent:
        direction = "UP" if r > 0 else "DOWN"
        fprint(f"{str(dt.date()):<12} {abs(r)*100:>6.2f}% {direction:>10}")

    fprint(f"\nStatistics:")
    fprint(f"  Mean absolute move: {np.mean(moves):.2f}%")
    fprint(f"  Median absolute move: {np.median(moves):.2f}%")
    fprint(f"  Max move: {max(moves):.2f}%")
    fprint(f"  Min move: {min(moves):.2f}%")
    fprint(f"  Std: {np.std(moves):.2f}%")
    fprint(f"  Moves > 3%: {sum(1 for m in moves if m > 3)}/{len(moves)} ({sum(1 for m in moves if m > 3)/len(moves)*100:.0f}%)")
    fprint(f"  Moves > 5%: {sum(1 for m in moves if m > 5)}/{len(moves)} ({sum(1 for m in moves if m > 5)/len(moves)*100:.0f}%)")
    fprint(f"  Up/Down split: {sum(1 for r in signed_moves if r > 0)}/{sum(1 for r in signed_moves if r < 0)}")

    # Percentiles
    p50 = np.percentile(moves, 50)
    p75 = np.percentile(moves, 75)
    p90 = np.percentile(moves, 90)
    p95 = np.percentile(moves, 95)
    fprint(f"\n  P50: {p50:.2f}% (${current_price*p50/100:.2f})")
    fprint(f"  P75: {p75:.2f}% (${current_price*p75/100:.2f})")
    fprint(f"  P90: {p90:.2f}% (${current_price*p90/100:.2f})")
    fprint(f"  P95: {p95:.2f}% (${current_price*p95/100:.2f})")

    # 2. Strike placement analysis
    fprint(f"\n{'='*70}")
    fprint("2. STRIKE PLACEMENT ANALYSIS")
    fprint(f"{'='*70}")

    # Proposed IC: sell 142P/152C, buy 139P/155C
    proposed = {
        'sell_put': 142, 'buy_put': 139,
        'sell_call': 152, 'buy_call': 155,
        'credit': 1.04  # estimated
    }

    put_dist = (current_price - proposed['sell_put']) / current_price * 100
    call_dist = (proposed['sell_call'] - current_price) / current_price * 100
    fprint(f"\nProposed IC:")
    fprint(f"  Sell {proposed['sell_put']}P ({put_dist:.1f}% below) / Sell {proposed['sell_call']}C ({call_dist:.1f}% above)")
    fprint(f"  Buy {proposed['buy_put']}P / Buy {proposed['buy_call']}C (${proposed['buy_call']-proposed['sell_call']} wide)")
    fprint(f"  Estimated credit: ${proposed['credit']*100:.0f}")
    fprint(f"  Max loss: ${(proposed['buy_call']-proposed['sell_call'])*100 - proposed['credit']*100:.0f}")

    # How often would this IC win historically?
    wins = 0
    losses = 0
    for dt, r in recent:
        move_price = current_price * (1 + r)
        if proposed['sell_put'] <= move_price <= proposed['sell_call']:
            wins += 1
        else:
            losses += 1
    fprint(f"\n  Historical win rate at these strikes: {wins}/{len(recent)} ({wins/len(recent)*100:.0f}%)")

    # 3. Alternative setups
    fprint(f"\n{'='*70}")
    fprint("3. ALTERNATIVE IC SETUPS")
    fprint(f"{'='*70}")

    alternatives = [
        ('Tight (2% wings)', 0.02, 3),
        ('Standard (3% wings)', 0.03, 3),
        ('Wide (4% wings)', 0.04, 3),
        ('Very wide (5% wings)', 0.05, 3),
        ('Narrow width ($2)', 0.03, 2),
        ('Wide width ($5)', 0.03, 5),
    ]

    fprint(f"\n{'Setup':<25} {'PutK':>6} {'CallK':>6} {'WinRate':>8} {'Est.Cr':>8} {'MaxLoss':>8} {'EV':>8}")
    fprint("-"*78)
    for name, wing_pct, width in alternatives:
        sell_put = round(current_price * (1 - wing_pct))
        sell_call = round(current_price * (1 + wing_pct))
        buy_put = sell_put - width
        buy_call = sell_call + width

        # Historical win rate
        w = sum(1 for _, r in recent if sell_put <= current_price*(1+r) <= sell_call)
        wr = w / len(recent)

        # Estimate credit (rough: proportional to closeness of short strikes)
        # Tighter wings = higher credit but lower WR
        credit_est = width * 100 * (1 - wing_pct/0.10)  # rough
        credit_est = max(20, min(credit_est, width*100*0.4))
        max_loss = width * 100 - credit_est

        ev = wr * credit_est - (1-wr) * max_loss

        fprint(f"{name:<25} ${sell_put:>5} ${sell_call:>5} {wr*100:>6.0f}%  ${credit_est:>6.0f} ${max_loss:>7.0f} ${ev:>7.0f}")

    # 4. Risk scenarios
    fprint(f"\n{'='*70}")
    fprint("4. RISK SCENARIOS FOR PROPOSED IC")
    fprint(f"{'='*70}")

    scenarios = [
        ('PG flat (±0.5%)', 0.005),
        ('Normal move (±2%)', 0.02),
        ('Large move (±3.5%)', 0.035),
        ('Big surprise (±5%)', 0.05),
        ('Shock (±8%)', 0.08),
    ]

    credit = proposed['credit'] * 100
    max_loss_val = (proposed['buy_call'] - proposed['sell_call']) * 100 - credit

    fprint(f"\n{'Scenario':<25} {'PG Price':>10} {'IC P&L':>8} {'Outcome':>10}")
    fprint("-"*58)
    for name, move_pct in scenarios:
        for direction in [1, -1]:
            move_price = current_price * (1 + direction * move_pct)
            dir_str = "up" if direction > 0 else "down"

            # Calculate IC P&L
            put_val = max(0, proposed['sell_put'] - move_price) - max(0, proposed['buy_put'] - move_price)
            call_val = max(0, move_price - proposed['sell_call']) - max(0, move_price - proposed['buy_call'])
            pnl = credit - (put_val + call_val) * 100

            outcome = "WIN" if pnl > 0 else "LOSS"
            fprint(f"{name+' '+dir_str:<25} ${move_price:>9.2f} ${pnl:>7.0f} {outcome:>10}")

    # 5. Recommendation
    fprint(f"\n{'='*70}")
    fprint("5. RECOMMENDATION")
    fprint(f"{'='*70}")

    p_win = wins / len(recent)
    ev = p_win * credit - (1-p_win) * max_loss_val
    fprint(f"\nProposed IC expected value: ${ev:.0f}")
    fprint(f"Win probability (historical): {p_win*100:.0f}%")
    fprint(f"Risk/reward: ${credit:.0f} credit vs ${max_loss_val:.0f} max loss")
    fprint(f"Capital at risk: ${max_loss_val:.0f} = {max_loss_val/645*100:.0f}% of $645 account")

    if ev > 0 and p_win > 0.70 and max_loss_val/645 < 0.35:
        fprint(f"\n✅ TRADE APPROVED. Positive EV (${ev:.0f}), high WR ({p_win*100:.0f}%), acceptable risk ({max_loss_val/645*100:.0f}% of capital).")
    elif ev > 0:
        fprint(f"\n⚠️ TRADE MARGINAL. Positive EV but {'high risk' if max_loss_val/645 > 0.35 else 'moderate WR'}.")
    else:
        fprint(f"\n❌ TRADE REJECTED. Negative EV (${ev:.0f}).")

    # Check volatility context
    fprint(f"\n=== VOLATILITY CONTEXT ===")
    vol_20d = float(close.pct_change().iloc[-20:].std() * np.sqrt(252) * 100)
    vol_60d = float(close.pct_change().iloc[-60:].std() * np.sqrt(252) * 100)
    fprint(f"PG 20-day realized vol: {vol_20d:.1f}%")
    fprint(f"PG 60-day realized vol: {vol_60d:.1f}%")
    fprint(f"Implied 1-day move at 20d vol: ±{vol_20d/np.sqrt(252):.2f}% (${current_price*vol_20d/100/np.sqrt(252):.2f})")

    # Save
    elapsed = (datetime.now() - t0).total_seconds()
    save_data = {
        'timestamp': t0.isoformat(),
        'pg_price': current_price,
        'proposed_ic': proposed,
        'historical_wr': p_win,
        'ev': round(ev, 2),
        'capital_at_risk_pct': round(max_loss_val/645*100, 1),
        'earnings_moves': [{'date': str(dt.date()), 'move_pct': round(r*100,2)} for dt, r in recent],
        'vol_20d': round(vol_20d, 1),
        'vol_60d': round(vol_60d, 1),
        'runtime_s': round(elapsed, 1)
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nRuntime: {elapsed:.0f}s")
    fprint(f"\n{'='*70}\nDONE — PG Earnings Deep Dive v1\n{'='*70}")

if __name__ == '__main__':
    main()
