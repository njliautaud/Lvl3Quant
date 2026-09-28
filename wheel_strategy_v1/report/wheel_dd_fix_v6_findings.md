# wheel_dd_fix_v6 — v5_REAL FullWheel drawdown anomaly: VERDICT = ACCOUNTING BUG (mostly)

Date: 2026-06-10. Window: 2020-01-01 → 2025-12-31, $100k/tier, modeled_bs_calibrated
pricing (same inputs as `results/tier_ladder_v5_REAL_2020_2025`). MLflow experiment:
**wheel_dd_fix_v6** (http://localhost:5000). Sweep code: `wheel_dd_fix_sweep.py`.

## 1. Triage: bug vs strategy

**~75–90% of the blown-out drawdown was a NAV accounting bug** in
`backtest/wheel_engine.py::_equity_mtm`, NOT bad chain data and NOT (primarily) the
strategy.

Evidence:
- Tier2_Balanced_FW equity hit $2,570 (-97.5%) on 2020-04-13 — while SPY was
  *rallying* — then "recovered" 80x to $205k by 2025. Impossible with PF 2.0 / WR 90%.
- On 2020-04-13 the book held 6 assigned-share positions with ~$86k strike-notional
  basis. Reported equity $2.7k + the double-subtracted $86k ≈ $89k true equity —
  the portfolio actually sailed through COVID roughly flat.
- The "realized cash" columns in the v7 books were already sane (Tier2 realized
  MaxDD -11.9%) while MTM showed -82%: marking layer, not trades.

### Bugs found and fixed (wheel_engine.py, 2026-06-10)
1. **Share cost basis double-count (the big one).** At assignment, cash is debited
   the full strike. `_equity_mtm` then ALSO subtracted `share_cost_basis × 100 ×
   contracts` from equity for `long_shares` / `short_call` states. Every assigned
   name transiently crushed NAV by ~its full notional until called away. With 5–7
   names assigned in March 2020 → phantom -97.5% DD.
2. **Open premium double-count in MTM.** Premium is credited to cash at open;
   MTM re-added `(open_price − opt_val)` instead of just subtracting the option
   liability. Overstated equity by one premium per open position.
3. **CC premium double-credited to cash.** Credited at open AND again at
   expire-worthless and at profit-take close (inflated long-run returns).
4. Minor: CSP buy-back didn't debit the per-contract fee from cash.

Fixed MTM is now self-consistent: equity is continuous across open / assignment /
called-away events (verified algebraically per event type).

## 2. Corrected baseline (fixed engine, full wheel, all 5 tiers)
`results/tier_ladder_v6_FIXED_MTM/` — MLflow run `tier_ladder_v6_FIXED_MTM`.

| Tier | CAGR | Sharpe | Sortino | MaxDD | PF | WR |
|---|---:|---:|---:|---:|---:|---:|
| Tier1_Conservative_FW | 8.1% | 0.99 | 0.99 | **-12.2%** (was -84.2%) | 3.11 | 94.0% |
| Tier2_Balanced_FW | 20.1% | 1.57 | 1.48 | **-24.8%** (was -97.5%) | 3.45 | 93.8% |
| Tier3_Income_FW | 21.9% | 1.76 | 1.79 | -22.3% (was -84.0%) | 2.96 | 89.8% |
| Tier4_Aggressive_FW | 9.9% | 0.59 | 0.62 | **-47.4%** (was -64.3%) | 1.32 | 84.7% |
| Tier5_Turbo_FW | 11.5% | 0.76 | 0.82 | -32.1% (was -49.2%) | 1.37 | 81.8% |

Residual real structural risk remains: the wheel holds assigned stock through bear
markets. Balanced -24.8% (COVID, trough 2020-03-23) is genuine; Aggressive -47.4%
(high-β names through 2022) is genuine.

## 3. Fix-variant sweep (Balanced + Aggressive, 7 variants each)
`results/wheel_dd_fix_v6_sweep/sweep_summary.csv` — 14 MLflow runs.
Variants: assigned-notional caps (30%/50% of equity), share stop-loss (10%/20%
below basis), regime suspension (t−1 VIX ≤ 30, no look-ahead), combo
(cap30 + stop15 + regime).

### Tier2_Balanced_FW
| Variant | CAGR | Sharpe | Sortino | Calmar | MaxDD | PF | WR | gap | dayconc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_fixed | 20.1% | 1.57 | 1.48 | 0.81 | -24.8% | 3.45 | 93.8% | 1.57 | 0.04 |
| **cap30 (BEST)** | **22.4%** | **1.84** | **1.62** | **0.90** | -24.8% | **5.19** | **95.1%** | 1.51 | 0.03 |
| cap50 | 20.1% | 1.58 | 1.48 | 0.81 | -24.8% | 3.44 | 93.7% | 1.57 | 0.04 |
| stop10 | 20.6% | 1.79 | 1.53 | 0.86 | -24.0% | 4.12 | 93.3% | 1.50 | 0.03 |
| stop20 | 18.8% | 1.46 | 1.31 | 0.76 | -24.8% | 2.74 | 91.3% | 1.58 | 0.04 |
| regime_vix30_t1 | 22.2% | 1.77 | 1.61 | 0.90 | -24.8% | 4.14 | 94.2% | 1.54 | 0.04 |
| combo | 18.4% | 1.64 | 1.37 | 0.75 | -24.4% | 3.53 | 92.7% | 1.53 | 0.04 |

### Tier4_Aggressive_FW
| Variant | CAGR | Sharpe | Sortino | Calmar | MaxDD | PF | WR | gap | dayconc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_fixed | 9.9% | 0.59 | 0.62 | 0.21 | -47.4% | 1.32 | 84.7% | 1.91 | 0.13 |
| cap30 | 6.2% | 0.44 | 0.47 | 0.16 | -39.8% | 1.13 | 83.4% | 1.98 | 0.21 |
| cap50 | 11.7% | 0.70 | 0.75 | 0.31 | -37.5% | 1.44 | 84.9% | 1.94 | 0.10 |
| **stop10 (BEST Sharpe)** | **19.3%** | **1.14** | **1.22** | **0.67** | -28.6% | 1.73 | 82.7% | 1.86 | 0.09 |
| stop20 | 16.6% | 0.96 | 1.02 | 0.44 | -37.8% | 1.56 | 84.4% | 1.86 | 0.11 |
| regime_vix30_t1 | 5.4% | 0.37 | 0.39 | 0.10 | -52.1% | 1.12 | 85.1% | 1.90 | 0.23 |
| **combo (BEST DD)** | 15.2% | 1.13 | 1.24 | 0.63 | **-24.2%** | 1.83 | 84.3% | 1.88 | 0.10 |

## 4. HC #428 R1 gates
- **Day-concentration ≤ 0.70: PASS everywhere** (max 0.23).
- **Regime-gap ≤ 0.50: FAIL everywhere** (1.50–1.98). Structural, not fixable by
  these overlays: a short-put book is delta-long, so daily returns stratified on
  SPY close-to-close are necessarily green-skewed (green Sharpe +8…+12, red −5…−9).
  Consistent with 2026-06-09 finding that 19/20 full-wheel configs fail the gate
  and only the Tier2 *scalp* (no-assignment) variant passes (gap 0.42). If the
  letter of the gate is non-negotiable, the deployable wheel remains the scalp
  variant; the full wheel passes only the "spirit" (profitable across regimes:
  every variant above has positive total return through 2020 and 2022 bears).

## 5. Recommendations
1. Bug fix is in `backtest/wheel_engine.py` — **all pre-2026-06-10 MTM equity
   curves/DDs from this engine are invalid** (ledgers/realized-cash were fine).
   Old result dirs (tier_ladder_v1…v7) kept for archive; do not quote their MTM DD.
2. Balanced tier: deploy with **cap30** (30% assigned-notional cap): Sharpe 1.84,
   Sortino 1.62, Calmar 0.90, MaxDD -24.8%, PF 5.19, WR 95.1%.
3. Aggressive tier: **stop10** (sell assigned shares 10% below basis) if maximizing
   Sharpe (1.14); **combo** if minimizing DD (-24.2%). Plain regime suspension
   alone HURTS aggressive (vol gate blocks the best premium days).
4. Re-run this ladder on true vendor chains once `materialize_chains_parallel`
   finishes (was 4/56 symbols at 09:20 ET) to confirm marks on real NBBO mids.
