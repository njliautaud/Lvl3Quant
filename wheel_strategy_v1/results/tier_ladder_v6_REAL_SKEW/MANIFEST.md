# v6_REAL_SKEW Ladder — Honest skew-calibrated wheel

**Pricing inputs:**
- IV: `iv_features_real_blend.parquet` (51% real DOLT IV, 49% modeled fallback)
- Skew: `strategy/iv_skew.py` parametric smile (slope -0.10, curvature +0.20 in log(K/S)/sqrt(T))
  applied via `_strike_from_delta_skew()` + `_sigma_at_strike()` helpers in `backtest/wheel_engine.py`.

**Engine changes (this version vs v5):**
- `wheel_engine.py` now imports iv_skew.iv_at_strike (via `_iv_at_strike_impl`) and applies it at
  ALL five sigma-usage sites: CSP open (strike + premium), CC open (strike + premium),
  CSP MTM during life, CC MTM during life, _equity_mtm for any open position.
- Two-pass strike solver: first pass with ATM σ to find K0; second pass with σ(K0) to refine.
- `USE_SKEW = True` toggle at module scope. Set to False to reproduce ATM-only v5 behaviour.

**Result vs v5 (REAL IV, ATM-only):**

| Tier         | v5 CAGR | v6 CAGR | Delta | v5 Sharpe | v6 Sharpe |
|--------------|--------:|--------:|------:|----------:|----------:|
| Conservative |   4.5%  |   3.7%  | -0.8% |     1.31  |     0.94  |
| Balanced     |  10.4%  |  11.2%  | +0.9% |     0.96  |     1.01  |
| Income       |   9.8%  |   5.6%  | -4.2% |     0.77  |     0.46  |
| Aggressive   |  16.3%  |   6.1%  |-10.3% |     0.98  |     0.32  |
| Turbo        |   9.3%  |   8.8%  | -0.5% |     0.71  |     0.66  |

**Blended 35/35/25/5 on $1M:**

|              | v5 ATM-only | v6 SKEW   |
|--------------|------------:|----------:|
| CAGR         |     10.21%  |    7.41%  |
| Sharpe       |      1.73   |    0.83   |
| Sortino      |      1.14   |    0.57   |
| Max DD       |    -10.4%   |  -14.1%   |
| End equity   |  $1,791,945 | $1,535,635|

**Why skew hurts mid/high delta tiers**: with σ(K) > σ_atm for OTM puts, the strike that
gives the target Δ moves CLOSER to spot. Closer strikes -> more assignment in down moves
-> more assigned-share loss. Effect is small for low-Δ (15-22) but material at 27Δ+.

**Honest conclusion**: skew calibration knocks ~3 percentage points off the blended CAGR
and roughly halves Sharpe. The skew-on numbers are the credible production estimate.

Window: 2020-01-01 -> 2025-12-31. Regime overlay ON. Full-wheel (allows assignment).
