"""DSR sensitivity analysis across effective N values."""
import json, math
from scipy import stats
import numpy as np


def expected_max_sr(N, T):
    gamma = 0.5772156649
    e_max = ((1 - gamma) * stats.norm.ppf(1 - 1 / N)
             + gamma * stats.norm.ppf(1 - 1 / (N * math.e)))
    return e_max * math.sqrt(252 / T)


def se_sr(sr_best, T):
    sr_daily = sr_best / math.sqrt(252)
    var_sr = (1 / T) * (1 + 0.5 * sr_daily**2)
    return math.sqrt(var_sr) * math.sqrt(252)


# ── OOT: SR=9.011, T=13
print("=== OOT Sweep: SR_best=9.011, T=13 days ===")
header = "  Effective N   E[max SR]   z-score    p-value   Pass 5%"
print(header)
for N in [10, 20, 36, 60, 100, 200, 500, 1000, 3996]:
    e_max = expected_max_sr(N, 13)
    se = se_sr(9.011, 13)
    z = (9.011 - e_max) / se
    p = 1 - stats.norm.cdf(z)
    flag = "PASS" if p < 0.05 else "FAIL"
    print(f"  {N:>12,}  {e_max:>10.4f}  {z:>8.4f}  {p:>10.6f}  {flag:>8}")

print()

# ── WF: SR=6.188, T=22
print("=== WF Sim: SR_best=6.188, T=22 days ===")
print(header)
for N in [10, 20, 21, 36, 60, 100, 200, 3996]:
    e_max = expected_max_sr(N, 22)
    se = se_sr(6.188, 22)
    z = (6.188 - e_max) / se
    p = 1 - stats.norm.cdf(z)
    flag = "PASS" if p < 0.05 else "FAIL"
    print(f"  {N:>12,}  {e_max:>10.4f}  {z:>8.4f}  {p:>10.6f}  {flag:>8}")

print()

# ── IS: SR=3.89, T=74
print("=== IS Debiased: SR_best=3.89, T=74 days ===")
print(header)
for N in [5, 10, 20, 36, 74, 128, 500, 3996]:
    e_max = expected_max_sr(N, 74)
    se = se_sr(3.89, 74)
    z = (3.89 - e_max) / se
    p = 1 - stats.norm.cdf(z)
    flag = "PASS" if p < 0.05 else "FAIL"
    print(f"  {N:>12,}  {e_max:>10.4f}  {z:>8.4f}  {p:>10.6f}  {flag:>8}")

print()

# ── Breakeven N for each dataset
print("=== Breakeven Effective N (DSR passes at p<0.05) ===")
datasets = [
    ("OOT T=13 SR=9.011", 9.011, 13),
    ("WF  T=22 SR=6.188", 6.188, 22),
    ("IS  T=74 SR=3.890", 3.890, 74),
]
for label, sr, T in datasets:
    se = se_sr(sr, T)
    e_max_target = sr - 1.645 * se
    e_max_z_target = e_max_target / math.sqrt(252 / T)
    lo, hi = 2, 10**9
    for _ in range(100):
        mid = int((lo + hi) / 2)
        gamma = 0.5772156649
        try:
            e = ((1 - gamma) * stats.norm.ppf(1 - 1 / mid)
                 + gamma * stats.norm.ppf(1 - 1 / (mid * math.e)))
        except Exception:
            hi = mid
            continue
        if e < e_max_z_target:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1:
            break
    print(f"  {label}: DSR passes if effective N <= {lo:,}")

print()
print("=== Context: Trade count limitation ===")
print("  OOT best config (vol70/conv15/10min): 68 trades over 13 days")
print("  WF best config (vol80/conv25/30min):  27 trades over 22 days")
print("  With so few trades, daily PnL std is highly variable, inflating daily Sharpe.")
print()
print("  Trade-count-based Sharpe restatement (per-trade rather than per-day):")
for label, n_trades, T, sr in [
    ("OOT best", 68, 13, 9.011),
    ("WF best", 27, 22, 6.188),
]:
    # Per-trade SR: annualized using n_trades as T
    se_trade = se_sr(sr, n_trades)
    z_null_1 = (sr - 0) / se_trade  # vs SR=0
    p_vs_zero = 1 - stats.norm.cdf(z_null_1)
    print(f"  {label}: n_trades={n_trades}, SR={sr:.3f}, SE={se_trade:.3f}, "
          f"z-vs-SR0={z_null_1:.3f}, p={p_vs_zero:.4f} ({'PASS' if p_vs_zero < 0.05 else 'FAIL'} vs H0:SR=0)")
