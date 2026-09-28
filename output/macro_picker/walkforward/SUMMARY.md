# Macro Picker — HC #561 R3 Walk-Forward Validation (GA v2 re-run)

Generated: 2026-06-10 16:16:17

Design: SLIDING 36m train / 12m OOT / 6m step (HC #0). GA fitness on the
TRAIN window ONLY; frozen params evaluated on the following 1y OOT window.
Pooled OOT stream uses every-other fold so segments are non-overlapping.
Costs: 5/30 bps ADV-bucketed round-trip turnover. Weekly rebalance,
long top decile / short bottom decile.

## Pooled OOT — combined macro picker (equal-weight sector sleeves)

| Metric | Value |
|---|---|
| Sharpe | -0.057 |
| Sortino | -0.085 |
| PF | 0.990 |
| WR | 0.491 |
| MaxDD | -0.504 |
| Calmar | -0.035 |
| CAGR | -0.018 |
| Sharpe (green days) | 3.059 |
| Sharpe (red days) | -4.123 |
| Sharpe (flat days) | -1.209 |
| Regime asymmetry | 1.742 (gate <= 0.50: FAIL) |
| OOT days pooled | 2011 |

## Verdict: **R&D ONLY — NOT DEPLOYABLE**

- HC #561 R4 gates on pooled OOT: FAILS CALMAR FLOOR
- HC #428 R1 regime symmetry: FAIL

## Per-sector pooled OOT

| Sector | folds | Sharpe | Sortino | PF | WR | MaxDD | Calmar | RegAsym |
|---|---|---|---|---|---|---|---|---|
| Technology | 15 | -0.249 | -0.343 | 0.956 | 0.496 | -0.714 | -0.109 | 1.611 |
| Financial Services | 15 | 0.257 | 0.385 | 1.048 | 0.506 | -0.340 | 0.083 | 1.974 |

## Per-fold OOT Sharpe (train Sharpe -> OOT Sharpe)

### Technology
- fold 00 OOT 2018-01-02..2019-01-02: train +2.47 -> OOT 0.03 (PF 1.00, WR 0.56, MaxDD -0.20)
- fold 01 OOT 2018-07-02..2019-07-02: train +2.04 -> OOT 0.02 (PF 1.00, WR 0.52, MaxDD -0.21)
- fold 02 OOT 2019-01-02..2020-01-02: train +2.22 -> OOT 0.71 (PF 1.14, WR 0.52, MaxDD -0.08)
- fold 03 OOT 2019-07-02..2020-07-02: train +2.36 -> OOT 2.82 (PF 1.62, WR 0.57, MaxDD -0.15)
- fold 04 OOT 2020-01-02..2021-01-02: train +1.65 -> OOT 0.20 (PF 1.04, WR 0.51, MaxDD -0.18)
- fold 05 OOT 2020-07-02..2021-07-02: train +1.32 -> OOT 0.30 (PF 1.05, WR 0.49, MaxDD -0.22)
- fold 06 OOT 2021-01-02..2022-01-02: train +1.98 -> OOT 0.20 (PF 1.03, WR 0.48, MaxDD -0.20)
- fold 07 OOT 2021-07-02..2022-07-02: train +1.74 -> OOT -1.55 (PF 0.77, WR 0.44, MaxDD -0.48)
- fold 08 OOT 2022-01-02..2023-01-02: train +1.64 -> OOT -1.50 (PF 0.78, WR 0.46, MaxDD -0.35)
- fold 09 OOT 2022-07-02..2023-07-02: train +1.37 -> OOT 1.42 (PF 1.28, WR 0.51, MaxDD -0.15)
- fold 10 OOT 2023-01-02..2024-01-02: train +1.28 -> OOT 0.38 (PF 1.07, WR 0.50, MaxDD -0.15)
- fold 11 OOT 2023-07-02..2024-07-02: train +2.19 -> OOT 1.00 (PF 1.20, WR 0.49, MaxDD -0.26)
- fold 12 OOT 2024-01-02..2025-01-02: train +1.67 -> OOT -1.81 (PF 0.71, WR 0.41, MaxDD -0.59)
- fold 13 OOT 2024-07-02..2025-07-02: train +2.58 -> OOT -0.67 (PF 0.89, WR 0.46, MaxDD -0.32)
- fold 14 OOT 2025-01-02..2026-01-02: train +2.02 -> OOT 1.24 (PF 1.24, WR 0.54, MaxDD -0.08)

### Financial Services
- fold 00 OOT 2018-01-02..2019-01-02: train +1.38 -> OOT 0.66 (PF 1.12, WR 0.53, MaxDD -0.08)
- fold 01 OOT 2018-07-02..2019-07-02: train +2.19 -> OOT 1.78 (PF 1.33, WR 0.58, MaxDD -0.05)
- fold 02 OOT 2019-01-02..2020-01-02: train +2.35 -> OOT -0.58 (PF 0.90, WR 0.54, MaxDD -0.08)
- fold 03 OOT 2019-07-02..2020-07-02: train +2.24 -> OOT 2.41 (PF 1.56, WR 0.57, MaxDD -0.10)
- fold 04 OOT 2020-01-02..2021-01-02: train +1.79 -> OOT 0.61 (PF 1.11, WR 0.49, MaxDD -0.10)
- fold 05 OOT 2020-07-02..2021-07-02: train +2.79 -> OOT -0.20 (PF 0.97, WR 0.52, MaxDD -0.17)
- fold 06 OOT 2021-01-02..2022-01-02: train +1.82 -> OOT -0.85 (PF 0.87, WR 0.47, MaxDD -0.17)
- fold 07 OOT 2021-07-02..2022-07-02: train +2.02 -> OOT -2.02 (PF 0.70, WR 0.43, MaxDD -0.47)
- fold 08 OOT 2022-01-02..2023-01-02: train +1.67 -> OOT -1.13 (PF 0.81, WR 0.49, MaxDD -0.24)
- fold 09 OOT 2022-07-02..2023-07-02: train +1.59 -> OOT 0.15 (PF 1.03, WR 0.50, MaxDD -0.09)
- fold 10 OOT 2023-01-02..2024-01-02: train +1.95 -> OOT 0.88 (PF 1.17, WR 0.54, MaxDD -0.11)
- fold 11 OOT 2023-07-02..2024-07-02: train +2.06 -> OOT -0.33 (PF 0.94, WR 0.49, MaxDD -0.18)
- fold 12 OOT 2024-01-02..2025-01-02: train +1.86 -> OOT 0.79 (PF 1.16, WR 0.48, MaxDD -0.17)
- fold 13 OOT 2024-07-02..2025-07-02: train +1.76 -> OOT 0.01 (PF 1.00, WR 0.52, MaxDD -0.20)
- fold 14 OOT 2025-01-02..2026-01-02: train +2.03 -> OOT 1.24 (PF 1.26, WR 0.51, MaxDD -0.11)

## Context

Prior GA v2 numbers were in-sample-selected (fitness = median OOT-fold
Sharpe). This harness removes that selection bias; the pooled OOT result
above is the honest deployability evidence for the macro picker.

---

# Beta-Hedged Residual Alpha (HC #561 R5 kill-or-keep test)

Generated: 2026-06-10 16:23:16

Hedge: daily strategy return minus rolling-beta x market return.
Beta = trailing 60d cov/var ending the PRIOR day (shift(1), no
look-ahead; min 20d warm-up unhedged). Market proxy = equal-weight
universe close-to-close mean return (SPY not in price cache; same
proxy as the regime labels). Reuses saved per-fold OOT return
streams — GA and simulator NOT re-run.

## Pooled hedged OOT — combined (equal-weight sector sleeves)

| Metric | Unhedged | Hedged |
|---|---|---|
| Sharpe | -0.057 | -0.243 |
| Sortino | -0.085 | -0.355 |
| PF | 0.990 | 0.958 |
| WR | 0.491 | 0.492 |
| MaxDD | -0.504 | -0.539 |
| Calmar | -0.035 | -0.077 |
| Sharpe green/red/flat | +3.06 / -4.12 / -1.21 | 0.29 / -0.78 / -1.21 |
| Regime asymmetry | 1.742 (FAIL) | 1.378 (FAIL) |
| OOT days pooled | 2011 | 2011 |

## Per-sector pooled hedged OOT

| Sector | Sharpe | Sortino | PF | WR | MaxDD | RegAsym | mean beta |
|---|---|---|---|---|---|---|---|
| financial_services | 0.083 | 0.125 | 1.015 | 0.499 | -0.377 | 0.568 | 0.143 |
| technology | -0.359 | -0.485 | 0.936 | 0.481 | -0.739 | 1.231 | 0.195 |

## Per-fold hedged OOT Sharpe

- financial_services: 00:1.07  01:1.88  02:-0.62  03:2.36  04:0.48  05:-0.59  06:-1.24  07:-1.63  08:-1.11  09:-0.03  10:0.83  11:-0.25  12:0.48  13:0.18  14:1.14
- technology: 00:0.21  01:-0.28  02:0.23  03:2.23  04:0.07  05:0.48  06:-0.05  07:-1.24  08:-1.29  09:1.21  10:0.36  11:0.17  12:-1.80  13:-0.37  14:1.31

## Verdict: **KILL LANE — no residual stock-selection alpha after beta hedge**

- Residual-alpha gate (hedged Sharpe >= 0.30 AND regime asymmetry <= 0.50): FAIL
