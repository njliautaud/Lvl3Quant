# v3.2 p_reversal_15s Family Analysis — Confluence + Per-Day + ToD

_2026-05-14T05:38:07Z — fold 0 OOT, 5 days_

## Survivor codes
- **A** = log_ret_60s   SHORT Top0.5%  (original headline survivor)
- **D** = log_ret_1s    SHORT Top0.1%
- **E** = p_reversal_15s SHORT Top20%  ← BIG new
- **F** = p_reversal_15s SHORT Top10%  ← BIG new (best $/edge balance)
- **G** = p_reversal_15s SHORT Top5%
- **H** = p_reversal_15s SHORT Top1%
- **I** = log_ret_60s   SHORT Top5%
- **J** = log_ret_60s   SHORT Top10%
- **K** = log_ret_60s   SHORT Top20%

## 1. Individual verification

| ID | Head | Band | n_fills | mean t | Sharpe | WR | passive_net | $ P&L (5d) |
|---|---|---|---|---|---|---|---|---|
| **A** | 60s_SHORT_T0.5% | - | 76 | +1.119 | 0.40 | 63.2% | +0.743 | $+706 |
| **F** | pr15s_SHORT_T10% | - | 8530 | +0.631 | 0.21 | 57.5% | +0.255 | $+27,209 |
| **G** | pr15s_SHORT_T5% | - | 4036 | +0.663 | 0.25 | 56.9% | +0.287 | $+14,483 |
| **H** | pr15s_SHORT_T1% | - | 827 | +0.583 | 0.24 | 55.7% | +0.207 | $+2,144 |
| **E** | pr15s_SHORT_T20% | - | 16841 | +0.625 | 0.20 | 57.8% | +0.249 | $+52,397 |
| **J** | 60s_SHORT_T10% | - | 998 | +0.539 | 0.19 | 55.8% | +0.163 | $+2,031 |
| **I** | 60s_SHORT_T5% | - | 531 | +0.511 | 0.18 | 54.2% | +0.135 | $+894 |
| **K** | 60s_SHORT_T20% | - | 1989 | +0.547 | 0.19 | 55.6% | +0.171 | $+4,244 |
| **D** | 1s_SHORT_T0.1% | - | 35 | +1.390 | 0.54 | 74.3% | +1.014 | $+444 |

## 2. Pairwise confluence (does combining heads help?)

| Pair | n_signals | n_fills | mean | Sharpe | WR | passive_net | $ P&L | frac_of_smaller |
|---|---|---|---|---|---|---|---|---|
| AxF | 29 | 20 | +1.751 | 0.61 | 75.0% | +1.375 | $+344 | 0.16 |
| AxG | 26 | 17 | +1.876 | 0.68 | 76.5% | +1.500 | $+319 | 0.14 |
| IxF | 32 | 22 | +1.831 | 0.67 | 77.3% | +1.455 | $+400 | 0.02 |
| IxG | 28 | 18 | +1.959 | 0.73 | 77.8% | +1.583 | $+356 | 0.01 |
| JxF | 35 | 23 | +1.593 | 0.55 | 73.9% | +1.217 | $+350 | 0.01 |
| KxE | 413 | 149 | +0.849 | 0.36 | 56.4% | +0.473 | $+881 | 0.06 |

## 3. Overlap matrix (Jaccard of signal sets)

Key question: does F (large-n) encompass A (small-n)?

| Pair | intersect | union | Jaccard | n1 | n2 |
|---|---|---|---|---|---|
| A_vs_F | 29 | 24074 | 0.001 | 186 | 23917 |
| A_vs_G | 26 | 12227 | 0.002 | 186 | 12067 |
| A_vs_E | 31 | 48057 | 0.001 | 186 | 47902 |
| F_vs_G | 12067 | 23917 | 0.505 | 23917 | 12067 |
| F_vs_J | 35 | 27623 | 0.001 | 23917 | 3741 |

## 4. Per-day robustness (5 OOT days, $ P&L)

| Date | A_n | A_$ | F_n | F_$ | G_n | G_$ | I_n | I_$ |
|---|---|---|---|---|---|---|---|---|
| 20260223 | 13 | $+225 | 151 | $+1,125 | 57 | $+725 | 15 | $+300 |
| 20260224 | 32 | $+319 | 731 | $-231 | 77 | $-175 | 289 | $+175 |
| 20260225 | 25 | $+156 | 6515 | $+26,309 | 3816 | $+13,933 | 188 | $+500 |
| 20260226 | 6 | $+6 | 1116 | $-19 | 86 | $-0 | 39 | $-81 |
| 20260227 | 0 | $+0 | 17 | $+25 | 0 | $+0 | 0 | $+0 |

## 5. ToD distribution (approx via event-index buckets, $ P&L)

**CAVEAT**: Same caveat as confluence_tod analysis — NPZ has no timestamps; ToD is approximated by event index assuming uniform RTH event rate.

| Bucket | F_n | F_passive | F_$ | A_n | A_passive | A_$ |
|---|---|---|---|---|---|---|
| 09:30-10:30 | 1139 | +0.537 | $+7,644 | 5 | -0.700 | $-44 |
| 10:30-11:30 | 1403 | +0.108 | $+1,897 | 3 | +nan | $+0 |
| 11:30-12:30 | 2216 | +0.419 | $+11,619 | 16 | +0.875 | $+175 |
| 12:30-13:30 | 2280 | +0.179 | $+5,112 | 15 | +1.167 | $+219 |
| 13:30-14:30 | 756 | +0.165 | $+1,556 | 10 | +0.400 | $+50 |
| 14:30-15:30 | 536 | -0.141 | $-944 | 19 | +1.263 | $+300 |
| 15:30-16:00 | 200 | +0.130 | $+325 | 8 | +0.687 | $+69 |

## Decision-quality interpretation

- If **F dominates A in $ P&L AND has 4-of-5 days positive** → F is the deployment-ready candidate (not A)
- If **F × A confluence has higher per-fill edge than F alone** → run BOTH, but only enter on confluence
- If **F is concentrated in same ToD bucket as A** → ToD-gate for the strategy is real
- If **F and A have <0.10 Jaccard** → they're orthogonal, BOTH should be deployed
