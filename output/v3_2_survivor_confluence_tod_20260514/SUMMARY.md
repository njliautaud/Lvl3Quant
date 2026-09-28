# v3.2 Survivor Confluence + ToD Analysis

_2026-05-14T05:25:59Z_

## Survivor letter codes
- **A** = log_ret_60s SHORT Top0.5%
- **B** = log_ret_60s SHORT Top1%  (B is a superset of A)
- **C** = log_ret_60s_q10 SHORT Top10%  (different head: q10 quantile)
- **D** = log_ret_1s SHORT Top0.1%

## 1. Individual verification

| ID | Head | Side | Band | n | mean t/fill | Sharpe | WR | passive_net |
|---|---|---|---|---|---|---|---|---|
| A | A_log_ret_60s_SHORT_Top0.5 | SHORT | - | 76 | +1.119 | 0.40 | 63.2% | +0.743 |
| B | B_log_ret_60s_SHORT_Top1 | SHORT | - | 137 | +0.803 | 0.29 | 59.9% | +0.427 |
| C | C_log_ret_60s_q10_SHORT_Top10 | SHORT | - | 171 | +0.587 | 0.20 | 55.0% | +0.211 |
| D | D_log_ret_1s_SHORT_Top0.1 | SHORT | - | 35 | +1.390 | 0.54 | 74.3% | +1.014 |

## 2. Pairwise confluence (do agreeing signals stack?)

| Pair | n_sigs | n_fills | mean | Sharpe | WR | passive_net |
|---|---|---|---|---|---|---|
| AxB | 186 | 76 | +1.119 | 0.40 | 63.2% | +0.743 |
| AxC | 124 | 41 | +1.108 | 0.41 | 61.0% | +0.732 |
| AxD | 31 | 11 | +0.785 | 0.27 | 63.6% | +0.409 |
| BxC | 254 | 83 | +0.394 | 0.14 | 54.2% | +0.018 |
| BxD | 45 | 15 | +0.843 | 0.29 | 66.7% | +0.467 |
| CxD | 57 | 19 | +0.744 | 0.26 | 63.2% | +0.368 |
| AxBxCxD | 31 | 11 | +0.785 | 0.27 | 63.6% | +0.409 |

## 3. ToD distribution (approximate via event-index buckets)

**CAVEAT**: NPZ does not carry per-event timestamps. ToD is approximated by event index assuming uniform RTH event rate. Real ToD requires re-running deep-sim with timestamps preserved.

| Bucket | A_fills | A_mean | B_fills | B_mean | C_fills | C_mean | D_fills | D_mean |
|---|---|---|---|---|---|---|---|---|
| 09:30-10:00 | 5 | -0.324 | 7 | +0.733 | 8 | +0.751 | 5 | +2.076 |
| 10:00-10:30 | 0 | +nan | 0 | +nan | 2 | +nan | 0 | +nan |
| 10:30-11:00 | 2 | +nan | 2 | +nan | 0 | +nan | 1 | +nan |
| 11:00-11:30 | 1 | +nan | 1 | +nan | 0 | +nan | 1 | +nan |
| 11:30-12:00 | 2 | +nan | 2 | +nan | 7 | +0.876 | 1 | +nan |
| 12:00-12:30 | 14 | +1.447 | 21 | +0.543 | 18 | -0.291 | 0 | +nan |
| 12:30-13:00 | 8 | +2.438 | 15 | +2.176 | 3 | +nan | 1 | +nan |
| 13:00-13:30 | 7 | +0.519 | 13 | +0.645 | 1 | +nan | 2 | +nan |
| 13:30-14:00 | 3 | +nan | 7 | +3.376 | 18 | +1.820 | 6 | +3.376 |
| 14:00-14:30 | 7 | -0.338 | 22 | -0.851 | 52 | +0.001 | 14 | +0.483 |
| 14:30-15:00 | 8 | +1.001 | 16 | +0.251 | 31 | +0.311 | 4 | +nan |
| 15:00-15:30 | 11 | +2.103 | 16 | +2.063 | 16 | +1.345 | 0 | +nan |
| 15:30-16:00 | 8 | +1.063 | 15 | +0.976 | 15 | +0.976 | 0 | +nan |

## 4. Per-day robustness

| Date | A_n | A_total | B_n | B_total | C_n | C_total | D_n | D_total |
|---|---|---|---|---|---|---|---|---|
| 20260223 | 13 | +22.9 | 14 | +26.3 | 0 | +0.0 | 1 | +3.4 |
| 20260224 | 32 | +37.5 | 69 | +39.9 | 132 | +78.6 | 18 | +13.3 |
| 20260225 | 25 | +21.9 | 45 | +37.9 | 39 | +21.7 | 13 | +31.4 |
| 20260226 | 6 | +2.8 | 9 | +5.9 | 0 | +0.0 | 3 | +0.6 |
| 20260227 | 0 | +0.0 | 0 | +0.0 | 0 | +0.0 | 0 | +0.0 |

## Interpretation guide

- If **B is essentially the same as A**: the signal isn't tightly localized → loose-band tradable
- If **A×D > A**: confluence with the 1s-horizon adds value (different timescales agree)
- If **all fills concentrate in ToD bucket X**: that's the actual edge window — static rule
- If **per-day shows 1 day with 80% of fills**: same day-2-only artifact as the all-night research
