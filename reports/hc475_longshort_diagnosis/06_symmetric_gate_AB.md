# HC #475 — Symmetric Gate A/B vs Baseline Per-Tail Top-Quantile Gate
Generated: 2026-05-22 03:28 ET. Wall: 15639.6s.

**Compliance**: HC #475 R1/R2/R3 (long/short balance, both-sides competency, short-only justification), HC #474 R6 (Sharpe/Sortino/PF/WR/trades-per-day order), HC #428 R1 (regime gate), HC #472 R2 (first-half / second-half stratification), HC #74 (canonical FIFO replay).

**Data scope**: 16-day OOT NPZ (same as baseline). Symmetric threshold calibrated on first 70% of dates (IS), applied to all 16 days. NO IS-on-OOT leakage.

**Bracket** (identical to baseline): TP=4t, SL=3t, hold=30s, cancel=10s, passive_at_touch.

---

## A/B Headline (HC #474 R6 column order)

| Config | Variant | Sharpe | Sortino | PF | WR | trades/day | long_share | short_share | Regime skew | Gates |
|---|---|---|---|---|---|---|---|---|---|---|
| pair01_logret1s+pup5s | baseline | -0.114 | -3.277 | 0.80 | 42.7% | 963.5 | 0.0% | 100.0% | 1.87 | FAIL |
| pair01_logret1s+pup5s | **symmetric** | -0.096 | -1.866 | 0.83 | 43.6% | 520.8 | 83.2% | 16.8% | 0.36 | FAIL |
| pair07_logret10s+logret60sq50 | baseline | -0.173 | -2.025 | 0.71 | 40.3% | 83.6 | 2.7% | 97.3% | 0.11 | FAIL |
| pair07_logret10s+logret60sq50 | **symmetric** | -0.180 | -1.485 | 0.70 | 39.9% | 153.0 | 2.9% | 97.1% | 0.43 | FAIL |
| pair08_logret5s+pup5s | baseline | -0.100 | -2.595 | 0.82 | 43.5% | 345.4 | 0.0% | 100.0% | 0.56 | FAIL |
| pair08_logret5s+pup5s | **symmetric** | -0.099 | -2.056 | 0.82 | 43.5% | 689.9 | 82.3% | 17.7% | 0.85 | FAIL |
| trip01_pup5s+logret1s+logret10s | baseline | -0.207 | -1.152 | 0.66 | 37.9% | 8.2 | 0.0% | 100.0% | 0.22 | FAIL |
| trip01_pup5s+logret1s+logret10s | **symmetric** | -0.095 | -2.186 | 0.83 | 43.7% | 368.4 | 94.3% | 5.7% | 1.09 | FAIL |
| trip03_logret5s+pup5s+logret1s | baseline | -0.106 | -4.703 | 0.81 | 43.2% | 429.2 | 0.0% | 100.0% | 1.79 | FAIL |
| trip03_logret5s+pup5s+logret1s | **symmetric** | -0.092 | -1.790 | 0.83 | 43.8% | 507.1 | 83.4% | 16.6% | 0.37 | FAIL |

**Gates**: pass_dayconc (≤0.70), pass_regime (skew ≤0.50), pass_wr_floor (WR ≥55% OR (Sharpe≥2 AND PF≥1.8)).

## First-half / Second-half stratification (HC #472 R2)

| Config | Variant | Sharpe_h1 | Sharpe_h2 | WR_h1 | WR_h2 |
|---|---|---|---|---|---|
| pair01_logret1s+pup5s | baseline | -0.106 | -0.124 | 43.1% | 42.2% |
| pair01_logret1s+pup5s | symmetric | -0.115 | -0.070 | 42.7% | 44.8% |
| pair07_logret10s+logret60sq50 | baseline | -0.123 | -0.226 | 43.0% | 37.4% |
| pair07_logret10s+logret60sq50 | symmetric | -0.159 | -0.199 | 41.3% | 38.7% |
| pair08_logret5s+pup5s | baseline | -0.092 | -0.108 | 44.0% | 43.0% |
| pair08_logret5s+pup5s | symmetric | -0.124 | -0.066 | 42.3% | 45.0% |
| trip01_pup5s+logret1s+logret10s | baseline | -0.308 | +0.168 | 34.0% | 53.8% |
| trip01_pup5s+logret1s+logret10s | symmetric | -0.119 | -0.062 | 42.6% | 45.2% |
| trip03_logret5s+pup5s+logret1s | baseline | -0.080 | -0.129 | 44.5% | 41.9% |
| trip03_logret5s+pup5s+logret1s | symmetric | -0.113 | -0.063 | 42.8% | 45.1% |

## Calibration metadata

```json
{
  "pair01_logret1s+pup5s": {
    "calib": {
      "pred_log_ret_1s": {
        "kind": "directional",
        "k_pos": 0.2099609375,
        "k_neg": 0.2734375
      },
      "pred_p_up_5s": {
        "kind": "directional",
        "k_pos": 2.2351741790771484e-08,
        "k_neg": 0.28515625
      }
    },
    "n_long": 7083,
    "n_short": 1369,
    "short_share_triggers": 0.1619734973970658
  },
  "pair07_logret10s+logret60sq50": {
    "calib": {
      "pred_log_ret_10s": {
        "kind": "directional",
        "k_pos": 0.25390625,
        "k_neg": 0.296875
      },
      "pred_log_ret_60s_q50": {
        "kind": "directional",
        "k_pos": 1.484375,
        "k_neg": 2.265625
      }
    },
    "n_long": 82,
    "n_short": 2730,
    "short_share_triggers": 0.9708392603129445
  },
  "pair08_logret5s+pup5s": {
    "calib": {
      "pred_log_ret_5s": {
        "kind": "directional",
        "k_pos": 0.2001953125,
        "k_neg": 0.328125
      },
      "pred_p_up_5s": {
        "kind": "directional",
        "k_pos": 2.2351741790771484e-08,
        "k_neg": 0.28515625
      }
    },
    "n_long": 10060,
    "n_short": 2053,
    "short_share_triggers": 0.1694873276644927
  },
  "trip01_pup5s+logret1s+logret10s": {
    "calib": {
      "pred_p_up_5s": {
        "kind": "directional",
        "k_pos": 2.2351741790771484e-08,
        "k_neg": 0.28515625
      },
      "pred_log_ret_1s": {
        "kind": "directional",
        "k_pos": 0.2099609375,
        "k_neg": 0.2734375
      },
      "pred_log_ret_10s": {
        "kind": "directional",
        "k_pos": 0.25390625,
        "k_neg": 0.296875
      }
    },
    "n_long": 4401,
    "n_short": 245,
    "short_share_triggers": 0.052733534222987516
  },
  "trip03_logret5s+pup5s+logret1s": {
    "calib": {
      "pred_log_ret_5s": {
        "kind": "directional",
        "k_pos": 0.2001953125,
        "k_neg": 0.328125
      },
      "pred_p_up_5s": {
        "kind": "directional",
        "k_pos": 2.2351741790771484e-08,
        "k_neg": 0.28515625
      },
      "pred_log_ret_1s": {
        "kind": "directional",
        "k_pos": 0.2099609375,
        "k_neg": 0.2734375
      }
    },
    "n_long": 6906,
    "n_short": 1311,
    "short_share_triggers": 0.15954728002920773
  }
}
```

## Verdict (HC #475 R3 short-only deploy decision)

- No symmetric-gate config clears all three gates. Per HC #475 R3, this is NOT a deploy candidate.
  Best Sharpe under symmetric gate, with failure reasons:
  `trip03_logret5s+pup5s+logret1s`: Sharpe=-0.092, fails [day-conc 0.00>0.70, WR 43.8%<55% and not (Sharpe≥2 AND PF≥1.8)].

- HC #475 R4 alpha redevelopment (sign-balanced loss, continuation targets, execution-tailored outputs) remains the medium-term recommendation regardless of this A/B outcome — the symmetric gate is a policy patch over a model whose predicted-magnitude distribution is asymmetric.