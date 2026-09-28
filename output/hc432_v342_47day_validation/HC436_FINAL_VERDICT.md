# HC #432 Final Verdict — 5-Candidate 47-Day Validation

FIFO market replay (HC #74) + regime stratification (HC #428 R1) + MFE-within-horizon (HC #428 R2). All metrics on realized fills only.

## Leaderboard

| config | n_fills | net_tk/fill | Sharpe(sqrt-N) | PF | WR | day_conc | R1 ratio | R1 | R2 | OVERALL |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|
| v3.4.2 1s long top-0.5% (existing) | 1,728 | -0.153 | -8.51 | 0.66 | 48.3% | 0.15 | 0.01 | P | P | FAIL |
| v3.4.2 5s short top-0.5% (t1422 R2-fixed) | 1,337 | -0.372 | -13.66 | 0.45 | 50.0% | 0.28 | 0.30 | P | P | FAIL |
| v3.4.2 10s short top-0.5% (t2831 R2-fixed) | 1,427 | -0.226 | -5.61 | 0.73 | 47.9% | 0.46 | 0.30 | P | P | FAIL |
| v3.4.2 5s long top-0.5% (ensemble leg) | 2,645 | -0.156 | -10.77 | 0.66 | 48.1% | 0.23 | 0.20 | P | P | FAIL |
| 5s long/short 50/50 ensemble | 3,982 | -0.114 | -17.09 | 0.57 | 48.7% | 0.19 | 0.10 | P | P | FAIL |
| v2 1s short top-0.5% (sanity baseline) | 2,676 | -0.173 | -12.12 | 0.62 | 47.0% | 0.18 | 0.28 | P | P | FAIL |

## Gates
- **R1 (HC #428):** |Sh_green − Sh_red| / max ≤ 0.50 AND day_conc ≤ 0.70
- **R2 (HC #428):** TP ≤ p90(MFE@horizon), hold ≤ 1.5×h, cancel ≤ h
- **OVERALL:** R1 PASS and R2 PASS and Sharpe(sqrt-N) > 0

## Harness sanity
v2 baseline DID NOT reproduce: got -0.173 tk/fill vs HC #413 expected +0.274 tk/fill (gap -0.447). Harness may have a bug — review FIFO replay before trusting v3.4.2 verdicts.

## Per-candidate verdict files
- v3.4.2 1s long top-0.5% (existing): `v342_long_1s_top0.5_verdict.md`
- v3.4.2 5s short top-0.5% (t1422 R2-fixed): `v342_short_5s_top0.5_t1422_R2fix_verdict.md`
- v3.4.2 10s short top-0.5% (t2831 R2-fixed): `v342_short_10s_top0.5_t2831_R2fix_verdict.md`
- v3.4.2 5s long top-0.5% (ensemble leg): `v342_long_5s_top0.5_for_ensemble_verdict.md`
- 5s long/short 50/50 ensemble: `v342_lshort_5s_ensemble_50_50_verdict.md`
- v2 1s short top-0.5% (sanity baseline): `v2_short_1s_top0.5_baseline_verdict.md`

