# HC #413 Sanity Gate Report

- Generated (UTC): 2026-05-18T02:16:02.809489Z
- npz: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz`
- ckpt: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt`
- model_family: `v3_3`
- data_dir: `None`
- elapsed_sec: `0.591`

## OVERALL: FAIL

| Check | Result | Failures | Notes |
|---|---|---|---|
| npz_health | FAIL | NaN in 9 heads (sample: [('target_log_ret_5s', 597), ('target_log_ret_10s', 1010), ('target_log_ret_30s', 2018)]) | n_pred_heads, n_target_heads, n_mask_heads |
| distribution | PASS | - | per_head |
| calibration | PASS | - | decile_hit_rates, decile_counts, slope |
| direction | FAIL | pred_p_up_10s: bunched — frac<0.4=0.430 frac>0.6=0.002 (need >= 0.02); pred_p_up_30s: bunched — frac<0.4=0.070 frac>0.6=0.000 (need >= 0.02) | sign_counts, long_short_ratio, pred_p_up_10s |
| book_gate | SKIP | - | reason |
| input_features | PASS | - | sources, per_array, channel_names_seen |
| e2e_smoke | FAIL | zero directional hits among tier fills (mock TP/SL) | n_total_good, tier_threshold_abs, sample_n |

## Per-check details
### npz_health
```json
{
  "check": "npz_health",
  "passed": false,
  "failures": [
    "NaN in 9 heads (sample: [('target_log_ret_5s', 597), ('target_log_ret_10s', 1010), ('target_log_ret_30s', 2018)])"
  ],
  "details": {
    "n_pred_heads": 32,
    "n_target_heads": 32,
    "n_mask_heads": 32,
    "ref_shape": [
      241351
    ],
    "nan_counts_n": 9,
    "inf_counts_n": 0,
    "total_rows": 241351
  }
}
```
### distribution
```json
{
  "check": "distribution",
  "passed": true,
  "failures": [],
  "details": {
    "per_head": {
      "pred_log_ret_1s": {
        "mean": 0.025678832083940506,
        "std": 0.3682124614715576,
        "min": -2.640625,
        "max": 2.1875,
        "n": 241351,
        "target_std": 1.6341161727905273
      },
      "pred_log_ret_5s": {
        "mean": 0.07043974101543427,
        "std": 0.40655627846717834,
        "min": -2.328125,
        "max": 2.3125,
        "n": 241351,
        "target_std": 3.4511966705322266
      },
      "pred_log_ret_10s": {
        "mean": 0.11657081544399261,
        "std": 0.4327429533004761,
        "min": -2.140625,
        "max": 2.28125,
        "n": 241351,
        "target_std": 4.843695640563965
      },
      "pred_log_ret_30s": {
        "mean": 0.23905937373638153,
        "std": 0.4524390995502472,
        "min": -1.78125,
        "max": 2.09375,
        "n": 241351,
        "target_std": 8.014437675476074
      },
      "pred_log_ret_60s": {
        "mean": 0.24564388394355774,
        "std": 0.10256706178188324,
        "min": -0.359375,
        "max": 0.890625,
        "n": 241351,
        "target_std": 0.0
      },
      "pred_log_ret_5min": {
        "mean": 0.43383222818374634,
        "std": 0.10350363701581955,
        "min": 0.11572265625,
        "max": 0.89453125,
        "n": 241351,
        "target_std": 0.0
      },
      "pred_p_up_5s": {
        "mean": -0.47168630361557007,
        "std": 0.3627011477947235,
        "min": -2.171875,
        "max": 1.0625,
        "n": 241351
      },
      "pred_p_up_10s": {
        "mean": -0.34356674551963806,
        "std": 0.27542421221733093,
        "min": -1.6640625,
        "max": 0.8203125,
        "n": 241351
      },
      "pred_p_up_30s": {
        "mean": -0.13577638566493988,
        "std": 0.17674386501312256,
        "min": -1.0234375,
        "max": 0.462890625,
        "n": 241351
      },
      "pred_p_up_60s": {
        "mean": 0.4003976285457611,
        "std": 0.0984061136841774,
        "min": 0.115234375,
        "max": 0.7109375,
        "n": 241351
      }
    }
  }
}
```
### calibration
```json
{
  "check": "calibration",
  "passed": true,
  "failures": [],
  "details": {
    "decile_hit_rates": [
      0.439,
      0.4469,
      0.4523,
      0.4599,
      0.4651,
      0.4699,
      0.4802,
      0.4853,
      0.5045,
      0.5141
    ],
    "decile_counts": [
      23964,
      23940,
      24033,
      23881,
      23771,
      24212,
      23975,
      24182,
      23905,
      24478
    ],
    "slope": 0.07143790301115922,
    "intercept": 0.4359943307472947,
    "head": "pred_log_ret_10s",
    "lift_top_minus_bottom": 0.07510246768415008
  }
}
```
### direction
```json
{
  "check": "direction",
  "passed": false,
  "failures": [
    "pred_p_up_10s: bunched \u2014 frac<0.4=0.430 frac>0.6=0.002 (need >= 0.02)",
    "pred_p_up_30s: bunched \u2014 frac<0.4=0.070 frac>0.6=0.000 (need >= 0.02)"
  ],
  "details": {
    "sign_counts": {
      "long": 140817,
      "short": 100534,
      "zero": 0
    },
    "long_short_ratio": 1.401,
    "pred_p_up_10s": {
      "raw_min": -1.6640625,
      "raw_max": 0.8203125,
      "frac_below_0.4": 0.4301,
      "frac_above_0.6": 0.0018,
      "mean": 0.41644321822514
    },
    "pred_p_up_30s": {
      "raw_min": -1.0234375,
      "raw_max": 0.462890625,
      "frac_below_0.4": 0.0702,
      "frac_above_0.6": 0.0003,
      "mean": 0.4663786849762765
    }
  }
}
```
### book_gate
```json
{
  "check": "book_gate",
  "passed": true,
  "skipped": true,
  "failures": [],
  "details": {
    "reason": "v3.3 has no book head"
  }
}
```
### input_features
```json
{
  "check": "input_features",
  "passed": true,
  "failures": [],
  "details": {
    "sources": [
      "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_feature_stats.npz"
    ],
    "per_array": {
      "mean_t1": {
        "min": -0.009843716397881508,
        "max": 0.23744527995586395,
        "mean": 0.019241616129875183,
        "n": 39
      },
      "std_t1": {
        "min": 9.999999747378752e-05,
        "max": 1.4467768669128418,
        "mean": 0.4554167091846466,
        "n": 39
      }
    },
    "channel_names_seen": null,
    "range_violations": []
  }
}
```
### e2e_smoke
```json
{
  "check": "e2e_smoke",
  "passed": false,
  "failures": [
    "zero directional hits among tier fills (mock TP/SL)"
  ],
  "details": {
    "n_total_good": 240341,
    "tier_threshold_abs": 1.1875,
    "sample_n": 1000,
    "sample_tier_fills": 2,
    "population_tier_fills": 1206,
    "sample_tier_hits": 0,
    "sample_tier_hit_rate": 0.0,
    "sample_long_count": 588,
    "sample_short_count": 412
  }
}
```