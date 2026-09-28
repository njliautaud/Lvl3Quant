# Razer Paper Trader CLI Wrapper (HC #351)

A thin config wrapper around the existing CNN-Mamba v2 paper trader.
Lets you tweak operational params from one command instead of editing
source on the Windows box.

## What it does
1. Resolves every operational param from layered config (CLI > env > JSON > default).
2. Validates ranges (thresholds in [0,1], positive sizes, etc.).
3. Dumps a timestamped `resolved_config_<UTC>.json` snapshot into `log_dir`.
4. Prints the resolved table to stdout.
5. Invokes the existing daemon (`paper_trading_mamba_v2.py`) as a subprocess
   with the matching native flags. Extra params are exported via the
   `RAZER_PAPER_RESOLVED_CONFIG` env var (daemon can opt-in later).

The wrapper does NOT contain any trading logic. If the daemon doesn't
accept a flag yet, the param still gets snapshotted and is available in
`RAZER_PAPER_RESOLVED_CONFIG` for future hookup.

## Priority (highest wins)
1. CLI flag, e.g. `--cnn-threshold 0.55`
2. Environment variable: `RAZER_PAPER_<UPPER_NAME>` (e.g. `RAZER_PAPER_CNN_THRESHOLD=0.55`)
3. JSON config file via `--config path.json`
4. Hard-coded default in `CONFIG_SPEC`

## Quick examples

```bash
# Validate config without launching
python paper_trader_cli.py --config example_config.json --dry-run

# Loosen the gate (more trades)
python paper_trader_cli.py --config example_config.json --cnn-threshold 0.55

# Disable PatchTST veto for a CNN-only run
python paper_trader_cli.py --config example_config.json --patchtst-veto-enabled false

# Long-only test
python paper_trader_cli.py --config example_config.json --side-bias long

# Tighter risk: 1-tick hard stop, $200 daily cap
python paper_trader_cli.py --config example_config.json \
    --hard-stop-ticks 1.0 --max-daily-loss -200

# Env-var override (useful inside scheduled tasks)
RAZER_PAPER_CNN_THRESHOLD=0.65 python paper_trader_cli.py --config example_config.json
```

## Param groups
- **gates** — confluence / veto / side bias
- **bands** — top-percentile cutoffs per horizon
- **sizing** — position / per-trade / vol scaling
- **timing** — cancel window, hold time, passive/aggressive
- **risk** — stops, trailing, daily loss, cooldowns
- **tod** — time-of-day blocks (HC #41)
- **warmup** — feature warmup, MBO buffer
- **models** — CNN-Mamba & PatchTST weights/stats paths
- **output** — log dir, verbosity, MLflow experiment
- **entrypoint** — daemon script path, python exe, symbol, exchange, device

Run `python paper_trader_cli.py --help` for the full list.

## JSON schema
Every key in `CONFIG_SPEC` (see `paper_trader_cli.py`) is a valid JSON key.
Unknown keys are ignored. Booleans are real JSON booleans. Numeric values
are coerced (e.g. `"60"` → `60` for ints).

## Env-var naming
`RAZER_PAPER_` + uppercase param name with underscores preserved.
Examples:
- `cnn_threshold` → `RAZER_PAPER_CNN_THRESHOLD`
- `tod_block_start` → `RAZER_PAPER_TOD_BLOCK_START`
- `patchtst_veto_enabled` → `RAZER_PAPER_PATCHTST_VETO_ENABLED` (`true`/`false`)

## Launching on Razer Windows via WMI (HC #308)

From Jupiter (Linux), use the existing WMI dispatch pattern:

```powershell
# Run on Razer:
Invoke-WmiMethod -ComputerName razer -Class Win32_Process -Name Create -ArgumentList @(
  'powershell -ExecutionPolicy Bypass -File C:\Users\claude\Lvl3Quant\live_trading\razer\launch_razer_paper_cli.ps1 -ConfigPath C:\Users\claude\Lvl3Quant\live_trading\razer\example_config.json'
)
```

Or one-line direct python:

```
python C:\Users\claude\Lvl3Quant\live_trading\razer\paper_trader_cli.py --config C:\Users\claude\Lvl3Quant\live_trading\razer\example_config.json
```

## Snapshot output
On every (non-`--no-snapshot`) launch the wrapper writes
`<log_dir>/resolved_config_<UTC>.json` containing:
```json
{
  "values":   { "cnn_threshold": 0.6, ... },
  "sources":  { "cnn_threshold": "cli", "max_position": "default", ... },
  "json_path": "example_config.json",
  "resolved_at_utc": "2026-05-14T12:34:56+00:00"
}
```
This is the reproducibility record — pair it with the MLflow run.

## Validation failure mode
Invalid values exit non-zero with a clear message, e.g.

```
Config validation failed:
  - cnn_threshold=1.5 failed validation
  - max_daily_loss=100.0 failed validation
```

## Where defaults came from
See docstring at the top of `paper_trader_cli.py`. Key references:
- CLAUDE.md cost constants (ES_RT_COMMISSION_TICKS=0.376, market crossing 1.376t)
- HC #46 (risk mgmt) / HC #226, #230, #231C
- HC #41 (TOD block 09:25–09:40 ET)
- HC #324 (quality > quantity)
- HC #340, #341 (cost-aware execution sizing)
- HC #308 (WMI dispatch to Razer)
- HC #351 (this wrapper)
