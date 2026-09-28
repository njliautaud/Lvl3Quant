#!/usr/bin/env python3
"""
Patch fill_sim.rs and fill_sim_main.rs to add conviction exit support.

Conviction exit = delayed signal flip exit. Instead of exiting immediately when
the prediction flips sign, require N consecutive bars of opposite-direction signal
before triggering the exit. This filters out noise and only exits when the model
persistently says the trade is wrong.

New parameters:
  --conviction-exit-bars N  (0 = disabled, 100 = 10 seconds at 100ms bars)
  --conviction-exit-mag M   (0.0 = any opposite signal, 1.0 = require |z| >= 1.0)

Usage:
  python3 patch_conviction_exit.py  (run on Jupiter in rust_cache_builder dir)
"""

import re
import sys
import os

FILL_SIM = "src/fill_sim.rs"
FILL_SIM_MAIN = "src/fill_sim_main.rs"

def patch_fill_sim():
    with open(FILL_SIM, 'r') as f:
        code = f.read()

    # 1. Add ConvictionExit to ExitReason enum (after RatchetStop)
    code = code.replace(
        "    RatchetStop,  // Progressive trailing stop based on MFE thresholds",
        "    RatchetStop,  // Progressive trailing stop based on MFE thresholds\n"
        "    ConvictionExit,  // Delayed signal flip: opposite signal persisted for N bars"
    )

    # 2. Add ConvictionExit to the market exit cost match arm
    code = code.replace(
        "| ExitReason::RatchetStop | ExitReason::MaeTimeExit => {",
        "| ExitReason::RatchetStop | ExitReason::MaeTimeExit | ExitReason::ConvictionExit => {"
    )

    # 3. Add conviction_exit_bars and conviction_exit_mag to SimConfig
    code = code.replace(
        "    /// If position is underwater by >= this many ticks AND held for >= mae_exit_hold_sec,",
        "    // ---- Conviction Exit (delayed signal flip) ----\n"
        "    /// Number of consecutive bars the signal must be opposite before triggering exit.\n"
        "    /// 0 = disabled. 100 = 10 seconds (at 100ms bars). Replaces instant signal_flip_exit.\n"
        "    pub conviction_exit_bars: u64,\n"
        "    /// Minimum magnitude (absolute z-score) of opposite signal to count toward conviction.\n"
        "    /// 0.0 = any opposite signal counts. 1.0 = require |z| >= 1.0.\n"
        "    pub conviction_exit_mag: f64,\n"
        "\n"
        "    /// If position is underwater by >= this many ticks AND held for >= mae_exit_hold_sec,"
    )

    # 4. Add defaults in SimConfig::default()
    code = code.replace(
        "            mae_exit_ticks: 0.0,\n"
        "            mae_exit_hold_sec: 0.0,",
        "            mae_exit_ticks: 0.0,\n"
        "            mae_exit_hold_sec: 0.0,\n"
        "            conviction_exit_bars: 0,\n"
        "            conviction_exit_mag: 0.0,"
    )

    # 5. Add conviction_flip_counter to VirtualOrder struct (after last_chase_check_ns)
    code = code.replace(
        "    pub last_chase_check_ns: u64,     // Last time we checked for reprice\n}",
        "    pub last_chase_check_ns: u64,     // Last time we checked for reprice\n"
        "\n"
        "    // Conviction exit tracking\n"
        "    pub conviction_flip_counter: u64,  // Consecutive bars of opposite signal\n}"
    )

    # 6. Add conviction_flip_counter: 0 to all three VirtualOrder construction sites
    code = code.replace(
        "                chase_reprices: 0,\n"
        "                last_chase_check_ns: signal.bar_ns,\n"
        "            };\n"
        "\n"
        "            // Set trailing stop if configured",
        "                chase_reprices: 0,\n"
        "                last_chase_check_ns: signal.bar_ns,\n"
        "                conviction_flip_counter: 0,\n"
        "            };\n"
        "\n"
        "            // Set trailing stop if configured"
    )

    # For the zero-latency and latency>0 constructors, add before the closing };
    # The zero-latency one pushes directly
    code = code.replace(
        "                last_chase_check_ns: signal.bar_ns,\n"
        "            };\n"
        "\n"
        "            self.virtual_orders.push(vo);\n"
        "            self.total_posted += 1;\n"
        "        } else {",
        "                last_chase_check_ns: signal.bar_ns,\n"
        "                conviction_flip_counter: 0,\n"
        "            };\n"
        "\n"
        "            self.virtual_orders.push(vo);\n"
        "            self.total_posted += 1;\n"
        "        } else {"
    )

    # The latency>0 constructor - find the third occurrence
    # It's after "state: VirtualOrderState::PendingEntry"
    # We need to find the last_chase_check_ns that's followed by the pending entry pattern
    code = code.replace(
        "                last_chase_check_ns: signal.bar_ns + latency,\n"
        "            };\n"
        "\n"
        "            self.virtual_orders.push(vo);\n"
        "            self.total_posted += 1;\n"
        "        }\n",
        "                last_chase_check_ns: signal.bar_ns + latency,\n"
        "                conviction_flip_counter: 0,\n"
        "            };\n"
        "\n"
        "            self.virtual_orders.push(vo);\n"
        "            self.total_posted += 1;\n"
        "        }\n"
    )

    # 7. Replace the check_signal_flip function with conviction-aware version
    old_fn = '''    /// Check if the current prediction signal has flipped sign relative to entry.
    /// If signal_flip_exit is enabled and a position is open, close it immediately
    /// at mid price with spread cost (market exit).
    /// Returns true if any position was closed.
    pub fn check_signal_flip(&mut self, current_prediction: f64, ts: u64) -> bool {
        if !self.config.signal_flip_exit {
            return false;
        }

        let config = self.config.clone();
        let current_mid = self.current_mid;
        let mut closed_any = false;

        for vo in self.virtual_orders.iter_mut() {
            if vo.state != VirtualOrderState::Filled {
                continue;
            }

            // Signal has flipped when entry and current prediction have opposite signs.
            // We use a strict sign check: entry positive means we're long,
            // if current prediction is negative (or zero) the signal has reversed.
            let entry_was_long = vo.signal_strength > 0.0;
            let current_is_short = current_prediction <= 0.0;
            let entry_was_short = vo.signal_strength < 0.0;
            let current_is_long = current_prediction >= 0.0;

            let flipped = (entry_was_long && current_is_short)
                || (entry_was_short && current_is_long);

            if flipped {
                close_position_impl(&config, vo, current_mid, ts, ExitReason::SignalFlip);
                closed_any = true;
            }
        }

        closed_any
    }'''

    new_fn = '''    /// Check if the current prediction signal has flipped sign relative to entry.
    /// Supports two modes:
    /// 1. Instant flip (signal_flip_exit=true, conviction_exit_bars=0): exit immediately on flip
    /// 2. Conviction exit (conviction_exit_bars>0): exit after N consecutive opposite bars
    ///    Optionally requires minimum magnitude (conviction_exit_mag>0).
    /// Returns true if any position was closed.
    pub fn check_signal_flip(&mut self, current_prediction: f64, ts: u64) -> bool {
        let use_instant = self.config.signal_flip_exit && self.config.conviction_exit_bars == 0;
        let use_conviction = self.config.conviction_exit_bars > 0;

        if !use_instant && !use_conviction {
            return false;
        }

        let config = self.config.clone();
        let current_mid = self.current_mid;
        let mut closed_any = false;

        for vo in self.virtual_orders.iter_mut() {
            if vo.state != VirtualOrderState::Filled {
                continue;
            }

            let entry_was_long = vo.signal_strength > 0.0;
            let current_is_short = current_prediction <= 0.0;
            let entry_was_short = vo.signal_strength < 0.0;
            let current_is_long = current_prediction >= 0.0;

            let flipped = (entry_was_long && current_is_short)
                || (entry_was_short && current_is_long);

            if use_instant {
                // Original behavior: exit immediately on any flip
                if flipped {
                    close_position_impl(&config, vo, current_mid, ts, ExitReason::SignalFlip);
                    closed_any = true;
                }
            } else if use_conviction {
                // Conviction mode: count consecutive opposite bars
                let mag_ok = config.conviction_exit_mag <= 0.0
                    || current_prediction.abs() >= config.conviction_exit_mag;

                if flipped && mag_ok {
                    vo.conviction_flip_counter += 1;
                } else {
                    // Reset counter: signal is back in our direction (or below magnitude)
                    vo.conviction_flip_counter = 0;
                }

                if vo.conviction_flip_counter >= config.conviction_exit_bars {
                    close_position_impl(&config, vo, current_mid, ts, ExitReason::ConvictionExit);
                    closed_any = true;
                }
            }
        }

        closed_any
    }'''

    code = code.replace(old_fn, new_fn)

    # 8. Update the signal flip check call site to also trigger for conviction_exit_bars
    code = code.replace(
        "                    if has_position && sim_config.signal_flip_exit {",
        "                    if has_position && (sim_config.signal_flip_exit || sim_config.conviction_exit_bars > 0) {"
    )

    with open(FILL_SIM, 'w') as f:
        f.write(code)

    print(f"Patched {FILL_SIM}")


def patch_fill_sim_main():
    with open(FILL_SIM_MAIN, 'r') as f:
        code = f.read()

    # 1. Add CLI args after mae_exit_hold_sec
    code = code.replace(
        '    /// Suppress progress output\n'
        '    #[arg(long)]\n'
        '    quiet: bool,',
        '    /// Conviction exit: number of consecutive bars the signal must be opposite\n'
        '    /// before triggering exit. 0 = disabled. 100 = 10 seconds.\n'
        '    /// When >0, replaces instant signal-flip-exit with delayed version.\n'
        '    #[arg(long, default_value_t = 0)]\n'
        '    conviction_exit_bars: u64,\n'
        '\n'
        '    /// Conviction exit: minimum |z-score| of opposite signal to count.\n'
        '    /// 0.0 = any opposite signal. 1.0 = require |prediction| >= 1.0.\n'
        '    #[arg(long, default_value_t = 0.0)]\n'
        '    conviction_exit_mag: f64,\n'
        '\n'
        '    /// Suppress progress output\n'
        '    #[arg(long)]\n'
        '    quiet: bool,'
    )

    # 2. Add to ConfigJson struct
    code = code.replace(
        '    #[serde(default)]\n'
        '    mae_exit_hold_sec: Option<f64>,\n'
        '}',
        '    #[serde(default)]\n'
        '    mae_exit_hold_sec: Option<f64>,\n'
        '    #[serde(default)]\n'
        '    conviction_exit_bars: Option<u64>,\n'
        '    #[serde(default)]\n'
        '    conviction_exit_mag: Option<f64>,\n'
        '}'
    )

    # 3. Add config file mapping (after mae_exit_hold_sec mapping)
    code = code.replace(
        '        if let Some(v) = cfg.mae_exit_hold_sec { sim_config.mae_exit_hold_sec = v; }\n'
        '    }',
        '        if let Some(v) = cfg.mae_exit_hold_sec { sim_config.mae_exit_hold_sec = v; }\n'
        '        if let Some(v) = cfg.conviction_exit_bars { sim_config.conviction_exit_bars = v; }\n'
        '        if let Some(v) = cfg.conviction_exit_mag { sim_config.conviction_exit_mag = v; }\n'
        '    }'
    )

    # 4. Add CLI override mapping (after mae_exit_hold_sec)
    code = code.replace(
        '    if args.mae_exit_ticks > 0.0 {\n'
        '        sim_config.mae_exit_ticks = args.mae_exit_ticks;',
        '    if args.conviction_exit_bars > 0 {\n'
        '        sim_config.conviction_exit_bars = args.conviction_exit_bars;\n'
        '        sim_config.conviction_exit_mag = args.conviction_exit_mag;\n'
        '    }\n'
        '    if args.mae_exit_ticks > 0.0 {\n'
        '        sim_config.mae_exit_ticks = args.mae_exit_ticks;'
    )

    with open(FILL_SIM_MAIN, 'w') as f:
        f.write(code)

    print(f"Patched {FILL_SIM_MAIN}")


if __name__ == "__main__":
    if not os.path.exists(FILL_SIM):
        print(f"ERROR: {FILL_SIM} not found. Run from rust_cache_builder dir.")
        sys.exit(1)

    # Backup originals
    for f in [FILL_SIM, FILL_SIM_MAIN]:
        backup = f + ".bak"
        if not os.path.exists(backup):
            import shutil
            shutil.copy2(f, backup)
            print(f"Backed up {f} -> {backup}")

    patch_fill_sim()
    patch_fill_sim_main()

    print("\nDone! Now run: cargo build --release --bin fill_sim_cli")
    print("\nNew CLI args:")
    print("  --conviction-exit-bars 100   # 10 seconds (100ms bars)")
    print("  --conviction-exit-mag 0.0    # any opposite signal")
    print("\nExample:")
    print("  fill_sim_cli --mbo-file ... --predictions ... --conviction-exit-bars 100 --output ...")
