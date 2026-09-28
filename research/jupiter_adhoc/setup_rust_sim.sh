#!/bin/bash
# Setup and launch Rust fill_sim_cli runs on Jupiter
# Run as: bash /home/jupiter/setup_rust_sim.sh

set -e

BINARY="/home/jupiter/lvl3quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR="/home/jupiter/lvl3quant/data/mbo"
PRED_DIR="/home/jupiter/lvl3quant/data/processed/rust_predictions"
OUT_DIR="/home/jupiter/lvl3quant/production/results/rust_sim"
CONFIG_DIR="/home/jupiter/lvl3quant/production/configs"

mkdir -p "$OUT_DIR"
mkdir -p "$CONFIG_DIR"

# Create SimConfig files for top parameter combos from Saturn sweep
# Based on best results: h15000_t8_s8, h15000_t6_s8, h20000_t8_s7, h30000_t4_s7

cat > "$CONFIG_DIR/h15000_t8_s8.json" << 'EOF'
{
  "hold_ms": 15000,
  "trailing_stop_ticks": 8.0,
  "take_profit_ticks": null,
  "max_wait_bars": 100,
  "market_exit_spread_cost": 0.5,
  "commission_ticks": 0.24,
  "eod_close": true,
  "signal_flip_exit": false
}
EOF

cat > "$CONFIG_DIR/h15000_t6_s8.json" << 'EOF'
{
  "hold_ms": 15000,
  "trailing_stop_ticks": 6.0,
  "take_profit_ticks": null,
  "max_wait_bars": 100,
  "market_exit_spread_cost": 0.5,
  "commission_ticks": 0.24,
  "eod_close": true,
  "signal_flip_exit": false
}
EOF

cat > "$CONFIG_DIR/h20000_t8_s7.json" << 'EOF'
{
  "hold_ms": 20000,
  "trailing_stop_ticks": 8.0,
  "take_profit_ticks": null,
  "max_wait_bars": 100,
  "market_exit_spread_cost": 0.5,
  "commission_ticks": 0.24,
  "eod_close": true,
  "signal_flip_exit": false
}
EOF

cat > "$CONFIG_DIR/h30000_t4_s7.json" << 'EOF'
{
  "hold_ms": 30000,
  "trailing_stop_ticks": 4.0,
  "take_profit_ticks": null,
  "max_wait_bars": 100,
  "market_exit_spread_cost": 0.5,
  "commission_ticks": 0.24,
  "eod_close": true,
  "signal_flip_exit": false
}
EOF

cat > "$CONFIG_DIR/h10000_t5_default.json" << 'EOF'
{
  "hold_ms": 10000,
  "trailing_stop_ticks": 5.0,
  "take_profit_ticks": null,
  "max_wait_bars": 100,
  "market_exit_spread_cost": 0.5,
  "commission_ticks": 0.24,
  "eod_close": true,
  "signal_flip_exit": false
}
EOF

echo "Config files created in $CONFIG_DIR"
ls -la "$CONFIG_DIR/"

# Check if prediction files are ready
echo ""
echo "Prediction files in $PRED_DIR:"
ls "$PRED_DIR/" 2>/dev/null | head -10 || echo "  (none yet)"

echo ""
echo "MBO files available:"
ls "$MBO_DIR/" | wc -l
