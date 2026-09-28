/// MBO Feature Expander — Binary entry point.
///
/// Usage:
///   mbo_feature_expander --cache-dir ./data/processed/medium_snapshots_cache
///                        --output-dir ./data/processed/mbo_features_cache
///   mbo_feature_expander --cache-dir ./cache --output-dir ./features --workers 8
///   mbo_feature_expander --cache-dir ./cache --output-dir ./features --date 2025-07-14

use anyhow::Result;
use clap::Parser;

mod mbo_feature_expander;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let args = mbo_feature_expander::ExpanderArgs::parse();
    mbo_feature_expander::run_expander(args)
}
