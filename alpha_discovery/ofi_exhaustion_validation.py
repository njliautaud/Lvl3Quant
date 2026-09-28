"""
OFI Exhaustion Signal — IS/OOT Validation + Threshold Sweep
Counter-trend exhaustion: fade OFI spikes aligned with recent trend direction
"""
import pandas as pd
import numpy as np
import glob
import sys

def load_data():
    files = sorted(glob.glob('/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1/*.parquet'))
    print(f'Loading {len(files)} files...')
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['ts_minute'] = pd.to_datetime(df['ts_minute'], utc=True)
    df = df.sort_values('ts_minute').reset_index(drop=True)
    df['date'] = df['ts_minute'].dt.date.astype(str)
    return df

def compute_features(df):
    ofi_col = 'ofi_1min'
    df['ofi_mean'] = df.groupby('date')[ofi_col].transform(lambda x: x.rolling(60, min_periods=20).mean())
    df['ofi_std'] = df.groupby('date')[ofi_col].transform(lambda x: x.rolling(60, min_periods=20).std())
    df['ofi_z'] = (df[ofi_col] - df['ofi_mean']) / df['ofi_std'].clip(lower=1e-6)

    df['vol_mean'] = df.groupby('date')['volume'].transform(lambda x: x.rolling(60, min_periods=20).mean())
    df['vol_std'] = df.groupby('date')['volume'].transform(lambda x: x.rolling(60, min_periods=20).std())
    df['vol_z'] = (df['volume'] - df['vol_mean']) / df['vol_std'].clip(lower=1e-6)

    df['ret_30m'] = df.groupby('date')['close'].transform(lambda x: x.pct_change(30))

    # Multiple forward horizons
    for h in [5, 10, 15, 30]:
        df[f'fwd_{h}m'] = df.groupby('date')['close'].transform(lambda x: x.shift(-h) / x - 1)

    return df

def calc_daily_sharpe(valid, long_mask, short_mask, date_set, fwd_col, avg_price, cost_ticks):
    dpnl = []
    for d in date_set:
        dm = valid['date'] == d
        dl = valid.loc[dm & long_mask, fwd_col].values
        ds = -valid.loc[dm & short_mask, fwd_col].values
        dt = np.concatenate([dl, ds]) if len(dl)+len(ds) > 0 else np.array([])
        if len(dt) > 0:
            dpnl.append((dt * avg_price / 0.25 - cost_ticks).sum() * 12.50)
    dpnl = np.array(dpnl)
    if len(dpnl) > 1 and np.std(dpnl) > 0:
        sh = np.mean(dpnl) / np.std(dpnl) * np.sqrt(252)
        down = dpnl[dpnl < 0]
        so = np.mean(dpnl) / np.std(down) * np.sqrt(252) if len(down) > 0 and np.std(down) > 0 else 0
        return sh, so, dpnl
    return 0, 0, dpnl

def main():
    df = load_data()
    print(f'Rows: {len(df)}, dates: {df.date.nunique()}, range: {df.date.min()} - {df.date.max()}')

    df = compute_features(df)
    valid = df.dropna(subset=['ofi_z','vol_z','ret_30m','fwd_15m']).copy()
    print(f'Valid rows: {len(valid)}')

    # Regime analysis
    daily = df.groupby('date')['close'].last().reset_index()
    daily['daily_ret'] = daily['close'].pct_change()
    daily['regime'] = np.where(daily['daily_ret'] > 0.001, 'green',
                      np.where(daily['daily_ret'] < -0.001, 'red', 'flat'))

    dates = sorted(valid['date'].unique())
    split_idx = int(len(dates) * 0.7)
    is_dates = dates[:split_idx]
    oot_dates = dates[split_idx:]

    is_daily = daily[daily['date'].isin(is_dates)]
    oot_daily = daily[daily['date'].isin(oot_dates)]

    print(f'\n--- Regime Distribution ---')
    print(f'IS  ({min(is_dates)} to {max(is_dates)}, {len(is_dates)}d): '
          f'Green {(is_daily.regime=="green").sum()} / Red {(is_daily.regime=="red").sum()} / Flat {(is_daily.regime=="flat").sum()}'
          f' | Avg ret: {is_daily.daily_ret.mean()*100:.3f}%')
    print(f'OOT ({min(oot_dates)} to {max(oot_dates)}, {len(oot_dates)}d): '
          f'Green {(oot_daily.regime=="green").sum()} / Red {(oot_daily.regime=="red").sum()} / Flat {(oot_daily.regime=="flat").sum()}'
          f' | Avg ret: {oot_daily.daily_ret.mean()*100:.3f}%')

    # Threshold sweep
    cost_ticks = 2.376
    avg_price = valid['close'].mean()

    header = f"{'OFI_z':>6} {'Vol_z':>6} {'Hold':>5} | {'N':>5} {'WR%':>5} {'PF':>5} {'All_Sh':>7} | {'IS_Sh':>7} {'OOT_Sh':>7} {'Gap':>5}"
    print(f'\n--- Threshold Sweep ---')
    print(header)
    print('-' * len(header))

    best_robust = None
    best_score = -999

    for ofi_thresh in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for vol_thresh in [0.5, 1.0, 1.5, 2.0]:
            for fwd_col in ['fwd_5m', 'fwd_10m', 'fwd_15m', 'fwd_30m']:
                hold = fwd_col.replace('fwd_','').replace('m','')

                long_mask = (valid['ofi_z'] < -ofi_thresh) & (valid['vol_z'] > vol_thresh) & (valid['ret_30m'] < 0)
                short_mask = (valid['ofi_z'] > ofi_thresh) & (valid['vol_z'] > vol_thresh) & (valid['ret_30m'] > 0)

                trades_all = np.concatenate([
                    valid.loc[long_mask, fwd_col].values,
                    -valid.loc[short_mask, fwd_col].values
                ])
                if len(trades_all) < 30:
                    continue

                net = trades_all * avg_price / 0.25 - cost_ticks
                wr = (net > 0).sum() / len(net) * 100
                gw = net[net>0].sum()
                gl = abs(net[net<0].sum())
                pf = gw/gl if gl > 0 else 0

                sh_all, _, _ = calc_daily_sharpe(valid, long_mask, short_mask, set(dates), fwd_col, avg_price, cost_ticks)
                sh_is, _, _ = calc_daily_sharpe(valid, long_mask, short_mask, set(is_dates), fwd_col, avg_price, cost_ticks)
                sh_oot, _, _ = calc_daily_sharpe(valid, long_mask, short_mask, set(oot_dates), fwd_col, avg_price, cost_ticks)

                # Regime gap
                gap = abs(sh_is - sh_oot) / max(abs(sh_is), abs(sh_oot), 0.01)

                # Robustness score: want both IS and OOT positive, low gap
                robustness = min(sh_is, sh_oot) * (1 - gap*0.5)

                if robustness > best_score and len(trades_all) >= 50:
                    best_score = robustness
                    best_robust = (ofi_thresh, vol_thresh, hold, len(trades_all), wr, pf, sh_all, sh_is, sh_oot)

                # Only print interesting rows
                if sh_all > 0.5 or (sh_is > 0 and sh_oot > 0):
                    print(f'{ofi_thresh:6.1f} {vol_thresh:6.1f} {hold:>5} | {len(trades_all):5d} {wr:5.1f} {pf:5.2f} {sh_all:7.2f} | {sh_is:7.2f} {sh_oot:7.2f} {gap:5.2f}')

    if best_robust:
        print(f'\n=== MOST ROBUST CONFIG ===')
        ofi_t, vol_t, hold, n, wr, pf, sh, sh_is, sh_oot = best_robust
        print(f'OFI_z={ofi_t}, Vol_z={vol_t}, Hold={hold}m')
        print(f'N={n}, WR={wr:.1f}%, PF={pf:.2f}, Sharpe(all)={sh:.2f}')
        print(f'IS Sharpe={sh_is:.2f}, OOT Sharpe={sh_oot:.2f}')

        # Detailed analysis of best config
        fwd_col = f'fwd_{hold}m'
        long_mask = (valid['ofi_z'] < -ofi_t) & (valid['vol_z'] > vol_t) & (valid['ret_30m'] < 0)
        short_mask = (valid['ofi_z'] > ofi_t) & (valid['vol_z'] > vol_t) & (valid['ret_30m'] > 0)

        # Per-regime analysis
        print(f'\n--- Per-Regime Breakdown (best config) ---')
        regime_map = dict(zip(daily['date'], daily['regime']))

        for regime in ['green', 'red', 'flat']:
            regime_dates = [d for d in dates if regime_map.get(d) == regime]
            if not regime_dates:
                continue
            sh_r, so_r, dpnl_r = calc_daily_sharpe(valid, long_mask, short_mask, set(regime_dates), fwd_col, avg_price, cost_ticks)
            dm = valid['date'].isin(regime_dates)
            n_trades = (dm & (long_mask | short_mask)).sum()
            print(f'{regime:>5}: {len(regime_dates):3d} days, {n_trades:4d} trades, Sharpe={sh_r:.2f}, Sortino={so_r:.2f}')

    # Also test: pure mean-reversion without trend filter
    print(f'\n\n--- PURE OFI MEAN-REVERSION (no trend filter) ---')
    print(f"{'OFI_z':>6} {'Vol_z':>6} {'Hold':>5} | {'N':>5} {'WR%':>5} {'PF':>5} {'All_Sh':>7} | {'IS_Sh':>7} {'OOT_Sh':>7}")

    for ofi_thresh in [2.0, 2.5, 3.0, 3.5]:
        for vol_thresh in [0.5, 1.0, 1.5]:
            for fwd_col in ['fwd_5m', 'fwd_10m', 'fwd_15m']:
                hold = fwd_col.replace('fwd_','').replace('m','')

                # Pure fade: just fade extreme OFI regardless of trend
                long_mask = (valid['ofi_z'] < -ofi_thresh) & (valid['vol_z'] > vol_thresh)
                short_mask = (valid['ofi_z'] > ofi_thresh) & (valid['vol_z'] > vol_thresh)

                trades_all = np.concatenate([
                    valid.loc[long_mask, fwd_col].values,
                    -valid.loc[short_mask, fwd_col].values
                ])
                if len(trades_all) < 50:
                    continue

                net = trades_all * avg_price / 0.25 - cost_ticks
                wr = (net > 0).sum() / len(net) * 100
                gw = net[net>0].sum(); gl = abs(net[net<0].sum())
                pf = gw/gl if gl > 0 else 0

                sh_all, _, _ = calc_daily_sharpe(valid, long_mask, short_mask, set(dates), fwd_col, avg_price, cost_ticks)
                sh_is, _, _ = calc_daily_sharpe(valid, long_mask, short_mask, set(is_dates), fwd_col, avg_price, cost_ticks)
                sh_oot, _, _ = calc_daily_sharpe(valid, long_mask, short_mask, set(oot_dates), fwd_col, avg_price, cost_ticks)

                if sh_all > 0.3 or (sh_is > 0 and sh_oot > 0):
                    print(f'{ofi_thresh:6.1f} {vol_thresh:6.1f} {hold:>5} | {len(trades_all):5d} {wr:5.1f} {pf:5.2f} {sh_all:7.2f} | {sh_is:7.2f} {sh_oot:7.2f}')

if __name__ == '__main__':
    main()
