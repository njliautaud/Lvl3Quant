#!/usr/bin/env python3
"""
Triple Confluence Scanner v1
==============================
HYPOTHESIS (HC #735 continuation):
  Vol compression is the universal edge concentrator. Every passing strategy
  uses it. This script tests whether combining ALL THREE proven signals
  produces stronger edge than any pair or individual.

THREE SIGNALS (must fire same day, same stock):
  1. OVERSOLD: RSI(5) < 20 OR single-day drop > 3%
  2. VOL COMPRESSED: 20d realized vol < 10th percentile of stock's own 252d history
  3. VOLUME CLIMAX: Daily volume > 2x 20-day average

VARIANTS: 7 signal combos × 2 hold periods = 14 variants
  Singles: oversold, vol_compressed, volume_climax (3)
  Pairs: OS+VC, OS+VClx, VC+VClx (3)
  Triple: OS+VC+VClx (1)
  Hold periods: 5d, 10d

VALIDATION GATES (all mandatory):
  1. Permutation test: 200 shuffles, p < 0.05
  2. Regime gap: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) < 0.50
  3. Per-year consistency: profitable in > 60% of years

UNIVERSE: ~200 S&P 500 stocks, 10+ years daily OHLCV from yfinance
WINDOW: SLIDING lookback (HC #0)
"""

import os, sys, json, time, warnings, traceback
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Configuration ───
START_DATE = '2014-01-01'
END_DATE = '2026-07-22'
RSI_PERIOD = 5
RSI_THRESHOLD = 20
DROP_THRESHOLD = 0.03       # 3% single-day drop
VOL_LOOKBACK = 20            # 20d realized vol
VOL_PCTL_HISTORY = 252       # 1-year for percentile
COMPRESSION_PCT = 10         # 10th percentile
VOLUME_AVG_LOOKBACK = 20     # 20-day avg volume
VOLUME_MULTIPLIER = 2.0      # 2x average

HOLD_PERIODS = [5, 10]
N_PERMS = 200
MIN_TRADES = 30
MIN_YEAR_PCT = 0.60

OUTPUT_DIR = Path('/home/nick/Lvl3Quant/output/triple_confluence_v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── S&P 500 Top ~200 (hardcoded) ───
UNIVERSE = [
    # Technology
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AVGO','ORCL','CRM',
    'AMD','ADBE','ACN','CSCO','INTC','IBM','TXN','QCOM','NOW','INTU',
    'AMAT','ADI','LRCX','KLAC','SNPS','CDNS','MRVL','FTNT','PANW','CRWD',
    'WDAY','ZS','DDOG','HUBS','NET','PYPL','SQ','SHOP','ABNB','PLTR',
    # Financials
    'JPM','BAC','WFC','GS','MS','C','BLK','SCHW','AXP','BK',
    'USB','PNC','TFC','COF','CME','ICE','MCO','SPGI','MSCI','FIS',
    'FISV','ADP','MMC','AON','CB','AFL','MET','PRU','ALL','TRV',
    # Healthcare
    'UNH','JNJ','LLY','PFE','MRK','ABBV','ABT','TMO','DHR','BMY',
    'AMGN','MDT','ISRG','ELV','SYK','GILD','VRTX','REGN','BSX','ZBH',
    'BDX','IQV','A','DXCM','IDXX','PODD','ALGN','HOLX','MTD','WAT',
    # Consumer Discretionary
    'HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG','MAR','HLT',
    'ORLY','AZO','ROST','DG','DLTR','BBY','DHI','LEN','PHM','LULU',
    'DECK','TPR','RL','EBAY','ETSY','GRMN','GPC','POOL','ON','NVR',
    # Consumer Staples
    'PG','KO','PEP','COST','WMT','PM','MO','CL','KMB','GIS',
    'HSY','MNST','STZ','MKC','CHD','CLX','EL','KHC','KDP','MDLZ',
    # Industrials
    'HON','UNP','UPS','CAT','RTX','DE','BA','LMT','GD','NOC',
    'GE','MMM','EMR','ROK','ITW','PH','ETN','IR','CARR','OTIS',
    'CSX','NSC','FDX','DAL','UAL','ODFL','SAIA','WAB','AME','DOV',
    # Energy
    'XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','DVN',
    'HES','HAL','BKR','FANG','CTRA','MRO','WMB','TRGP','OVV','APA',
    # Materials + Utilities + REITs
    'LIN','APD','ECL','SHW','DD','NEM','FCX','NUE','STLD','CF',
    'NEE','DUK','SO','D','AEP','SRE','XEL','WEC','ED','AEE',
    'PLD','AMT','CCI','EQIX','PSA','O','SPG','DLR','WELL','AVB',
]

# ─── Helpers ───

def compute_rsi(prices, period=14):
    """RSI calculation."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_realized_vol(prices, lookback=20):
    """Annualized realized vol from log returns."""
    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(252)
    return vol


def compute_vol_percentile(vol_series, hist_lookback=252, pct=10):
    """Is current vol below the pct-th percentile of its own hist_lookback history?"""
    result = pd.Series(False, index=vol_series.index)
    for i in range(hist_lookback, len(vol_series)):
        window = vol_series.iloc[i - hist_lookback:i]
        threshold = np.nanpercentile(window.dropna(), pct)
        if not np.isnan(vol_series.iloc[i]) and vol_series.iloc[i] <= threshold:
            result.iloc[i] = True
    return result


def get_spy_regime(spy_close):
    """Classify each day as green (up) or red (down) based on SPY returns."""
    ret = spy_close.pct_change()
    regime = pd.Series('flat', index=spy_close.index)
    regime[ret > 0] = 'green'
    regime[ret < 0] = 'red'
    return regime


def sharpe_from_returns(returns):
    """Annualized Sharpe from a series of trade returns."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def profit_factor(returns):
    """Profit factor = sum(gains) / abs(sum(losses))."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return 999.0 if gains > 0 else 0.0
    return gains / losses


# ═══════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════

def download_data():
    """Download OHLCV for universe + SPY."""
    import yfinance as yf

    tickers = UNIVERSE + ['SPY']
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}...")

    all_data = {}
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            df = yf.download(batch, start=START_DATE, end=END_DATE,
                           group_by='ticker', progress=False, threads=True)
            if len(batch) == 1:
                ticker = batch[0]
                if 'Close' in df.columns and len(df) > 252:
                    all_data[ticker] = df[['Open','High','Low','Close','Volume']].copy()
            else:
                for ticker in batch:
                    try:
                        td = df[ticker][['Open','High','Low','Close','Volume']].copy()
                        td.dropna(subset=['Close'], inplace=True)
                        if len(td) > 252:
                            all_data[ticker] = td
                    except:
                        pass
        except Exception as e:
            print(f"  Batch {i//batch_size} error: {e}")
        if i % 100 == 0 and i > 0:
            print(f"  Downloaded {len(all_data)} tickers so far...")
            time.sleep(1)

    print(f"Successfully downloaded {len(all_data)} tickers")
    return all_data


# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_signals(all_data, spy_data):
    """Generate per-stock, per-day signal flags for all three conditions."""
    print("\nGenerating signals for each stock...")

    signals = {}  # ticker -> DataFrame with columns: oversold, vol_compressed, volume_climax, daily_ret

    for ticker, df in all_data.items():
        if ticker == 'SPY':
            continue
        try:
            close = df['Close'].astype(float)
            volume = df['Volume'].astype(float)

            # 1. Oversold: RSI(5) < 20 OR single-day drop > 3%
            rsi = compute_rsi(close, RSI_PERIOD)
            daily_ret = close.pct_change()
            oversold = (rsi < RSI_THRESHOLD) | (daily_ret < -DROP_THRESHOLD)

            # 2. Vol compressed: 20d realized vol < 10th percentile of 252d history
            rvol = compute_realized_vol(close, VOL_LOOKBACK)
            vol_compressed = compute_vol_percentile(rvol, VOL_PCTL_HISTORY, COMPRESSION_PCT)

            # 3. Volume climax: daily volume > 2x 20-day average
            vol_avg = volume.rolling(VOLUME_AVG_LOOKBACK, min_periods=VOLUME_AVG_LOOKBACK).mean()
            volume_climax = volume > (VOLUME_MULTIPLIER * vol_avg)

            sig_df = pd.DataFrame({
                'oversold': oversold,
                'vol_compressed': vol_compressed,
                'volume_climax': volume_climax,
                'close': close,
                'daily_ret': daily_ret,
            }, index=df.index)
            sig_df.dropna(subset=['close'], inplace=True)
            signals[ticker] = sig_df

        except Exception as e:
            pass  # skip problematic tickers

    print(f"Signals generated for {len(signals)} tickers")
    return signals


# ═══════════════════════════════════════════════════════════════════
# TRADE GENERATION & EVALUATION
# ═══════════════════════════════════════════════════════════════════

COMBOS = {
    # Singles
    'oversold_only': ['oversold'],
    'vol_compressed_only': ['vol_compressed'],
    'volume_climax_only': ['volume_climax'],
    # Pairs
    'oversold+vol_compressed': ['oversold', 'vol_compressed'],
    'oversold+volume_climax': ['oversold', 'volume_climax'],
    'vol_compressed+volume_climax': ['vol_compressed', 'volume_climax'],
    # Triple
    'TRIPLE_ALL': ['oversold', 'vol_compressed', 'volume_climax'],
}


def generate_trades(signals, spy_regime, combo_name, combo_signals, hold_period):
    """Generate trades for a specific signal combination and hold period."""
    trades = []

    for ticker, sig_df in signals.items():
        # Align with SPY regime
        aligned_regime = spy_regime.reindex(sig_df.index).fillna('flat')

        # Find entry days: all combo signals must be True
        entry_mask = sig_df[combo_signals[0]].copy()
        for s in combo_signals[1:]:
            entry_mask = entry_mask & sig_df[s]

        entry_dates = sig_df.index[entry_mask]

        for entry_date in entry_dates:
            entry_idx = sig_df.index.get_loc(entry_date)
            exit_idx = entry_idx + hold_period

            if exit_idx >= len(sig_df):
                continue

            entry_price = sig_df['close'].iloc[entry_idx]
            exit_price = sig_df['close'].iloc[exit_idx]
            ret = (exit_price - entry_price) / entry_price

            trades.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'exit_date': sig_df.index[exit_idx],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': ret,
                'regime': aligned_regime.iloc[entry_idx],
                'year': entry_date.year if hasattr(entry_date, 'year') else pd.Timestamp(entry_date).year,
            })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def evaluate_variant(trades_df, n_perms=200):
    """Evaluate a variant against all three gates."""
    if len(trades_df) < MIN_TRADES:
        return {
            'n_trades': len(trades_df),
            'pass': False,
            'reject_reason': f'insufficient trades ({len(trades_df)} < {MIN_TRADES})',
        }

    returns = trades_df['return'].values
    mean_ret = np.mean(returns)
    wr = np.mean(returns > 0)
    sharpe = sharpe_from_returns(pd.Series(returns))
    pf = profit_factor(pd.Series(returns))

    # ─── Gate 1: Permutation test ───
    observed_mean = mean_ret
    perm_count = 0
    shuffled = returns.copy()
    for _ in range(n_perms):
        np.random.shuffle(shuffled)
        # Permutation: shuffle entry-return assignment (breaks temporal structure)
        if np.mean(shuffled) >= observed_mean:
            perm_count += 1
    perm_p = (perm_count + 1) / (n_perms + 1)

    # ─── Gate 2: Regime gap ───
    green_rets = trades_df[trades_df['regime'] == 'green']['return']
    red_rets = trades_df[trades_df['regime'] == 'red']['return']

    sharpe_green = sharpe_from_returns(green_rets) if len(green_rets) >= 5 else 0.0
    sharpe_red = sharpe_from_returns(red_rets) if len(red_rets) >= 5 else 0.0

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0.0

    # ─── Gate 3: Per-year consistency ───
    yearly = trades_df.groupby('year')['return'].mean()
    years_profitable = (yearly > 0).sum()
    total_years = len(yearly)
    year_pct = years_profitable / total_years if total_years > 0 else 0.0

    # ─── Gates ───
    pass_perm = perm_p < 0.05
    pass_regime = regime_gap < 0.50
    pass_years = year_pct >= MIN_YEAR_PCT

    passed = pass_perm and pass_regime and pass_years

    reject_reasons = []
    if not pass_perm:
        reject_reasons.append(f'perm_p={perm_p:.3f}')
    if not pass_regime:
        reject_reasons.append(f'regime_gap={regime_gap:.3f}')
    if not pass_years:
        reject_reasons.append(f'year_pct={year_pct:.2f}')

    return {
        'n_trades': len(trades_df),
        'mean_return': float(mean_ret),
        'win_rate': float(wr),
        'sharpe': float(sharpe),
        'profit_factor': float(pf),
        'perm_p': float(perm_p),
        'sharpe_green': float(sharpe_green),
        'sharpe_red': float(sharpe_red),
        'regime_gap': float(regime_gap),
        'n_green_trades': len(green_rets),
        'n_red_trades': len(red_rets),
        'years_profitable': int(years_profitable),
        'total_years': int(total_years),
        'year_pct': float(year_pct),
        'yearly_returns': {str(k): float(v) for k, v in yearly.items()},
        'pass_perm': pass_perm,
        'pass_regime': pass_regime,
        'pass_years': pass_years,
        'pass': passed,
        'reject_reason': ', '.join(reject_reasons) if reject_reasons else 'PASS',
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("TRIPLE CONFLUENCE SCANNER v1")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Download data
    all_data = download_data()
    if 'SPY' not in all_data:
        print("FATAL: Could not download SPY data")
        sys.exit(1)

    spy_close = all_data['SPY']['Close'].astype(float)
    spy_regime = get_spy_regime(spy_close)

    # Generate signals
    signals = generate_signals(all_data, all_data['SPY'])

    # Run all variants
    results = {}
    pass_count = 0
    total_variants = len(COMBOS) * len(HOLD_PERIODS)

    print(f"\n{'='*70}")
    print(f"RUNNING {total_variants} VARIANTS ({len(COMBOS)} combos × {len(HOLD_PERIODS)} hold periods)")
    print(f"{'='*70}\n")

    for combo_name, combo_signals in COMBOS.items():
        for hold in HOLD_PERIODS:
            variant_name = f"{combo_name}_hold{hold}d"
            print(f"\n--- {variant_name} ---")

            trades_df = generate_trades(signals, spy_regime, combo_name, combo_signals, hold)
            print(f"  Trades: {len(trades_df)}")

            if len(trades_df) == 0:
                results[variant_name] = {
                    'combo': combo_name,
                    'signals': combo_signals,
                    'hold_period': hold,
                    'n_trades': 0,
                    'pass': False,
                    'reject_reason': 'no trades generated',
                }
                print(f"  SKIP: no trades")
                continue

            eval_result = evaluate_variant(trades_df, N_PERMS)
            eval_result['combo'] = combo_name
            eval_result['signals'] = combo_signals
            eval_result['hold_period'] = hold

            results[variant_name] = eval_result

            status = "PASS ✓" if eval_result['pass'] else f"FAIL ({eval_result['reject_reason']})"
            print(f"  Trades: {eval_result['n_trades']}, "
                  f"Sharpe: {eval_result.get('sharpe', 0):.2f}, "
                  f"WR: {eval_result.get('win_rate', 0):.1%}, "
                  f"PF: {eval_result.get('profit_factor', 0):.2f}, "
                  f"Perm-p: {eval_result.get('perm_p', 1):.3f}, "
                  f"Regime gap: {eval_result.get('regime_gap', 9):.3f}, "
                  f"Year%: {eval_result.get('year_pct', 0):.1%}")
            print(f"  → {status}")

            if eval_result['pass']:
                pass_count += 1

    # ─── Summary ───
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total variants: {total_variants}")
    print(f"Passed all gates: {pass_count}/{total_variants}")
    print(f"Elapsed: {elapsed:.0f}s")

    # Rank by Sharpe
    ranked = sorted(
        [(k, v) for k, v in results.items() if v.get('n_trades', 0) >= MIN_TRADES],
        key=lambda x: x[1].get('sharpe', 0),
        reverse=True
    )

    print(f"\n--- TOP 5 BY SHARPE (regardless of gates) ---")
    for name, r in ranked[:5]:
        status = "PASS" if r.get('pass') else "FAIL"
        print(f"  {name}: Sharpe={r.get('sharpe',0):.2f}, WR={r.get('win_rate',0):.1%}, "
              f"PF={r.get('profit_factor',0):.2f}, trades={r['n_trades']}, "
              f"regime_gap={r.get('regime_gap',9):.3f} [{status}]")

    print(f"\n--- PASSED VARIANTS ---")
    passed_variants = [(k, v) for k, v in results.items() if v.get('pass')]
    if not passed_variants:
        print("  None passed all gates.")
    else:
        for name, r in sorted(passed_variants, key=lambda x: x[1].get('sharpe', 0), reverse=True):
            print(f"  {name}: Sharpe={r.get('sharpe',0):.2f}, WR={r.get('win_rate',0):.1%}, "
                  f"PF={r.get('profit_factor',0):.2f}, trades={r['n_trades']}, "
                  f"perm_p={r.get('perm_p',1):.3f}, regime_gap={r.get('regime_gap',9):.3f}, "
                  f"year%={r.get('year_pct',0):.1%}")

    # Key comparison: does triple beat pairs?
    print(f"\n--- CONFLUENCE COMPARISON (10d hold) ---")
    for combo in ['oversold_only', 'vol_compressed_only', 'volume_climax_only',
                   'oversold+vol_compressed', 'oversold+volume_climax',
                   'vol_compressed+volume_climax', 'TRIPLE_ALL']:
        key = f"{combo}_hold10d"
        if key in results and results[key].get('n_trades', 0) >= MIN_TRADES:
            r = results[key]
            status = "PASS" if r.get('pass') else "FAIL"
            print(f"  {combo:35s}: Sharpe={r.get('sharpe',0):+.2f}, "
                  f"WR={r.get('win_rate',0):.1%}, trades={r['n_trades']:5d}, "
                  f"regime_gap={r.get('regime_gap',9):.3f} [{status}]")
        else:
            nt = results.get(key, {}).get('n_trades', 0)
            print(f"  {combo:35s}: trades={nt} (insufficient)")

    # ─── Save results ───
    output = {
        'metadata': {
            'script': 'triple_confluence_v1.py',
            'run_date': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'universe_size': len(signals),
            'date_range': f'{START_DATE} to {END_DATE}',
            'parameters': {
                'rsi_period': RSI_PERIOD,
                'rsi_threshold': RSI_THRESHOLD,
                'drop_threshold': DROP_THRESHOLD,
                'vol_lookback': VOL_LOOKBACK,
                'vol_pctl_history': VOL_PCTL_HISTORY,
                'compression_pct': COMPRESSION_PCT,
                'volume_avg_lookback': VOLUME_AVG_LOOKBACK,
                'volume_multiplier': VOLUME_MULTIPLIER,
                'n_perms': N_PERMS,
            },
        },
        'total_variants': total_variants,
        'passed_count': pass_count,
        'results': results,
    }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*70}")
    print(f"DONE — {datetime.now().isoformat()}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
