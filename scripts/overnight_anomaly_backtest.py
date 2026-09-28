#!/usr/bin/env python3
"""
Overnight Return Anomaly Backtest — Growth Stocks
==================================================
Tests whether buying at close and selling at open generates alpha,
especially on growth/tech stocks available on Robinhood.

6 Variants tested against 5-gate validation framework.
OOT period: Jan 2022 – Jul 2026.

Author: Claude (quant research)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'SNOW', 'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS',
    'COIN', 'SQ', 'RBLX', 'RIVN', 'UBER', 'LYFT', 'ROKU', 'NET',
    'DDOG', 'TTD', 'SHOP', 'SE', 'MELI', 'NU'
]

START_DATE = '2021-06-01'   # extra lookback for indicators
OOT_START = '2022-01-01'
END_DATE = '2026-07-28'
CAPITAL = 645.0
STOCK_SLIP_BPS = 2          # 0.02% each way
N_PERMUTATIONS = 1000
REGIME_GAP_THRESH = 0.50
SHARPE_THRESH = 0.50
MAX_DD_THRESH = -0.50
MIN_TRADES = 20

# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download OHLC for universe + SPY (for regime)."""
    tickers = UNIVERSE + ['SPY', '^VIX']
    print(f"Downloading data for {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[ticker] = df
        except Exception as e:
            print(f"  WARN: {ticker} failed: {e}")
    print(f"  Downloaded {len(data)} tickers successfully.")
    return data


def build_regime(spy_df):
    """Bull = SPY above 200-SMA, Bear = below."""
    close = spy_df['Close'].squeeze()
    sma200 = close.rolling(200).mean()
    regime = pd.Series('bull', index=spy_df.index)
    bear_mask = close < sma200
    regime.loc[bear_mask] = 'bear'
    return regime


# ─── Helper: overnight return for a single stock ────────────────────────────

def compute_overnight_returns(df):
    """close-to-open return = Open[t+1] / Close[t] - 1."""
    close = df['Close'].values.flatten()
    opn = df['Open'].values.flatten()
    # overnight return: buy at close[t], sell at open[t+1]
    overnight_ret = opn[1:] / close[:-1] - 1.0
    dates = df.index[:-1]  # entry date
    return pd.Series(overnight_ret, index=dates, name='overnight_ret')


# ─── Variant Strategies ─────────────────────────────────────────────────────

def variant_a_basic(data, oot_start):
    """A) Buy all stocks at close, sell at open. Equal weight."""
    daily_rets = []
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker]
        ret = compute_overnight_returns(df)
        ret = ret[ret.index >= oot_start]
        # apply slippage: 0.02% each way = 0.04% round trip
        ret = ret - 0.0004
        daily_rets.append(ret)
    if not daily_rets:
        return pd.Series(dtype=float), 0
    # equal weight across all stocks each day
    combined = pd.concat(daily_rets, axis=1)
    port_ret = combined.mean(axis=1)
    n_trades = combined.notna().sum().sum()
    return port_ret.dropna(), int(n_trades)


def variant_b_momentum(data, oot_start):
    """B) Only buy stocks with positive 20d momentum (close > close_20d_ago)."""
    daily_rets = []
    trade_count = 0
    all_dates = None

    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker]
        close = df['Close'].squeeze()
        mom20 = close / close.shift(20) - 1.0
        ret = compute_overnight_returns(df)

        # filter: only enter when 20d momentum > 0
        mask = mom20.reindex(ret.index) > 0
        ret_filtered = ret.where(mask, 0.0)
        ret_filtered = ret_filtered - 0.0004 * mask.astype(float)
        ret_filtered = ret_filtered[ret_filtered.index >= oot_start]
        trade_count += mask.reindex(ret_filtered.index).fillna(False).sum()
        daily_rets.append(ret_filtered)

    if not daily_rets:
        return pd.Series(dtype=float), 0
    combined = pd.concat(daily_rets, axis=1)
    port_ret = combined.mean(axis=1)
    return port_ret.dropna(), int(trade_count)


def variant_c_volatility(data, oot_start):
    """C) Only buy stocks with ATR(14)/price > 2%."""
    daily_rets = []
    trade_count = 0

    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker]
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        close = df['Close'].squeeze()

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_pct = atr14 / close

        ret = compute_overnight_returns(df)
        mask = atr_pct.reindex(ret.index) > 0.02
        ret_filtered = ret.where(mask, 0.0)
        ret_filtered = ret_filtered - 0.0004 * mask.astype(float)
        ret_filtered = ret_filtered[ret_filtered.index >= oot_start]
        trade_count += mask.reindex(ret_filtered.index).fillna(False).sum()
        daily_rets.append(ret_filtered)

    if not daily_rets:
        return pd.Series(dtype=float), 0
    combined = pd.concat(daily_rets, axis=1)
    port_ret = combined.mean(axis=1)
    return port_ret.dropna(), int(trade_count)


def variant_d_top5(data, oot_start):
    """D) Buy only the 5 stocks with strongest recent (10d) overnight returns."""
    # First build overnight return series for all stocks
    all_rets = {}
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker]
        ret = compute_overnight_returns(df)
        all_rets[ticker] = ret

    if not all_rets:
        return pd.Series(dtype=float), 0

    ret_df = pd.DataFrame(all_rets)
    # rolling 10d average overnight return
    rolling_avg = ret_df.rolling(10).mean()

    # each day, pick top-5 by rolling avg
    oot_mask = ret_df.index >= oot_start
    ret_oot = ret_df[oot_mask]
    rolling_oot = rolling_avg[oot_mask]

    port_rets = []
    trade_count = 0
    for date in ret_oot.index:
        scores = rolling_oot.loc[date].dropna()
        if len(scores) < 5:
            continue
        top5 = scores.nlargest(5).index
        day_ret = ret_oot.loc[date, top5].mean() - 0.0004
        port_rets.append((date, day_ret))
        trade_count += 5

    if not port_rets:
        return pd.Series(dtype=float), 0
    port_ret = pd.DataFrame(port_rets, columns=['date', 'ret']).set_index('date')['ret']
    return port_ret, int(trade_count)


def variant_e_options(data, oot_start):
    """E) Buy ATM calls at close, sell at open. Model with spread cost."""
    # Model: overnight call return ≈ delta * stock_overnight_return
    # ATM call delta ≈ 0.50
    # Cost: $0.65/contract each way = $1.30 RT per contract
    # Bid-ask spread: ~5% of premium each way = 10% RT
    # For a $645 account split across ~5 positions = ~$129/position
    # With ~$3 premium on growth stocks, ~43 shares equivalent per contract
    # But we can only buy 1-2 contracts per position realistically

    daily_rets = []
    trade_count = 0

    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker]
        close = df['Close'].squeeze()
        ret = compute_overnight_returns(df)

        # ATM call: delta ~0.50, gamma effect small for overnight
        # Premium ~ 1-3% of stock price for weekly ATM
        # Model premium as 2% of stock price
        premium_pct = 0.02

        # Call overnight return = delta * stock_ret / premium_pct
        # But capped by reality — call can't go below 0 (intrinsic floor near ATM)
        call_ret = 0.50 * ret / premium_pct

        # Costs: 5% spread each way (10% RT) + commission
        # Commission per contract: $1.30 RT. If premium=$3, that's ~$1.30/$300 = 0.43%
        # Total cost: 10% + 0.43% ≈ 10.4% of premium per RT
        cost_per_trade = 0.104
        call_ret = call_ret - cost_per_trade

        call_ret = call_ret[call_ret.index >= oot_start]
        trade_count += len(call_ret)
        daily_rets.append(call_ret)

    if not daily_rets:
        return pd.Series(dtype=float), 0
    combined = pd.concat(daily_rets, axis=1)
    # only hold ~5 positions at a time (capital constraint)
    # randomly sample or take first 5 each day for simplicity
    port_ret = combined.mean(axis=1)
    return port_ret.dropna(), int(trade_count)


def variant_f_vix_selective(data, oot_start):
    """F) Only enter overnight when VIX > 20."""
    vix_key = '^VIX'
    if vix_key not in data:
        print("  WARN: VIX data not available, skipping variant F")
        return pd.Series(dtype=float), 0

    vix_close = data[vix_key]['Close'].squeeze()

    daily_rets = []
    trade_count = 0

    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker]
        ret = compute_overnight_returns(df)

        # only enter when VIX > 20
        vix_aligned = vix_close.reindex(ret.index, method='ffill')
        mask = vix_aligned > 20
        ret_filtered = ret.where(mask, 0.0)
        ret_filtered = ret_filtered - 0.0004 * mask.astype(float)
        ret_filtered = ret_filtered[ret_filtered.index >= oot_start]
        trade_count += mask.reindex(ret_filtered.index).fillna(False).sum()
        daily_rets.append(ret_filtered)

    if not daily_rets:
        return pd.Series(dtype=float), 0
    combined = pd.concat(daily_rets, axis=1)
    port_ret = combined.mean(axis=1)
    return port_ret.dropna(), int(trade_count)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(returns, n_trades, regime_series):
    """Compute Sharpe, Sortino, MaxDD, PF, WR, regime gap, etc."""
    if len(returns) < 10 or n_trades < 1:
        return None

    ret = returns.values
    ann_factor = np.sqrt(252)

    sharpe = (ret.mean() / ret.std()) * ann_factor if ret.std() > 0 else 0.0

    downside = ret[ret < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-6
    sortino = (ret.mean() / downside_std) * ann_factor if downside_std > 0 else 0.0

    # Max drawdown
    cum = (1 + pd.Series(ret)).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Win rate and profit factor
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    wr = len(wins) / len(ret) if len(ret) > 0 else 0.0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and abs(losses.sum()) > 0 else float('inf')

    # Total return
    total_ret = cum.iloc[-1] - 1.0 if len(cum) > 0 else 0.0

    # Annual return
    n_years = len(ret) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0

    # Regime-stratified Sharpe
    regime_aligned = regime_series.reindex(returns.index, method='ffill')
    bull_mask = regime_aligned == 'bull'
    bear_mask = regime_aligned == 'bear'

    bull_ret = ret[bull_mask.values] if bull_mask.any() else np.array([0.0])
    bear_ret = ret[bear_mask.values] if bear_mask.any() else np.array([0.0])

    sharpe_bull = (bull_ret.mean() / bull_ret.std()) * ann_factor if bull_ret.std() > 0 and len(bull_ret) > 5 else 0.0
    sharpe_bear = (bear_ret.mean() / bear_ret.std()) * ann_factor if bear_ret.std() > 0 and len(bear_ret) > 5 else 0.0

    max_sharpe = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_sharpe if max_sharpe > 0 else 0.0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd, 4),
        'win_rate': round(wr, 4),
        'profit_factor': round(min(pf, 99.0), 3),
        'total_return': round(total_ret, 4),
        'ann_return': round(ann_ret, 4),
        'n_trades': n_trades,
        'n_days': len(ret),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 4),
        'bull_days': int(bull_mask.sum()),
        'bear_days': int(bear_mask.sum()),
    }


def permutation_test(returns, n_perms=1000):
    """Shuffle dates, compute Sharpe under null. Return p-value."""
    ret = returns.values
    if len(ret) < 10:
        return 1.0

    observed_sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0.0

    rng = np.random.default_rng(42)
    count_exceeds = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(ret)
        s = (shuffled.mean() / shuffled.std()) * np.sqrt(252) if shuffled.std() > 0 else 0.0
        if s >= observed_sharpe:
            count_exceeds += 1

    p_value = count_exceeds / n_perms
    return p_value


def validate_gates(metrics, p_value):
    """Check all 5 gates. Return dict of gate results."""
    gates = {}
    gates['sharpe_gt_0.5'] = {'value': metrics['sharpe'], 'pass': metrics['sharpe'] > SHARPE_THRESH}
    gates['perm_p_lt_0.05'] = {'value': round(p_value, 4), 'pass': p_value < 0.05}
    gates['regime_gap_lt_0.5'] = {'value': metrics['regime_gap'], 'pass': metrics['regime_gap'] < REGIME_GAP_THRESH}
    gates['max_dd_gt_neg50'] = {'value': metrics['max_dd'], 'pass': metrics['max_dd'] > MAX_DD_THRESH}
    gates['min_20_trades'] = {'value': metrics['n_trades'], 'pass': metrics['n_trades'] >= MIN_TRADES}
    gates['all_pass'] = all(g['pass'] for g in gates.values() if isinstance(g, dict))
    return gates


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("OVERNIGHT RETURN ANOMALY BACKTEST — Growth Stocks")
    print(f"OOT Period: {OOT_START} to {END_DATE}  |  Capital: ${CAPITAL}")
    print("=" * 80)

    data = download_data()

    if 'SPY' not in data:
        print("FATAL: SPY data not available. Cannot compute regime.")
        return

    regime = build_regime(data['SPY'])

    variants = {
        'A_basic_overnight': variant_a_basic,
        'B_momentum_filtered': variant_b_momentum,
        'C_volatility_filtered': variant_c_volatility,
        'D_top5_overnight': variant_d_top5,
        'E_options_version': variant_e_options,
        'F_vix_selective': variant_f_vix_selective,
    }

    results = {}

    for name, func in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Running Variant: {name}")
        print(f"{'─' * 60}")

        returns, n_trades = func(data, OOT_START)

        if len(returns) < 10:
            print(f"  SKIP: insufficient data ({len(returns)} days)")
            results[name] = {'status': 'SKIP', 'reason': 'insufficient data'}
            continue

        metrics = compute_metrics(returns, n_trades, regime)
        if metrics is None:
            results[name] = {'status': 'SKIP', 'reason': 'metrics computation failed'}
            continue

        print(f"  Computing permutation test ({N_PERMUTATIONS} iterations)...")
        p_value = permutation_test(returns, N_PERMUTATIONS)

        gates = validate_gates(metrics, p_value)

        # Dollar P&L estimate
        dollar_pnl = CAPITAL * metrics['total_return']

        result = {
            'metrics': metrics,
            'p_value': round(p_value, 4),
            'gates': gates,
            'dollar_pnl': round(dollar_pnl, 2),
            'status': 'PASS' if gates['all_pass'] else 'FAIL',
        }
        results[name] = result

        # Print summary
        status_str = "PASS ALL GATES" if gates['all_pass'] else "FAIL"
        print(f"\n  >>> {name}: {status_str}")
        print(f"  Sharpe: {metrics['sharpe']:>7.3f}  |  Sortino: {metrics['sortino']:>7.3f}")
        print(f"  MaxDD:  {metrics['max_dd']:>7.4f}  |  WR: {metrics['win_rate']:>7.4f}  |  PF: {metrics['profit_factor']:>7.3f}")
        print(f"  Total Return: {metrics['total_return']:>7.4f}  ({metrics['total_return']*100:.1f}%)")
        print(f"  Dollar P&L:   ${dollar_pnl:>8.2f}  (on ${CAPITAL})")
        print(f"  Trades: {metrics['n_trades']}  |  Days: {metrics['n_days']}")
        print(f"  Sharpe Bull: {metrics['sharpe_bull']:>6.3f}  |  Sharpe Bear: {metrics['sharpe_bear']:>6.3f}  |  Regime Gap: {metrics['regime_gap']:.4f}")
        print(f"  Permutation p-value: {p_value:.4f}")

        gate_detail = []
        for gname, gval in gates.items():
            if gname == 'all_pass':
                continue
            symbol = "PASS" if gval['pass'] else "FAIL"
            gate_detail.append(f"    {gname}: {gval['value']} [{symbol}]")
        print("  Gates:")
        for g in gate_detail:
            print(g)

    # ─── Summary Table ───────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = f"{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6} {'Perm-p':>7} {'RGap':>6} {'$PnL':>9} {'Status':>8}"
    print(header)
    print("-" * 100)

    for name, res in results.items():
        if res.get('status') == 'SKIP':
            print(f"{name:<25} {'SKIP':>7}")
            continue
        m = res['metrics']
        print(f"{name:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['max_dd']:>8.4f} {m['win_rate']:>6.1%} {m['profit_factor']:>6.2f} {res['p_value']:>7.4f} {m['regime_gap']:>6.4f} {res['dollar_pnl']:>9.2f} {res['status']:>8}")

    print("-" * 100)

    passing = [n for n, r in results.items() if r.get('status') == 'PASS']
    if passing:
        print(f"\nPASSING VARIANTS: {', '.join(passing)}")
    else:
        print(f"\nNO VARIANTS PASSED ALL 5 GATES.")

    # ─── Save results ────────────────────────────────────────────────────
    output_path = Path('/home/jupiter/Lvl3Quant/data/overnight_anomaly_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj

    serializable = make_serializable(results)

    save_obj = {
        'strategy': 'overnight_return_anomaly',
        'universe': UNIVERSE,
        'oot_period': f'{OOT_START} to {END_DATE}',
        'capital': CAPITAL,
        'run_date': datetime.now().isoformat(),
        'cost_model': {
            'stock_slippage_bps': STOCK_SLIP_BPS,
            'options_spread_pct': '5% each way',
            'options_commission': '$0.65/contract each way',
        },
        'validation_gates': {
            'sharpe_min': SHARPE_THRESH,
            'perm_p_max': 0.05,
            'regime_gap_max': REGIME_GAP_THRESH,
            'max_dd_min': MAX_DD_THRESH,
            'min_trades': MIN_TRADES,
        },
        'results': serializable,
    }

    with open(output_path, 'w') as f:
        json.dump(save_obj, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
