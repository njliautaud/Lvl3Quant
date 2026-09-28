#!/usr/bin/env python3
"""
Commodity Cross-Momentum Backtest
Tests 6 commodity strategies designed to be UNCORRELATED with equities.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────────
TICKERS = ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBB', 'PDBC']
BENCHMARKS = ['SPY', 'TIP', 'TLT']
ALL_TICKERS = list(set(TICKERS + BENCHMARKS))
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'
DOWNLOAD_START = '2020-01-01'  # extra history for indicators
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
PERM_ITERS = 500
ANNUALIZE = 252


# ── DATA DOWNLOAD ──────────────────────────────────────────────────────────
def download_data():
    print("Downloading data...")
    data = yf.download(ALL_TICKERS, start=DOWNLOAD_START, end=OOT_END,
                       auto_adjust=True, progress=False)
    close = data['Close'].copy()
    close = close.ffill().dropna(how='all')
    # Ensure all tickers present
    missing = [t for t in ALL_TICKERS if t not in close.columns]
    if missing:
        print(f"WARNING: Missing tickers: {missing}")
    return close


# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────
def compute_returns(prices):
    return prices.pct_change()

def sma(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def zscore(series, window):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s

def apply_slippage(returns, positions):
    """Apply slippage on position changes. Handles both Series and DataFrame positions."""
    pos_changes = positions.diff().abs()
    if isinstance(pos_changes, pd.DataFrame):
        slippage_cost = pos_changes.sum(axis=1) * SLIPPAGE_PCT
    else:
        slippage_cost = pos_changes * SLIPPAGE_PCT
    slippage_cost = slippage_cost.fillna(0)
    return returns - slippage_cost

def strategy_equity(daily_returns, starting_capital=STARTING_CAPITAL):
    return starting_capital * (1 + daily_returns).cumprod()

def count_trades(positions):
    """Count number of position changes."""
    if isinstance(positions, pd.DataFrame):
        changes = (positions.diff().abs().sum(axis=1) > 0.01).sum()
    else:
        changes = (positions.diff().abs() > 0.01).sum()
    return int(changes)


# ── METRICS ────────────────────────────────────────────────────────────────
def compute_metrics(daily_rets, spy_rets, spy_prices, name):
    daily_rets = daily_rets.fillna(0)
    if len(daily_rets) < 20:
        return None
    # Check it's actually a Series not DataFrame
    if isinstance(daily_rets, pd.DataFrame):
        daily_rets = daily_rets.iloc[:, 0]

    total_ret = (1 + daily_rets).prod() - 1
    ann_ret = (1 + total_ret) ** (ANNUALIZE / len(daily_rets)) - 1
    ann_vol = daily_rets.std() * np.sqrt(ANNUALIZE)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = daily_rets[daily_rets < 0].std() * np.sqrt(ANNUALIZE)
    sortino = ann_ret / downside if downside > 0 else 0

    # Profit factor
    gross_profit = daily_rets[daily_rets > 0].sum()
    gross_loss = abs(daily_rets[daily_rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win rate
    wr = (daily_rets > 0).sum() / len(daily_rets) if len(daily_rets) > 0 else 0

    # MDD
    cum = (1 + daily_rets).cumprod()
    drawdown = cum / cum.cummax() - 1
    mdd = drawdown.min()

    # Correlation with SPY
    aligned = pd.DataFrame({'strat': daily_rets, 'spy': spy_rets}).dropna()
    corr_spy = aligned['strat'].corr(aligned['spy']) if len(aligned) > 20 else 0

    # Per-regime Sharpe
    spy_sma200 = spy_prices.rolling(200).mean()
    bull_mask = spy_prices > spy_sma200
    bear_mask = ~bull_mask

    # Align masks with returns
    bull_aligned = bull_mask.reindex(daily_rets.index).fillna(False)
    bear_aligned = bear_mask.reindex(daily_rets.index).fillna(False)

    bull_rets = daily_rets[bull_aligned]
    bear_rets = daily_rets[bear_aligned]

    def regime_sharpe(r):
        if len(r) < 20:
            return 0.0
        ar = r.mean() * ANNUALIZE
        av = r.std() * np.sqrt(ANNUALIZE)
        return ar / av if av > 0 else 0.0

    sharpe_bull = regime_sharpe(bull_rets)
    sharpe_bear = regime_sharpe(bear_rets)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.01)

    return {
        'name': name,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'win_rate': round(wr, 4),
        'mdd': round(mdd, 4),
        'total_return': round(total_ret, 4),
        'annual_return': round(ann_ret, 4),
        'corr_spy': round(corr_spy, 4),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
    }


def permutation_test(daily_rets, n_iters=PERM_ITERS):
    """Permutation test: shuffle daily returns, compute Sharpe each time."""
    rets = daily_rets.dropna().values
    if len(rets) < 20:
        return 1.0
    observed_sharpe = np.mean(rets) / np.std(rets) * np.sqrt(ANNUALIZE) if np.std(rets) > 0 else 0
    count_better = 0
    rng = np.random.default_rng(42)
    for _ in range(n_iters):
        shuffled = rng.permutation(rets)
        s = np.mean(shuffled) / np.std(shuffled) * np.sqrt(ANNUALIZE) if np.std(shuffled) > 0 else 0
        if s >= observed_sharpe:
            count_better += 1
    return round(count_better / n_iters, 4)


# ── STRATEGIES ─────────────────────────────────────────────────────────────
def strategy_A_momentum_rotation(close, oot_idx):
    """Commodity Momentum Rotation: rank by 20d return, long top 2 if > 0, rebalance weekly."""
    tickers = ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBB', 'PDBC']
    prices = close[tickers]
    ret20 = prices.pct_change(20)

    positions = pd.DataFrame(0.0, index=prices.index, columns=tickers)

    last_rebal = None
    current_pos = pd.Series(0.0, index=tickers)

    for i, date in enumerate(prices.index):
        if date < oot_idx[0]:
            positions.iloc[i] = 0
            continue

        # Use lagged signal (shift 1)
        if i < 1:
            positions.iloc[i] = current_pos
            continue

        prev_date = prices.index[i - 1]

        # Rebalance weekly (every 5 trading days)
        if last_rebal is None or (date - last_rebal).days >= 5:
            r = ret20.loc[prev_date].dropna()
            positive = r[r > 0].sort_values(ascending=False)
            if len(positive) >= 2:
                top2 = positive.index[:2]
                current_pos = pd.Series(0.0, index=tickers)
                current_pos[top2] = 0.5
            elif len(positive) == 1:
                current_pos = pd.Series(0.0, index=tickers)
                current_pos[positive.index[0]] = 1.0
            else:
                current_pos = pd.Series(0.0, index=tickers)
            last_rebal = date

        positions.iloc[i] = current_pos

    # Compute portfolio returns — positions already built with 1-day lag (using prev_date)
    asset_rets = prices.pct_change()
    port_rets = (positions * asset_rets).sum(axis=1)
    port_rets = port_rets.loc[oot_idx].fillna(0)
    port_rets = apply_slippage(port_rets, positions.loc[oot_idx])

    n_trades = count_trades(positions.loc[oot_idx])
    return port_rets, n_trades


def strategy_B_gold_silver_ratio(close, oot_idx):
    """Gold-Silver Ratio Mean Reversion."""
    ratio = close['GLD'] / close['SLV']
    z = zscore(ratio, 60)

    # Build position signal with shift(1) lag
    signal = pd.Series(0.0, index=close.index)
    position = 0.0
    for i in range(1, len(close.index)):
        prev_z = z.iloc[i - 1]  # lagged signal
        if np.isnan(prev_z):
            signal.iloc[i] = position
            continue

        if position == 0:
            if prev_z > 1.5:
                position = 1.0  # long SLV (silver undervalued)
            elif prev_z < -1.5:
                position = -1.0  # long GLD (gold undervalued)
        else:
            # Exit when z crosses zero
            if position == 1.0 and prev_z < 0:
                position = 0.0
            elif position == -1.0 and prev_z > 0:
                position = 0.0
        signal.iloc[i] = position

    # Translate: +1 = long SLV, -1 = long GLD
    slv_rets = close['SLV'].pct_change()
    gld_rets = close['GLD'].pct_change()

    port_rets = signal * slv_rets + (-signal).clip(lower=0) * gld_rets
    # Simpler: when signal=1 → long SLV; when signal=-1 → long GLD
    port_rets_clean = pd.Series(0.0, index=close.index)
    for i in range(len(close.index)):
        if signal.iloc[i] == 1.0:
            port_rets_clean.iloc[i] = slv_rets.iloc[i] if not np.isnan(slv_rets.iloc[i]) else 0
        elif signal.iloc[i] == -1.0:
            port_rets_clean.iloc[i] = gld_rets.iloc[i] if not np.isnan(gld_rets.iloc[i]) else 0

    port_rets_clean = port_rets_clean.loc[oot_idx]
    slippage_cost = signal.diff().abs().loc[oot_idx] * SLIPPAGE_PCT
    port_rets_clean = port_rets_clean - slippage_cost

    n_trades = count_trades(signal.loc[oot_idx])
    return port_rets_clean, n_trades


def strategy_C_energy_momentum(close, oot_idx):
    """Energy Momentum: long USO/UNG when 50d ret > 0 AND above 200-SMA."""
    positions = pd.DataFrame(0.0, index=close.index, columns=['USO', 'UNG'])

    for ticker in ['USO', 'UNG']:
        ret50 = close[ticker].pct_change(50)
        sma200 = close[ticker].rolling(200).mean()

        # Lagged signals
        mom_signal = (ret50 > 0).shift(1)
        sma_signal = (close[ticker] > sma200).shift(1)
        combined = mom_signal & sma_signal
        positions[ticker] = combined.astype(float)

    # Equal weight when both trigger, full weight when one
    total_on = positions.sum(axis=1)
    weights = positions.copy()
    weights[total_on == 2] = 0.5
    weights[total_on == 1] = positions[total_on == 1]  # full weight to the one that's on

    asset_rets = close[['USO', 'UNG']].pct_change()
    port_rets = (weights * asset_rets).sum(axis=1)
    port_rets = port_rets.loc[oot_idx].fillna(0)
    port_rets = apply_slippage(port_rets, weights.loc[oot_idx])

    n_trades = count_trades(weights.loc[oot_idx])
    return port_rets, n_trades


def strategy_D_carry_proxy(close, oot_idx):
    """Commodity Carry Proxy: long PDBC in contango, short in backwardation. Half size."""
    ret20 = close['PDBC'].pct_change(20)
    ret1 = close['PDBC'].pct_change(1)

    # Contango proxy: 20d return < 1d return (spot < futures → roll yield negative → contango)
    # Actually: contango = futures > spot. With ETFs, contango → drag (negative roll yield).
    # Better proxy: if 20d momentum is negative while 1d is positive → contango drag
    # Simplified: use ret20 - ret1 as carry signal
    carry_signal = (ret20 - ret1).shift(1)  # lagged

    position = pd.Series(0.0, index=close.index)
    position[carry_signal > 0] = 0.5   # "contango" → long (half size)
    position[carry_signal < 0] = -0.5  # "backwardation" → short (half size)

    pdbc_rets = close['PDBC'].pct_change()
    port_rets = position * pdbc_rets
    port_rets = port_rets.loc[oot_idx]
    slippage_cost = position.diff().abs().loc[oot_idx] * SLIPPAGE_PCT
    port_rets = port_rets - slippage_cost

    n_trades = count_trades(position.loc[oot_idx])
    return port_rets, n_trades


def strategy_E_inflation_hedge(close, oot_idx):
    """Inflation Hedge Basket: long [GLD, USO, DBA] when TIP/TLT > 200-SMA."""
    tip_tlt = close['TIP'] / close['TLT']
    tip_tlt_sma = tip_tlt.rolling(200).mean()

    # Lagged signal
    inflation_on = (tip_tlt > tip_tlt_sma).shift(1)

    basket = ['GLD', 'USO', 'DBA']
    asset_rets = close[basket].pct_change()

    positions = pd.DataFrame(0.0, index=close.index, columns=basket)
    for t in basket:
        positions[t] = inflation_on.astype(float) / 3.0  # equal weight

    port_rets = (positions * asset_rets).sum(axis=1)
    port_rets = port_rets.loc[oot_idx]
    slippage_cost = positions.diff().abs().sum(axis=1).loc[oot_idx] * SLIPPAGE_PCT
    port_rets = port_rets - slippage_cost

    n_trades = count_trades(positions.loc[oot_idx])
    return port_rets, n_trades


def strategy_F_counter_trend_gold(close, oot_idx):
    """Counter-Trend Gold: long when RSI(14) < 35 AND above 200-SMA. Cash when RSI > 70."""
    gld = close['GLD']
    gld_rsi = rsi(gld, 14)
    gld_sma200 = gld.rolling(200).mean()

    # Lagged signals
    rsi_low = (gld_rsi < 35).shift(1)
    above_sma = (gld > gld_sma200).shift(1)
    rsi_high = (gld_rsi > 70).shift(1)

    position = pd.Series(0.0, index=close.index)

    # Stateful: enter on RSI < 35 + above SMA, exit on RSI > 70 or below SMA
    state = 0.0
    for i in range(len(close.index)):
        if i == 0:
            continue
        rl = rsi_low.iloc[i] if not pd.isna(rsi_low.iloc[i]) else False
        ab = above_sma.iloc[i] if not pd.isna(above_sma.iloc[i]) else False
        rh = rsi_high.iloc[i] if not pd.isna(rsi_high.iloc[i]) else False

        if state == 0:
            if rl and ab:
                state = 1.0
        else:
            if rh or not ab:
                state = 0.0
        position.iloc[i] = state

    gld_rets = gld.pct_change()
    port_rets = position * gld_rets
    port_rets = port_rets.loc[oot_idx]
    slippage_cost = position.diff().abs().loc[oot_idx] * SLIPPAGE_PCT
    port_rets = port_rets - slippage_cost

    n_trades = count_trades(position.loc[oot_idx])
    return port_rets, n_trades


# ── BENCHMARKS ─────────────────────────────────────────────────────────────
def benchmark_buy_hold(close, ticker, oot_idx):
    rets = close[ticker].pct_change().loc[oot_idx]
    n_trades = 1
    return rets, n_trades


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    close = download_data()
    oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
    oot_idx = close.index[oot_mask]

    spy_rets = close['SPY'].pct_change().loc[oot_idx]
    spy_prices = close['SPY']

    print(f"\nOOT period: {oot_idx[0].date()} to {oot_idx[-1].date()} ({len(oot_idx)} days)")
    print(f"Starting capital: ${STARTING_CAPITAL}")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}%\n")

    strategies = {
        'A_Commodity_Momentum_Rotation': strategy_A_momentum_rotation,
        'B_Gold_Silver_Ratio_MeanRev': strategy_B_gold_silver_ratio,
        'C_Energy_Momentum': strategy_C_energy_momentum,
        'D_Commodity_Carry_Proxy': strategy_D_carry_proxy,
        'E_Inflation_Hedge_Basket': strategy_E_inflation_hedge,
        'F_Counter_Trend_Gold': strategy_F_counter_trend_gold,
    }

    results = {}

    # Run strategies
    for name, func in strategies.items():
        print(f"Running {name}...")
        try:
            rets, n_trades = func(close, oot_idx)
            metrics = compute_metrics(rets, spy_rets, spy_prices, name)
            if metrics is None:
                print(f"  SKIP: insufficient data")
                continue
            metrics['n_trades'] = n_trades

            # Permutation test
            print(f"  Running permutation test ({PERM_ITERS} iters)...")
            perm_p = permutation_test(rets, PERM_ITERS)
            metrics['perm_p'] = perm_p

            # Final equity
            eq = strategy_equity(rets)
            metrics['final_equity'] = round(float(eq.iloc[-1]), 2)

            # 5-gate validation
            gates = {
                'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
                'perm_p_lt_0.05': perm_p < 0.05,
                'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
                'mdd_gt_neg50': metrics['mdd'] > -0.50,
                'trades_gte_20': n_trades >= 20,
            }
            metrics['gates'] = gates
            metrics['gates_passed'] = sum(gates.values())
            metrics['all_gates_passed'] = all(gates.values())

            results[name] = metrics

            # Print summary
            print(f"  Sharpe={metrics['sharpe']:.3f} Sortino={metrics['sortino']:.3f} "
                  f"PF={metrics['profit_factor']:.2f} WR={metrics['win_rate']:.1%} "
                  f"MDD={metrics['mdd']:.1%} Ret={metrics['total_return']:.1%} "
                  f"Corr(SPY)={metrics['corr_spy']:.3f} Trades={n_trades} "
                  f"Perm_p={perm_p:.4f} Gates={metrics['gates_passed']}/5"
                  f"{' *** PASS ***' if metrics['all_gates_passed'] else ''}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Benchmarks
    print("\n--- BENCHMARKS ---")
    for ticker in ['PDBC', 'GLD', 'SPY']:
        bname = f"BH_{ticker}"
        rets, n_trades = benchmark_buy_hold(close, ticker, oot_idx)
        metrics = compute_metrics(rets, spy_rets, spy_prices, bname)
        if metrics:
            metrics['n_trades'] = n_trades
            metrics['perm_p'] = None
            metrics['final_equity'] = round(float(strategy_equity(rets).iloc[-1]), 2)
            metrics['gates'] = None
            metrics['gates_passed'] = None
            metrics['all_gates_passed'] = None
            results[bname] = metrics
            print(f"  {bname}: Sharpe={metrics['sharpe']:.3f} Sortino={metrics['sortino']:.3f} "
                  f"Ret={metrics['total_return']:.1%} MDD={metrics['mdd']:.1%} "
                  f"Corr(SPY)={metrics['corr_spy']:.3f}")

    # Summary table
    print("\n" + "="*120)
    print(f"{'Strategy':<35} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
          f"{'MDD':>7} {'Return':>8} {'CorrSPY':>8} {'Trades':>7} {'Perm_p':>7} {'Gates':>6} {'Pass':>5}")
    print("-"*120)
    for name, m in results.items():
        perm_str = f"{m['perm_p']:.4f}" if m['perm_p'] is not None else "  N/A"
        gates_str = f"{m['gates_passed']}/5" if m['gates_passed'] is not None else " N/A"
        pass_str = "YES" if m.get('all_gates_passed') else ("NO" if m.get('all_gates_passed') is not None else "N/A")
        print(f"  {name:<33} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>5.1%} {m['mdd']:>7.1%} {m['total_return']:>7.1%} "
              f"{m['corr_spy']:>8.3f} {m['n_trades']:>7} {perm_str:>7} {gates_str:>6} {pass_str:>5}")
    print("="*120)

    # Correlation matrix of strategy returns
    print("\n--- STRATEGY RETURN CORRELATIONS ---")
    strat_rets = {}
    for name in results:
        if name.startswith('BH_'):
            continue
        try:
            r, _ = strategies[name](close, oot_idx)
            if isinstance(r, pd.Series):
                strat_rets[name[:20]] = r.fillna(0)
        except:
            pass
    if len(strat_rets) >= 2:
        corr_df = pd.DataFrame(strat_rets).corr()
        print(corr_df.round(3).to_string())
    else:
        print("Not enough strategies for correlation matrix.")

    # Save results
    output = {
        'metadata': {
            'run_date': datetime.now().isoformat(),
            'oot_start': OOT_START,
            'oot_end': OOT_END,
            'starting_capital': STARTING_CAPITAL,
            'slippage_pct': SLIPPAGE_PCT,
            'perm_iterations': PERM_ITERS,
            'tickers': TICKERS,
        },
        'strategies': {},
        'benchmarks': {},
    }

    for name, m in results.items():
        # Convert gates bools to serializable
        entry = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                 for k, v in m.items()}
        if entry.get('gates'):
            entry['gates'] = {k: bool(v) for k, v in entry['gates'].items()}
        if name.startswith('BH_'):
            output['benchmarks'][name] = entry
        else:
            output['strategies'][name] = entry

    output_path = '/home/jupiter/Lvl3Quant/data/commodity_cross_momentum_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Final verdict
    print("\n--- VERDICT ---")
    passed = [n for n, m in results.items() if m.get('all_gates_passed')]
    low_corr = [n for n, m in results.items() if abs(m.get('corr_spy', 1)) < 0.3 and not n.startswith('BH_')]

    if passed:
        print(f"Strategies passing ALL 5 gates: {passed}")
    else:
        print("No strategy passed all 5 gates.")

    if low_corr:
        print(f"Strategies with |corr(SPY)| < 0.3 (decorrelation candidates): {low_corr}")
    else:
        print("No strategy achieved |corr(SPY)| < 0.3")

    best_decorr = min(
        [(n, m) for n, m in results.items() if not n.startswith('BH_')],
        key=lambda x: abs(x[1].get('corr_spy', 1)),
        default=None
    )
    if best_decorr:
        print(f"Most decorrelated from SPY: {best_decorr[0]} (corr={best_decorr[1]['corr_spy']:.3f})")


if __name__ == '__main__':
    main()
