#!/usr/bin/env python3
"""
Smart Dollar Cost Averaging Backtest
Academic basis: Brennan, Li & Torous (2005) — enhanced DCA strategies

6 Variants tested against SPY buy-and-hold:
A. Naive DCA ($100/month)
B. Value Averaging (target $100/month growth)
C. RSI DCA (double when RSI<30, half when RSI>70)
D. Drawdown DCA (double when >5% below 52wk high)
E. Moving Average DCA ($200 below 200-SMA, $50 when >10% above)
F. Combined Smart DCA (score-based: RSI<40 + drawdown>5% + below 200-SMA)

OOT: Jan 2022 - Jul 2026
Starting: $645 + $100/month
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Parameters ──
START_DATE = '2021-01-01'  # extra lookback for 200-SMA + RSI warmup
TRADE_START = '2022-01-01'
TRADE_END = '2026-07-30'
INITIAL_CAPITAL = 645.0
MONTHLY_CONTRIBUTION = 100.0
SLIPPAGE_PCT = 0.0002  # 0.02%
N_PERMUTATIONS = 1000
RISK_FREE_RATE = 0.0  # simplified

print("=" * 70)
print("SMART DCA BACKTEST — Brennan, Li & Torous (2005) Enhanced DCA")
print("=" * 70)

# ── Download Data ──
print("\nDownloading SPY data...")
spy = yf.download('SPY', start=START_DATE, end=TRADE_END, auto_adjust=True, progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy = spy.sort_index()

# Flatten index
spy.index = spy.index.tz_localize(None) if spy.index.tz else spy.index

print(f"Data: {spy.index[0].date()} to {spy.index[-1].date()} ({len(spy)} days)")

# ── Compute Indicators ──
spy['SMA200'] = spy['Close'].rolling(200).mean()
spy['RSI'] = _compute_rsi(spy['Close'], 14) if False else None

# RSI calculation
delta = spy['Close'].diff()
gain = delta.where(delta > 0, 0.0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
rs = gain / loss
spy['RSI'] = 100 - (100 / (1 + rs))

# 52-week high
spy['High52W'] = spy['Close'].rolling(252).max()

# Drawdown from 52wk high
spy['DD_from_52W'] = (spy['Close'] - spy['High52W']) / spy['High52W']

# Distance from 200-SMA
spy['Dist_SMA200'] = (spy['Close'] - spy['SMA200']) / spy['SMA200']

# Regime: Bull = above 200-SMA, Bear = below
spy['Regime'] = np.where(spy['Close'] > spy['SMA200'], 'Bull', 'Bear')

# Filter to trading period
trade_mask = spy.index >= pd.Timestamp(TRADE_START)
spy_trade = spy[trade_mask].copy()

print(f"Trading period: {spy_trade.index[0].date()} to {spy_trade.index[-1].date()} ({len(spy_trade)} days)")

# ── Generate monthly investment dates (first trading day of each month) ──
monthly_dates = []
current = pd.Timestamp(TRADE_START)
end = pd.Timestamp(TRADE_END)
while current <= end:
    # Find first trading day on or after current
    mask = spy_trade.index >= current
    if mask.any():
        monthly_dates.append(spy_trade.index[mask][0])
    current = current + pd.DateOffset(months=1)
    current = current.replace(day=1)

# Remove duplicates
monthly_dates = sorted(set(monthly_dates))
print(f"Monthly investment dates: {len(monthly_dates)} months")


def apply_slippage(price, direction='buy'):
    """Apply slippage to price"""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def compute_metrics(portfolio_values, dates):
    """Compute Sharpe, Sortino, MaxDD from portfolio value series"""
    pv = pd.Series(portfolio_values, index=dates)
    returns = pv.pct_change().dropna()

    if len(returns) < 2 or returns.std() == 0:
        return {'sharpe': 0, 'sortino': 0, 'max_dd': 0, 'total_return_pct': 0, 'final_value': pv.iloc[-1]}

    # Annualize
    ann_factor = np.sqrt(252)
    sharpe = (returns.mean() / returns.std()) * ann_factor

    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * ann_factor
    else:
        sortino = sharpe * 2  # no downside = great

    # Max drawdown
    cummax = pv.cummax()
    drawdown = (pv - cummax) / cummax
    max_dd = drawdown.min()

    total_return = (pv.iloc[-1] / pv.iloc[0] - 1) * 100

    return {
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_dd': round(float(max_dd) * 100, 2),  # as percentage
        'total_return_pct': round(float(total_return), 2),
        'final_value': round(float(pv.iloc[-1]), 2)
    }


def run_dca_strategy(name, spy_data, monthly_dates, invest_fn):
    """
    Run a DCA strategy.
    invest_fn(date, spy_row, portfolio_state) -> amount_to_invest (can be negative for selling)
    Returns: daily portfolio values, total invested, investment decisions
    """
    cash = INITIAL_CAPITAL
    shares = 0.0
    total_invested = INITIAL_CAPITAL  # track total capital contributed
    decisions = []  # (date, amount, reason)

    daily_values = []
    daily_dates = []

    monthly_set = set(monthly_dates)

    for date, row in spy_data.iterrows():
        price = float(row['Close'])

        if date in monthly_set:
            # Get investment amount from strategy
            state = {
                'cash': cash,
                'shares': shares,
                'portfolio_value': cash + shares * price,
                'total_invested': total_invested,
                'month_number': len([d for d in decisions]) + 1
            }
            amount, reason = invest_fn(date, row, state)

            # Add monthly contribution to cash
            cash += MONTHLY_CONTRIBUTION
            total_invested += MONTHLY_CONTRIBUTION

            # Clamp investment to available cash
            if amount > 0:
                actual_invest = min(amount, cash)
                if actual_invest > 0:
                    buy_price = apply_slippage(price, 'buy')
                    new_shares = actual_invest / buy_price
                    shares += new_shares
                    cash -= actual_invest
                decisions.append((date, actual_invest, reason))
            elif amount < 0:
                # Sell shares
                sell_amount = min(abs(amount), shares * price)
                if sell_amount > 0:
                    sell_price = apply_slippage(price, 'sell')
                    shares_to_sell = sell_amount / sell_price
                    shares_to_sell = min(shares_to_sell, shares)
                    shares -= shares_to_sell
                    cash += shares_to_sell * sell_price
                decisions.append((date, -sell_amount, reason))
            else:
                decisions.append((date, 0, reason))

        # Record daily portfolio value
        portfolio_value = cash + shares * price
        daily_values.append(portfolio_value)
        daily_dates.append(date)

    return {
        'name': name,
        'daily_values': daily_values,
        'daily_dates': daily_dates,
        'total_invested': total_invested,
        'final_cash': cash,
        'final_shares': shares,
        'decisions': decisions
    }


# ── Strategy A: Naive DCA ──
def naive_dca(date, row, state):
    return MONTHLY_CONTRIBUTION, "naive_100"

# ── Strategy B: Value Averaging ──
def value_averaging(date, row, state):
    target_value = INITIAL_CAPITAL + state['month_number'] * MONTHLY_CONTRIBUTION
    current_equity = state['shares'] * float(row['Close'])
    gap = target_value - current_equity
    if gap > 0:
        reason = f"va_invest_{gap:.0f}"
        return min(gap, state['cash'] + MONTHLY_CONTRIBUTION), reason
    else:
        reason = f"va_sell_{abs(gap):.0f}"
        return max(-abs(gap), -current_equity * 0.5), reason  # cap selling at 50% of equity

# ── Strategy C: RSI DCA ──
def rsi_dca(date, row, state):
    rsi = float(row['RSI'])
    if rsi < 30:
        return 200, f"rsi_oversold_{rsi:.0f}"
    elif rsi > 70:
        return 50, f"rsi_overbought_{rsi:.0f}"
    else:
        return 100, f"rsi_normal_{rsi:.0f}"

# ── Strategy D: Drawdown DCA ──
def drawdown_dca(date, row, state):
    dd = float(row['DD_from_52W'])
    if dd < -0.05:
        return 200, f"dd_buying_dip_{dd:.1%}"
    else:
        return 100, f"dd_normal_{dd:.1%}"

# ── Strategy E: Moving Average DCA ──
def ma_dca(date, row, state):
    dist = float(row['Dist_SMA200'])
    if dist < 0:  # Below 200-SMA
        return 200, f"ma_below_sma_{dist:.1%}"
    elif dist > 0.10:  # >10% above
        return 50, f"ma_stretched_{dist:.1%}"
    else:
        return 100, f"ma_normal_{dist:.1%}"

# ── Strategy F: Combined Smart DCA ──
def combined_smart_dca(date, row, state):
    score = 0
    reasons = []
    rsi = float(row['RSI'])
    dd = float(row['DD_from_52W'])
    dist = float(row['Dist_SMA200'])

    if rsi < 40:
        score += 1
        reasons.append("rsi<40")
    if dd < -0.05:
        score += 1
        reasons.append("dd>5%")
    if dist < 0:
        score += 1
        reasons.append("below_sma")

    amount = 50 * (1 + score)
    return amount, f"combined_score{score}_{'+'.join(reasons) if reasons else 'bullish'}"


# ── Run All Strategies ──
strategies = {
    'A_Naive_DCA': naive_dca,
    'B_Value_Averaging': value_averaging,
    'C_RSI_DCA': rsi_dca,
    'D_Drawdown_DCA': drawdown_dca,
    'E_MA_DCA': ma_dca,
    'F_Combined_Smart': combined_smart_dca,
}

results = {}
for sname, sfunc in strategies.items():
    res = run_dca_strategy(sname, spy_trade, monthly_dates, sfunc)
    metrics = compute_metrics(res['daily_values'], res['daily_dates'])
    metrics['total_invested'] = round(res['total_invested'], 2)
    metrics['final_cash'] = round(res['final_cash'], 2)
    metrics['final_shares'] = round(res['final_shares'], 4)

    # Regime-stratified returns
    pv = pd.Series(res['daily_values'], index=res['daily_dates'])
    returns = pv.pct_change().dropna()
    regime_series = spy_trade.loc[returns.index, 'Regime']

    bull_ret = returns[regime_series == 'Bull']
    bear_ret = returns[regime_series == 'Bear']

    ann = np.sqrt(252)
    if len(bull_ret) > 1 and bull_ret.std() > 0:
        metrics['sharpe_bull'] = round(float(bull_ret.mean() / bull_ret.std() * ann), 4)
    else:
        metrics['sharpe_bull'] = 0.0
    if len(bear_ret) > 1 and bear_ret.std() > 0:
        metrics['sharpe_bear'] = round(float(bear_ret.mean() / bear_ret.std() * ann), 4)
    else:
        metrics['sharpe_bear'] = 0.0

    # Regime gap
    max_regime = max(abs(metrics['sharpe_bull']), abs(metrics['sharpe_bear']), 0.0001)
    metrics['regime_gap'] = round(abs(metrics['sharpe_bull'] - metrics['sharpe_bear']) / max_regime, 4)

    # Investment decisions summary
    invest_amounts = [d[1] for d in res['decisions']]
    metrics['avg_monthly_invest'] = round(np.mean(invest_amounts), 2)
    metrics['min_monthly_invest'] = round(min(invest_amounts), 2)
    metrics['max_monthly_invest'] = round(max(invest_amounts), 2)

    results[sname] = metrics

    print(f"\n{'=' * 50}")
    print(f"Strategy: {sname}")
    print(f"  Final Value:     ${metrics['final_value']:,.2f}")
    print(f"  Total Invested:  ${metrics['total_invested']:,.2f}")
    print(f"  Total Return:    {metrics['total_return_pct']:.2f}%")
    print(f"  Sharpe:          {metrics['sharpe']:.4f}")
    print(f"  Sortino:         {metrics['sortino']:.4f}")
    print(f"  Max Drawdown:    {metrics['max_dd']:.2f}%")
    print(f"  Sharpe (Bull):   {metrics['sharpe_bull']:.4f}")
    print(f"  Sharpe (Bear):   {metrics['sharpe_bear']:.4f}")
    print(f"  Regime Gap:      {metrics['regime_gap']:.4f}")
    print(f"  Avg Mo. Invest:  ${metrics['avg_monthly_invest']:.2f}")

# ── SPY Buy-and-Hold Benchmark ──
# Same total invested as Naive DCA, but lump sum at start + contributions invested immediately
print(f"\n{'=' * 50}")
print("Benchmark: SPY Buy-and-Hold (lump sum equivalent)")

total_naive_invested = results['A_Naive_DCA']['total_invested']
# For B&H: invest everything available on each monthly date (same total capital as naive)
bh_res = run_dca_strategy('BuyHold', spy_trade, monthly_dates, naive_dca)
bh_metrics = compute_metrics(bh_res['daily_values'], bh_res['daily_dates'])
bh_metrics['total_invested'] = round(bh_res['total_invested'], 2)

print(f"  (Note: B&H with same $100/month schedule = Naive DCA)")
print(f"  For true B&H, using lump-sum of all capital at start)")

# True lump sum: invest $645 + present value of all contributions at day 1
lump_total = INITIAL_CAPITAL + MONTHLY_CONTRIBUTION * len(monthly_dates)
first_price = float(spy_trade['Close'].iloc[0])
buy_price_ls = apply_slippage(first_price, 'buy')
shares_ls = lump_total / buy_price_ls

ls_values = []
ls_dates = []
for date, row in spy_trade.iterrows():
    ls_values.append(shares_ls * float(row['Close']))
    ls_dates.append(date)

ls_metrics = compute_metrics(ls_values, ls_dates)
ls_metrics['total_invested'] = round(lump_total, 2)
ls_metrics['description'] = f"Lump sum ${lump_total:.0f} invested at start"

print(f"  Lump Sum Total:  ${lump_total:,.2f}")
print(f"  Final Value:     ${ls_metrics['final_value']:,.2f}")
print(f"  Total Return:    {ls_metrics['total_return_pct']:.2f}%")
print(f"  Sharpe:          {ls_metrics['sharpe']:.4f}")
print(f"  Sortino:         {ls_metrics['sortino']:.4f}")
print(f"  Max Drawdown:    {ls_metrics['max_dd']:.2f}%")

results['Lump_Sum_BH'] = ls_metrics

# ── Permutation Test (for strategies B-F vs A) ──
print(f"\n{'=' * 50}")
print("PERMUTATION TESTS (Smart vs Naive DCA)")
print(f"Shuffling buy-more/buy-less decisions, {N_PERMUTATIONS} permutations")

naive_result = results['A_Naive_DCA']

for sname in ['B_Value_Averaging', 'C_RSI_DCA', 'D_Drawdown_DCA', 'E_MA_DCA', 'F_Combined_Smart']:
    # Get the strategy's actual investment amounts
    actual_res = run_dca_strategy(sname, spy_trade, monthly_dates, strategies[sname])
    actual_amounts = [d[1] for d in actual_res['decisions']]
    actual_metrics = compute_metrics(actual_res['daily_values'], actual_res['daily_dates'])
    actual_sharpe = actual_metrics['sharpe']

    # Permutation: shuffle which months get which investment amounts
    perm_sharpes = []
    rng = np.random.RandomState(42)
    for _ in range(N_PERMUTATIONS):
        shuffled_amounts = actual_amounts.copy()
        rng.shuffle(shuffled_amounts)

        # Re-run with shuffled amounts
        idx = [0]
        def perm_invest(date, row, state, _amounts=shuffled_amounts, _idx=idx):
            i = _idx[0]
            _idx[0] += 1
            if i < len(_amounts):
                return _amounts[i], "perm"
            return MONTHLY_CONTRIBUTION, "perm_default"

        idx[0] = 0
        perm_res = run_dca_strategy(f'perm_{sname}', spy_trade, monthly_dates, perm_invest)
        perm_metrics = compute_metrics(perm_res['daily_values'], perm_res['daily_dates'])
        perm_sharpes.append(perm_metrics['sharpe'])

    # p-value: fraction of permuted Sharpes >= actual
    p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])
    results[sname]['perm_p_value'] = round(float(p_value), 4)
    results[sname]['perm_mean_sharpe'] = round(float(np.mean(perm_sharpes)), 4)

    print(f"  {sname}: Sharpe={actual_sharpe:.4f}, Perm mean={np.mean(perm_sharpes):.4f}, p={p_value:.4f}")


# ── 5-Gate Validation ──
print(f"\n{'=' * 70}")
print("5-GATE VALIDATION (Smart B-F vs Naive A)")
print("=" * 70)

naive_sharpe = results['A_Naive_DCA']['sharpe']

gate_results = {}
for sname in ['B_Value_Averaging', 'C_RSI_DCA', 'D_Drawdown_DCA', 'E_MA_DCA', 'F_Combined_Smart']:
    m = results[sname]
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates['G1_sharpe_gt_0.5'] = m['sharpe'] > 0.5

    # Gate 2: Permutation p < 0.05
    gates['G2_perm_p_lt_0.05'] = m.get('perm_p_value', 1.0) < 0.05

    # Gate 3: Regime gap < 0.5
    gates['G3_regime_gap_lt_0.5'] = m['regime_gap'] < 0.5

    # Gate 4: MaxDD > -50%
    gates['G4_maxdd_gt_neg50'] = m['max_dd'] > -50.0

    # Gate 5: Smart beats Naive by > 0.2 Sharpe
    gates['G5_beats_naive_0.2'] = (m['sharpe'] - naive_sharpe) > 0.2

    passed = sum(gates.values())
    gates['total_passed'] = f"{passed}/5"
    gates['verdict'] = 'PASS' if passed == 5 else 'FAIL'

    gate_results[sname] = gates
    results[sname]['gates'] = gates

    print(f"\n  {sname}:")
    for gname, gval in gates.items():
        if gname in ('total_passed', 'verdict'):
            continue
        status = "PASS" if gval else "FAIL"
        print(f"    {gname}: {status}")
    print(f"    VERDICT: {gates['verdict']} ({gates['total_passed']})")


# ── Comparison Table ──
print(f"\n{'=' * 70}")
print("COMPARISON TABLE")
print("=" * 70)
print(f"{'Strategy':<25} {'Final $':>10} {'Return%':>9} {'Sharpe':>8} {'Sortino':>9} {'MaxDD%':>8} {'Invested':>10}")
print("-" * 85)
for sname in ['A_Naive_DCA', 'B_Value_Averaging', 'C_RSI_DCA', 'D_Drawdown_DCA', 'E_MA_DCA', 'F_Combined_Smart', 'Lump_Sum_BH']:
    m = results[sname]
    print(f"{sname:<25} ${m['final_value']:>9,.2f} {m['total_return_pct']:>8.2f}% {m['sharpe']:>8.4f} {m['sortino']:>9.4f} {m['max_dd']:>7.2f}% ${m.get('total_invested', 0):>9,.2f}")

# ── vs Naive DCA delta ──
print(f"\n{'Strategy':<25} {'Sharpe Delta':>14} {'Return Delta':>14}")
print("-" * 55)
for sname in ['B_Value_Averaging', 'C_RSI_DCA', 'D_Drawdown_DCA', 'E_MA_DCA', 'F_Combined_Smart']:
    m = results[sname]
    sd = m['sharpe'] - naive_sharpe
    rd = m['total_return_pct'] - results['A_Naive_DCA']['total_return_pct']
    print(f"{sname:<25} {sd:>+14.4f} {rd:>+13.2f}%")
    results[sname]['sharpe_delta_vs_naive'] = round(sd, 4)
    results[sname]['return_delta_vs_naive'] = round(rd, 2)


# ── Save Results ──
output_path = '/home/jupiter/Lvl3Quant/data/smart_dca_results.json'

# Clean up non-serializable items
save_results = {}
for k, v in results.items():
    save_results[k] = {kk: vv for kk, vv in v.items() if not isinstance(vv, (pd.Series, pd.DataFrame, list))}
    # Keep gates if present
    if 'gates' in v:
        save_results[k]['gates'] = v['gates']

save_results['metadata'] = {
    'backtest_date': datetime.now().isoformat(),
    'oot_period': f'{TRADE_START} to {TRADE_END}',
    'initial_capital': INITIAL_CAPITAL,
    'monthly_contribution': MONTHLY_CONTRIBUTION,
    'slippage_pct': SLIPPAGE_PCT,
    'n_permutations': N_PERMUTATIONS,
    'academic_basis': 'Brennan, Li & Torous (2005)',
    'description': 'Smart DCA variants vs naive DCA and lump-sum buy-hold'
}

with open(output_path, 'w') as f:
    json.dump(save_results, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("\nDONE.")
