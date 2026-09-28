#!/usr/bin/env python3
"""
Options Premium Capture Backtest
6 variants (A-F) simulating systematic options selling on liquid ETFs.

Since historical options prices aren't available from yfinance, we simulate
option premiums using Black-Scholes with VIX as implied volatility proxy.

Variants:
  A) Weekly CSP on SPY (98% strike)
  B) Monthly CSP on QQQ (95% strike)
  C) VIX-gated CSP on SPY (only when VIX>20)
  D) Iron Condor on SPY (monthly, defined risk)
  E) Put Credit Spread on QQQ (monthly, defined risk)
  F) Covered Call Overwrite on QQQ (monthly)

Capital: $645 fractional sizing. OOT: Jan 2022 – Jul 2026.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ─── Parameters ───
START = '2021-10-01'  # Extra buffer for warmup
OOT_START = '2022-01-03'
END = '2026-07-28'
CAPITAL = 645.0
COMMISSION_PER_CONTRACT = 0.65  # RH options commission
RF_RATE = 0.05  # Risk-free rate
PERMUTATION_ITERS = 1000
np.random.seed(42)


# ─── Black-Scholes Functions ───
def bs_d1(S, K, T, r, sigma):
    """d1 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S, K, T, r, sigma):
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ─── Download Data ───
print("Downloading market data...")
tickers = ['SPY', 'QQQ', '^VIX']
raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

close = raw['Close'].copy()
# Flatten multi-index columns if needed
if hasattr(close.columns, 'get_level_values'):
    close.columns = [c if isinstance(c, str) else c for c in close.columns]

# Rename VIX
rename_map = {}
for col in close.columns:
    if 'VIX' in str(col).upper() and 'VIXY' not in str(col).upper():
        rename_map[col] = 'VIX'
close.rename(columns=rename_map, inplace=True)

close = close.dropna(subset=['VIX', 'SPY', 'QQQ'])
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")

# Precompute
vix = close['VIX']
spy = close['SPY']
qqq = close['QQQ']
spy_200sma = spy.rolling(200).mean()

# Regime: bull/bear based on SPY vs 200-SMA
regime = pd.Series('bull', index=close.index)
regime[spy < spy_200sma] = 'bear'

# Day of week (0=Mon, 4=Fri)
dow = close.index.dayofweek


# ─── Fractional Sizing ───
# We scale to $645 notional. 1 "contract" = $645 / spot_price shares.
# Premium and P&L are scaled proportionally.
def fractional_multiplier(spot_price):
    """How many shares $645 buys."""
    return CAPITAL / spot_price


def scaled_commission(spot_price):
    """Commission scaled to fractional position.
    Standard: $0.65 per contract (100 shares).
    Fractional: $0.65 * (fractional_shares / 100)."""
    frac = fractional_multiplier(spot_price)
    return COMMISSION_PER_CONTRACT * (frac / 100.0)


# ─── Strategy A: Weekly CSP on SPY (98% strike) ───
def strategy_a_weekly_csp_spy():
    """Sell ATM-ish weekly put on SPY every Friday. 7 DTE, 98% strike."""
    equity = CAPITAL
    equity_curve = []
    trades = []

    position = None  # (entry_date, strike, premium_collected, spot_at_entry)

    dates = close.index[close.index >= OOT_START]

    for i, date in enumerate(dates):
        spot = spy[date]
        iv = vix[date] / 100.0  # VIX as annualized IV for SPY

        # If we have a position, check if it expired (7 days later)
        if position is not None:
            entry_date, strike, premium, spot_entry = position
            days_held = (date - entry_date).days

            if days_held >= 7:
                # Expiry: settle
                spot_now = spot
                if spot_now < strike:
                    # Assigned: loss = strike - spot (we must buy at strike)
                    # But we also collected premium
                    intrinsic = strike - spot_now
                    frac = fractional_multiplier(spot_entry)
                    pnl = (premium - intrinsic) * frac - scaled_commission(spot_entry)
                else:
                    # Expired worthless: keep full premium
                    frac = fractional_multiplier(spot_entry)
                    pnl = premium * frac - scaled_commission(spot_entry)

                equity += pnl
                trades.append({
                    'entry': entry_date.strftime('%Y-%m-%d'),
                    'exit': date.strftime('%Y-%m-%d'),
                    'pnl': pnl,
                    'regime': regime[entry_date]
                })
                position = None

        # Enter new position on Fridays (or last trading day of week)
        if position is None and dow[i if i < len(dow) else -1] == 4:  # Friday
            strike = spot * 0.98
            T = 7.0 / 365.0
            premium = bs_put_price(spot, strike, T, RF_RATE, iv)
            position = (date, strike, premium, spot)

        equity_curve.append({'date': date, 'equity': equity})

    return pd.DataFrame(equity_curve).set_index('date'), trades


# ─── Strategy B: Monthly CSP on QQQ (95% strike) ───
def strategy_b_monthly_csp_qqq():
    """Sell monthly put on QQQ, 95% strike, 30 DTE."""
    equity = CAPITAL
    equity_curve = []
    trades = []
    position = None

    dates = close.index[close.index >= OOT_START]
    last_entry_month = None

    for i, date in enumerate(dates):
        spot = qqq[date]
        iv = vix[date] * 1.2 / 100.0  # QQQ IV = VIX * 1.2

        if position is not None:
            entry_date, strike, premium, spot_entry = position
            days_held = (date - entry_date).days

            if days_held >= 30:
                spot_now = spot
                frac = fractional_multiplier(spot_entry)
                if spot_now < strike:
                    intrinsic = strike - spot_now
                    pnl = (premium - intrinsic) * frac - scaled_commission(spot_entry)
                else:
                    pnl = premium * frac - scaled_commission(spot_entry)

                equity += pnl
                trades.append({
                    'entry': entry_date.strftime('%Y-%m-%d'),
                    'exit': date.strftime('%Y-%m-%d'),
                    'pnl': pnl,
                    'regime': regime[entry_date]
                })
                position = None

        # Enter on first trading day of month
        current_month = (date.year, date.month)
        if position is None and current_month != last_entry_month:
            strike = spot * 0.95
            T = 30.0 / 365.0
            premium = bs_put_price(spot, strike, T, RF_RATE, iv)
            position = (date, strike, premium, spot)
            last_entry_month = current_month

        equity_curve.append({'date': date, 'equity': equity})

    return pd.DataFrame(equity_curve).set_index('date'), trades


# ─── Strategy C: VIX-gated CSP on SPY ───
def strategy_c_vix_gated_csp():
    """Sell weekly put on SPY only when VIX > 20. Cash otherwise."""
    equity = CAPITAL
    equity_curve = []
    trades = []
    position = None

    dates = close.index[close.index >= OOT_START]

    for i, date in enumerate(dates):
        spot = spy[date]
        iv = vix[date] / 100.0
        current_vix = vix[date]

        if position is not None:
            entry_date, strike, premium, spot_entry = position
            days_held = (date - entry_date).days

            if days_held >= 7:
                spot_now = spot
                frac = fractional_multiplier(spot_entry)
                if spot_now < strike:
                    intrinsic = strike - spot_now
                    pnl = (premium - intrinsic) * frac - scaled_commission(spot_entry)
                else:
                    pnl = premium * frac - scaled_commission(spot_entry)

                equity += pnl
                trades.append({
                    'entry': entry_date.strftime('%Y-%m-%d'),
                    'exit': date.strftime('%Y-%m-%d'),
                    'pnl': pnl,
                    'regime': regime[entry_date]
                })
                position = None

        # Only sell when VIX > 20 and it's Friday
        if position is None and dow[i if i < len(dow) else -1] == 4 and current_vix > 20:
            strike = spot * 0.98
            T = 7.0 / 365.0
            premium = bs_put_price(spot, strike, T, RF_RATE, iv)
            position = (date, strike, premium, spot)

        equity_curve.append({'date': date, 'equity': equity})

    return pd.DataFrame(equity_curve).set_index('date'), trades


# ─── Strategy D: Iron Condor on SPY (monthly) ───
def strategy_d_iron_condor_spy():
    """
    Monthly iron condor on SPY:
    - Sell put at 97%, buy put at 95% (put spread)
    - Sell call at 103%, buy call at 105% (call spread)
    Max loss = spread width - premium collected (per share).
    """
    equity = CAPITAL
    equity_curve = []
    trades = []
    position = None
    last_entry_month = None

    dates = close.index[close.index >= OOT_START]

    for i, date in enumerate(dates):
        spot = spy[date]
        iv = vix[date] / 100.0

        if position is not None:
            entry_date, put_sell_k, put_buy_k, call_sell_k, call_buy_k, net_premium, spot_entry = position
            days_held = (date - entry_date).days

            if days_held >= 30:
                spot_now = spot
                frac = fractional_multiplier(spot_entry)

                # Put spread P&L at expiry
                put_sell_intrinsic = max(put_sell_k - spot_now, 0)
                put_buy_intrinsic = max(put_buy_k - spot_now, 0)
                put_spread_loss = put_sell_intrinsic - put_buy_intrinsic  # We sold higher, bought lower

                # Call spread P&L at expiry
                call_sell_intrinsic = max(spot_now - call_sell_k, 0)
                call_buy_intrinsic = max(spot_now - call_buy_k, 0)
                call_spread_loss = call_sell_intrinsic - call_buy_intrinsic

                pnl = (net_premium - put_spread_loss - call_spread_loss) * frac
                # 4 legs = 4 commissions (open + close = 8 but we hold to expiry so 4)
                pnl -= 4 * scaled_commission(spot_entry)

                equity += pnl
                trades.append({
                    'entry': entry_date.strftime('%Y-%m-%d'),
                    'exit': date.strftime('%Y-%m-%d'),
                    'pnl': pnl,
                    'regime': regime[entry_date]
                })
                position = None

        current_month = (date.year, date.month)
        if position is None and current_month != last_entry_month:
            T = 30.0 / 365.0

            put_sell_k = spot * 0.97
            put_buy_k = spot * 0.95
            call_sell_k = spot * 1.03
            call_buy_k = spot * 1.05

            # Net premium = sell premiums - buy premiums
            put_sell_prem = bs_put_price(spot, put_sell_k, T, RF_RATE, iv)
            put_buy_prem = bs_put_price(spot, put_buy_k, T, RF_RATE, iv)
            call_sell_prem = bs_call_price(spot, call_sell_k, T, RF_RATE, iv)
            call_buy_prem = bs_call_price(spot, call_buy_k, T, RF_RATE, iv)

            net_premium = (put_sell_prem - put_buy_prem) + (call_sell_prem - call_buy_prem)

            position = (date, put_sell_k, put_buy_k, call_sell_k, call_buy_k, net_premium, spot)
            last_entry_month = current_month

        equity_curve.append({'date': date, 'equity': equity})

    return pd.DataFrame(equity_curve).set_index('date'), trades


# ─── Strategy E: Put Credit Spread on QQQ (monthly) ───
def strategy_e_put_spread_qqq():
    """
    Monthly put credit spread on QQQ:
    - Sell 97% put, buy 94% put
    Max loss = 3% spread width - premium.
    """
    equity = CAPITAL
    equity_curve = []
    trades = []
    position = None
    last_entry_month = None

    dates = close.index[close.index >= OOT_START]

    for i, date in enumerate(dates):
        spot = qqq[date]
        iv = vix[date] * 1.2 / 100.0

        if position is not None:
            entry_date, sell_k, buy_k, net_premium, spot_entry = position
            days_held = (date - entry_date).days

            if days_held >= 30:
                spot_now = spot
                frac = fractional_multiplier(spot_entry)

                sell_intrinsic = max(sell_k - spot_now, 0)
                buy_intrinsic = max(buy_k - spot_now, 0)
                spread_loss = sell_intrinsic - buy_intrinsic

                pnl = (net_premium - spread_loss) * frac - 2 * scaled_commission(spot_entry)

                equity += pnl
                trades.append({
                    'entry': entry_date.strftime('%Y-%m-%d'),
                    'exit': date.strftime('%Y-%m-%d'),
                    'pnl': pnl,
                    'regime': regime[entry_date]
                })
                position = None

        current_month = (date.year, date.month)
        if position is None and current_month != last_entry_month:
            T = 30.0 / 365.0

            sell_k = spot * 0.97
            buy_k = spot * 0.94

            sell_prem = bs_put_price(spot, sell_k, T, RF_RATE, iv)
            buy_prem = bs_put_price(spot, buy_k, T, RF_RATE, iv)
            net_premium = sell_prem - buy_prem

            position = (date, sell_k, buy_k, net_premium, spot)
            last_entry_month = current_month

        equity_curve.append({'date': date, 'equity': equity})

    return pd.DataFrame(equity_curve).set_index('date'), trades


# ─── Strategy F: Covered Call Overwrite on QQQ ───
def strategy_f_covered_call_qqq():
    """
    Buy and hold QQQ, sell 102% OTM covered calls monthly.
    Premium income + stock appreciation (capped at strike).
    Track: stock_value + cumulative_premium_income.
    """
    equity_curve = []
    trades = []

    dates = close.index[close.index >= OOT_START]

    # Buy QQQ on day 1
    entry_spot = qqq[dates[0]]
    shares = CAPITAL / entry_spot
    cumulative_premium = 0.0

    call_position = None  # (entry_date, strike, premium_per_share)
    last_call_month = None

    for i, date in enumerate(dates):
        spot = qqq[date]
        iv = vix[date] * 1.2 / 100.0

        # Mark stock to market
        stock_value = shares * spot

        # Check if call expired
        if call_position is not None:
            call_entry_date, call_strike, call_premium = call_position
            days_held = (date - call_entry_date).days

            if days_held >= 30:
                comm = scaled_commission(spot)
                if spot > call_strike:
                    # Called away: sell at strike, rebuy at market
                    # Net effect: lose (spot - strike) per share, keep premium
                    assignment_cost = (spot - call_strike) * shares
                    call_pnl = call_premium * shares - assignment_cost - comm
                else:
                    # Expired worthless: keep full premium
                    call_pnl = call_premium * shares - comm

                cumulative_premium += call_pnl
                trades.append({
                    'entry': call_entry_date.strftime('%Y-%m-%d'),
                    'exit': date.strftime('%Y-%m-%d'),
                    'pnl': call_pnl,
                    'regime': regime[call_entry_date]
                })
                call_position = None

        # Sell new covered call on first of month
        current_month = (date.year, date.month)
        if call_position is None and current_month != last_call_month:
            T = 30.0 / 365.0
            call_strike = spot * 1.02
            call_premium = bs_call_price(spot, call_strike, T, RF_RATE, iv)
            call_position = (date, call_strike, call_premium)
            last_call_month = current_month

        # Total equity = stock value + cumulative premium income
        total_eq = stock_value + cumulative_premium
        equity_curve.append({'date': date, 'equity': total_eq})

    return pd.DataFrame(equity_curve).set_index('date'), trades


# ─── Metrics Engine ───
def compute_metrics(equity_curve, trades, name):
    """Compute risk-adjusted metrics + 5-gate validation."""
    eq = equity_curve['equity']

    # Daily returns
    daily_ret = eq.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)

    # Basic metrics
    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / max(len(daily_ret), 1)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 1e-6
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 1e-6
    sortino = ann_ret / downside_vol

    # Max drawdown
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n_trades
        profit_factor = (sum(wins) / abs(sum(losses))) if sum(losses) != 0 else float('inf')
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
    else:
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0

    # Regime analysis
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']

    def regime_sharpe(trade_list):
        if len(trade_list) < 2:
            return 0.0
        pnls = [t['pnl'] for t in trade_list]
        if np.std(pnls) == 0:
            return 0.0
        return np.mean(pnls) / np.std(pnls) * np.sqrt(252 / max(len(trade_list), 1))

    sharpe_bull = regime_sharpe(bull_trades)
    sharpe_bear = regime_sharpe(bear_trades)

    max_regime = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_regime if max_regime > 0 else 0

    # Permutation test: randomly assign trade P&Ls to random dates
    # to test if the TIMING of entries matters (not just the distribution)
    if n_trades >= 5:
        pnl_array = np.array([t['pnl'] for t in trades])
        actual_total = np.sum(pnl_array)
        actual_sharpe_trade = np.mean(pnl_array) / (np.std(pnl_array) + 1e-10)

        count_better = 0
        for _ in range(PERMUTATION_ITERS):
            # Randomly flip signs of P&Ls (tests if the strategy has real edge
            # vs random long/short assignment)
            signs = np.random.choice([-1, 1], size=len(pnl_array))
            shuffled_pnl = pnl_array * signs
            shuf_total = np.sum(shuffled_pnl)
            if shuf_total >= actual_total:
                count_better += 1
        perm_p = count_better / PERMUTATION_ITERS
    else:
        perm_p = 1.0

    # 5-gate validation
    gate_sharpe = bool(sharpe > 0.5)
    gate_perm = bool(perm_p < 0.05)
    gate_regime = bool(regime_gap < 0.5)
    gate_mdd = bool(max_dd > -0.50)
    gate_trades = bool(n_trades >= 20)
    gates_passed = sum([gate_sharpe, gate_perm, gate_regime, gate_mdd, gate_trades])

    result = {
        'variant': name,
        'total_return_pct': round(total_ret * 100, 2),
        'ann_return_pct': round(ann_ret * 100, 2),
        'ann_volatility_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 3),
        'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 999.0,
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'final_equity': round(eq.iloc[-1], 2),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
        'perm_p_value': round(perm_p, 4),
        'validation': {
            'sharpe_gt_0.5': gate_sharpe,
            'perm_p_lt_0.05': gate_perm,
            'regime_gap_lt_0.5': gate_regime,
            'mdd_gt_neg50': gate_mdd,
            'trades_gte_20': gate_trades,
            'gates_passed': f"{gates_passed}/5",
            'PASS': bool(gates_passed == 5)
        }
    }

    return result


# ─── Run All Variants ───
print("\n" + "="*70)
print("OPTIONS PREMIUM CAPTURE BACKTEST")
print(f"OOT Period: {OOT_START} to {END} | Capital: ${CAPITAL}")
print("="*70)

strategies = {
    'A_Weekly_CSP_SPY': strategy_a_weekly_csp_spy,
    'B_Monthly_CSP_QQQ': strategy_b_monthly_csp_qqq,
    'C_VIX_Gated_CSP': strategy_c_vix_gated_csp,
    'D_Iron_Condor_SPY': strategy_d_iron_condor_spy,
    'E_Put_Spread_QQQ': strategy_e_put_spread_qqq,
    'F_Covered_Call_QQQ': strategy_f_covered_call_qqq,
}

results = []

for name, fn in strategies.items():
    print(f"\nRunning {name}...")
    try:
        eq_curve, trades = fn()
        metrics = compute_metrics(eq_curve, trades, name)
        results.append(metrics)

        v = metrics['validation']
        status = "PASS" if v['PASS'] else f"FAIL ({v['gates_passed']})"
        print(f"  Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
              f"WR={metrics['win_rate']:.1%}  PF={metrics['profit_factor']:.2f}  "
              f"MDD={metrics['max_drawdown_pct']:.1f}%  Trades={metrics['n_trades']}  "
              f"Return={metrics['total_return_pct']:.1f}%  [{status}]")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({'variant': name, 'error': str(e)})

# ─── Summary Table ───
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD':>7} {'Return':>8} {'Trades':>6} {'Gate':>6}")
print("-" * 85)

for r in results:
    if 'error' in r:
        print(f"{r['variant']:<25} ERROR: {r['error']}")
        continue
    v = r['validation']
    status = "PASS" if v['PASS'] else v['gates_passed']
    print(f"{r['variant']:<25} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
          f"{r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
          f"{r['max_drawdown_pct']:>6.1f}% {r['total_return_pct']:>7.1f}% "
          f"{r['n_trades']:>6} {status:>6}")

# ─── Correlation Analysis ───
print("\n" + "="*70)
print("CORRELATION WITH BUY-AND-HOLD SPY")
print("="*70)
spy_oot = spy[spy.index >= OOT_START]
spy_ret = spy_oot.pct_change().dropna()

for name, fn in strategies.items():
    try:
        eq_curve, _ = fn()
        strat_ret = eq_curve['equity'].pct_change().dropna()
        # Align
        common = spy_ret.index.intersection(strat_ret.index)
        if len(common) > 10:
            corr = spy_ret.loc[common].corr(strat_ret.loc[common])
            print(f"  {name}: {corr:.3f}")
    except:
        pass

# ─── Save Results ───
output = {
    'metadata': {
        'description': 'Options Premium Capture Backtest - BS-simulated premiums',
        'oot_period': f'{OOT_START} to {END}',
        'capital': CAPITAL,
        'method': 'Black-Scholes with VIX as IV proxy',
        'iv_model': 'SPY: VIX/100, QQQ: VIX*1.2/100',
        'risk_free_rate': RF_RATE,
        'commission': f'${COMMISSION_PER_CONTRACT}/contract (fractionally scaled)',
        'sizing': 'Fractional: $645 notional per position',
        'permutation_iterations': PERMUTATION_ITERS,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    },
    'results': results,
    'ranking': sorted(
        [r for r in results if 'error' not in r],
        key=lambda x: x['sharpe'],
        reverse=True
    )
}

output_path = '/home/jupiter/Lvl3Quant/data/options_premium_capture_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")

# ─── Recommendation ───
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
passed = [r for r in results if 'error' not in r and r['validation']['PASS']]
if passed:
    best = max(passed, key=lambda x: x['sharpe'])
    print(f"Best passing variant: {best['variant']}")
    print(f"  Sharpe={best['sharpe']:.3f}, Sortino={best['sortino']:.3f}, "
          f"WR={best['win_rate']:.1%}, MDD={best['max_drawdown_pct']:.1f}%")
    print(f"  ${CAPITAL} -> ${best['final_equity']:.2f} ({best['total_return_pct']:.1f}% total)")
else:
    # Show best even if not passing all gates
    valid = [r for r in results if 'error' not in r]
    if valid:
        best = max(valid, key=lambda x: x['sharpe'])
        print(f"No variant passed all 5 gates. Best: {best['variant']}")
        print(f"  Sharpe={best['sharpe']:.3f}, Gates={best['validation']['gates_passed']}")
        print(f"  Failing gates: ", end="")
        for k, v in best['validation'].items():
            if k not in ('gates_passed', 'PASS') and not v:
                print(f"{k} ", end="")
        print()
    else:
        print("All variants errored.")
