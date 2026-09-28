#!/usr/bin/env python3
"""
Intraweek Reversal + Breadth Backtest
Academic basis: Birru 2018, Bogousslavsky 2016
6 variants tested OOT Jan 2022 - Jul 2026
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# Config
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
    'CRM', 'ADBE', 'NFLX', 'AVGO', 'COST', 'PEP', 'LLY', 'UNH',
    'V', 'MA', 'JPM', 'HD', 'INTC', 'MU', 'QCOM', 'PYPL'
]
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP']
ALL_TICKERS = list(set(STOCK_UNIVERSE + SECTOR_ETFS + ['SPY']))

START_DATE = '2021-06-01'
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002

RESULTS_PATH = Path('/home/jupiter/Lvl3Quant/data/intraweek_reversal_results.json')


def download_data():
    print("Downloading price data...")
    data = {}
    for ticker in ALL_TICKERS:
        try:
            df = yf.download(ticker, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 200:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[ticker] = df
        except Exception as e:
            print(f"  Skip {ticker}: {e}")
    print(f"  Downloaded {len(data)} tickers")
    return data


def compute_indicators(data):
    indicators = {}
    for ticker, df in data.items():
        d = df.copy()
        d['SMA200'] = d['Close'].rolling(200).mean()
        delta = d['Close'].diff()
        gain = delta.clip(lower=0).rolling(5).mean()
        loss = (-delta.clip(upper=0)).rolling(5).mean()
        rs = gain / loss.replace(0, np.nan)
        d['RSI5'] = 100 - (100 / (1 + rs))
        d['VolAvg20'] = d['Volume'].rolling(20).mean()
        d['DayOfWeek'] = d.index.dayofweek
        indicators[ticker] = d
    return indicators


class Portfolio:
    def __init__(self, capital):
        self.initial_capital = capital
        self.capital = capital
        self.equity_curve = []
        self.trades = []

    def execute_trade(self, ticker, entry_date, entry_price, exit_date, exit_price, shares):
        slip_entry = entry_price * (1 + SLIPPAGE_PCT)
        slip_exit = exit_price * (1 - SLIPPAGE_PCT)
        cost = shares * slip_entry
        revenue = shares * slip_exit
        pnl = revenue - cost
        ret = pnl / cost if cost > 0 else 0
        self.trades.append({
            'ticker': ticker,
            'entry_date': str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
            'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
            'entry_price': round(float(entry_price), 2),
            'exit_price': round(float(exit_price), 2),
            'shares': int(shares),
            'pnl': round(float(pnl), 2),
            'return': round(float(ret), 6),
        })
        self.capital += pnl

    def record_equity(self, date):
        self.equity_curve.append({
            'date': str(date.date()) if hasattr(date, 'date') else str(date),
            'equity': round(self.capital, 2)
        })

    def stats(self):
        if not self.trades:
            return {k: 0 for k in ['num_trades','total_return_pct','final_equity','sharpe',
                                    'sortino','profit_factor','win_rate','max_drawdown_pct',
                                    'avg_return_pct','avg_win_pct','avg_loss_pct']}
        rets = np.array([t['return'] for t in self.trades])
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        total_ret = (self.capital - self.initial_capital) / self.initial_capital

        if len(rets) > 1 and np.std(rets) > 0:
            trades_per_year = max(len(rets) / 4.5, 1)
            sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        downside = rets[rets < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            trades_per_year = max(len(rets) / 4.5, 1)
            sortino = (np.mean(rets) / np.std(downside)) * np.sqrt(trades_per_year)
        else:
            sortino = sharpe

        gross_profit = float(wins.sum()) if len(wins) > 0 else 0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.001
        pf = gross_profit / gross_loss

        eq = np.array([e['equity'] for e in self.equity_curve]) if self.equity_curve else np.array([self.initial_capital, self.capital])
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = float(dd.min())

        wr = float(len(wins) / len(rets)) if len(rets) > 0 else 0

        return {
            'num_trades': int(len(rets)),
            'total_return_pct': round(total_ret * 100, 2),
            'final_equity': round(float(self.capital), 2),
            'sharpe': round(float(sharpe), 3),
            'sortino': round(float(sortino), 3),
            'profit_factor': round(float(pf), 3),
            'win_rate': round(float(wr), 4),
            'max_drawdown_pct': round(float(max_dd * 100), 2),
            'avg_return_pct': round(float(np.mean(rets) * 100), 4),
            'avg_win_pct': round(float(np.mean(wins) * 100), 4) if len(wins) > 0 else 0,
            'avg_loss_pct': round(float(np.mean(losses) * 100), 4) if len(losses) > 0 else 0,
        }


def get_trading_weeks(spy_data, oot_start, oot_end):
    df = spy_data.loc[oot_start:oot_end].copy()
    df['week'] = df.index.isocalendar().week.astype(int)
    df['year'] = df.index.year
    df['dow'] = df.index.dayofweek

    weeks = []
    for (y, w), grp in df.groupby(['year', 'week']):
        days = grp.sort_index()
        dows = days['dow'].values
        dates = days.index
        mon_mask = dows <= 1
        wed_mask = (dows >= 2) & (dows <= 2)
        fri_mask = dows >= 3
        if mon_mask.any() and wed_mask.any() and fri_mask.any():
            mon = dates[mon_mask][0]
            wed = dates[wed_mask][-1]
            fri = dates[fri_mask][-1]
            weeks.append((mon, wed, fri))
    return weeks


def get_prior_friday(data_df, monday_date):
    prior = data_df.loc[:monday_date - timedelta(days=1)]
    if len(prior) > 0:
        return prior.index[-1]
    return None


def position_size(capital, price):
    if price <= 0 or capital <= 0:
        return 0
    return max(int(capital / price), 0)


# Strategy A: Wed Close Dip Buy
def strategy_a(indicators, weeks):
    print("\n[A] Wed Close Dip Buy (>3% drop Mon-Wed, above SMA200)")
    port = Portfolio(STARTING_CAPITAL)
    for mon, wed, fri in weeks:
        for ticker in STOCK_UNIVERSE:
            d = indicators.get(ticker)
            if d is None: continue
            prior_fri = get_prior_friday(d, mon)
            if prior_fri is None or wed not in d.index or fri not in d.index: continue
            close_prior_fri = float(d.loc[prior_fri, 'Close'])
            close_wed = float(d.loc[wed, 'Close'])
            sma200_wed = d.loc[wed, 'SMA200']
            if pd.isna(sma200_wed): continue
            mon_wed_ret = (close_wed - close_prior_fri) / close_prior_fri
            if mon_wed_ret < -0.03 and close_wed > float(sma200_wed):
                shares = position_size(port.capital, close_wed)
                if shares > 0:
                    close_fri = float(d.loc[fri, 'Close'])
                    port.execute_trade(ticker, wed, close_wed, fri, close_fri, shares)
        port.record_equity(fri)
    return port


# Strategy B: Breadth-Filtered Reversal
def strategy_b(indicators, weeks):
    print("[B] Breadth-Filtered Reversal (SPY RSI(5) < 30)")
    port = Portfolio(STARTING_CAPITAL)
    spy = indicators.get('SPY')
    if spy is None: return port
    for mon, wed, fri in weeks:
        if wed not in spy.index: continue
        spy_rsi = spy.loc[wed, 'RSI5']
        if pd.isna(spy_rsi) or float(spy_rsi) >= 30: continue
        for ticker in STOCK_UNIVERSE:
            d = indicators.get(ticker)
            if d is None: continue
            prior_fri = get_prior_friday(d, mon)
            if prior_fri is None or wed not in d.index or fri not in d.index: continue
            close_prior_fri = float(d.loc[prior_fri, 'Close'])
            close_wed = float(d.loc[wed, 'Close'])
            sma200_wed = d.loc[wed, 'SMA200']
            if pd.isna(sma200_wed): continue
            mon_wed_ret = (close_wed - close_prior_fri) / close_prior_fri
            if mon_wed_ret < -0.03 and close_wed > float(sma200_wed):
                shares = position_size(port.capital, close_wed)
                if shares > 0:
                    close_fri = float(d.loc[fri, 'Close'])
                    port.execute_trade(ticker, wed, close_wed, fri, close_fri, shares)
        port.record_equity(fri)
    return port


# Strategy C: Best 3 Weekly Dips
def strategy_c(indicators, weeks):
    print("[C] Best 3 Weekly Dips (bottom 3 by Mon-Wed return, >2% drop)")
    port = Portfolio(STARTING_CAPITAL)
    for mon, wed, fri in weeks:
        candidates = []
        for ticker in STOCK_UNIVERSE:
            d = indicators.get(ticker)
            if d is None: continue
            prior_fri = get_prior_friday(d, mon)
            if prior_fri is None or wed not in d.index or fri not in d.index: continue
            close_prior_fri = float(d.loc[prior_fri, 'Close'])
            close_wed = float(d.loc[wed, 'Close'])
            mon_wed_ret = (close_wed - close_prior_fri) / close_prior_fri
            if mon_wed_ret < -0.02:
                candidates.append((ticker, mon_wed_ret, close_wed, fri))
        candidates.sort(key=lambda x: x[1])
        n_picks = min(3, len(candidates))
        for ticker, ret, entry_p, exit_date in candidates[:n_picks]:
            d = indicators[ticker]
            shares = position_size(port.capital / max(n_picks, 1), entry_p)
            if shares > 0:
                exit_p = float(d.loc[exit_date, 'Close'])
                port.execute_trade(ticker, wed, entry_p, exit_date, exit_p, shares)
        port.record_equity(fri)
    return port


# Strategy D: Oversold + Volume Spike
def strategy_d(indicators, weeks):
    print("[D] Oversold + Volume Spike (>3% drop + 1.5x avg volume)")
    port = Portfolio(STARTING_CAPITAL)
    for mon, wed, fri in weeks:
        for ticker in STOCK_UNIVERSE:
            d = indicators.get(ticker)
            if d is None: continue
            prior_fri = get_prior_friday(d, mon)
            if prior_fri is None or wed not in d.index or fri not in d.index: continue
            close_prior_fri = float(d.loc[prior_fri, 'Close'])
            close_wed = float(d.loc[wed, 'Close'])
            vol_wed = float(d.loc[wed, 'Volume'])
            vol_avg = d.loc[wed, 'VolAvg20']
            if pd.isna(vol_avg) or float(vol_avg) == 0: continue
            mon_wed_ret = (close_wed - close_prior_fri) / close_prior_fri
            if mon_wed_ret < -0.03 and vol_wed > 1.5 * float(vol_avg):
                shares = position_size(port.capital, close_wed)
                if shares > 0:
                    close_fri = float(d.loc[fri, 'Close'])
                    port.execute_trade(ticker, wed, close_wed, fri, close_fri, shares)
        port.record_equity(fri)
    return port


# Strategy E: Multi-Day Reversal
def strategy_e(indicators):
    print("[E] Multi-Day Reversal (>5% drop over 3 days, above SMA200, hold 5 days)")
    port = Portfolio(STARTING_CAPITAL)
    for ticker in STOCK_UNIVERSE:
        d = indicators.get(ticker)
        if d is None: continue
        df_oot = d.loc[OOT_START:OOT_END]
        if len(df_oot) < 10: continue
        i = 2
        while i < len(df_oot):
            idx = df_oot.index
            close_today = float(df_oot.iloc[i]['Close'])
            close_3ago = float(df_oot.iloc[i - 2]['Close'])
            sma200 = df_oot.iloc[i]['SMA200']
            if pd.isna(sma200):
                i += 1
                continue
            ret_3d = (close_today - close_3ago) / close_3ago
            if ret_3d < -0.05 and close_today > float(sma200):
                exit_idx = min(i + 5, len(df_oot) - 1)
                exit_price = float(df_oot.iloc[exit_idx]['Close'])
                shares = position_size(port.capital, close_today)
                if shares > 0:
                    port.execute_trade(ticker, idx[i], close_today, idx[exit_idx], exit_price, shares)
                    port.record_equity(idx[exit_idx])
                i = exit_idx + 1
            else:
                i += 1
    port.equity_curve.sort(key=lambda x: x['date'])
    port.trades.sort(key=lambda x: x['entry_date'])
    return port


# Strategy F: Sector Rotation Reversal
def strategy_f(indicators, weeks):
    print("[F] Sector Rotation Reversal (worst sector ETF, >1% drop, hold 5 days)")
    port = Portfolio(STARTING_CAPITAL)
    for mon, wed, fri in weeks:
        candidates = []
        for etf in SECTOR_ETFS:
            d = indicators.get(etf)
            if d is None: continue
            if mon not in d.index or fri not in d.index or wed not in d.index: continue
            close_mon = float(d.loc[mon, 'Close'])
            close_wed = float(d.loc[wed, 'Close'])
            week_ret = (close_wed - close_mon) / close_mon
            if week_ret < -0.01:
                candidates.append((etf, week_ret, close_wed))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            etf, ret, entry_p = candidates[0]
            d = indicators[etf]
            wed_loc = d.index.get_loc(wed)
            exit_idx = min(wed_loc + 5, len(d) - 1)
            exit_date = d.index[exit_idx]
            exit_p = float(d.iloc[exit_idx]['Close'])
            shares = position_size(port.capital, entry_p)
            if shares > 0:
                port.execute_trade(etf, wed, entry_p, exit_date, exit_p, shares)
        port.record_equity(fri)
    return port


# Regime Analysis
def regime_analysis(port, spy_data):
    if not port.trades:
        return {'bull_sharpe': 0, 'bear_sharpe': 0, 'regime_gap': 0, 'bull_trades': 0, 'bear_trades': 0}
    spy = spy_data.copy()
    spy['SMA200'] = spy['Close'].rolling(200).mean()
    bull_rets, bear_rets = [], []
    for t in port.trades:
        entry_date = pd.Timestamp(t['entry_date'])
        mask = spy.index <= entry_date
        if not mask.any(): continue
        nearest = spy.index[mask][-1]
        sma_val = spy.loc[nearest, 'SMA200']
        is_bull = float(spy.loc[nearest, 'Close']) > float(sma_val) if not pd.isna(sma_val) else True
        if is_bull:
            bull_rets.append(t['return'])
        else:
            bear_rets.append(t['return'])

    def sharpe_from_rets(rets):
        if len(rets) < 2: return 0.0
        r = np.array(rets)
        if np.std(r) == 0: return 0.0
        return float((np.mean(r) / np.std(r)) * np.sqrt(max(len(r) / 4.5, 1)))

    bull_s = sharpe_from_rets(bull_rets)
    bear_s = sharpe_from_rets(bear_rets)
    denom = max(abs(bull_s), abs(bear_s), 0.001)
    gap = abs(bull_s - bear_s) / denom
    return {
        'bull_sharpe': round(bull_s, 3), 'bear_sharpe': round(bear_s, 3),
        'bull_trades': len(bull_rets), 'bear_trades': len(bear_rets),
        'regime_gap': round(gap, 3),
    }


# Permutation Test
def permutation_test(port, n_perms=1000):
    if len(port.trades) < 5: return 1.0
    rets = np.array([t['return'] for t in port.trades])
    observed_mean = np.mean(rets)
    count_ge = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(rets))
        perm_mean = np.mean(rets * signs)
        if perm_mean >= observed_mean:
            count_ge += 1
    return round(count_ge / n_perms, 4)


# 5-Gate Validation
def validate_5gate(stats, regime, perm_p):
    gates = {
        'G1_sharpe_gt_0.5': stats['sharpe'] > 0.5,
        'G2_perm_p_lt_0.05': perm_p < 0.05,
        'G3_regime_gap_lt_0.5': regime['regime_gap'] < 0.5,
        'G4_maxdd_gt_neg50': stats['max_drawdown_pct'] > -50,
        'G5_trades_gte_20': stats['num_trades'] >= 20,
    }
    gates['all_passed'] = all(gates.values())
    return gates


def main():
    data = download_data()
    indicators = compute_indicators(data)
    spy = data.get('SPY')
    if spy is None:
        print("ERROR: Could not download SPY data")
        return

    weeks = get_trading_weeks(spy, OOT_START, OOT_END)
    print(f"  {len(weeks)} trading weeks in OOT period")

    strategies = {
        'A_wed_close_dip_buy': lambda: strategy_a(indicators, weeks),
        'B_breadth_filtered_reversal': lambda: strategy_b(indicators, weeks),
        'C_best_3_weekly_dips': lambda: strategy_c(indicators, weeks),
        'D_oversold_volume_spike': lambda: strategy_d(indicators, weeks),
        'E_multi_day_reversal': lambda: strategy_e(indicators),
        'F_sector_rotation_reversal': lambda: strategy_f(indicators, weeks),
    }

    results = {}
    for name, run_fn in strategies.items():
        port = run_fn()
        stats = port.stats()
        regime = regime_analysis(port, spy)
        perm_p = permutation_test(port)
        gates = validate_5gate(stats, regime, perm_p)
        results[name] = {
            'stats': stats, 'regime': regime,
            'perm_p_value': perm_p, 'gates': gates,
        }
        passed = "PASS" if gates['all_passed'] else "FAIL"
        print(f"  [{passed}] {name}: {stats['num_trades']} trades, "
              f"Sharpe={stats['sharpe']}, Sortino={stats['sortino']}, "
              f"PF={stats['profit_factor']}, WR={stats['win_rate']:.1%}, "
              f"MaxDD={stats['max_drawdown_pct']:.1f}%, "
              f"Return={stats['total_return_pct']:.1f}%, "
              f"p={perm_p}, RegimeGap={regime['regime_gap']:.2f}")

    output = {
        'metadata': {
            'strategy': 'Intraweek Reversal + Breadth',
            'academic_basis': 'Birru 2018, Bogousslavsky 2016',
            'oot_period': f'{OOT_START} to {OOT_END}',
            'starting_capital': STARTING_CAPITAL,
            'slippage_pct': SLIPPAGE_PCT,
            'commission': 0,
            'universe_stocks': STOCK_UNIVERSE,
            'universe_sectors': SECTOR_ETFS,
            'run_timestamp': datetime.now().isoformat(),
        },
        'variants': results,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\n" + "=" * 100)
    print(f"{'Variant':<35} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'Ret%':>7} {'5-Gate':>7}")
    print("-" * 100)
    for name, r in results.items():
        s = r['stats']
        g = r['gates']
        label = "PASS" if g['all_passed'] else "FAIL"
        print(f"{name:<35} {s['num_trades']:>6} {s['sharpe']:>7.3f} {s['sortino']:>8.3f} "
              f"{s['profit_factor']:>6.2f} {s['win_rate']:>5.1%} {s['max_drawdown_pct']:>6.1f}% "
              f"{s['total_return_pct']:>6.1f}% {label:>7}")
    print("=" * 100)

    print("\n5-Gate Detail:")
    for name, r in results.items():
        g = r['gates']
        fails = [k for k, v in g.items() if k != 'all_passed' and not v]
        if fails:
            print(f"  {name}: FAILED gates: {', '.join(fails)}")
        else:
            print(f"  {name}: ALL GATES PASSED")


if __name__ == '__main__':
    main()
