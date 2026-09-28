#!/usr/bin/env python3
"""
Multi-Timeframe Trend Following — Walk-Forward Backtest
HC #685 (growth lanes), HC #684 (dynamic exits), HC #683 (real data)

Macro Filter: SPY vs 200/50-day SMA → risk-on / cautious / risk-off
Stock Selection: 12-1 month momentum, above 50-day SMA, top 20 equal weight
Dynamic Exits: 7% trailing stop, 3-day SMA breakdown, macro exit, partial profit
Walk-forward: 60-month lookback, 1-month OOT, sliding window
Regime test: R1 regime-agnostic (gap < 0.50)
Permutation test: 1000 shuffles
Benchmark: SPY buy-and-hold
Costs: 0.3% slippage per trade (each way)
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research'
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'trend_following_results.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SLIPPAGE_PER_TRADE = 0.003  # 0.3% per trade (each way)
N_HOLDINGS = 20
TRAILING_STOP_PCT = 0.07    # 7% from high
SMA_BREAKDOWN_DAYS = 3      # consecutive days below 50-SMA to exit
PARTIAL_PROFIT_THRESHOLD = 0.50  # sell half at 50% gain
TRAIN_MONTHS = 60
TEST_MONTHS = 1

###############################################################################
# 1. DATA ACQUISITION
###############################################################################

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia with large fallback."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        if len(tickers) > 200:
            return tickers
    except:
        pass
    # Large fallback: ~200 of the most liquid S&P 500 names
    return [
        'AAPL','MSFT','AMZN','GOOGL','GOOG','META','NVDA','TSLA','BRK-B','UNH',
        'JNJ','XOM','JPM','V','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP','KO',
        'AVGO','COST','TMO','MCD','WMT','CSCO','ACN','ABT','DHR','NEE','LIN',
        'BMY','PM','TXN','UNP','RTX','AMGN','HON','LOW','QCOM','INTC','COP',
        'IBM','SBUX','CAT','GS','BA','MDLZ','BLK','ADP','DE','ADI','GILD',
        'ISRG','SYK','VRTX','BKNG','REGN','TJX','ZTS','CI','CB','PLD',
        'SO','DUK','CME','SLB','CL','USB','ITW','BDX','MO','EOG','WM','APD',
        'NOC','ICE','FDX','GD','FCX','PNC','ORLY','AZO','SHW','NSC','EMR',
        'MCK','TGT','PSX','VLO','OXY','AIG','AFL','D','HUM','MET','PRU',
        'MSCI','TRV','ALL','AEP','SPG','WELL','PSA','O','AMT','CCI','EQIX',
        # Extend with more S&P 500 names
        'LRCX','KLAC','MCHP','ON','SNPS','CDNS','ANSS','FTNT','PANW','CRWD',
        'ZS','DDOG','SNOW','NOW','CRM','ADBE','ORCL','INTU','PYPL','SQ',
        'ABNB','UBER','DASH','COIN','ANET','NFLX','DIS','CMCSA','T','VZ',
        'TMUS','CHTR','NXPI','MRVL','MU','AMD','TER','SWKS','MPWR','KEYS',
        'A','WAT','MTD','TECH','IQV','IDXX','DXCM','ALGN','HOLX','EW',
        'BSX','MDT','BDX','ZBH','BAX','ABT','TMO','DHR','SYK','ISRG',
        'MSI','TYL','CDNS','SNPS','ANSS','PTC','MANH','FICO','PAYC','PCTY',
        'CPRT','CTAS','PAYX','ADP','BR','FIS','FISV','GPN','FLT','WEX',
        'AJG','AON','MMC','MARSH','WTW','BRO','RJF','SCHW','MS','GS',
        'BLK','BEN','TROW','IVZ','STT','BK','NTRS','CME','ICE','NDAQ',
        'SPGI','MSCI','MCO','VRSK','FDS','MKTX','TW','CBOE','LPLA',
        'PH','ROK','EMR','ETN','AME','GNRC','DOV','XYL','ROP','IEX',
        'ITW','CMI','PCAR','FAST','GWW','WSO','LII','SNA','SWK','TTC',
        'CAT','DE','AGCO','CNH','TT','CARR','JCI','LHX','HII','NOC',
        'LMT','RTX','GD','TDG','HEI','TXT','SPR','AXON','LDOS','BAH',
        'ECL','APD','LIN','SHW','PPG','RPM','VMC','MLM','EXP','CX',
        'NUE','STLD','RS','CLF','X','AA','FCX','NEM','GOLD','FNV',
        'WPM','AEM','KGC','RGLD','SCCO','TECK',
        'NKE','LULU','DECK','CROX','TPR','RL','PVH','HBI','VFC','UAA',
        'ROST','TJX','BURL','GPS','ANF','AEO','FIVE','DG','DLTR','WMT',
        'COST','TGT','BJ','OLLI',
        'F','GM','TM','HMC','STLA','RIVN','LCID',
        'UPS','FDX','CHRW','XPO','JBHT','ODFL','SAIA','KNX','LSTR',
        'DAL','UAL','AAL','LUV','ALK','JBLU',
        'MAR','HLT','H','WH','IHG','ABNB',
        'SBUX','MCD','CMG','YUM','DPZ','QSR','WING','TXRH','DRI','CAKE',
        'PEP','KO','MNST','KDP','STZ','DEO','BF-B','SAM',
        'PG','CL','KMB','CHD','CLX','SPB','EPC','EL','COTY',
        'WBA','CVS','MCK','CAH','ABC','CI','HUM','UNH','ELV','CNC','MOH',
        'LH','DGX','ILMN','TMO','A','BIO','PKI','TECH','TFX','WAT',
    ]


def download_prices(tickers, start='2013-01-01', end='2026-07-13'):
    """Download adjusted close prices with batching."""
    import yfinance as yf

    all_prices = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        ticker_str = ' '.join(batch)
        try:
            data = yf.download(ticker_str, start=start, end=end, progress=False,
                             group_by='ticker', threads=True)
            for t in batch:
                try:
                    if len(batch) > 1:
                        price_col = 'Adj Close' if ('Adj Close' in data[t].columns
                                                     if hasattr(data[t], 'columns') else False) else 'Close'
                        col = data[t][price_col].dropna()
                    else:
                        price_col = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
                        col = data[price_col].dropna()
                        if isinstance(col, pd.DataFrame):
                            col = col.iloc[:, 0]
                    if len(col) > 252:
                        all_prices[t] = col
                except:
                    pass
        except Exception as e:
            print(f"  Batch {i//batch_size} error: {e}")
        time.sleep(0.5)

    prices = pd.DataFrame(all_prices)
    prices.index = pd.to_datetime(prices.index)
    print(f"  Downloaded {prices.shape[1]} stocks, {prices.shape[0]} days")
    return prices


def download_spy(start='2013-01-01', end='2026-07-13'):
    """Download SPY and SHY (bond ETF for risk-off) separately."""
    import yfinance as yf

    def _download_single(ticker, start, end):
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close'].iloc[:, 0] if 'Close' in data.columns.get_level_values(0) else data.iloc[:, 0]
        else:
            close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        return close.dropna().squeeze()

    spy_close = _download_single('SPY', start, end)

    try:
        shy_close = _download_single('SHY', start, end)
        if len(shy_close) < 100:
            shy_close = None
    except:
        shy_close = None

    return spy_close, shy_close


###############################################################################
# 2. MACRO TREND FILTER
###############################################################################

def compute_macro_regime(spy_close):
    """
    Classify market regime daily:
      risk_on (1.0):  SPY > 200-SMA
      cautious (0.5): SPY < 200-SMA but > 50-SMA
      risk_off (0.0): SPY < both 200-SMA and 50-SMA
    """
    sma200 = spy_close.rolling(200).mean()
    sma50 = spy_close.rolling(50).mean()

    regime = pd.Series(0.0, index=spy_close.index)
    regime[spy_close > sma200] = 1.0
    regime[(spy_close <= sma200) & (spy_close > sma50)] = 0.5
    regime[(spy_close <= sma200) & (spy_close <= sma50)] = 0.0

    return regime


###############################################################################
# 3. STOCK SELECTION
###############################################################################

def compute_momentum_12_1(prices):
    """12-month return minus most recent month (Fama-French classic)."""
    # 12-month return = pct_change(252)
    # Skip last month = shift by 21 days first
    return prices.shift(21).pct_change(231)


def compute_sma50(prices):
    """50-day simple moving average for each stock."""
    return prices.rolling(50).mean()


def select_stocks(prices, date, n=20):
    """
    Select top N momentum stocks that are above their 50-day SMA.
    Returns list of tickers.
    """
    # Get data up to this date
    hist = prices.loc[:date]
    if len(hist) < 252:
        return []

    # 12-1 month momentum
    mom12_1 = compute_momentum_12_1(hist)
    if len(mom12_1) < 1:
        return []
    mom_scores = mom12_1.iloc[-1].dropna()

    # 50-day SMA filter: stock must be above its SMA
    sma50 = hist.iloc[-50:].mean() if len(hist) >= 50 else hist.mean()
    current_price = hist.iloc[-1]

    # Filter: above 50-SMA
    above_sma = current_price > sma50
    valid = above_sma[above_sma].index.intersection(mom_scores.index)

    if len(valid) < n:
        # Relax: just use all with valid momentum
        valid = mom_scores.index

    scores = mom_scores[valid].sort_values(ascending=False)

    # Filter out negative momentum
    scores = scores[scores > 0]

    return scores.head(n).index.tolist()


###############################################################################
# 4. DYNAMIC EXIT MANAGEMENT
###############################################################################

class Position:
    """Track a single stock position with dynamic exit logic."""
    def __init__(self, ticker, entry_price, weight):
        self.ticker = ticker
        self.entry_price = entry_price
        self.weight = weight  # portfolio weight
        self.high = entry_price
        self.below_sma_count = 0
        self.partial_taken = False

    def update(self, current_price, sma50_val):
        """Update position state. Returns (remaining_weight, exit_weight, exit_reason)."""
        exit_weight = 0.0
        exit_reason = None

        # Update trailing high
        if current_price > self.high:
            self.high = current_price

        # Check trailing stop: 7% from high
        drawdown = (self.high - current_price) / self.high
        if drawdown >= TRAILING_STOP_PCT:
            exit_weight = self.weight
            self.weight = 0.0
            return self.weight, exit_weight, 'trailing_stop'

        # Check SMA breakdown: 3 consecutive days below 50-SMA
        if not np.isnan(sma50_val) and current_price < sma50_val:
            self.below_sma_count += 1
        else:
            self.below_sma_count = 0

        if self.below_sma_count >= SMA_BREAKDOWN_DAYS:
            exit_weight = self.weight
            self.weight = 0.0
            return self.weight, exit_weight, 'sma_breakdown'

        # Partial profit: sell half at 50% gain
        if not self.partial_taken:
            gain = (current_price - self.entry_price) / self.entry_price
            if gain >= PARTIAL_PROFIT_THRESHOLD:
                exit_weight = self.weight / 2
                self.weight -= exit_weight
                self.partial_taken = True
                return self.weight, exit_weight, 'partial_profit'

        return self.weight, 0.0, None


###############################################################################
# 5. WALK-FORWARD BACKTEST
###############################################################################

def run_trend_following_backtest(prices, spy_close, shy_close=None):
    """
    Walk-forward trend following backtest:
    - 60-month lookback, 1-month OOT, sliding
    - Macro filter controls allocation %
    - Daily dynamic exits
    - 0.3% slippage per trade
    """
    # Precompute macro regime
    macro_regime = compute_macro_regime(spy_close)

    # Precompute 50-day SMAs for all stocks
    sma50_all = prices.rolling(50).mean()

    # SHY returns for risk-off periods
    if shy_close is not None:
        shy_returns = shy_close.pct_change()
    else:
        shy_returns = None

    # Monthly rebalance dates
    monthly = prices.resample('ME').last().index
    monthly = monthly[monthly >= prices.index[max(252, 0)]]

    if len(monthly) < TRAIN_MONTHS + TEST_MONTHS + 1:
        print(f"Not enough data: {len(monthly)} months")
        return None, {}

    all_returns = []
    all_dates = []
    trade_count = 0
    time_in_market_days = 0
    total_days = 0
    exit_reasons = {'trailing_stop': 0, 'sma_breakdown': 0, 'partial_profit': 0, 'macro_exit': 0, 'rebalance': 0}

    positions = {}  # ticker -> Position

    for i in range(TRAIN_MONTHS, len(monthly) - TEST_MONTHS + 1):
        rebal_date = monthly[i]

        # Determine test period
        if i + TEST_MONTHS < len(monthly):
            test_end = monthly[i + TEST_MONTHS]
        else:
            test_end = prices.index[-1]

        test_prices = prices.loc[rebal_date:test_end]
        if len(test_prices) < 2:
            continue

        # Get macro regime at rebalance
        if rebal_date in macro_regime.index:
            alloc_pct = macro_regime.loc[rebal_date]
        else:
            # Find nearest date
            nearest = macro_regime.index[macro_regime.index <= rebal_date]
            alloc_pct = macro_regime.iloc[-1] if len(nearest) == 0 else macro_regime.loc[nearest[-1]]

        # Select stocks if risk-on or cautious
        if alloc_pct > 0:
            selected = select_stocks(prices, rebal_date, n=N_HOLDINGS)
        else:
            selected = []

        # Build new positions — keep existing positions that are still selected
        prev_tickers = set(positions.keys())
        new_tickers = set(selected)
        exits = prev_tickers - new_tickers
        entries = new_tickers - prev_tickers
        keeps = prev_tickers & new_tickers

        # Count trades (exits + entries)
        trade_count += len(exits) + len(entries)
        for t in exits:
            exit_reasons['rebalance'] += 1

        per_stock_weight = alloc_pct / N_HOLDINGS if N_HOLDINGS > 0 else 0

        # Remove exited positions
        for t in exits:
            del positions[t]

        # Reweight kept positions
        for t in keeps:
            positions[t].weight = per_stock_weight

        # Add new entries with slippage cost on day 0
        entry_slippage_cost = 0.0
        for t in entries:
            if t in test_prices.columns and not pd.isna(test_prices[t].iloc[0]):
                entry_p = test_prices[t].iloc[0]
                positions[t] = Position(t, entry_p, per_stock_weight)
                entry_slippage_cost += per_stock_weight * SLIPPAGE_PER_TRADE

        # Also slippage on exits
        exit_slippage_cost = len(exits) * per_stock_weight * SLIPPAGE_PER_TRADE

        # Cash weight (not invested in stocks)
        cash_weight = 1.0 - sum(p.weight for p in positions.values())

        # Record rebalance cost on first day
        rebal_slippage = entry_slippage_cost + exit_slippage_cost

        # Daily simulation
        first_day_of_window = True
        for d in range(1, len(test_prices)):
            date = test_prices.index[d]
            total_days += 1

            # Check macro regime daily — if risk-off, exit ALL
            if date in macro_regime.index:
                current_regime = macro_regime.loc[date]
            else:
                nearest = macro_regime.index[macro_regime.index <= date]
                current_regime = macro_regime.loc[nearest[-1]] if len(nearest) > 0 else 0

            # Macro exit: if regime goes to 0, exit everything
            if current_regime == 0 and len(positions) > 0:
                for t in list(positions.keys()):
                    exit_reasons['macro_exit'] += 1
                    trade_count += 1
                positions = {}
                cash_weight = 1.0

            # Compute daily return
            day_return = 0.0
            exited_this_day = []

            for t, pos in list(positions.items()):
                if t not in test_prices.columns:
                    continue
                p = test_prices[t].iloc[d]
                p_prev = test_prices[t].iloc[d-1]
                if pd.isna(p) or pd.isna(p_prev) or p_prev == 0:
                    continue

                stock_ret = (p - p_prev) / p_prev
                day_return += pos.weight * stock_ret

                # Get SMA50 for this stock on this date
                try:
                    sma50_val = sma50_all.loc[:date, t].iloc[-1]
                except:
                    sma50_val = np.nan

                # Dynamic exit check
                remaining, exit_w, reason = pos.update(p, sma50_val)
                if reason:
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                    # Apply slippage on exit
                    day_return -= exit_w * SLIPPAGE_PER_TRADE
                    trade_count += 1
                    cash_weight += exit_w
                    if remaining <= 0:
                        exited_this_day.append(t)

            # Remove fully exited positions
            for t in exited_this_day:
                del positions[t]

            # SHY return on cash portion (risk-off allocation)
            if shy_returns is not None and date in shy_returns.index and cash_weight > 0:
                shy_ret = shy_returns.loc[date]
                if not pd.isna(shy_ret):
                    day_return += cash_weight * shy_ret

            # Apply rebalance slippage on first day of window
            if first_day_of_window:
                day_return -= rebal_slippage
                first_day_of_window = False

            if len(positions) > 0:
                time_in_market_days += 1

            all_returns.append(day_return)
            all_dates.append(date)

    if not all_returns:
        return None, {}

    returns = pd.Series(all_returns, index=pd.DatetimeIndex(all_dates))
    returns = returns[~returns.index.duplicated(keep='last')]
    returns = returns.sort_index()

    # Apply entry slippage (approximation: spread across rebalance periods)
    # Already handled per-exit above; for entries, deduct at rebalance
    # We approximate: each entry costs SLIPPAGE_PER_TRADE on weight
    # Total entry slippage already implicitly low since we're equal-weight monthly

    meta = {
        'total_trades': trade_count,
        'pct_time_in_market': time_in_market_days / max(total_days, 1),
        'exit_reasons': exit_reasons,
        'total_days': total_days,
    }

    return returns, meta


###############################################################################
# 6. METRICS & ANALYSIS
###############################################################################

def compute_metrics(returns, name="Strategy"):
    """Compute risk-adjusted metrics."""
    if returns is None or len(returns) < 30:
        return {}

    ann_factor = 252
    total_days = len(returns)
    years = total_days / ann_factor

    cum_ret = (1 + returns).prod() - 1
    cagr = (1 + cum_ret) ** (1 / max(years, 0.1)) - 1

    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor)) if returns.std() > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(ann_factor) if len(returns[returns < 0]) > 0 else 1e-6
    sortino = (returns.mean() * ann_factor) / downside

    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    win_rate = (returns > 0).mean()

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'name': name,
        'CAGR': round(cagr, 4),
        'CAGR_pct': f"{cagr:.1%}",
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Max_DD': round(max_dd, 4),
        'Max_DD_pct': f"{max_dd:.1%}",
        'Win_Rate': round(win_rate, 4),
        'Win_Rate_pct': f"{win_rate:.1%}",
        'Profit_Factor': round(pf, 2),
        'Total_Return': f"{cum_ret:.1%}",
        'N_Days': total_days,
        'Ann_Vol': f"{ann_vol:.1%}",
    }


def regime_analysis(returns, spy_close):
    """
    Stratify by bull/bear regime (SPY above/below 200-day MA).
    For R1 test: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|) < 0.50
    """
    sma200 = spy_close.rolling(200).mean()

    # Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
    bull_mask = spy_close > sma200
    bear_mask = spy_close <= sma200

    common = returns.index.intersection(spy_close.index)
    if len(common) < 60:
        return None, None, None, {}

    r = returns.loc[common]
    bull = bull_mask.reindex(common)
    bear = bear_mask.reindex(common)

    bull_returns = r[bull == True]
    bear_returns = r[bear == True]

    bull_metrics = compute_metrics(bull_returns, "Bull Regime (SPY>200MA)")
    bear_metrics = compute_metrics(bear_returns, "Bear Regime (SPY<200MA)")

    # Also classify by daily SPY return (green/red/flat day)
    spy_daily = spy_close.pct_change()
    spy_daily_common = spy_daily.reindex(common).fillna(0)
    green_mask = spy_daily_common > 0.001
    red_mask = spy_daily_common < -0.001
    flat_mask = ~green_mask & ~red_mask

    green_ret = r[green_mask]
    red_ret = r[red_mask]
    flat_ret = r[flat_mask]

    day_metrics = {
        'green_days': compute_metrics(green_ret, "Green Days") if len(green_ret) > 30 else {},
        'red_days': compute_metrics(red_ret, "Red Days") if len(red_ret) > 30 else {},
        'flat_days': compute_metrics(flat_ret, "Flat Days") if len(flat_ret) > 30 else {},
    }

    # Regime gap
    if bull_metrics and bear_metrics and 'Sharpe' in bull_metrics and 'Sharpe' in bear_metrics:
        s_bull = bull_metrics['Sharpe']
        s_bear = bear_metrics['Sharpe']
        denom = max(abs(s_bull), abs(s_bear))
        regime_gap = abs(s_bull - s_bear) / denom if denom > 0 else 0
    else:
        regime_gap = None

    return bull_metrics, bear_metrics, regime_gap, day_metrics


def permutation_test(returns, spy_returns_aligned, n_perms=1000):
    """
    Permutation test for a timing strategy.
    We compare actual strategy returns against randomly timed entry/exit.
    Method: for each permutation, randomly assign which days are 'in market'
    vs 'in cash' with the same % time in market, compute Sharpe.
    """
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Determine fraction of days with nonzero return (approx time in market)
    nonzero_mask = returns.abs() > 1e-8
    pct_in_market = nonzero_mask.mean()

    # Use SPY returns as the underlying — if we were randomly timing the market
    if spy_returns_aligned is None or len(spy_returns_aligned) < 30:
        # Fallback: block shuffle of returns (preserves autocorrelation better)
        rng = np.random.RandomState(42)
        perm_sharpes = []
        vals = returns.values
        block_size = 21  # monthly blocks
        n = len(vals)
        n_blocks = n // block_size
        for _ in range(n_perms):
            # Block bootstrap shuffle
            blocks = [vals[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
            rng.shuffle(blocks)
            shuffled = np.concatenate(blocks)
            s = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
            perm_sharpes.append(s)
        p_value = (np.sum(np.array(perm_sharpes) >= actual_sharpe) + 1) / (n_perms + 1)
        return actual_sharpe, p_value, perm_sharpes

    # Better approach: random timing of SPY exposure
    rng = np.random.RandomState(42)
    spy_vals = spy_returns_aligned.values
    n = len(spy_vals)
    n_in = int(pct_in_market * n)
    perm_sharpes = []

    for _ in range(n_perms):
        # Randomly pick which days to be in market
        in_market = np.zeros(n, dtype=bool)
        in_market[rng.choice(n, size=n_in, replace=False)] = True
        perm_ret = np.where(in_market, spy_vals, 0.0)
        s = perm_ret.mean() / perm_ret.std() * np.sqrt(252) if perm_ret.std() > 0 else 0
        perm_sharpes.append(s)

    p_value = (np.sum(np.array(perm_sharpes) >= actual_sharpe) + 1) / (n_perms + 1)
    return actual_sharpe, p_value, perm_sharpes


def per_year_analysis(returns):
    """Break down performance by calendar year."""
    yearly = {}
    for year in sorted(returns.index.year.unique()):
        yr = returns[returns.index.year == year]
        if len(yr) > 20:
            m = compute_metrics(yr, f"Year {year}")
            m['year'] = int(year)
            yearly[int(year)] = m
    return yearly


###############################################################################
# 7. MAIN
###############################################################################

def main():
    print("=" * 70)
    print("MULTI-TIMEFRAME TREND FOLLOWING — Walk-Forward Backtest")
    print("HC #685 (growth lanes) | HC #684 (dynamic exits) | HC #683 (real data)")
    print("=" * 70)

    # 1. Get universe
    print("\n[1/6] Getting S&P 500 ticker universe...")
    tickers = get_sp500_tickers()
    print(f"  Universe: {len(tickers)} tickers")

    # 2. Download data
    print("\n[2/6] Downloading price data (2013-2026)...")
    prices = download_prices(tickers, start='2013-01-01', end='2026-07-13')

    print("  Downloading SPY + SHY...")
    spy_close, shy_close = download_spy(start='2013-01-01', end='2026-07-13')
    print(f"  SPY: {len(spy_close)} days, SHY: {len(shy_close) if shy_close is not None else 'N/A'} days")

    # 3. Run backtest
    print("\n[3/6] Running walk-forward trend following backtest...")
    print(f"  Config: {TRAIN_MONTHS}mo lookback, {TEST_MONTHS}mo OOT, sliding")
    print(f"  Holdings: top {N_HOLDINGS}, trailing stop {TRAILING_STOP_PCT:.0%}, "
          f"SMA breakdown {SMA_BREAKDOWN_DAYS}d, partial profit at {PARTIAL_PROFIT_THRESHOLD:.0%}")
    print(f"  Slippage: {SLIPPAGE_PER_TRADE:.1%} per trade")

    returns, meta = run_trend_following_backtest(prices, spy_close, shy_close)

    if returns is None or len(returns) < 60:
        print("ERROR: Insufficient returns data")
        return

    print(f"  Backtest period: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Total days: {len(returns)}, Trades: {meta.get('total_trades', 0)}")
    print(f"  Time in market: {meta.get('pct_time_in_market', 0):.1%}")
    print(f"  Exit reasons: {meta.get('exit_reasons', {})}")

    # 4. Compute metrics
    print("\n[4/6] Computing metrics...")
    metrics = compute_metrics(returns, "Trend Following")
    print("\n  --- OVERALL RESULTS ---")
    for k, v in metrics.items():
        if k != 'name':
            print(f"    {k}: {v}")

    # Per-year breakdown
    yearly = per_year_analysis(returns)
    print("\n  --- PER-YEAR BREAKDOWN ---")
    for year, m in sorted(yearly.items()):
        print(f"    {year}: CAGR={m.get('CAGR_pct','?')}, Sharpe={m.get('Sharpe','?')}, "
              f"MaxDD={m.get('Max_DD_pct','?')}, WR={m.get('Win_Rate_pct','?')}")

    # 5. Regime analysis (R1 test)
    print("\n[5/6] Regime analysis (R1 test)...")
    bull, bear, regime_gap, day_metrics = regime_analysis(returns, spy_close)

    if bull:
        print(f"\n  Bull Regime: Sharpe={bull.get('Sharpe','?')}, CAGR={bull.get('CAGR_pct','?')}, "
              f"WR={bull.get('Win_Rate_pct','?')}, MaxDD={bull.get('Max_DD_pct','?')}")
    if bear:
        print(f"  Bear Regime: Sharpe={bear.get('Sharpe','?')}, CAGR={bear.get('CAGR_pct','?')}, "
              f"WR={bear.get('Win_Rate_pct','?')}, MaxDD={bear.get('Max_DD_pct','?')}")

    if regime_gap is not None:
        status = "PASS" if regime_gap < 0.50 else "FAIL"
        print(f"\n  REGIME GAP: {regime_gap:.3f} — {status} (threshold < 0.50)")
    else:
        print("\n  REGIME GAP: Could not compute")

    if day_metrics:
        for label in ['green_days', 'red_days', 'flat_days']:
            m = day_metrics.get(label, {})
            if m:
                print(f"  {label}: Sharpe={m.get('Sharpe','?')}, CAGR={m.get('CAGR_pct','?')}")

    # Benchmark: SPY buy-and-hold
    spy_returns = spy_close.pct_change().dropna()
    common_idx = returns.index.intersection(spy_returns.index)

    # 6. Permutation test
    print("\n[6/6] Permutation test (1000 shuffles)...")
    spy_ret_aligned = spy_returns.reindex(returns.index).fillna(0) if len(common_idx) > 30 else None
    actual_s, p_val, _ = permutation_test(returns, spy_ret_aligned, n_perms=1000)
    print(f"  Actual Sharpe: {actual_s:.3f}")
    print(f"  p-value: {p_val:.4f} — {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}")
    spy_bench = {}
    if len(common_idx) > 30:
        spy_bench = compute_metrics(spy_returns.loc[common_idx], "SPY Buy&Hold")
        strat_on_common = compute_metrics(returns.loc[common_idx], "Trend Following (same period)")
        print("\n  --- BENCHMARK COMPARISON (same period) ---")
        print(f"  Strategy: Sharpe={strat_on_common.get('Sharpe','?')}, CAGR={strat_on_common.get('CAGR_pct','?')}, "
              f"MaxDD={strat_on_common.get('Max_DD_pct','?')}")
        print(f"  SPY B&H:  Sharpe={spy_bench.get('Sharpe','?')}, CAGR={spy_bench.get('CAGR_pct','?')}, "
              f"MaxDD={spy_bench.get('Max_DD_pct','?')}")

    # Save results
    results = {
        'strategy': 'Multi-Timeframe Trend Following',
        'description': 'Macro trend filter (SPY 200/50 SMA) + momentum stock selection + daily dynamic exits',
        'config': {
            'n_holdings': N_HOLDINGS,
            'trailing_stop_pct': TRAILING_STOP_PCT,
            'sma_breakdown_days': SMA_BREAKDOWN_DAYS,
            'partial_profit_threshold': PARTIAL_PROFIT_THRESHOLD,
            'slippage_per_trade': SLIPPAGE_PER_TRADE,
            'train_months': TRAIN_MONTHS,
            'test_months': TEST_MONTHS,
            'window': 'sliding',
        },
        'overall': metrics,
        'per_year': yearly,
        'bull_regime': bull,
        'bear_regime': bear,
        'regime_gap': regime_gap,
        'regime_gap_pass': regime_gap < 0.50 if regime_gap is not None else None,
        'day_classification': day_metrics,
        'permutation_p_value': p_val,
        'permutation_significant': p_val < 0.05,
        'benchmark_spy': spy_bench,
        'meta': {
            'total_trades': meta.get('total_trades', 0),
            'pct_time_in_market': round(meta.get('pct_time_in_market', 0), 4),
            'exit_reasons': meta.get('exit_reasons', {}),
            'n_stocks_universe': prices.shape[1],
            'backtest_period': f"{returns.index[0].date()} to {returns.index[-1].date()}",
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")

    # Save daily returns CSV
    csv_path = os.path.join(OUTPUT_DIR, 'trend_following_returns.csv')
    returns.to_csv(csv_path)
    print(f"  Daily returns saved to {csv_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
