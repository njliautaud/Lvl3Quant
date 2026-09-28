# BACKTEST SCRIPT AUDIT — 12 VALIDATED STRATEGIES

## Summary Table

| Strategy | Script Path | Daily Returns File | Format | Date Range | Rows |
|----------|-------------|--------------------|--------|------------|------|
| CTA Trend Following | `/scripts/growth_research/trend_following_cta_v1.py` | `/output/growth_research/trend_following_returns.csv` | CSV | 2019-02-01 to 2023-02-10 | 1,841 |
| Commodity Trend | `/scripts/growth_research/ml_commodity_trend.py` | `/output/ml_commodity_trend/portfolio_returns.parquet` | Parquet | 2016-11-08 to 2026-05-20 | 115 rows |
| Sector Rotation | `/scripts/growth_research/ml_sector_momentum_v1.py` | `/output/growth_research/sector_momentum_returns.csv` | CSV | 2020-02-03 to 2024-02-12 | ? |
| Currency Carry | `/scripts/growth_research/ml_currency_carry.py` | `/output/ml_currency_carry/portfolio_returns.parquet` | Parquet | 2012-01-03 to 2026-06-15 | 174 rows |
| Gold/Silver Ratio | `/scripts/growth_research/ml_gold_silver_ratio.py` | **NO DAILY CSV/PARQUET** | JSON only | 2006-2026 (in results.json) | Metrics only |
| Yield Curve | `/scripts/growth_research/ml_yield_curve_trade.py` | **NO DAILY CSV/PARQUET** | JSON only | 2006-2026 (in results.json) | Metrics only |
| Tail Risk Hedging | `/scripts/growth_research/ml_tail_risk_hedging.py` | `/output/ml_tail_risk_hedging/portfolio_returns.parquet` | Parquet | 2019-04-09 to 2026-06-15 | 87 rows |
| Bond Duration Timing | `/scripts/growth_research/ml_bond_duration_timing.py` | `/output/ml_bond_duration_timing/portfolio_returns.parquet` | Parquet | 2022-05-31 to 2026-06-05 | 49 rows |
| Stat Arb Pairs | `/scripts/growth_research/ml_stat_arb.py` | **NOT GENERATED** | None | (strategy doesn't save returns) | N/A |
| Carry+Momentum | `/scripts/growth_research/ml_carry_momentum.py` | `/output/ml_carry_momentum/portfolio_returns.parquet` | Parquet | 2015-04-20 to 2026-05-27 | 134 rows |
| Vol Breakout | `/scripts/growth_research/ml_vol_breakout.py` | `/output/ml_vol_breakout/equity_curve.csv` | CSV | 2011-04-04 to 2026-07-09 | 891 rows |
| Thematic Rotation | `/scripts/growth_research/ml_thematic_rotation.py` | `/output/ml_thematic_rotation/portfolio_returns.parquet` | Parquet | 2016-11-01 to 2026-06-11 | 116 rows |

---

## Detailed File Locations

### Group A: Complete Daily Return Series (Ready to Combine)
**8 strategies with daily returns in readable format:**

1. **CTA Trend Following**
   - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/trend_following_cta_v1.py`
   - Daily Returns: `/home/jupiter/Lvl3Quant/output/growth_research/trend_following_returns.csv`
   - Results: `/home/jupiter/Lvl3Quant/output/growth_research/trend_following_results.json`
   - Date Range: 2019-02-01 to 2023-02-10 (1,841 rows)
   - Format: CSV with single return column

2. **Commodity Trend**
   - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_commodity_trend.py`
   - Daily Returns: `/home/jupiter/Lvl3Quant/output/ml_commodity_trend/portfolio_returns.parquet`
   - Results: `/home/jupiter/Lvl3Quant/output/ml_commodity_trend/results.json`
   - Date Range: 2016-11-08 to 2026-05-20 (115 rows)
   - Format: Parquet with columns: `date`, `ml_return`, `ew_return`
   - Note: Saves MONTHLY rebalance returns (not daily)

3. **Sector Rotation**
   - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_sector_momentum_v1.py`
   - Daily Returns: `/home/jupiter/Lvl3Quant/output/growth_research/sector_momentum_returns.csv`
   - Results: `/home/jupiter/Lvl3Quant/output/growth_research/sector_momentum_results.json`
   - Date Range: 2020-02-03 to 2024-02-12
   - Format: CSV

4. **Currency Carry**
   - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_currency_carry.py`
   - Daily Returns: `/home/jupiter/Lvl3Quant/output/ml_currency_carry/portfolio_returns.parquet`
   - Results: `/home/jupiter/Lvl3Quant/output/ml_currency_carry/results.json`
   - Date Range: 2012-01-03 to 2026-06-15 (174 rows)
   - Format: Parquet with columns: `date`, `ml_return`, `ew_return`
   - Note: Monthly rebalance returns

5. **Tail Risk Hedging**
   - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_tail_risk_hedging.py`
   - Daily Returns: `/home/jupiter/Lvl3Quant/output/ml_tail_risk_hedging/portfolio_returns.parquet`
   - Results: `/home/jupiter/Lvl3Quant/output/ml_tail_risk_hedging/results.json`
   - Date Range: 2019-04-09 to 2026-06-15 (87 rows)
   - Format: Parquet
   - Note: Monthly rebalance returns

6. **Bond Duration Timing**
   - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_bond_duration_timing.py`
   - Daily Returns: `/home/jupiter/Lvl3Quant/output/ml_bond_duration_timing/portfolio_returns.parquet`
   - Results: `/home/jupiter/Lvl3Quant/output/ml_bond_duration_timing/results.json`
   - Date Range: 2022-05-31 to 2026-06-05 (49 rows)
   - Format: Parquet
   - Note: Monthly rebalance returns (SHORT HISTORY)

7. **Carry+Momentum**
   - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_carry_momentum.py`
   - Daily Returns: `/home/jupiter/Lvl3Quant/output/ml_carry_momentum/portfolio_returns.parquet`
   - Results: `/home/jupiter/Lvl3Quant/output/ml_carry_momentum/results.json`
   - Date Range: 2015-04-20 to 2026-05-27 (134 rows)
   - Format: Parquet with columns: `date`, `ml_return`, `ew_return`
   - Note: Monthly rebalance returns

8. **Vol Breakout**
   - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_vol_breakout.py`
   - Daily Returns: `/home/jupiter/Lvl3Quant/output/ml_vol_breakout/equity_curve.csv`
   - Date Range: 2011-04-04 to 2026-07-09 (891 rows)
   - Format: CSV with single `capital` column (equity curve, NOT daily returns)
   - Note: This is cumulative equity curve, NOT returns — need to compute pct_change()

---

### Group B: Results JSON Only (No Daily Return Series)
**3 strategies with NO daily returns file — only summary metrics in JSON:**

9. **Gold/Silver Ratio**
   - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_gold_silver_ratio.py`
   - Daily Returns: **NONE SAVED** (script computes but doesn't export daily returns)
   - Results: `/home/jupiter/Lvl3Quant/output/ml_gold_silver_ratio/results.json`
   - Metrics Available: Sharpe=1.278, Sortino=1.83, CAGR=29.7%, MaxDD=-23.8%, WR=52.7%
   - Note: Script runs full walk-forward but only saves summary JSON

10. **Yield Curve**
    - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_yield_curve_trade.py`
    - Daily Returns: **NONE SAVED** (script computes but doesn't export daily returns)
    - Results: `/home/jupiter/Lvl3Quant/output/ml_yield_curve_trade/results.json`
    - Metrics Available: Sharpe=2.006, Sortino=3.195, CAGR=31.8%, MaxDD=-15.1%, WR=52.2%
    - Note: Script runs full walk-forward but only saves summary JSON

11. **Stat Arb Pairs**
    - Script: `/home/jupiter/Lvl3Quant/scripts/growth_research/ml_stat_arb.py`
    - Daily Returns: **NOT GENERATED** (script doesn't compute or save returns)
    - Results: **NONE** in output directory
    - Status: Script exists but has NOT been run or doesn't output results
    - Note: Market-neutral pairs trading strategy

---

## Key Observations for Combined Portfolio Backtest

### Data Alignment Issues:
- **Date Range Mismatch**: Strategies have different start/end dates:
  - Vol Breakout: 2011-2026 (longest)
  - CTA Trend: 2019-2023 (shortest, STALE — ends 2023)
  - Bond Duration: 2022-2026 (limited history)
  
- **Rebalance Frequency**: Most scripts use MONTHLY rebalance (21-day advance), so daily returns are actually MONTHLY returns
  - Commodity Trend, Currency Carry, Carry+Momentum: Monthly
  - Vol Breakout: Daily (from equity curve)
  - Sector Rotation, CTA Trend: Need verification
  
### Missing Data:
- Gold/Silver and Yield Curve: Need to modify scripts to save daily returns (currently only save results.json)
- Stat Arb: Script doesn't save any results
- Bond Duration: Only 49 months of data (too short for robust portfolio backtest)

### To Build Real Combined Backtest:
1. **Regenerate** Gold/Silver and Yield Curve with daily returns export
2. **Fix** Stat Arb script to save daily returns
3. **Convert** Vol Breakout equity curve to daily returns (pct_change)
4. **Align** all series to common date range (recommend 2012-2026 using common data availability)
5. **Normalize** rebalance frequencies (all monthly or all daily?)
6. **Check** CTA Trend data — it stops in 2023, likely needs rerun

---

## To Get Started:

**Immediately usable (no modifications needed):**
```python
import pandas as pd

# Load all available daily returns
returns = {}
returns['cta'] = pd.read_csv('/home/jupiter/Lvl3Quant/output/growth_research/trend_following_returns.csv', index_col=0)
returns['commodity'] = pd.read_parquet('/home/jupiter/Lvl3Quant/output/ml_commodity_trend/portfolio_returns.parquet')
returns['sector'] = pd.read_csv('/home/jupiter/Lvl3Quant/output/growth_research/sector_momentum_returns.csv', index_col=0)
returns['currency'] = pd.read_parquet('/home/jupiter/Lvl3Quant/output/ml_currency_carry/portfolio_returns.parquet')
returns['tail_risk'] = pd.read_parquet('/home/jupiter/Lvl3Quant/output/ml_tail_risk_hedging/portfolio_returns.parquet')
returns['bond_duration'] = pd.read_parquet('/home/jupiter/Lvl3Quant/output/ml_bond_duration_timing/portfolio_returns.parquet')
returns['carry_momentum'] = pd.read_parquet('/home/jupiter/Lvl3Quant/output/ml_carry_momentum/portfolio_returns.parquet')
returns['vol_breakout'] = pd.read_csv('/home/jupiter/Lvl3Quant/output/ml_vol_breakout/equity_curve.csv', index_col=0).pct_change()
returns['thematic'] = pd.read_parquet('/home/jupiter/Lvl3Quant/output/ml_thematic_rotation/portfolio_returns.parquet')
```

**Needs script modifications (currently no daily returns export):**
- Gold/Silver Ratio
- Yield Curve Trade
- Stat Arb Pairs

