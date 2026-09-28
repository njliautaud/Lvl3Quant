# Confluence Stacking v1: CNN-Mamba v2 x PatchTST

Analysis date: 2026-05-23
Overlapping OOT days: 42
Total aligned events: 1,998,849
Prediction correlation: 0.0252
Horizon: 10s

Cost thresholds: hybrid=0.876, market=1.376

## Confluence Results (SHORT side)

Best confluence filter: Both top 2%
  Avg realized: +1.187 ticks
  Net hybrid: +0.311 ticks
  Net market: -0.189 ticks
  Win rate: 53.2%
  Daily Sharpe: +9.91
  Events: 1,187 (32/day)

## Full Results Table

           filter      label  pct_threshold  n_events  n_days  avg_realized_ticks  median_realized_ticks  std_realized_ticks  net_hybrid  net_market  win_rate  daily_sharpe  daily_profitable_frac  events_per_day
 CNN-Mamba top 1%    CM_only              1     19989      41            0.826555                    1.0            6.563581   -0.049445   -0.549445  0.564460      7.223272               0.414634      487.536585
  PatchTST top 1%    PT_only              1     20001      39           -0.765462                    0.0           85.541275   -1.641462   -2.141462  0.465727     -0.595174               0.128205      512.846154
      Both top 1% Confluence              1       361      34            0.937673                    0.0            8.437787    0.061673   -0.438327  0.476454      7.238501               0.441176       10.617647
 CNN-Mamba top 2%    CM_only              2     39978      42            0.728963                    1.0            6.020751   -0.147037   -0.647037  0.559408      8.088373               0.357143      951.857143
  PatchTST top 2%    PT_only              2     39991      42           -0.391838                    0.0           60.694302   -1.267838   -1.767838  0.469281      1.657728               0.166667      952.166667
      Both top 2% Confluence              2      1187      37            1.187447                    0.5            8.244041    0.311447   -0.188553  0.532435      9.911359               0.594595       32.081081
 CNN-Mamba top 5%    CM_only              5     99943      42            0.630374                    0.5            5.647001   -0.245626   -0.745626  0.549783     10.314984               0.309524     2379.595238
  PatchTST top 5%    PT_only              5    100015      42           -0.173174                    0.0           38.699169   -1.049174   -1.549174  0.468660      3.034951               0.142857     2381.309524
      Both top 5% Confluence              5      6160      39            0.795860                    0.5            6.297908   -0.080140   -0.580140  0.546591      7.485495               0.307692      157.948718
CNN-Mamba top 10%    CM_only             10    199885      42            0.583723                    0.5            5.675621   -0.292277   -0.792277  0.539680     13.844576               0.119048     4759.166667
 PatchTST top 10%    PT_only             10    200056      42           -0.093376                    0.0           27.639242   -0.969376   -1.469376  0.469319      2.742206               0.095238     4763.238095
     Both top 10% Confluence             10     22236      42            0.624280                    0.5            5.448905   -0.251720   -0.751720  0.544073      7.271204               0.238095      529.428571
CNN-Mamba top 20%    CM_only             20    399770      42            0.503147                    0.5            5.843655   -0.372853   -0.872853  0.527281     19.821450               0.119048     9518.333333
 PatchTST top 20%    PT_only             20    399791      42           -0.069679                    0.0           19.959705   -0.945679   -1.445679  0.465543      2.321971               0.071429     9518.833333
     Both top 20% Confluence             20     85001      42            0.488241                    0.5            5.445722   -0.387759   -0.887759  0.524959      7.949203               0.119048     2023.833333

## Combined Z-Score Results

          filter  pct_threshold  n_events  avg_realized_ticks  net_hybrid  net_market  win_rate  daily_sharpe  events_per_day
 Combined top 1%              1     19989            0.707389   -0.168611   -0.668611  0.549802      6.919830      475.928571
 Combined top 2%              2     39977            0.602046   -0.273954   -0.773954  0.543437      6.983098      951.833333
 Combined top 5%              5     99943            0.559379   -0.316621   -0.816621  0.534375      8.712468     2379.595238
Combined top 10%             10    199885            0.412700   -0.463300   -0.963300  0.520624      8.337099     4759.166667
Combined top 20%             20    399770            0.340189   -0.535811   -1.035811  0.507759     10.139471     9518.333333

## Verdict

Confluence stacking DOES push realized ticks above hybrid breakeven.
Best config: Both top 2% at net +0.311 ticks/trade.