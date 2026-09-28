# HC #439 Phase 1 — MFE/MAE deep analysis

Cost regime: passive_net = mean_label - 0.376 ticks
             market_net  = mean_label - 1.376 ticks
All values in ticks (1 tick = 0.25 ES points = $12.50).

## OVERALL — top 20 (side, horizon, band) by passive-net EV (n>=500)

```
side  | horizon | band    | n_signals | label_mean | passive_net | market_net | wr    | hit_1tk | hit_2tk | mfe_mean | mae_mean
------+---------+---------+-----------+------------+-------------+------------+-------+---------+---------+----------+---------
short | 1s      | top0.1% | 1695      | 1.099      | 0.723       | -0.277     | 0.639 | 0.768   | 0.519   | 2.654    | 2.280   
long  | 1s      | top0.1% | 1695      | 0.955      | 0.579       | -0.421     | 0.614 | 0.566   | 0.314   | 1.580    | 1.369   
short | 1s      | top0.5% | 8479      | 0.899      | 0.523       | -0.477     | 0.616 | 0.564   | 0.271   | 1.324    | 1.217   
short | 1s      | top1%   | 16959     | 0.823      | 0.447       | -0.553     | 0.608 | 0.523   | 0.231   | 1.092    | 1.014   
short | 1s      | top2%   | 33918     | 0.737      | 0.361       | -0.639     | 0.594 | 0.504   | 0.210   | 0.967    | 0.919   
short | 1s      | top5%   | 84795     | 0.651      | 0.275       | -0.725     | 0.576 | 0.499   | 0.204   | 0.914    | 0.891   
long  | 1s      | top0.5% | 8479      | 0.615      | 0.239       | -0.761     | 0.577 | 0.481   | 0.197   | 0.962    | 0.944   
short | 1s      | top10%  | 169591    | 0.579      | 0.203       | -0.797     | 0.557 | 0.498   | 0.203   | 0.904    | 0.895   
long  | 1s      | top1%   | 16959     | 0.577      | 0.201       | -0.799     | 0.574 | 0.467   | 0.175   | 0.856    | 0.863   
long  | 1s      | top5%   | 84795     | 0.569      | 0.193       | -0.807     | 0.572 | 0.454   | 0.166   | 0.799    | 0.798   
long  | 1s      | top2%   | 33918     | 0.567      | 0.191       | -0.809     | 0.572 | 0.455   | 0.164   | 0.808    | 0.821   
long  | 1s      | top10%  | 169591    | 0.557      | 0.181       | -0.819     | 0.559 | 0.471   | 0.181   | 0.847    | 0.826   
long  | 1s      | top20%  | 339183    | 0.505      | 0.129       | -0.871     | 0.533 | 0.487   | 0.198   | 0.908    | 0.878   
short | 1s      | top20%  | 339183    | 0.495      | 0.119       | -0.881     | 0.532 | 0.501   | 0.205   | 0.911    | 0.915   
long  | 5s      | top0.1% | 1695      | nan        | nan         | nan        | 0.601 | 0.757   | 0.512   | 2.660    | 2.495   
long  | 5s      | top0.5% | 8479      | nan        | nan         | nan        | 0.568 | 0.744   | 0.461   | 2.160    | 2.127   
long  | 5s      | top1%   | 16959     | nan        | nan         | nan        | 0.571 | 0.729   | 0.442   | 2.025    | 2.020   
long  | 5s      | top2%   | 33918     | nan        | nan         | nan        | 0.563 | 0.722   | 0.434   | 1.989    | 2.003   
long  | 5s      | top5%   | 84795     | nan        | nan         | nan        | 0.559 | 0.726   | 0.442   | 2.047    | 2.023   
long  | 5s      | top10%  | 169591    | nan        | nan         | nan        | 0.549 | 0.738   | 0.465   | 2.187    | 2.125   
```

## Top 15 — Volatility (filter_vol_500ev_tk) terciles (n>=200)

```
tag                      | side  | horizon | band    | n_signals | label_mean | passive_net | market_net | wr    | hit_1tk | mfe_mean | mae_mean
-------------------------+-------+---------+---------+-----------+------------+-------------+------------+-------+---------+----------+---------
filter_vol_500ev_tk_High | long  | 10s     | top0.1% | 575       | 2.165      | 1.789       | 0.789      | 0.586 | 0.873   | 5.817    | 4.948   
filter_vol_500ev_tk_High | long  | 5s      | top0.1% | 575       | 1.995      | 1.619       | 0.619      | 0.626 | 0.852   | 4.313    | 3.574   
filter_vol_500ev_tk_High | short | 1s      | top0.1% | 575       | 1.903      | 1.527       | 0.527      | 0.696 | 0.826   | 3.045    | 2.607   
filter_vol_500ev_tk_High | long  | 1s      | top0.1% | 575       | 1.463      | 1.087       | 0.087      | 0.666 | 0.699   | 2.482    | 2.054   
filter_vol_500ev_tk_Mid  | long  | 5s      | top0.1% | 560       | 1.444      | 1.068       | 0.068      | 0.614 | 0.793   | 3.000    | 2.575   
filter_vol_500ev_tk_Low  | short | 10s     | top0.1% | 560       | 1.439      | 1.063       | 0.063      | 0.566 | 0.800   | 3.634    | 3.602   
filter_vol_500ev_tk_Mid  | long  | 10s     | top0.1% | 560       | 1.303      | 0.927       | -0.073     | 0.577 | 0.879   | 4.141    | 3.657   
filter_vol_500ev_tk_Mid  | short | 10s     | top0.1% | 560       | 1.230      | 0.854       | -0.146     | 0.593 | 0.911   | 4.750    | 4.386   
filter_vol_500ev_tk_Mid  | long  | 1s      | top0.1% | 560       | 1.104      | 0.728       | -0.272     | 0.643 | 0.621   | 1.652    | 1.357   
filter_vol_500ev_tk_High | short | 1s      | top0.5% | 2875      | 1.062      | 0.686       | -0.314     | 0.614 | 0.629   | 1.743    | 1.678   
filter_vol_500ev_tk_Mid  | short | 10s     | top0.5% | 2802      | 1.031      | 0.655       | -0.345     | 0.566 | 0.856   | 3.789    | 3.650   
filter_vol_500ev_tk_Low  | short | 5s      | top0.5% | 2802      | 1.012      | 0.636       | -0.364     | 0.578 | 0.740   | 1.994    | 1.897   
filter_vol_500ev_tk_Mid  | short | 5s      | top0.1% | 560       | 1.010      | 0.634       | -0.366     | 0.595 | 0.875   | 3.752    | 3.430   
filter_vol_500ev_tk_Low  | short | 5s      | top0.1% | 560       | 0.955      | 0.579       | -0.421     | 0.536 | 0.762   | 2.918    | 2.773   
filter_vol_500ev_tk_Mid  | short | 10s     | top1%   | 5604      | 0.954      | 0.578       | -0.422     | 0.567 | 0.844   | 3.576    | 3.489   
```

## Top 15 — Event-rate (filter_evt_per_sec_30s) terciles (n>=200)

```
tag                         | side  | horizon | band    | n_signals | label_mean | passive_net | market_net | wr    | hit_1tk | mfe_mean | mae_mean
----------------------------+-------+---------+---------+-----------+------------+-------------+------------+-------+---------+----------+---------
filter_evt_per_sec_30s_High | short | 1s      | top0.1% | 575       | 1.796      | 1.420       | 0.420      | 0.680 | 0.861   | 3.475    | 2.494   
filter_evt_per_sec_30s_Low  | short | 10s     | top0.1% | 560       | 1.562      | 1.186       | 0.186      | 0.546 | 0.721   | 2.830    | 3.041   
filter_evt_per_sec_30s_Low  | short | 5s      | top0.1% | 560       | 1.452      | 1.076       | 0.076      | 0.554 | 0.664   | 2.261    | 2.507   
filter_evt_per_sec_30s_High | long  | 1s      | top0.1% | 575       | 1.363      | 0.987       | -0.013     | 0.643 | 0.770   | 2.450    | 1.950   
filter_evt_per_sec_30s_Mid  | short | 10s     | top0.1% | 559       | 1.230      | 0.854       | -0.146     | 0.630 | 0.903   | 4.252    | 4.081   
filter_evt_per_sec_30s_High | short | 1s      | top0.5% | 2878      | 1.017      | 0.641       | -0.359     | 0.625 | 0.750   | 2.073    | 1.863   
filter_evt_per_sec_30s_Mid  | short | 10s     | top0.5% | 2797      | 1.012      | 0.636       | -0.364     | 0.588 | 0.867   | 3.350    | 3.238   
filter_evt_per_sec_30s_Mid  | short | 5s      | top0.1% | 559       | 0.945      | 0.569       | -0.431     | 0.612 | 0.866   | 3.311    | 3.363   
filter_evt_per_sec_30s_Mid  | short | 5s      | top0.5% | 2797      | 0.938      | 0.562       | -0.438     | 0.608 | 0.812   | 2.440    | 2.380   
filter_evt_per_sec_30s_Mid  | short | 1s      | top0.1% | 559       | 0.928      | 0.552       | -0.448     | 0.651 | 0.748   | 2.064    | 1.946   
filter_evt_per_sec_30s_Mid  | long  | 1s      | top0.1% | 559       | 0.898      | 0.522       | -0.478     | 0.633 | 0.551   | 1.304    | 1.193   
filter_evt_per_sec_30s_Mid  | long  | 10s     | top0.1% | 559       | 0.886      | 0.510       | -0.490     | 0.572 | 0.825   | 3.379    | 3.256   
filter_evt_per_sec_30s_High | short | 1s      | top1%   | 5756      | 0.881      | 0.505       | -0.495     | 0.605 | 0.718   | 1.767    | 1.671   
filter_evt_per_sec_30s_Mid  | short | 5s      | top1%   | 5595      | 0.873      | 0.497       | -0.503     | 0.597 | 0.803   | 2.270    | 2.233   
filter_evt_per_sec_30s_Mid  | short | 10s     | top1%   | 5595      | 0.869      | 0.493       | -0.507     | 0.572 | 0.850   | 3.142    | 3.114   
```

## Top 15 — Buy-aggression (filter_buy_aggr_50) terciles (n>=200)

```
tag                     | side  | horizon | band    | n_signals | label_mean | passive_net | market_net | wr    | hit_1tk | mfe_mean | mae_mean
------------------------+-------+---------+---------+-----------+------------+-------------+------------+-------+---------+----------+---------
filter_buy_aggr_50_Mid  | short | 1s      | top0.1% | 568       | 1.490      | 1.114       | 0.114      | 0.680 | 0.794   | 2.917    | 2.058   
filter_buy_aggr_50_Low  | short | 1s      | top0.1% | 587       | 1.440      | 1.064       | 0.064      | 0.608 | 0.695   | 2.399    | 2.092   
filter_buy_aggr_50_Low  | long  | 10s     | top0.1% | 587       | 1.283      | 0.907       | -0.093     | 0.588 | 0.847   | 4.148    | 3.486   
filter_buy_aggr_50_Low  | short | 1s      | top0.5% | 2939      | 1.198      | 0.822       | -0.178     | 0.627 | 0.550   | 1.273    | 1.071   
filter_buy_aggr_50_Low  | long  | 5s      | top0.1% | 587       | 1.124      | 0.748       | -0.252     | 0.620 | 0.777   | 2.816    | 2.387   
filter_buy_aggr_50_High | long  | 10s     | top0.1% | 539       | 1.041      | 0.665       | -0.335     | 0.575 | 0.826   | 3.545    | 3.310   
filter_buy_aggr_50_Low  | short | 1s      | top1%   | 5879      | 1.010      | 0.634       | -0.366     | 0.618 | 0.522   | 1.098    | 0.926   
filter_buy_aggr_50_Low  | long  | 1s      | top0.1% | 587       | 0.994      | 0.618       | -0.382     | 0.625 | 0.574   | 1.811    | 1.400   
filter_buy_aggr_50_Mid  | long  | 1s      | top0.1% | 568       | 0.980      | 0.604       | -0.396     | 0.611 | 0.579   | 1.592    | 1.463   
filter_buy_aggr_50_High | long  | 5s      | top0.1% | 539       | 0.929      | 0.553       | -0.447     | 0.581 | 0.746   | 2.436    | 2.347   
filter_buy_aggr_50_High | long  | 1s      | top0.1% | 539       | 0.923      | 0.547       | -0.453     | 0.610 | 0.579   | 1.384    | 1.195   
filter_buy_aggr_50_Mid  | short | 1s      | top0.5% | 2841      | 0.895      | 0.519       | -0.481     | 0.615 | 0.575   | 1.390    | 1.225   
filter_buy_aggr_50_Low  | long  | 10s     | top0.5% | 2939      | 0.849      | 0.473       | -0.527     | 0.556 | 0.825   | 3.126    | 2.983   
filter_buy_aggr_50_Low  | short | 1s      | top2%   | 11758     | 0.844      | 0.468       | -0.532     | 0.598 | 0.509   | 1.006    | 0.894   
filter_buy_aggr_50_High | short | 1s      | top0.1% | 539       | 0.826      | 0.450       | -0.550     | 0.629 | 0.779   | 2.690    | 2.529   
```

## Top 15 — Spread-proxy terciles (n>=200)

```
tag                         | side  | horizon | band    | n_signals | label_mean | passive_net | market_net | wr    | hit_1tk | mfe_mean | mae_mean
----------------------------+-------+---------+---------+-----------+------------+-------------+------------+-------+---------+----------+---------
filter_spread_proxy_tk_High | long  | 10s     | top0.1% | 494       | 2.130      | 1.754       | 0.754      | 0.615 | 0.895   | 5.490    | 4.314   
filter_spread_proxy_tk_High | short | 5s      | top0.1% | 494       | 2.067      | 1.691       | 0.691      | 0.648 | 0.913   | 5.132    | 4.605   
filter_spread_proxy_tk_High | short | 10s     | top0.1% | 494       | 1.841      | 1.465       | 0.465      | 0.611 | 0.941   | 6.565    | 5.905   
filter_spread_proxy_tk_High | short | 1s      | top0.1% | 494       | 1.796      | 1.420       | 0.420      | 0.680 | 0.836   | 2.962    | 2.468   
filter_spread_proxy_tk_High | long  | 5s      | top0.1% | 494       | 1.762      | 1.386       | 0.386      | 0.640 | 0.848   | 3.988    | 3.146   
filter_spread_proxy_tk_High | long  | 1s      | top0.1% | 494       | 1.435      | 1.059       | 0.059      | 0.680 | 0.652   | 2.221    | 1.626   
filter_spread_proxy_tk_Mid  | short | 10s     | top0.1% | 565       | 1.377      | 1.001       | 0.001      | 0.628 | 0.904   | 5.071    | 5.133   
filter_spread_proxy_tk_Mid  | long  | 5s      | top0.1% | 565       | 1.312      | 0.936       | -0.064     | 0.607 | 0.825   | 3.025    | 2.710   
filter_spread_proxy_tk_Mid  | long  | 10s     | top0.1% | 565       | 1.279      | 0.903       | -0.097     | 0.563 | 0.871   | 4.186    | 3.876   
filter_spread_proxy_tk_Mid  | long  | 1s      | top0.1% | 565       | 1.229      | 0.853       | -0.147     | 0.667 | 0.646   | 1.855    | 1.584   
filter_spread_proxy_tk_High | short | 5s      | top0.5% | 2473      | 1.195      | 0.819       | -0.181     | 0.606 | 0.814   | 3.490    | 3.484   
filter_spread_proxy_tk_Mid  | short | 5s      | top0.1% | 565       | 1.155      | 0.779       | -0.221     | 0.607 | 0.867   | 4.076    | 4.094   
filter_spread_proxy_tk_High | short | 10s     | top0.5% | 2473      | 1.101      | 0.725       | -0.275     | 0.580 | 0.872   | 4.745    | 4.751   
filter_spread_proxy_tk_Mid  | short | 1s      | top0.1% | 565       | 1.089      | 0.713       | -0.287     | 0.642 | 0.777   | 2.726    | 2.396   
filter_spread_proxy_tk_High | long  | 10s     | top0.5% | 2473      | 1.086      | 0.710       | -0.290     | 0.566 | 0.846   | 4.368    | 4.165   
```

## Top 15 — Time-of-day buckets (n>=200)

```
tag                  | side  | horizon | band    | n_signals | label_mean | passive_net | market_net | wr    | hit_1tk | mfe_mean | mae_mean
---------------------+-------+---------+---------+-----------+------------+-------------+------------+-------+---------+----------+---------
tod_bucket=evening   | short | 5s      | top0.5% | 341       | 3.163      | 2.787       | 1.787      | 0.557 | 0.405   | 1.727    | 1.891   
tod_bucket=evening   | short | 10s     | top0.5% | 341       | 3.028      | 2.652       | 1.652      | 0.545 | 0.499   | 2.504    | 2.516   
tod_bucket=evening   | short | 1s      | top0.5% | 341       | 2.843      | 2.467       | 1.467      | 0.402 | 0.196   | 1.114    | 0.762   
tod_bucket=evening   | short | 5s      | top1%   | 683       | 2.015      | 1.639       | 0.639      | 0.572 | 0.414   | 1.419    | 1.338   
tod_bucket=evening   | short | 10s     | top1%   | 683       | 1.884      | 1.508       | 0.508      | 0.559 | 0.534   | 2.047    | 1.899   
tod_bucket=evening   | short | 1s      | top1%   | 683       | 1.680      | 1.304       | 0.304      | 0.441 | 0.186   | 0.750    | 0.701   
tod_bucket=evening   | short | 5s      | top2%   | 1366      | 1.398      | 1.022       | 0.022      | 0.579 | 0.417   | 1.140    | 1.105   
tod_bucket=1030-1100 | short | 10s     | top0.5% | 749       | 1.340      | 0.964       | -0.036     | 0.577 | 0.849   | 4.276    | 3.858   
tod_bucket=evening   | short | 10s     | top2%   | 1366      | 1.282      | 0.906       | -0.094     | 0.583 | 0.559   | 1.740    | 1.682   
tod_bucket=1400-1430 | short | 5s      | top0.5% | 397       | 1.272      | 0.896       | -0.104     | 0.622 | 0.773   | 2.423    | 1.950   
tod_bucket=0930-1000 | long  | 5s      | top0.1% | 206       | 1.252      | 0.876       | -0.124     | 0.583 | 0.908   | 3.840    | 3.267   
tod_bucket=pre-open  | short | 10s     | top0.5% | 342       | 1.222      | 0.846       | -0.154     | 0.594 | 0.775   | 3.257    | 3.158   
tod_bucket=pre-open  | short | 5s      | top0.5% | 342       | 1.213      | 0.837       | -0.163     | 0.640 | 0.687   | 2.295    | 2.281   
tod_bucket=1000-1030 | short | 10s     | top0.5% | 831       | 1.192      | 0.816       | -0.184     | 0.573 | 0.892   | 4.513    | 3.924   
tod_bucket=1400-1430 | short | 5s      | top1%   | 795       | 1.187      | 0.811       | -0.189     | 0.626 | 0.762   | 2.199    | 1.779   
```

## Top 15 — Day-of-week (n>=200)

```
tag   | side  | horizon | band    | n_signals | label_mean | passive_net | market_net | wr    | hit_1tk | mfe_mean | mae_mean
------+-------+---------+---------+-----------+------------+-------------+------------+-------+---------+----------+---------
dow=6 | short | 10s     | top5%   | 372       | 2.825      | 2.449       | 1.449      | 0.446 | 0.524   | 2.578    | 2.376   
dow=6 | short | 5s      | top5%   | 372       | 2.809      | 2.433       | 1.433      | 0.449 | 0.457   | 1.710    | 1.976   
dow=6 | short | 1s      | top5%   | 372       | 2.660      | 2.284       | 1.284      | 0.422 | 0.234   | 1.070    | 0.965   
dow=6 | short | 10s     | top10%  | 745       | 1.752      | 1.376       | 0.376      | 0.487 | 0.619   | 2.952    | 2.744   
dow=6 | short | 1s      | top10%  | 745       | 1.701      | 1.325       | 0.325      | 0.462 | 0.279   | 1.005    | 0.828   
dow=2 | short | 10s     | top0.1% | 317       | 1.689      | 1.313       | 0.313      | 0.631 | 0.845   | 4.098    | 3.505   
dow=6 | short | 5s      | top10%  | 745       | 1.681      | 1.305       | 0.305      | 0.472 | 0.515   | 2.030    | 2.035   
dow=1 | short | 1s      | top0.1% | 342       | 1.640      | 1.264       | 0.264      | 0.658 | 0.816   | 3.178    | 2.871   
dow=1 | long  | 1s      | top0.1% | 342       | 1.525      | 1.149       | 0.149      | 0.673 | 0.655   | 2.047    | 1.746   
dow=3 | short | 1s      | top0.1% | 413       | 1.362      | 0.986       | -0.014     | 0.695 | 0.770   | 2.310    | 1.971   
dow=3 | long  | 10s     | top0.1% | 413       | 1.357      | 0.981       | -0.019     | 0.557 | 0.864   | 4.535    | 3.613   
dow=3 | long  | 5s      | top0.1% | 413       | 1.259      | 0.883       | -0.117     | 0.622 | 0.797   | 3.174    | 2.600   
dow=3 | long  | 1s      | top0.1% | 413       | 1.211      | 0.835       | -0.165     | 0.642 | 0.627   | 2.010    | 1.467   
dow=2 | short | 10s     | top0.5% | 1585      | 1.197      | 0.821       | -0.179     | 0.599 | 0.819   | 3.251    | 2.874   
dow=3 | short | 5s      | top0.1% | 413       | 1.190      | 0.814       | -0.186     | 0.608 | 0.884   | 3.746    | 3.644   
```

## ALL SLICES with positive MARKET-order EV (n>=200)

```
source | tag                         | side  | horizon | band    | n_signals | label_mean | passive_net | market_net | wr   
-------+-----------------------------+-------+---------+---------+-----------+------------+-------------+------------+------
tod    | tod_bucket=evening          | short | 5s      | top0.5% | 341       | 3.163      | 2.787       | 1.787      | 0.557
tod    | tod_bucket=evening          | short | 10s     | top0.5% | 341       | 3.028      | 2.652       | 1.652      | 0.545
tod    | tod_bucket=evening          | short | 1s      | top0.5% | 341       | 2.843      | 2.467       | 1.467      | 0.402
dow    | dow=6                       | short | 10s     | top5%   | 372       | 2.825      | 2.449       | 1.449      | 0.446
dow    | dow=6                       | short | 5s      | top5%   | 372       | 2.809      | 2.433       | 1.433      | 0.449
dow    | dow=6                       | short | 1s      | top5%   | 372       | 2.660      | 2.284       | 1.284      | 0.422
vol    | filter_vol_500ev_tk_High    | long  | 10s     | top0.1% | 575       | 2.165      | 1.789       | 0.789      | 0.586
spread | filter_spread_proxy_tk_High | long  | 10s     | top0.1% | 494       | 2.130      | 1.754       | 0.754      | 0.615
spread | filter_spread_proxy_tk_High | short | 5s      | top0.1% | 494       | 2.067      | 1.691       | 0.691      | 0.648
tod    | tod_bucket=evening          | short | 5s      | top1%   | 683       | 2.015      | 1.639       | 0.639      | 0.572
vol    | filter_vol_500ev_tk_High    | long  | 5s      | top0.1% | 575       | 1.995      | 1.619       | 0.619      | 0.626
vol    | filter_vol_500ev_tk_High    | short | 1s      | top0.1% | 575       | 1.903      | 1.527       | 0.527      | 0.696
tod    | tod_bucket=evening          | short | 10s     | top1%   | 683       | 1.884      | 1.508       | 0.508      | 0.559
spread | filter_spread_proxy_tk_High | short | 10s     | top0.1% | 494       | 1.841      | 1.465       | 0.465      | 0.611
evt    | filter_evt_per_sec_30s_High | short | 1s      | top0.1% | 575       | 1.796      | 1.420       | 0.420      | 0.680
spread | filter_spread_proxy_tk_High | short | 1s      | top0.1% | 494       | 1.796      | 1.420       | 0.420      | 0.680
spread | filter_spread_proxy_tk_High | long  | 5s      | top0.1% | 494       | 1.762      | 1.386       | 0.386      | 0.640
dow    | dow=6                       | short | 10s     | top10%  | 745       | 1.752      | 1.376       | 0.376      | 0.487
dow    | dow=6                       | short | 1s      | top10%  | 745       | 1.701      | 1.325       | 0.325      | 0.462
dow    | dow=2                       | short | 10s     | top0.1% | 317       | 1.689      | 1.313       | 0.313      | 0.631
dow    | dow=6                       | short | 5s      | top10%  | 745       | 1.681      | 1.305       | 0.305      | 0.472
tod    | tod_bucket=evening          | short | 1s      | top1%   | 683       | 1.680      | 1.304       | 0.304      | 0.441
dow    | dow=1                       | short | 1s      | top0.1% | 342       | 1.640      | 1.264       | 0.264      | 0.658
evt    | filter_evt_per_sec_30s_Low  | short | 10s     | top0.1% | 560       | 1.562      | 1.186       | 0.186      | 0.546
dow    | dow=1                       | long  | 1s      | top0.1% | 342       | 1.525      | 1.149       | 0.149      | 0.673
```
