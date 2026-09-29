# Backtest report: diamond_harbour

Selected model: `A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(et=0.75,lgbm-l15-m20=0.25)`. Selection set: test years up to 2019; final set: later years. Truth: QC-passed gauge readings. A hit is within ±30 min and ±0.30 m.

**Caveat:** tides in the 2021–2023 final years run about 10 min later than in 2000–2016 (M2 phase about +5° against the same months), consistent with a gauge-site change; final scores there understate a model fitted to earlier years.

## Final set, all horizons

```text
                                                                                                       candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(et=0.75,lgbm-l15-m20=0.25)      1251     91.127        13.021        30.303         -7.942         0.110         0.279         -0.028       0      0          0.259
                                                                                                current_pipeline      1251     88.889        13.143        32.583         -7.049         0.122         0.313          0.027       0      0          0.291
                                                                                                      utide_only      1251     62.750        22.701        49.069        -18.386         0.155         0.401          0.020       0      0          0.291
                                                                                A-wall-auto+shallow-side-notrend      1251     78.897        17.749        44.170        -12.486         0.138         0.336         -0.008       0      0          0.272
```

## Final set by horizon

```text
 horizon                                                                                                        candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
       1 A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(et=0.75,lgbm-l15-m20=0.25)       311     92.605        11.774        28.328         -7.003         0.113         0.290         -0.041       0      0          0.249
       2 A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(et=0.75,lgbm-l15-m20=0.25)       571     89.842        13.405        32.070         -7.872         0.114         0.289         -0.045       0      0          0.256
       3 A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(et=0.75,lgbm-l15-m20=0.25)       369     91.870        13.477        30.114         -8.842         0.102         0.259          0.008       0      0          0.273
       1                                                                                                 current_pipeline       311     92.926        10.710        27.559         -4.821         0.122         0.288         -0.001       0      0          0.275
       2                                                                                                 current_pipeline       571     86.865        14.203        34.586         -7.948         0.125         0.342          0.017       0      0          0.274
       3                                                                                                 current_pipeline       369     88.618        13.553        36.074         -7.535         0.118         0.307          0.067       0      0          0.328
       1                                                                                                       utide_only       311     65.916        22.617        44.458        -18.317         0.140         0.350         -0.046       0      0          0.275
       2                                                                                                       utide_only       571     63.748        20.566        45.118        -16.740         0.171         0.418          0.051       0      0          0.274
       3                                                                                                       utide_only       369     58.537        26.075        55.300        -20.990         0.144         0.400          0.029       0      0          0.328
       1                                                                                 A-wall-auto+shallow-side-notrend       311     83.601        16.050        36.537        -10.771         0.132         0.330         -0.045       0      0          0.256
       2                                                                                 A-wall-auto+shallow-side-notrend       571     74.781        18.546        47.063        -12.930         0.142         0.343         -0.005       0      0          0.263
       3                                                                                 A-wall-auto+shallow-side-notrend       369     81.301        17.947        41.378        -13.243         0.136         0.337          0.018       0      0          0.297
```

## Final set by season

```text
      season                                                                                                        candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
         dry A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(et=0.75,lgbm-l15-m20=0.25)       565     95.221        11.783        27.406         -6.891         0.094         0.254         -0.035       0      0          0.237
     monsoon A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(et=0.75,lgbm-l15-m20=0.25)       261     85.441        13.753        34.237        -10.265         0.128         0.321         -0.018       0      0          0.274
post_monsoon A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(et=0.75,lgbm-l15-m20=0.25)       310     93.548        13.111        28.222         -5.858         0.102         0.242         -0.068       0      0          0.240
 pre_monsoon A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(et=0.75,lgbm-l15-m20=0.25)       115     77.391        17.193        34.817        -13.456         0.169         0.368          0.087       0      0          0.362
         dry                                                                                                 current_pipeline       565     96.106        11.228        27.514         -5.166         0.100         0.240          0.014       0      0          0.282
     monsoon                                                                                                 current_pipeline       261     77.395        15.487        36.461        -10.769         0.159         0.419          0.056       0      0          0.300
post_monsoon                                                                                                 current_pipeline       310     94.839        13.122        28.463         -5.573         0.096         0.206         -0.016       0      0          0.249
 pre_monsoon                                                                                                 current_pipeline       115     63.478        17.289        40.040        -11.838         0.216         0.416          0.143       0      0          0.400
         dry                                                                                                       utide_only       565     65.310        23.537        48.861        -18.728         0.124         0.327         -0.044       0      0          0.282
     monsoon                                                                                                       utide_only       261     58.621        21.042        46.000        -17.852         0.213         0.506          0.128       0      0          0.300
post_monsoon                                                                                                       utide_only       310     68.065        20.165        42.113        -15.804         0.135         0.316         -0.014       0      0          0.249
 pre_monsoon                                                                                                       utide_only       115     45.217        29.194        56.974        -24.875         0.232         0.472          0.187       0      0          0.400
         dry                                                                                 A-wall-auto+shallow-side-notrend       565     87.611        15.424        33.702        -10.475         0.121         0.286         -0.040       0      0          0.253
     monsoon                                                                                 A-wall-auto+shallow-side-notrend       261     67.433        20.304        50.155        -16.789         0.161         0.372          0.039       0      0          0.284
post_monsoon                                                                                 A-wall-auto+shallow-side-notrend       310     80.968        17.065        41.497         -9.681         0.126         0.312         -0.042       0      0          0.243
 pre_monsoon                                                                                 A-wall-auto+shallow-side-notrend       115     56.522        25.209        54.600        -20.157         0.197         0.376          0.133       0      0          0.381
```

## Error-range coverage on the final set (%)

90% ranges calibrated on the selection folds; the band is 88-92. Columns: all, each horizon, each season.

```text
                1     2     3   all   dry  monsoon  post_monsoon  pre_monsoon
output                                                                       
high_height  85.7  91.8  78.9  86.5  87.9     89.1          94.1         52.6
high_time    59.7  53.4  58.9  56.6  60.6     43.8          61.4         52.6
level        79.5  81.6  83.1  81.5  83.3     79.0          83.7         72.4
low_height   83.4  95.2  96.2  92.6  92.6     92.5          97.5         79.3
low_time     79.6  71.0  70.1  72.9  77.0     75.9          66.9         62.1
```

## Promotion checks against the current pipeline

- joint_better: FAIL
- time_mae_not_worse: PASS
- height_mae_not_worse: PASS
- no_season_worse_than_1pp: FAIL
- coverage_88_92: FAIL
- beats_tables: PASS

Joint share difference: +2.24 points (95% interval -0.50 to +5.29).
Season differences (points): dry -0.9, pre_monsoon +13.9, monsoon +8.0, post_monsoon -1.3.
Against the official tables (points): no overlap.
Publishable horizons: [].
Overall: NOT PASSED.
