# Backtest report: haldia

Selected model: `A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)`. Selection set: test years up to 2019; final set: later years. Truth: QC-passed gauge readings. A hit is within ±30 min and ±0.30 m.

## Final set, all horizons

```text
                                                                                              candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)     21156     88.788         8.779        22.440         -0.764         0.148         0.356         -0.081       0      0          0.198
                                                                                       current_pipeline     21156     87.309         9.455        24.631          0.420         0.148         0.366         -0.011       0      0          0.209
                                                                                             utide_only     21156     85.432        11.776        28.773         -6.707         0.151         0.365         -0.005       0      0          0.209
                                                                       A-wall-auto+shallow-side-notrend     21156     87.332        10.382        26.446         -3.935         0.147         0.362         -0.052       0      0          0.207
```

## Final set by horizon

```text
 horizon                                                                                               candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
       1 A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)      7052     89.265         8.698        22.241         -0.737         0.146         0.349         -0.076       0      0          0.197
       2 A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)      7052     88.939         8.789        22.473         -0.752         0.148         0.354         -0.082       0      0          0.198
       3 A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)      7052     88.159         8.850        22.571         -0.803         0.149         0.363         -0.086       0      0          0.199
       1                                                                                        current_pipeline      7052     87.961         9.345        24.202          1.389         0.144         0.360         -0.017       0      0          0.210
       2                                                                                        current_pipeline      7052     85.607         9.448        24.906          0.391         0.154         0.381         -0.006       0      0          0.210
       3                                                                                        current_pipeline      7052     88.358         9.573        24.684         -0.518         0.146         0.354         -0.010       0      0          0.207
       1                                                                                              utide_only      7052     85.210        11.745        28.789         -6.655         0.152         0.367         -0.006       0      0          0.210
       2                                                                                              utide_only      7052     85.111        11.786        28.759         -6.710         0.153         0.366         -0.004       0      0          0.210
       3                                                                                              utide_only      7052     85.976        11.798        28.759         -6.755         0.149         0.360         -0.006       0      0          0.207
       1                                                                        A-wall-auto+shallow-side-notrend      7052     87.323        10.357        26.429         -3.893         0.147         0.361         -0.050       0      0          0.207
       2                                                                        A-wall-auto+shallow-side-notrend      7052     87.252        10.389        26.461         -3.943         0.147         0.363         -0.052       0      0          0.207
       3                                                                        A-wall-auto+shallow-side-notrend      7052     87.422        10.400        26.476         -3.969         0.147         0.361         -0.055       0      0          0.207
```

## Final set by season

```text
      season                                                                                               candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
         dry A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)      5235     92.741         8.006        20.539         -1.483         0.122         0.316         -0.078       0      0          0.158
     monsoon A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)      7074     89.087         8.664        22.098          0.487         0.144         0.375         -0.082       0      0          0.196
post_monsoon A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)      5310     84.407         8.814        21.851         -1.833         0.181         0.383         -0.117       0      0          0.228
 pre_monsoon A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)      3537     88.917        10.099        26.104         -0.598         0.141         0.345         -0.033       0      0          0.208
         dry                                                                                        current_pipeline      5235     94.499         8.323        21.970          0.941         0.115         0.285         -0.006       0      0          0.178
     monsoon                                                                                        current_pipeline      7074     87.772         9.186        24.106          0.610         0.144         0.372         -0.007       0      0          0.205
post_monsoon                                                                                        current_pipeline      5310     80.546         9.499        23.481          0.102         0.181         0.414         -0.037       0      0          0.227
 pre_monsoon                                                                                        current_pipeline      3537     85.892        11.604        30.630         -0.251         0.155         0.352          0.014       0      0          0.230
         dry                                                                                              utide_only      5235     89.112        12.709        28.391         -9.274         0.129         0.332         -0.038       0      0          0.178
     monsoon                                                                                              utide_only      7074     87.334         9.884        26.000         -2.943         0.148         0.359          0.030       0      0          0.205
post_monsoon                                                                                              utide_only      5310     82.298        11.931        27.938         -8.429         0.169         0.401         -0.050       0      0          0.227
 pre_monsoon                                                                                              utide_only      3537     80.888        13.948        34.497         -7.849         0.163         0.363          0.038       0      0          0.230
         dry                                                                        A-wall-auto+shallow-side-notrend      5235     91.213        10.641        26.294         -5.919         0.126         0.316         -0.050       0      0          0.166
     monsoon                                                                        A-wall-auto+shallow-side-notrend      7074     87.645         9.553        25.021         -1.900         0.140         0.357         -0.051       0      0          0.205
post_monsoon                                                                        A-wall-auto+shallow-side-notrend      5310     83.974        10.306        25.000         -4.721         0.176         0.392         -0.088       0      0          0.234
 pre_monsoon                                                                        A-wall-auto+shallow-side-notrend      3537     86.005        11.769        31.150         -3.890         0.149         0.367         -0.005       0      0          0.222
```

## Against the official tables (one year ahead)

```text
 test_year                                                                                               candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
      2024                                                                                         official_tables      1413     79.335        16.580        33.477         15.900         0.161         0.400         -0.119       0      0            NaN
      2024 A-wall-auto+shallow-side-notrend/lv:blend(mlp=0.75,lgbm-l31-m100-huber=0.25)/ev:blend(mlp=0.75,et=0.25)      1414     90.311         7.449        19.739         -0.083         0.135         0.346         -0.050       0      0          0.198
      2024                                                                                        current_pipeline      1414     87.765         9.500        23.937          5.299         0.143         0.360         -0.044       0      0          0.205
      2024                                                                                              utide_only      1414     87.199         9.787        23.894         -5.111         0.148         0.372          0.021       0      0          0.205
      2024                                                                        A-wall-auto+shallow-side-notrend      1414     89.533         8.761        23.058         -2.728         0.141         0.363         -0.026       0      0          0.200
```

## Error-range coverage on the final set (%)

90% ranges calibrated on the selection folds; the band is 88-92. Columns: all, each horizon, each season.

```text
                1     2     3   all   dry  monsoon  post_monsoon  pre_monsoon
output                                                                       
high_height  84.9  84.4  86.0  85.1  89.5     86.9          78.3         85.3
high_time    83.6  84.5  84.3  84.1  86.0     84.2          80.5         86.7
level        83.9  84.2  84.6  84.2  85.0     87.7          78.5         84.9
low_height   84.7  85.3  84.6  84.9  84.5     90.5          75.9         87.5
low_time     88.1  88.3  88.4  88.3  89.0     88.9          86.6         88.4
```

## Promotion checks against the current pipeline

- joint_better: FAIL
- time_mae_not_worse: PASS
- height_mae_not_worse: PASS
- no_season_worse_than_1pp: FAIL
- coverage_88_92: FAIL
- beats_tables: PASS

Joint share difference: +1.48 points (95% interval -1.07 to +4.20).
Season differences (points): dry -1.8, pre_monsoon +3.0, monsoon +1.3, post_monsoon +3.9.
Against the official tables (points): 2024 +11.0.
Publishable horizons: [].
Overall: NOT PASSED.
