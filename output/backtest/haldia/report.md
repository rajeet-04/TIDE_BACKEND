# Backtest report: haldia

Selected model A: `A-wall-auto+shallow-side-notrend`. Selection set: test years up to 2019; final set: later years. Truth: QC-passed gauge readings. A hit is within ±30 min and ±0.30 m.

## Final set, all horizons

```text
                       candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
A-wall-auto+shallow-side-notrend     21156     87.332        10.382        26.446         -3.935         0.147         0.362         -0.052       0      0          0.207
                current_pipeline     21156     87.261         9.455        24.631          0.420         0.148         0.367         -0.011       0      0          0.209
                      utide_only     21156     85.432        11.776        28.773         -6.707         0.151         0.365         -0.005       0      0          0.209
```

## Final set by horizon

```text
 horizon                        candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
       1 A-wall-auto+shallow-side-notrend      7052     87.323        10.357        26.429         -3.893         0.147         0.361         -0.050       0      0          0.207
       2 A-wall-auto+shallow-side-notrend      7052     87.252        10.389        26.461         -3.943         0.147         0.363         -0.052       0      0          0.207
       3 A-wall-auto+shallow-side-notrend      7052     87.422        10.400        26.476         -3.969         0.147         0.361         -0.055       0      0          0.207
       1                 current_pipeline      7052     87.876         9.345        24.202          1.389         0.144         0.361         -0.017       0      0          0.210
       2                 current_pipeline      7052     85.579         9.448        24.906          0.391         0.154         0.380         -0.006       0      0          0.210
       3                 current_pipeline      7052     88.330         9.573        24.684         -0.518         0.146         0.354         -0.010       0      0          0.207
       1                       utide_only      7052     85.210        11.745        28.789         -6.655         0.152         0.367         -0.006       0      0          0.210
       2                       utide_only      7052     85.111        11.786        28.759         -6.710         0.153         0.366         -0.004       0      0          0.210
       3                       utide_only      7052     85.976        11.798        28.759         -6.755         0.149         0.360         -0.006       0      0          0.207
```

## Final set by season

```text
      season                        candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
         dry A-wall-auto+shallow-side-notrend      5235     91.213        10.641        26.294         -5.919         0.126         0.316         -0.050       0      0          0.166
     monsoon A-wall-auto+shallow-side-notrend      7074     87.645         9.553        25.021         -1.900         0.140         0.357         -0.051       0      0          0.205
post_monsoon A-wall-auto+shallow-side-notrend      5310     83.974        10.306        25.000         -4.721         0.176         0.392         -0.088       0      0          0.234
 pre_monsoon A-wall-auto+shallow-side-notrend      3537     86.005        11.769        31.150         -3.890         0.149         0.367         -0.005       0      0          0.222
         dry                 current_pipeline      5235     94.499         8.323        21.970          0.941         0.114         0.285         -0.006       0      0          0.178
     monsoon                 current_pipeline      7074     87.716         9.186        24.106          0.610         0.144         0.373         -0.007       0      0          0.205
post_monsoon                 current_pipeline      5310     80.490         9.499        23.481          0.102         0.181         0.412         -0.037       0      0          0.227
 pre_monsoon                 current_pipeline      3537     85.807        11.604        30.630         -0.251         0.155         0.351          0.013       0      0          0.230
         dry                       utide_only      5235     89.112        12.709        28.391         -9.274         0.129         0.332         -0.038       0      0          0.178
     monsoon                       utide_only      7074     87.334         9.884        26.000         -2.943         0.148         0.359          0.030       0      0          0.205
post_monsoon                       utide_only      5310     82.298        11.931        27.938         -8.429         0.169         0.401         -0.050       0      0          0.227
 pre_monsoon                       utide_only      3537     80.888        13.948        34.497         -7.849         0.163         0.363          0.038       0      0          0.230
```

## Against the official tables (one year ahead)

```text
 test_year                        candidate  observed  joint_pct  time_mae_min  time_p95_min  time_bias_min  height_mae_m  height_p95_m  height_bias_m  missed  extra  hourly_rmse_m
      2024                  official_tables      1413     79.335        16.580        33.477         15.900         0.161         0.400         -0.119       0      0            NaN
      2024 A-wall-auto+shallow-side-notrend      1414     89.533         8.761        23.058         -2.728         0.141         0.363         -0.026       0      0          0.200
      2024                 current_pipeline      1414     87.694         9.500        23.937          5.299         0.143         0.360         -0.044       0      0          0.205
      2024                       utide_only      1414     87.199         9.787        23.894         -5.111         0.148         0.372          0.021       0      0          0.205
```

## Promotion checks against the current pipeline

- joint_better: FAIL
- time_mae_not_worse: FAIL
- height_mae_not_worse: PASS
- no_season_worse_than_1pp: FAIL

Joint share difference: +0.07 points (95% interval -1.76 to +2.14).
Season differences (points): dry -3.3, pre_monsoon +0.2, monsoon -0.1, post_monsoon +3.5.
Publishable horizons: [1, 2, 3]. Coverage check: pending: error ranges arrive in plan 1b.
Overall: NOT PASSED.
