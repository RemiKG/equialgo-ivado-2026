# Submission results

Best verified result: **94.78% accuracy / 94.56% macro F1**.
The indicative score equalled macro F1 for every scored attempt. First three scores were reported by the participant; subsequent scores were directly observed on HxBuddy.

| # | Attempt / preserved CSV | Accuracy | Macro F1 |
|---:|---|---:|---:|
| 1 | [corrected-policy-audited](predictions_history/20261003T163644542154Z_corrected-policy-audited_0120f838/predictions.csv) | 94.63% | 94.40% |
| 2 | [v2-structured_merit](predictions_history/20261003T170400518781Z_v2-structured-merit_e83a4b59/predictions.csv) | 92.23% | 91.90% |
| 3 | [v2-academic-anchor](predictions_history/20261003T170401370588Z_v2-academic-anchor_fe0fda5e/predictions.csv) | 93.13% | 92.84% |
| 4 | [v3-feedback-conditioned](predictions_history/20261003T172906132935Z_v3-feedback-conditioned_96ac7e4b/predictions.csv) | 94.58% | 94.35% |
| 5 | [income_negative](predictions_history/20261003T173553661117Z_income-negative_eadf4645/predictions.csv) | 93.68% | 93.41% |
| 6 | [income_positive](predictions_history/20261003T173553508116Z_income-positive_c9bddf03/predictions.csv) | 93.83% | 93.57% |
| 7 | [firstgen_positive](predictions_history/20261003T173553860733Z_firstgen-positive_43ce21b9/predictions.csv) | 93.43% | 93.15% |
| 8 | [work_022](predictions_history/20261003T173553343153Z_work-022_5e56596c/predictions.csv) | 93.68% | 93.41% |
| 9 | [commute_positive](predictions_history/20261003T173554176733Z_commute-positive_678ed1a6/predictions.csv) | 94.08% | 93.83% |
| 10 | [remote_positive](predictions_history/20261003T174409483857Z_remote-positive_0125662d/predictions.csv) | 93.33% | 93.05% |
| 11 | [maxentropy_10](predictions_history/20261003T174643107811Z_maxentropy-10_be5c684f/predictions.csv) | 94.68% | 94.45% |
| 12 | [maxentropy_11](predictions_history/20261003T174849668846Z_maxentropy-11_2038f8dc/predictions.csv) | 94.78% | 94.56% |
| 13 | [additive_spline_C0.1](predictions_history/20261003T175021851824Z_additive-spline-c0-1_9f9f514c/predictions.csv) | 94.73% | 94.50% |
| 14 | [maxentropy_13](predictions_history/20261003T175209397846Z_maxentropy-13_80a004de/predictions.csv) | 94.78% | 94.56% |

Active champion: `20261003T174849668846Z_maxentropy-11_2038f8dc`. Root `predictions.csv` contains 1,600 grants among 4,000 applications.
The original method scored 94.63% / 94.40%; improvement is 0.15 percentage points accuracy and 0.16 points macro F1.
The last observed competing leader scored 94.73% / 94.50%. Our lead over that observation is narrow; the leaderboard has not been rechecked since desktop control was disabled.

Calibration uses aggregate leaderboard feedback on this evaluation cohort. These are real portal scores, but they do not establish out-of-sample generalization or 98% accuracy.

Unscored candidates and historical-label diagnostics are listed separately in [the complete history](predictions_history/README.md).
Every snapshot preserves predictions, code and metadata; official results are append-only. Some exploratory code snapshots require the shared root modules.

Desktop automation is disabled at the participant's request. Background work does not interact with keyboard, mouse, windows, clipboard or browser UI.
