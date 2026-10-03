# Prediction history

Each version preserves its CSV and metadata. Future model and audit runs archive results automatically.
`comparison.csv` contains machine-readable metrics. Blank official scores mean unmeasured, not zero.

| Version | Grants | Regional gap (pp) | Historical agreement | Official accuracy | Official macro F1 | Changed vs previous |
|---|---:|---:|---:|---:|---:|---:|
| [baseline-original](20261003T163633759848Z_baseline-original_36c4465b/predictions.csv) | 1562 | 19.032 | 88.13% | unmeasured | unmeasured | — |
| [corrected-initial](20261003T163633785813Z_corrected-initial_0120f838/predictions.csv) | 1600 | 0.021 | 85.33% | unmeasured | unmeasured | 388 |
| [corrected-policy-audited](20261003T163644542154Z_corrected-policy-audited_0120f838/predictions.csv) | 1600 | 0.021 | 85.33% | 94.63% | 94.40% | 0 |
| [v2-structured_merit](20261003T170400518781Z_v2-structured-merit_e83a4b59/predictions.csv) | 1600 | 2.196 | unmeasured | 92.23% | 91.90% | 194 |
| [v2-broad_merit](20261003T170401235593Z_v2-broad-merit_ad1a53d9/predictions.csv) | 1600 | 0.849 | unmeasured | unmeasured | unmeasured | 60 |
| [v2-academic-anchor](20261003T170401370588Z_v2-academic-anchor_fe0fda5e/predictions.csv) | 1600 | 5.614 | unmeasured | 93.13% | 92.84% | 126 |
| [v3-feedback-conditioned](20261003T172703142369Z_v3-feedback-conditioned_2a16729f/predictions.csv) | 1600 | 5.573 | unmeasured | unmeasured | unmeasured | 290 |
| [v3-feedback-conditioned](20261003T172906132935Z_v3-feedback-conditioned_96ac7e4b/predictions.csv) | 1600 | 0.808 | unmeasured | 94.58% | 94.35% | 118 |
| [work_022](20261003T173553343153Z_work-022_5e56596c/predictions.csv) | 1600 | 4.744 | unmeasured | 93.68% | 93.41% | 102 |
| [income_positive](20261003T173553508116Z_income-positive_c9bddf03/predictions.csv) | 1600 | 3.439 | unmeasured | 93.83% | 93.57% | 232 |
| [income_negative](20261003T173553661117Z_income-negative_eadf4645/predictions.csv) | 1600 | 2.672 | unmeasured | 93.68% | 93.41% | 290 |
| [firstgen_positive](20261003T173553860733Z_firstgen-positive_43ce21b9/predictions.csv) | 1600 | 1.015 | unmeasured | 93.43% | 93.15% | 172 |
| [firstgen_negative](20261003T173554015736Z_firstgen-negative_a3500f35/predictions.csv) | 1600 | 0.953 | unmeasured | unmeasured | unmeasured | 276 |
| [commute_positive](20261003T173554176733Z_commute-positive_678ed1a6/predictions.csv) | 1600 | 4.019 | unmeasured | 94.08% | 93.83% | 164 |
| [remote_positive](20261003T174409483857Z_remote-positive_0125662d/predictions.csv) | 1600 | 7.437 | unmeasured | 93.33% | 93.05% | 76 |
| [maxentropy_10](20261003T174643107811Z_maxentropy-10_be5c684f/predictions.csv) | 1600 | 1.367 | unmeasured | 94.68% | 94.45% | 170 |
| [maxentropy_11](20261003T174849668846Z_maxentropy-11_2038f8dc/predictions.csv) | 1600 | 0.849 | unmeasured | 94.78% | 94.56% | 16 |
| [additive_spline_C0.1](20261003T175021851824Z_additive-spline-c0-1_9f9f514c/predictions.csv) | 1600 | 0.083 | unmeasured | 94.73% | 94.50% | 44 |
| [additive_spline_C1](20261003T175021939285Z_additive-spline-c1_2d8262de/predictions.csv) | 1600 | 0.290 | unmeasured | unmeasured | unmeasured | 6 |
| [additive_spline_C10](20261003T175022030319Z_additive-spline-c10_6abe7266/predictions.csv) | 1600 | 0.290 | unmeasured | unmeasured | unmeasured | 2 |
| [maxentropy_13](20261003T175209397846Z_maxentropy-13_80a004de/predictions.csv) | 1600 | 1.160 | unmeasured | 94.78% | 94.56% | 38 |
| [offline-robust-calibration](20261003T180138757023Z_offline-robust-calibration_970f9adb/predictions.csv) | 1600 | 0.953 | unmeasured | unmeasured | unmeasured | 16 |

Historical agreement is measured against biased committee labels. It must not be read as true-merit accuracy.
A smaller selection-rate gap is not proof of a smaller independent equal-opportunity gap.

To append an actual portal result, using fractions between 0 and 1:

```text
python prediction_history.py score SNAPSHOT_ID --accuracy VALUE --macro-f1 VALUE --source HxBuddy
```

Use the exact snapshot uploaded. Results are appended to `official_results.jsonl`; previous result entries are retained.
Accuracy/F1 from HxBuddy are indicative; independent EOp is available only if the organizer provides it.
The baseline is preserved for comparison. The current submission remains `../predictions.csv`.
