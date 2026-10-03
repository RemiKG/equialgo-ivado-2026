# Prediction history

Each version preserves its CSV and metadata. Future model and audit runs archive results automatically.
`comparison.csv` contains machine-readable metrics. Blank official scores mean unmeasured, not zero.

| Version | Grants | Regional gap (pp) | Historical agreement | Official accuracy | Changed vs previous |
|---|---:|---:|---:|---:|---:|
| [baseline-original](20261003T163633759848Z_baseline-original_36c4465b/predictions.csv) | 1562 | 19.032 | 88.13% | unmeasured | — |
| [corrected-initial](20261003T163633785813Z_corrected-initial_0120f838/predictions.csv) | 1600 | 0.021 | 85.33% | unmeasured | 388 |
| [corrected-policy-audited](20261003T163644542154Z_corrected-policy-audited_0120f838/predictions.csv) | 1600 | 0.021 | 85.33% | unmeasured | 0 |

Historical agreement is measured against biased committee labels. It must not be read as true-merit accuracy.
A smaller selection-rate gap is not proof of a smaller independent equal-opportunity gap.

To append an actual portal result, using fractions between 0 and 1:

```text
python prediction_history.py score SNAPSHOT_ID --accuracy VALUE --macro-f1 VALUE --source HxBuddy
```

Use the exact snapshot uploaded. Results are appended to `official_results.jsonl`; previous result entries are retained.
Accuracy/F1 from HxBuddy are indicative; independent EOp is available only if the organizer provides it.
The baseline is preserved for comparison. The current submission remains `../predictions.csv`.
