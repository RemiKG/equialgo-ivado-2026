# Applicant clustering

Standalone cohort exploration using HDBSCAN, DBSCAN, OPTICS, K-means,
MiniBatch K-means, Ward agglomerative clustering, BIRCH and Gaussian mixtures.
Uses the existing `config/requirements.txt` dependencies.

```bash
python experiments/clustering/run.py --method hdbscan
python experiments/clustering/run.py --method all --n-clusters 5
python experiments/clustering/run.py --input applicants.csv --method dbscan --eps 0.3
```

Academic scores and weekly work hours are standardized per input cohort.
`--features` accepts an explicit list of numeric columns; IDs and decisions are
excluded. Inputs require unique `id_candidat` values and finite features.

Outputs go to `.local/clustering/`: ID-aligned labels and a JSON report per method.
Reports record parameters, scaling, cluster sizes, noise fraction and silhouette
on at most 1,000 non-noise rows. Undefined silhouettes are `null`; noise is `-1`.
`--min-samples` and `--min-cluster-size` control density methods; `--n-clusters`
controls fixed-count methods. DBSCAN's `--eps` is in standardized feature units.

Cluster IDs are local to each fit. These diagnostics support cohort analysis and
future feature experiments; they do not measure allocation accuracy or fairness.
The allocation model does not consume these outputs.
