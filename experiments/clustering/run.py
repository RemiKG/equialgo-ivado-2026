"""Explore applicant structure with complementary clustering methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import (
    HDBSCAN, OPTICS, AgglomerativeClustering, Birch, DBSCAN,
    KMeans, MiniBatchKMeans,
)
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ("cote_r_equivalent", "heures_travail_semaine")
METHODS = (
    "hdbscan", "dbscan", "optics", "kmeans", "minibatch_kmeans",
    "agglomerative", "birch", "gaussian_mixture",
)


def make_model(method, *, n_clusters=5, min_samples=10,
               min_cluster_size=25, eps=0.5, seed=42):
    factories = {
        "hdbscan": lambda: HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True),
        "dbscan": lambda: DBSCAN(eps=eps, min_samples=min_samples),
        "optics": lambda: OPTICS(
            min_samples=min_samples, min_cluster_size=min_cluster_size),
        "kmeans": lambda: KMeans(
            n_clusters=n_clusters, n_init=10, random_state=seed),
        "minibatch_kmeans": lambda: MiniBatchKMeans(
            n_clusters=n_clusters, n_init=10, random_state=seed),
        "agglomerative": lambda: AgglomerativeClustering(n_clusters=n_clusters),
        "birch": lambda: Birch(n_clusters=n_clusters),
        "gaussian_mixture": lambda: GaussianMixture(
            n_components=n_clusters, n_init=3, random_state=seed),
    }
    if method not in factories:
        raise ValueError(f"Unknown method: {method}")
    return factories[method]()


def cluster(frame, method="hdbscan", *, features=FEATURES, **params):
    """Fit one cohort; return ID-aligned labels and diagnostics."""
    features = list(features)
    if not features or len(features) != len(set(features)):
        raise ValueError("Features must be nonempty and unique.")
    if {"id_candidat", "decision_octroi"}.intersection(features):
        raise ValueError("IDs and decisions cannot be clustering features.")
    if len(frame) < 3 or frame.id_candidat.isna().any() or frame.id_candidat.duplicated().any():
        raise ValueError("At least three rows with unique, nonmissing IDs are required.")
    values = frame[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Clustering features must be finite numeric values.")
    scaler = StandardScaler()
    x = scaler.fit_transform(values)
    model = make_model(method, **params)
    with threadpool_limits(limits=1):
        labels = model.fit_predict(x)
    assigned = labels >= 0
    counts = pd.Series(labels).value_counts().sort_index()
    n_clusters = len(set(labels[assigned]))
    # Bound diagnostics; exclude noise.
    sample = np.flatnonzero(assigned)
    seed = params.get("seed", 42)
    if len(sample) > 1000:
        sample = np.random.default_rng(seed).choice(sample, 1000, replace=False)
    silhouette = None
    if 1 < len(set(labels[sample])) < len(sample):
        silhouette = float(silhouette_score(x[sample], labels[sample]))
    report = {
        "method": method, "rows": len(frame), "features": features,
        "parameters": {
            key: str(value) if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in model.get_params().items()
        },
        "sklearn_version": sklearn.__version__,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "clusters": n_clusters, "noise_fraction": float((~assigned).mean()),
        "cluster_sizes": {str(k): int(v) for k, v in counts.items()},
        "silhouette": silhouette, "silhouette_rows": len(sample), "seed": seed,
    }
    output = frame[["id_candidat"]].copy()
    output["cluster"] = labels
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/equialgo-participants/data/candidats_evaluation.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local/clustering")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="hdbscan")
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={"id_candidat": str})
    methods = METHODS if args.method == "all" else (args.method,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        labels, report = cluster(
            frame, method, features=args.features, n_clusters=args.n_clusters,
            min_samples=args.min_samples, min_cluster_size=args.min_cluster_size,
            eps=args.eps, seed=args.seed,
        )
        labels.to_csv(args.output_dir / f"{method}.csv", index=False)
        (args.output_dir / f"{method}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{method}: {report['clusters']} clusters, "
              f"{report['noise_fraction']:.1%} noise")


if __name__ == "__main__":
    main()
