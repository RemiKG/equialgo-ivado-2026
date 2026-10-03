"""EquiAlgo: auditable score repair and exact-budget regional allocation.

Historical decisions are observations of the committee, not fair ground truth.
No accuracy against the undisclosed jury reference is claimed by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "equialgo-participants" / "data"
REMOTE = {"Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"}
REGIONS = REMOTE | {"Montreal", "Capitale-Nationale"}
NUMERIC = ["cote_r_equivalent", "revenu_familial_estime", "heures_travail_semaine",
           "distance_domicile_campus_km", "premiere_generation_universitaire"]
CATEGORICAL = ["programme_etudes", "region_administrative"]
PROGRAMS = ["Arts et lettres", "Genie", "Sante", "Sciences", "Sciences sociales"]
FEATURES = ["id_candidat", "cote_r_equivalent", "programme_etudes", "region_administrative",
            "code_postal_3", "revenu_familial_estime", "heures_travail_semaine",
            "distance_domicile_campus_km", "premiere_generation_universitaire"]
TARGET = "decision_octroi"
DEFAULT_GAP = 0.02  # Policy chosen in advance; never tuned on hidden labels.


def groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.region_administrative.isin(REMOTE).to_numpy(dtype=int)


def validate_data(frame: pd.DataFrame, labeled: bool = False) -> None:
    required = FEATURES + ([TARGET] if labeled else [])
    if set(frame.columns) != set(required):
        raise ValueError(f"Expected exactly these columns: {required}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Empty data or missing values; send the batch for review.")
    if frame.id_candidat.duplicated().any() or not frame.id_candidat.str.fullmatch(r"C\d+").all():
        raise ValueError("Candidate IDs must be unique C-prefixed numeric identifiers.")
    if not set(frame.region_administrative).issubset(REGIONS):
        raise ValueError("Unknown region; do not silently assign it to a group.")
    if not set(frame.programme_etudes).issubset(PROGRAMS):
        raise ValueError("Unknown study program; review the input schema.")
    if not np.isfinite(frame[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError("All numeric features must be finite.")
    if not frame.premiere_generation_universitaire.isin([0, 1]).all():
        raise ValueError("First-generation indicator must be binary.")
    if not frame.cote_r_equivalent.between(15, 40).all():
        raise ValueError("R scores must be in the published range [15, 40].")
    if (frame[["revenu_familial_estime", "heures_travail_semaine", "distance_domicile_campus_km"]] < 0).any().any():
        raise ValueError("Income, hours and distance cannot be negative.")
    if labeled and not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Historical decisions must be binary.")


def load_data(data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(data_dir / "donnees_demandes.csv")
    evaluation = pd.read_csv(data_dir / "candidats_evaluation.csv")
    validate_data(history, True)
    validate_data(evaluation)
    if set(history.id_candidat) & set(evaluation.id_candidat):
        raise ValueError("Historical and evaluation candidate IDs overlap.")
    return history, evaluation


@dataclass
class RepairedScore:
    """Learn conditional associations, then apply explicit monotonic policy rules.

    Region, distance and program are nuisance controls during estimation. Postal
    codes are omitted because they are redundant geographic controls. The final
    individual score has no geographic or program contribution. A positive
    wealth advantage and a negative first-generation coefficient are removed.
    This is a policy assumption, not an identified causal or hidden-merit model.
    """
    means: np.ndarray
    scales: np.ndarray
    columns: list[str]
    coefficients: np.ndarray
    intercept: float
    repaired: np.ndarray

    @staticmethod
    def features(frame: pd.DataFrame) -> pd.DataFrame:
        raw = frame[NUMERIC + CATEGORICAL].copy()
        # Fixed categories make individual and subset predictions consistent.
        raw["programme_etudes"] = pd.Categorical(raw.programme_etudes, categories=PROGRAMS)
        raw["region_administrative"] = pd.Categorical(raw.region_administrative, categories=sorted(REGIONS))
        return pd.get_dummies(raw, columns=CATEGORICAL,
                              drop_first=True, dtype=float).astype(float)

    @classmethod
    def fit(cls, history: pd.DataFrame) -> "RepairedScore":
        raw = cls.features(history)
        means = raw.mean().to_numpy()
        scales = raw.std(ddof=0).replace(0, 1).to_numpy()
        x = (raw.to_numpy() - means) / scales
        y = history[TARGET].to_numpy(dtype=float)

        def loss(theta):
            z = theta[0] + x @ theta[1:]
            residual = expit(z) - y
            # Small ridge penalty for stable nuisance estimation.
            value = np.logaddexp(0, z).sum() - y @ z + 0.5 * np.dot(theta[1:], theta[1:])
            gradient = np.r_[residual.sum(), x.T @ residual + theta[1:]]
            return value, gradient

        opt = minimize(loss, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7})
        if not opt.success:
            raise RuntimeError(f"Model did not converge: {opt.message}")
        coefficients = opt.x[1:]
        repaired = np.zeros_like(coefficients)
        for feature in ["cote_r_equivalent", "heures_travail_semaine", "premiere_generation_universitaire"]:
            i = raw.columns.get_loc(feature)
            repaired[i] = max(0.0, coefficients[i])
        i = raw.columns.get_loc("revenu_familial_estime")
        repaired[i] = min(0.0, coefficients[i])
        return cls(means, scales, list(raw.columns), coefficients, float(opt.x[0]), repaired)

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.features(frame).reindex(columns=self.columns, fill_value=0).to_numpy()
        return (x - self.means) / self.scales

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.matrix(frame) @ self.repaired

    def historical_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return expit(self.intercept + self.matrix(frame) @ self.coefficients)

    def save(self, path: Path) -> None:
        payload = {"means": self.means.tolist(), "scales": self.scales.tolist(),
                   "columns": self.columns, "coefficients": self.coefficients.tolist(),
                   "intercept": self.intercept, "repaired": self.repaired.tolist(),
                   "warning": "Policy ranking score; not a calibrated probability of true merit."}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def tie_keys(ids: np.ndarray) -> np.ndarray:
    """Stable seeded hash breaks ties without exploiting candidate number/order."""
    return np.array([hashlib.sha256(("equialgo-v1:" + str(v)).encode()).hexdigest() for v in ids])


def allocate(scores: np.ndarray, group: np.ndarray, ids: np.ndarray,
             grant_rate: float = 0.4, max_gap: float | None = DEFAULT_GAP) -> np.ndarray:
    """Maximize summed ranking utility with exactly round(n*rate) binary grants.

    For each feasible remote-group count, select the highest scoring candidates
    within each group. Exhaustive count enumeration solves this two-group
    constrained problem exactly, including the finite-batch parity constraint.
    Group membership is used here and must be disclosed; ranking is group blind.
    """
    scores = np.asarray(scores, dtype=float)
    group, ids = np.asarray(group), np.asarray(ids)
    n = len(scores)
    if n == 0 or len(group) != n or len(ids) != n or not np.isfinite(scores).all():
        raise ValueError("Scores, groups and IDs must be aligned, finite and nonempty.")
    if len(set(ids)) != n or not set(group).issubset({0, 1}):
        raise ValueError("Unique IDs and binary groups are required.")
    if not 0.36 <= grant_rate <= 0.44:
        raise ValueError("Grant rate must meet the challenge budget [0.36, 0.44].")
    if max_gap is not None and not 0 <= max_gap <= 1:
        raise ValueError("Parity tolerance must be between zero and one.")
    k = int(round(n * grant_rate))
    if not 0.36 <= k / n <= 0.44:
        raise ValueError("No requested integer budget is valid for this batch size.")
    keys = tie_keys(ids)
    order = np.lexsort((keys, -scores))
    pred = np.zeros(n, dtype=int)
    if max_gap is None:
        pred[order[:k]] = 1
        return pred
    centre, remote = order[group[order] == 0], order[group[order] == 1]
    if not len(centre) or not len(remote):
        raise ValueError("Both groups are required to enforce regional parity.")
    nr = np.arange(max(0, k - len(centre)), min(k, len(remote)) + 1)
    nc = k - nr
    gaps = np.abs(nr / len(remote) - nc / len(centre))
    eligible = gaps <= max_gap + 1e-12
    if not eligible.any():
        raise ValueError("Requested parity tolerance is infeasible with integer grants.")
    cr = np.r_[0.0, np.cumsum(scores[remote])]
    cc = np.r_[0.0, np.cumsum(scores[centre])]
    utility = cr[nr] + cc[nc]
    utility[~eligible] = -np.inf
    best = int(np.argmax(utility))
    pred[remote[:nr[best]]] = 1
    pred[centre[:nc[best]]] = 1
    return pred


def verify_predictions(evaluation: pd.DataFrame, submission: pd.DataFrame) -> dict:
    if list(submission.columns) != ["id_candidat", TARGET]:
        raise ValueError("Submission must have exactly the two required columns.")
    if len(submission) != 4000 or len(evaluation) != 4000:
        raise ValueError("The official evaluation submission requires exactly 4,000 rows.")
    if not submission.id_candidat.equals(evaluation.id_candidat):
        raise ValueError("Candidate IDs or their order do not match the evaluation file.")
    if submission.id_candidat.duplicated().any() or not submission[TARGET].isin([0, 1]).all():
        raise ValueError("Duplicate candidates or nonbinary decisions.")
    rate = float(submission[TARGET].mean())
    if not 0.36 <= rate <= 0.44:
        raise ValueError("Grant rate violates the official budget.")
    g = groups(evaluation)
    rates = [float(submission.loc[g == i, TARGET].mean()) for i in [0, 1]]
    return {"rows": len(submission), "grants": int(submission[TARGET].sum()),
            "grant_rate": rate, "selection_rate_centre": rates[0],
            "selection_rate_remote": rates[1], "demographic_parity_gap": abs(rates[1] - rates[0]),
            "reference_accuracy": None, "reference_equal_opportunity_gap": None,
            "status": "Format, IDs and budget valid; hidden-reference metrics unavailable."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "predictions.csv")
    parser.add_argument("--max-gap", type=float, default=DEFAULT_GAP)
    args = parser.parse_args()
    history, evaluation = load_data(args.data_dir)
    model = RepairedScore.fit(history)
    pred = allocate(model.score(evaluation), groups(evaluation), evaluation.id_candidat.to_numpy(), max_gap=args.max_gap)
    submission = pd.DataFrame({"id_candidat": evaluation.id_candidat, TARGET: pred})
    report = verify_predictions(evaluation, submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from prediction_history import archive_prediction, preserve_existing
    history_dir = args.output.parent / "predictions_history"
    preserve_existing(args.output, evaluation, history_dir)
    submission.to_csv(args.output, index=False)
    artifacts = args.output.parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    model.save(artifacts / "model_parameters.json")
    (artifacts / "submission_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    archived = archive_prediction(evaluation, submission, "corrected-policy", history_dir=history_dir,
                                  note=f"Full historical fit; max_gap={args.max_gap}; no official result available.")
    print(f"Archived predictions: {archived}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
