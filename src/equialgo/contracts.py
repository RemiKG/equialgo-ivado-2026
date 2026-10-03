"""Shared input contracts and deterministic budget allocation; no fitted model."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

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

DEFAULT_GAP = 0.02

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
