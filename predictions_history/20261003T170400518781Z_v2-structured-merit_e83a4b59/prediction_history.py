"""Archive prediction versions and append official results without losing prior runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "predictions_history"


def file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def refresh_comparison(history_dir: Path = HISTORY) -> None:
    rows = []
    results_file = history_dir / "official_results.jsonl"
    official = {}
    if results_file.exists():
        for line in results_file.read_text(encoding="utf-8").splitlines():
            result = json.loads(line)
            official[result["snapshot"]] = {**official.get(result["snapshot"], {}), **result}
    for path in sorted(history_dir.glob("*/metadata.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        actual = official.get(m["snapshot"], {})
        rows.append({"snapshot": m["snapshot"], "created_utc": m["created_utc"], "model": m["model"],
                     "grants": m["validation"]["grants"], "grant_rate": m["validation"]["grant_rate"],
                     "selection_rate_centre": m["validation"]["selection_rate_centre"],
                     "selection_rate_remote": m["validation"]["selection_rate_remote"],
                     "demographic_parity_gap": m["validation"]["demographic_parity_gap"],
                     "historical_holdout_agreement": m.get("historical_holdout_agreement"),
                     "official_accuracy": actual.get("accuracy"),
                     "official_macro_f1": actual.get("macro_f1"),
                     "official_reference_eop_gap": actual.get("reference_eop_gap"),
                     "official_result_source": actual.get("source"),
                     "changed_decisions_from_previous": m.get("changed_decisions_from_previous"),
                     "csv_sha256": m["csv_sha256"],
                     "predictions_file": f"{m['snapshot']}/predictions.csv"})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(history_dir / "comparison.csv", index=False)
    lines = ["# Prediction history", "",
             "Each version preserves its CSV and metadata. Future model and audit runs archive results automatically.",
             "`comparison.csv` contains machine-readable metrics. Blank official scores mean unmeasured, not zero.", "",
             "| Version | Grants | Regional gap (pp) | Historical agreement | Official accuracy | Changed vs previous |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        hist = "unmeasured" if row["historical_holdout_agreement"] is None else f"{row['historical_holdout_agreement']:.2%}"
        acc = "unmeasured" if row["official_accuracy"] is None else f"{row['official_accuracy']:.2%}"
        change = "—" if row["changed_decisions_from_previous"] is None else str(row["changed_decisions_from_previous"])
        lines.append(f"| [{row['model']}]({row['snapshot']}/predictions.csv) | {row['grants']} | "
                     f"{row['demographic_parity_gap']*100:.3f} | {hist} | {acc} | {change} |")
    lines += ["", "Historical agreement is measured against biased committee labels. It must not be read as true-merit accuracy.",
              "A smaller selection-rate gap is not proof of a smaller independent equal-opportunity gap.", "",
              "To append an actual portal result, using fractions between 0 and 1:", "", "```text",
              "python prediction_history.py score SNAPSHOT_ID --accuracy VALUE --macro-f1 VALUE --source HxBuddy",
              "```", "", "Use the exact snapshot uploaded. Results are appended to `official_results.jsonl`; previous result entries are retained.",
              "Accuracy/F1 from HxBuddy are indicative; independent EOp is available only if the organizer provides it.",
              "The baseline is preserved for comparison. The current submission remains `../predictions.csv`."]
    (history_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_prediction(evaluation: pd.DataFrame, prediction: pd.DataFrame, name: str,
                       historical_agreement: float | None = None, note: str = "",
                       history_dir: Path = HISTORY, model_files: list[Path] | None = None) -> Path:
    from model_corrige import DATA, verify_predictions
    validation = verify_predictions(evaluation, prediction)
    now = datetime.now(timezone.utc)
    payload = prediction.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:45] or "prediction"
    snapshot = f"{now:%Y%m%dT%H%M%S%fZ}_{slug}_{digest[:8]}"
    history_dir.mkdir(exist_ok=True, parents=True)
    previous = sorted(history_dir.glob("*/metadata.json"))
    changed, previous_snapshot = None, None
    if previous:
        previous_snapshot = previous[-1].parent.name
        old = pd.read_csv(previous[-1].parent / "predictions.csv").set_index("id_candidat")
        current = prediction.set_index("id_candidat")
        if set(old.index) == set(current.index):
            changed = int((old.loc[current.index, "decision_octroi"] != current.decision_octroi).sum())
    folder = history_dir / snapshot
    folder.mkdir(exist_ok=False)
    (folder / "predictions.csv").write_bytes(payload)
    source = DATA.parent / "baseline_model.ipynb" if name.startswith("baseline") else ROOT / "model_corrige.py"
    (folder / source.name).write_bytes(source.read_bytes())
    metadata = {"snapshot": snapshot, "created_utc": now.isoformat(), "model": name,
                "csv_sha256": digest, "validation": validation,
                "historical_holdout_agreement": historical_agreement,
                "previous_snapshot": previous_snapshot, "changed_decisions_from_previous": changed,
                "history_data_sha256": file_sha(DATA / "donnees_demandes.csv"),
                "evaluation_data_sha256": file_sha(DATA / "candidats_evaluation.csv"),
                "model_source_sha256": file_sha(source), "model_source_filename": source.name, "note": note}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for dependency in [ROOT / "data_contract.py", ROOT / "merit_inference.py", ROOT / "prediction_history.py"]:
        if dependency.exists():
            (folder / dependency.name).write_bytes(dependency.read_bytes())
    for artifact in model_files or []:
        (folder / artifact.name).write_bytes(artifact.read_bytes())
    if (ROOT / "artifacts/model_parameters.json").exists() and name.startswith("corrected"):
        (folder / "model_parameters.json").write_bytes((ROOT / "artifacts/model_parameters.json").read_bytes())
    refresh_comparison(history_dir)
    return folder


def preserve_existing(output: Path, evaluation: pd.DataFrame, history_dir: Path = HISTORY) -> None:
    """Capture an untracked output before replacement; do not duplicate known CSVs."""
    if not output.exists():
        return
    old = pd.read_csv(output)
    digest = hashlib.sha256(old.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    for path in history_dir.glob("*/metadata.json"):
        if json.loads(path.read_text(encoding="utf-8"))["csv_sha256"] == digest:
            return
    archive_prediction(evaluation, old, "previous-output", note="Automatically preserved before overwrite.", history_dir=history_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Append an actual HxBuddy or jury result.")
    score.add_argument("snapshot")
    score.add_argument("--accuracy", type=float)
    score.add_argument("--macro-f1", type=float)
    score.add_argument("--reference-eop-gap", type=float)
    score.add_argument("--source", required=True)
    commands.add_parser("refresh", help="Rebuild the comparison table.")
    args = parser.parse_args()
    if args.command == "score":
        if Path(args.snapshot).name != args.snapshot or not (HISTORY / args.snapshot / "metadata.json").is_file():
            parser.error("Unknown snapshot. Use an exact directory name from predictions_history.")
        supplied = {k: getattr(args, k) for k in ["accuracy", "macro_f1", "reference_eop_gap"] if getattr(args, k) is not None}
        if not supplied or any(not 0 <= value <= 1 for value in supplied.values()):
            parser.error("Provide at least one measured metric as a fraction between 0 and 1.")
        event = {"snapshot": args.snapshot, "recorded_utc": datetime.now(timezone.utc).isoformat(), "source": args.source, **supplied}
        with (HISTORY / "official_results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
    refresh_comparison()
    print(HISTORY / "comparison.csv")


if __name__ == "__main__":
    main()
