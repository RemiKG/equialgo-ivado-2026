"""Validate deliverables and create a portable local archive for the judges."""
import hashlib
import json
import zipfile
from pathlib import Path

import nbformat
import pandas as pd
from pypdf import PdfReader

from model_corrige import ROOT, load_data, verify_predictions


def main():
    _, evaluation = load_data()
    prediction = pd.read_csv(ROOT / "predictions.csv")
    report = verify_predictions(evaluation, prediction)
    assert report["grants"] == 1600
    assert report["demographic_parity_gap"] <= .02
    required = ["predictions.csv", "audit_rapport.ipynb", "model_corrige.py", "presentation.pdf"]
    assert all((ROOT / name).is_file() for name in required)
    notebook = nbformat.read(ROOT / "audit_rapport.ipynb", as_version=4)
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.get("outputs", []))
    reader = PdfReader(ROOT / "presentation.pdf")
    assert len(reader.pages) == 7
    assert all(len(page.extract_text() or "") > 200 for page in reader.pages)
    history_dirs = list((ROOT / "predictions_history").glob("*/metadata.json"))
    assert len(history_dirs) >= 2
    for metadata_path in history_dirs:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        csv = metadata_path.parent / "predictions.csv"
        assert hashlib.sha256(csv.read_bytes()).hexdigest() == metadata["csv_sha256"]
        verify_predictions(evaluation, pd.read_csv(csv))
        source = metadata_path.parent / metadata["model_source_filename"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["model_source_sha256"]
    files = []
    for pattern in ["*.py", "*.md", "*.ipynb", "*.csv", "*.pdf", "requirements*.txt", ".gitignore", ".gitattributes"]:
        files.extend(ROOT.glob(pattern))
    for folder in ["artifacts", "tests", "predictions_history", "data/equialgo-participants"]:
        files.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files = sorted(set(files))
    manifest = {"format": 1, "required_deliverables": required, "validation": report,
                "files": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    manifest_path = ROOT / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output = ROOT / "submissions"
    output.mkdir(exist_ok=True)
    bundle = output / "equialgo_submission.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files + [manifest_path]:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert all(name in archive.namelist() for name in required)
    (output / "predictions.csv").write_bytes((ROOT / "predictions.csv").read_bytes())
    (output / "README.md").write_text(
        "# Ready-to-upload submission\n\n"
        "- `predictions.csv`: upload to HxBuddy, IVADO - EquiAlgo.\n"
        "- `equialgo_submission.zip`: all deliverables, original participant data, prediction history and checksums.\n"
        "- Presentation and executed audit are also at the project root.\n\n"
        "Official accuracy and equal-opportunity results are unmeasured. See `../DEVPOST.md` for the event handoff.\n",
        encoding="utf-8")
    print(json.dumps({"archive": str(bundle), "archive_bytes": bundle.stat().st_size,
                      "archived_files": len(files)+1, "prediction_versions": len(history_dirs),
                      "executed_audit_cells": len(cells), "presentation_pages": len(reader.pages),
                      "validation": report}, indent=2))


if __name__ == "__main__":
    main()
