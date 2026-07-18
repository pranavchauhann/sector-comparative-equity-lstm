"""
src/tracking.py
================
Shared MLflow helpers for Phase 2 experiment tracking.

Design
------
* **Local SQLite store.** Run metadata lands in ``<repo>/mlflow.db`` and
  artifacts in ``<repo>/mlruns/`` (both gitignored) — no remote server
  needed. MLflow 3.x put the plain filesystem backend into maintenance mode,
  so SQLite is the recommended local store. Launch the dashboard with
  ``mlflow ui --backend-store-uri sqlite:///mlflow.db`` from the repo root
  (see MLFLOW_SETUP.md).
* **One MLflow experiment per sector** (IT, Banking, Energy, FMCG), so a
  sector's baseline and LSTM runs sit side by side and are directly
  comparable, rather than all 40 stocks' runs landing in one flat list.
* **MLflow complements DVC, it does not replace it.** DVC versions the
  *artifacts and data lineage* (which data + code produced which model);
  MLflow records the *experiment* (which hyperparameters produced which
  metrics) so any two runs can be compared. The DVC stages call into this
  module as a side effect and still produce their tracked outputs unchanged.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_DIR = ROOT / "mlruns"


def setup_mlflow(sector: str) -> None:
    """Point MLflow at the local SQLite store and select the sector's experiment.

    Paths are anchored to the repo root (not cwd) so direct runs and
    ``dvc repro`` write to the same store. The experiment's artifact location
    is pinned to ``<repo>/mlruns`` at creation for the same reason.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(sector) is None:
        mlflow.create_experiment(
            sector, artifact_location=(ARTIFACT_DIR / sector.replace(" ", "_")).as_uri())
    mlflow.set_experiment(sector)
