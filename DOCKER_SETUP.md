# Docker Containerization (Phase 5)

Run the entire project — training pipeline and dashboard — with nothing
installed but Docker. No Python, no venv, no dependency wrangling.

> **Status: authored but not yet build-verified.** No Docker runtime was
> available on the development machine when this phase was written (Docker
> Desktop had been uninstalled), so `docker compose build/up` has not been
> executed against these files yet. The checklist at the bottom is the
> exact verification to run once a runtime (Docker Desktop or Colima) is
> installed. This note gets deleted when that happens.

## Quickstart

```bash
# 1. The dashboard (serves on http://localhost:8501)
docker compose up -d app

# 2. The training pipeline (on demand — full DVC repro inside the container)
cp .env.example .env          # placeholders are fine for a local run
docker compose run --rm training

# 3. Individual stages / other DVC commands work too:
docker compose run --rm training repro train_lstm
docker compose run --rm training status
```

## The pieces

| File | What it builds |
|---|---|
| [Dockerfile](Dockerfile) | `equity-lstm-training` — full pipeline env (TF, DVC, MLflow). Entrypoint wraps `dvc` with no-SCM init; default command `repro`. |
| [Dockerfile.app](Dockerfile.app) | `equity-lstm-app` — Streamlit dashboard, inference-only deps (streamlit/pandas/plotly). |
| [docker-compose.yml](docker-compose.yml) | `app` (continuous, port 8501) + `training` (on-demand, behind a compose profile), bridged by shared volumes. |
| [.dockerignore](.dockerignore) | Keeps data/, models/, mlruns/, .git, notebooks out of both build contexts. |
| [.env.example](.env.example) | Placeholder DVC-remote credentials; copy to `.env` (gitignored). |

## Why two images instead of one

The app reads precomputed CSVs — it never imports TensorFlow. One combined
image would mean the continuously-running web container carries a multi-GB
ML toolchain it never executes: slower pulls, bigger attack surface, and
every dependency bump to the training stack forces a redeploy of the app.
Separated, the app image is an order of magnitude smaller, rebuilds in
seconds, and the training image can churn freely without touching the
serving path.

## The shared volume (and what breaks without it)

`training` writes `results/` + `models/` into named volumes; `app` mounts
the same `artifacts` volume over `/app/results`. Remove the volume and
nothing crashes — which is exactly the danger: the training container
completes, its outputs vanish with it, and the app keeps serving the
results that were baked into its image at build time. Silently stale is
worse than loudly broken; the volume is what makes retraining actually
reach the dashboard. (First `up` seeds the volume from the app image's
baked-in results, so the dashboard works before training has ever run.)

## Base image: why 3.12-slim, not 3.10-slim

The dependency set is pinned to Python 3.12 behaviour: `pandas-ta`
requires ≥3.12 (documented in [app/requirements.txt](app/requirements.txt))
and the TensorFlow/Keras import-order deadlock this repo works around
(module docstring in [src/lstm_model.py](src/lstm_model.py)) was diagnosed
and verified on 3.12. A 3.10 container would ship an environment the
project was never tested on — the opposite of what containerization is
for.

## Image size: expectations and mitigations

Applied in the Dockerfiles:

- `python:3.12-slim` base (~120 MB vs ~1 GB for the full image)
- `pip install --no-cache-dir` (drops ~1 GB of wheel cache from the layer)
- dependency layer isolated from the code layer (code edits don't
  re-download TF)
- `.dockerignore` keeps data/models/notebooks out of the build context

Honest expectations: the training image will still land in the 2.5–3 GB
range — TensorFlow's wheels plus NumPy/SciPy/pandas are simply that big,
and a multi-stage build doesn't help when the "build-time" dependencies
*are* the runtime dependencies (there is no compile step to discard; the
wheels are prebuilt). The app image should land around 400–500 MB, the
gap being the whole argument for the two-image split. Record actuals with
`docker images` after the first build.

## Docker Hub

Not pushed yet — publishing requires the repo owner's Docker Hub account
(`docker login` credentials that shouldn't pass through an agent). To
publish once logged in:

```bash
docker tag equity-lstm-training <username>/equity-lstm-training:latest
docker tag equity-lstm-app      <username>/equity-lstm-app:latest
docker push <username>/equity-lstm-training:latest
docker push <username>/equity-lstm-app:latest
```

## Verification checklist (run when a Docker runtime is available)

1. `docker compose build` — both images build clean.
2. `docker compose up -d app` → `curl -s localhost:8501` returns the
   Streamlit shell; dashboard renders in a browser.
3. `docker compose run --rm training` — full `dvc repro` completes inside
   the container (fetch → features → baselines → LSTM).
4. Restart the app (`docker compose restart app`) and confirm it serves
   the training run's fresh `results/` from the shared volume.
5. The zero-local-Python proof: `deactivate` any venv (or use a machine
   with no Python at all) and repeat 1–4 — nothing on the host is
   consulted except Docker.
6. `docker images | grep equity-lstm` — record the size table here.
