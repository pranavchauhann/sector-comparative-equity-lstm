# Docker Containerization (Phase 5)

Run the entire project — training pipeline and dashboard — with nothing
installed but Docker. No Python, no venv, no dependency wrangling.

> **Status: build-verified end-to-end** (2026-07-18, Colima on macOS
> arm64): both images built, the containerized dashboard served on :8501,
> the training container ran the full `dvc repro` (fetch → features →
> baselines → 40 LSTMs, ~25 min), and after `docker compose restart app`
> the dashboard served the training run's fresh `results/` from the shared
> volume — proving the retrain → serve loop. Measured sizes below.

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
| [Dockerfile](../Dockerfile) | `equity-lstm-training` — full pipeline env (TF, DVC, MLflow). Entrypoint wraps `dvc` with no-SCM init; default command `repro`. |
| [Dockerfile.app](../Dockerfile.app) | `equity-lstm-app` — Streamlit dashboard, inference-only deps (streamlit/pandas/plotly). |
| [docker-compose.yml](../docker-compose.yml) | `app` (continuous, port 8501) + `training` (on-demand, behind a compose profile), bridged by shared volumes. |
| [.dockerignore](../.dockerignore) | Keeps data/, models/, mlruns/, .git, notebooks out of both build contexts. |
| [.env.example](../.env.example) | Placeholder DVC-remote credentials; copy to `.env` (gitignored). |

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
requires ≥3.12 (documented in [app/requirements.txt](../app/requirements.txt))
and the TensorFlow/Keras import-order deadlock this repo works around
(module docstring in [src/lstm_model.py](../src/lstm_model.py)) was diagnosed
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

Measured (Colima, linux/arm64):

| image | size |
|---|---|
| `equity-lstm-training` | **4.16 GB** |
| `equity-lstm-app` | **886 MB** |

The training image exceeds the 2.5–3 GB expectation: TensorFlow +
NumPy/SciPy/pandas wheels plus the DVC/MLflow toolchain are simply that
big, and a multi-stage build doesn't help when the "build-time"
dependencies *are* the runtime dependencies (no compile step to discard —
the wheels are prebuilt). The app image's 886 MB is ~450 MB of Python +
streamlit/pandas/plotly and ~400 MB of baked-in `results/` seed data
(mostly the per-horizon prediction CSVs); trimming those seeds to just
what the dashboard reads is the obvious next size win. The 4.7× gap
between the images remains the argument for the two-image split.

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

## Verification checklist — all executed 2026-07-18 (Colima/macOS arm64)

1. ✅ `docker compose --profile training build` — both images built clean.
2. ✅ `docker compose up -d app` → HTTP 200 on localhost:8501; dashboard
   fully rendered (screenshot: [screenshots/docker_app.png](screenshots/docker_app.png)).
3. ✅ `docker compose run --rm training` — full `dvc repro` inside the
   container: fetch → features → baselines → 40 LSTMs in ~25 min,
   `results/lstm_metrics.csv` + `models/lstm/` written to the volumes.
4. ✅ `docker compose restart app` — dashboard served the *fresh*
   `lstm_metrics.csv` (volume timestamp minutes old, values differing
   from the image-baked seed), proving the retrain → serve loop.
5. ✅ Zero-local-Python: every command above used only the Docker CLI —
   the containers resolve their own interpreters and dependencies; the
   host venv was never consulted.
6. ✅ Sizes recorded in the table above.

Note for macOS + external-drive setups: this run hosted the Colima VM on
an APFS sparsebundle on an exFAT external drive (internal disk was full).
`hdiutil create -type SPARSEBUNDLE -fs APFS`, mount, move `~/.colima`
there, symlink back — works; exFAT can't host the VM disk directly.
