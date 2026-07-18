# Phase 5: training-pipeline image.
#
# Runs the full DVC pipeline (fetch -> engineer -> baselines -> LSTM) in a
# reproducible environment. Heavy by nature: TensorFlow alone is ~600 MB of
# wheels; see DOCKER_SETUP.md's size notes for what was done about it.
#
# NOTE on the base image: python:3.12-slim, not 3.10-slim. This project's
# dependency set is pinned to 3.12 behaviour (pandas-ta requires >=3.12 —
# documented in app/requirements.txt — and the TF-import-order deadlock
# documented in src/lstm_model.py was diagnosed and verified on 3.12).
# Containerizing on 3.10 would ship an environment the project was never
# tested on, defeating the purpose of the container.

FROM python:3.12-slim

WORKDIR /work

# Layer 1: dependencies (cached until requirements.txt changes).
# --no-cache-dir keeps pip's wheel cache out of the layer (~1 GB saved).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "dvc>=3.0"

# Layer 2: pipeline code + config. data/, models/, mlruns/ etc. are excluded
# by .dockerignore — outputs belong in the shared volume, not the image.
COPY dvc.yaml dvc.lock ./
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/

# The pipeline stages invoke `.venv/bin/python`; inside the container the
# interpreter is the system python. Symlink so dvc.yaml works unmodified.
RUN mkdir -p .venv/bin && ln -s "$(which python)" .venv/bin/python

# DVC without git: initialise in no-SCM mode at first run, then hand off to
# `dvc repro` (or whatever stage/args the caller passes).
COPY docker/entrypoint-training.sh /usr/local/bin/entrypoint-training.sh
RUN chmod +x /usr/local/bin/entrypoint-training.sh

ENTRYPOINT ["entrypoint-training.sh"]
CMD ["repro"]
