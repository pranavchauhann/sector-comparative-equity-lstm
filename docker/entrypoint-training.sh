#!/bin/sh
# Training-container entrypoint: make /work a DVC repo (no git inside the
# container), wire the optional gdrive remote from env, then run dvc.
set -e

if [ ! -d .dvc ]; then
    dvc init --no-scm -q
fi

# Optional: DVC gdrive remote for pull/push, credentials from the .env file
# (see .env.example). `dvc repro` itself needs no remote.
if [ -n "$DVC_GDRIVE_URL" ]; then
    dvc remote add -f -d gdrive "$DVC_GDRIVE_URL" -q
    if [ -n "$GDRIVE_SERVICE_ACCOUNT_JSON_FILE" ]; then
        dvc remote modify gdrive gdrive_use_service_account true -q
        dvc remote modify gdrive gdrive_service_account_json_file_path \
            "$GDRIVE_SERVICE_ACCOUNT_JSON_FILE" -q
    fi
fi

exec dvc "$@"
