# Phase 5: Streamlit dashboard image — deliberately much leaner than the
# training image. app/app.py reads only precomputed artifacts (universe
# JSON + results/ CSVs); it never imports TensorFlow, scikit-learn, or
# statsmodels, so none of them are installed here. That is the entire
# reason training and app are separate images: one ships a ~multi-GB ML
# toolchain that runs occasionally, the other a ~400 MB web app that runs
# continuously. See DOCKER_SETUP.md.

FROM python:3.12-slim

WORKDIR /app

# Inference-only dependency set (streamlit / pandas / plotly).
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# Only what the dashboard reads. results/ is baked in as a seed so the app
# works standalone; when compose mounts the shared `artifacts` volume over
# /app/results, the training container's fresher outputs take precedence
# (a named volume is seeded from the image content on first use).
COPY app/app.py app/app.py
COPY config/universe.json config/universe.json
COPY results/ results/

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
