FROM python:3.12-slim

WORKDIR /opt/ml

RUN pip install --no-cache-dir mlflow==2.19.0 pandas numpy scikit-learn

RUN mkdir -p /opt/ml/model
COPY Membangun_model/mlruns /opt/ml/mlruns
ENV MLFLOW_TRACKING_URI=/opt/ml/mlruns

EXPOSE 5000

CMD mlflow models serve \
    --model-uri "runs:/e875be6f544a41d293523945d387b360/regressor_model" \
    --port 5000 \
    --host 0.0.0.0
