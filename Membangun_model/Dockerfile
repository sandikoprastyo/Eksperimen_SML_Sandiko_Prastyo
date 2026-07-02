FROM python:3.12-slim

WORKDIR /opt/ml/model

COPY mlruns/0/e875be6f544a41d293523945d387b360/artifacts/regressor_model/ .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD mlflow models serve \
    --model-uri /opt/ml/model \
    --port 5000 \
    --host 0.0.0.0
