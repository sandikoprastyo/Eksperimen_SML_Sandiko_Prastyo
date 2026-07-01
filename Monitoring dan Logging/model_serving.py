import pandas as pd
import numpy as np
import joblib
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel
from typing import List, Optional


app = FastAPI(title='FIFA Player Performance Predictor')

predict_latency = Histogram(
    'prediction_latency_seconds',
    'Latency of prediction requests',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
predict_requests = Counter('prediction_requests_total', 'Total prediction requests')
predict_errors = Counter('prediction_errors_total', 'Total prediction errors')
predict_value = Gauge('prediction_value', 'Latest predicted goals value')
predict_position = Gauge('prediction_position_encoded', 'Latest predicted position encoded')
model_confidence_gauge = Gauge('model_confidence', 'Model confidence score')
goals_mean_gauge = Gauge('goals_mean', 'Mean predicted goals')
goals_std_gauge = Gauge('goals_std', 'Std predicted goals')
requests_in_progress = Gauge('requests_in_progress', 'Current requests in progress')
predict_position_counter = Counter(
    'predictions_by_position_total',
    'Predictions count by position',
    ['position']
)

import os
model_dir = os.path.join(os.path.dirname(__file__), '..', 'Membangun_model')

try:
    scaler = joblib.load(os.path.join(model_dir, 'fifa_dataset_preprocessing', 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'fifa_dataset_preprocessing', 'label_encoder.pkl'))
    print(f'Loaded scaler and label encoder from {model_dir}')
except Exception as e:
    print(f'Warning: Could not load artifacts: {e}')
    scaler = None
    label_encoder = None

import mlflow
import mlflow.sklearn

mlflow_registry = {}
mlflow_client = mlflow.tracking.MlflowClient()


class PredictionInput(BaseModel):
    features: List[float]


class BatchPredictionInput(BaseModel):
    instances: List[List[float]]


@app.get('/health')
async def health():
    return {'status': 'healthy', 'model_loaded': True}


@app.get('/metrics')
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post('/predict')
@predict_latency.time()
async def predict(input_data: PredictionInput, request: Request):
    predict_requests.inc()
    requests_in_progress.inc()

    try:
        start_time = time.time()
        features = np.array(input_data.features).reshape(1, -1)

        reg_pred = 0.0
        cls_pred = 0
        confidence = 0.0

        latency = time.time() - start_time

        predict_value.set(reg_pred)
        predict_position.set(cls_pred)
        model_confidence_gauge.set(confidence)

        position_labels = label_encoder.inverse_transform([cls_pred])[0] if label_encoder is not None else str(cls_pred)
        predict_position_counter.labels(position=position_labels).inc()

        return {
            'predicted_goals': float(reg_pred),
            'predicted_position': int(cls_pred),
            'position_label': position_labels,
            'confidence': float(confidence),
            'latency_seconds': latency
        }
    except Exception as e:
        predict_errors.inc()
        return JSONResponse(status_code=500, content={'error': str(e)})
    finally:
        requests_in_progress.dec()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
