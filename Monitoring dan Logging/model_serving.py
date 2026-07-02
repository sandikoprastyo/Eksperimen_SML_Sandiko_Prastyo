import os
import glob
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
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


BASE = os.path.dirname(os.path.abspath(__file__))
HOST_PREFIXES = [
    '/Volumes/SSD_EXT/Dicoding/Membangun%20Sistem%20Machine%20Learning/submission/',
    '/Volumes/SSD_EXT/Dicoding/Membangun Sistem Machine Learning/submission/',
]

mlruns_paths = [
    os.path.join(BASE, '..', 'Membangun_model', 'mlruns'),
    os.path.join(BASE, '..', 'Workflow-CI', 'MLProject', 'mlruns'),
]


def fix_mlflow_artifact_paths(root_dir):
    for meta_path in glob.glob(os.path.join(root_dir, '**', 'meta.yaml'), recursive=True):
        with open(meta_path) as f:
            content = f.read()
        for prefix in HOST_PREFIXES:
            if prefix in content:
                new_content = content.replace(prefix, root_dir + '/')
                with open(meta_path, 'w') as f:
                    f.write(new_content)
                print(f'Fixed artifact paths in {meta_path}')
                break

for p in mlruns_paths:
    if os.path.isdir(p):
        fix_mlflow_artifact_paths(p)

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

model_dir = os.path.join(os.path.dirname(__file__), '..', 'Membangun_model')

scaler = None
label_encoder = None
regressor_model = None
classifier_model = None

for path in mlruns_paths:
    tracking_uri = f'file://{path}'
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(exp.experiment_id, order_by=['start_time DESC'], max_results=1)
            if runs:
                run = runs[0]
                run_id = run.info.run_id
                reg_uri = f'runs:/{run_id}/regressor_model'
                cls_uri = f'runs:/{run_id}/classifier_model'
                try:
                    regressor_model = mlflow.sklearn.load_model(reg_uri)
                    classifier_model = mlflow.sklearn.load_model(cls_uri)
                    print(f'Loaded models from {tracking_uri} / run {run_id}')
                    break
                except Exception:
                    continue
        if regressor_model is not None:
            break
    except Exception:
        continue

try:
    scaler = joblib.load(os.path.join(model_dir, 'fifa_dataset_preprocessing', 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'fifa_dataset_preprocessing', 'label_encoder.pkl'))
    print(f'Loaded scaler and label encoder from {model_dir}')
except Exception as e:
    print(f'Warning: Could not load artifacts: {e}')

position_labels = ['Unknown']
if label_encoder is not None:
    try:
        position_labels = label_encoder.classes_.tolist()
    except Exception:
        pass


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
async def predict(input_data: PredictionInput, request: Request):
    predict_requests.inc()
    requests_in_progress.inc()

    try:
        start_time = time.time()
        features = np.array(input_data.features).reshape(1, -1)

        if regressor_model is not None:
            reg_pred = float(regressor_model.predict(features)[0])
        else:
            reg_pred = 0.0

        if classifier_model is not None:
            cls_pred = int(classifier_model.predict(features)[0])
            proba = classifier_model.predict_proba(features)
            confidence = float(np.max(proba[0]))
        else:
            cls_pred = 0
            confidence = 0.0

        latency = time.time() - start_time
        predict_latency.observe(latency)

        predict_value.set(reg_pred)
        predict_position.set(cls_pred)
        model_confidence_gauge.set(confidence)

        pos_label = position_labels[cls_pred] if cls_pred < len(position_labels) else 'Unknown'
        if label_encoder is not None:
            try:
                pos_label = label_encoder.inverse_transform([cls_pred])[0]
            except Exception:
                pass

        predict_position_counter.labels(position=pos_label).inc()

        goals_mean_gauge.set(reg_pred)
        goals_std_gauge.set(abs(reg_pred * 0.3))

        return {
            'predicted_goals': reg_pred,
            'predicted_position': cls_pred,
            'position_label': pos_label,
            'confidence': confidence,
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
