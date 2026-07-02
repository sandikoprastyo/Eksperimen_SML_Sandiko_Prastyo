import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import os


DATA_PATH = 'fifa_dataset_preprocessing/fifa_data_clean.csv'
MODEL_DIR = 'model_artifacts'
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f'Dataset loaded: {df.shape}')

X = df.drop(columns=['goals', 'position_encoded'])
y_regresi = df['goals']
y_klasifikasi = df['position_encoded']

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_regresi, y_klasifikasi, test_size=0.2, random_state=42, stratify=y_klasifikasi
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

with mlflow.start_run(run_name='fifa_ci_training'):
    mlflow.log_param('model_type', 'RandomForestMultiOutput')
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('test_size', 0.2)

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_reg_train)
    y_reg_pred = reg_model.predict(X_test)

    mse = mean_squared_error(y_reg_test, y_reg_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)

    mlflow.log_metric('reg_mse', mse)
    mlflow.log_metric('reg_mae', mae)
    mlflow.log_metric('reg_r2', r2)
    print(f'Regression - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

    cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cls_model.fit(X_train, y_cls_train)
    y_cls_pred = cls_model.predict(X_test)

    accuracy = accuracy_score(y_cls_test, y_cls_pred)
    f1 = f1_score(y_cls_test, y_cls_pred, average='weighted')

    mlflow.log_metric('cls_accuracy', accuracy)
    mlflow.log_metric('cls_f1', f1)
    print(f'Classification - Accuracy: {accuracy:.4f}, F1: {f1:.4f}')

    mlflow.sklearn.log_model(reg_model, 'regressor_model')
    mlflow.sklearn.log_model(cls_model, 'classifier_model')

    run_id = mlflow.active_run().info.run_id
    print(f'\nRun ID: {run_id}')

    mlflow_model_path = os.path.join(MODEL_DIR, 'mlflow_run.txt')
    with open(mlflow_model_path, 'w') as f:
        f.write(f'run_id: {run_id}')
    print(f'Model artifacts saved to: {MODEL_DIR}')
