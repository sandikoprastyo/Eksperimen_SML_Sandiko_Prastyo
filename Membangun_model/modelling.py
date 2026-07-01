import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow


DATA_PATH = 'fifa_dataset_preprocessing/fifa_data_clean.csv'

df = pd.read_csv(DATA_PATH)
print(f'Dataset loaded: {df.shape}')

X = df.drop(columns=['goals', 'position_encoded'])
y_regresi = df['goals']
y_klasifikasi = df['position_encoded']

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_regresi, y_klasifikasi, test_size=0.2, random_state=42, stratify=y_klasifikasi
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

mlflow.sklearn.autolog()

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)
print(f'Regression - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

mlflow.sklearn.autolog(disable=True)

with mlflow.start_run(run_name='fifa_classifier'):
    cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cls_model.fit(X_train, y_cls_train)
    y_cls_pred = cls_model.predict(X_test)

    accuracy = accuracy_score(y_cls_test, y_cls_pred)
    f1 = f1_score(y_cls_test, y_cls_pred, average='weighted')
    precision = precision_score(y_cls_test, y_cls_pred, average='weighted')
    recall = recall_score(y_cls_test, y_cls_pred, average='weighted')

    mlflow.log_param('model_type', 'RandomForestClassifier')
    mlflow.log_param('n_estimators', 100)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)

    mlflow.sklearn.log_model(cls_model, 'classifier_model',
                             input_example=X_test.iloc[:5])

    print(f'Classification - Accuracy: {accuracy:.4f}, F1: {f1:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')

    run_id = mlflow.active_run().info.run_id
    print(f'\nRun ID: {run_id}')
    print(f'MLflow UI: mlflow ui --host 0.0.0.0')
