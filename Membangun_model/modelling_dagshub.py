import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import dagshub
import os
import json


DATA_PATH = 'fifa_dataset_preprocessing/fifa_data_clean.csv'
ARTIFACT_DIR = 'artifacts_dagshub'
os.makedirs(ARTIFACT_DIR, exist_ok=True)


dagshub.init(repo_owner='sandikoprastyo', repo_name='Membangun_Model_SML', mlflow=True)

df = pd.read_csv(DATA_PATH)
print(f'Dataset loaded: {df.shape}')

X = df.drop(columns=['goals', 'position_encoded'])
y_regresi = df['goals']
y_klasifikasi = df['position_encoded']

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_regresi, y_klasifikasi, test_size=0.2, random_state=42, stratify=y_klasifikasi
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

position_labels = ['Defender', 'Forward', 'Goalkeeper', 'Midfielder']


def plot_feature_importance(model, feature_names, title, filename):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(title)
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, filename))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Position Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, filename))
    plt.close()


def plot_residuals(y_true, y_pred, filename):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Goals')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', color='lightgreen')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, filename))
    plt.close()


mlflow.sklearn.autolog(disable=True)

with mlflow.start_run(run_name='fifa_dagshub_advanced') as run:
    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    mlflow.log_param('model_type', 'RandomForestMultiOutput')
    mlflow.log_param('param_grid', str(param_grid))
    mlflow.log_param('test_size', 0.2)
    mlflow.log_param('random_state', 42)
    mlflow.log_param('dataset_shape', str(df.shape))
    mlflow.log_param('n_features', X.shape[1])
    mlflow.set_tag('model_type', 'multi_output_regression_classification')
    mlflow.set_tag('dataset', 'FIFA World Cup 2026')

    print('=== Training Regression Model ===')
    reg_base = RandomForestRegressor(random_state=42)
    reg_grid = GridSearchCV(reg_base, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    reg_grid.fit(X_train, y_reg_train)
    best_reg = reg_grid.best_estimator_

    mlflow.log_param('reg_best_n_estimators', reg_grid.best_params_['n_estimators'])
    mlflow.log_param('reg_best_max_depth', reg_grid.best_params_['max_depth'])

    y_reg_pred = best_reg.predict(X_test)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)

    mlflow.log_metric('reg_mse', mse)
    mlflow.log_metric('reg_mae', mae)
    mlflow.log_metric('reg_r2', r2)
    print(f'Regression - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

    print('\n=== Training Classification Model ===')
    cls_base = RandomForestClassifier(random_state=42)
    cls_grid = GridSearchCV(cls_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    cls_grid.fit(X_train, y_cls_train)
    best_cls = cls_grid.best_estimator_

    mlflow.log_param('cls_best_n_estimators', cls_grid.best_params_['n_estimators'])
    mlflow.log_param('cls_best_max_depth', cls_grid.best_params_['max_depth'])

    y_cls_pred = best_cls.predict(X_test)
    accuracy = accuracy_score(y_cls_test, y_cls_pred)
    f1 = f1_score(y_cls_test, y_cls_pred, average='weighted')
    precision = precision_score(y_cls_test, y_cls_pred, average='weighted')
    recall = recall_score(y_cls_test, y_cls_pred, average='weighted')

    mlflow.log_metric('cls_accuracy', accuracy)
    mlflow.log_metric('cls_f1', f1)
    mlflow.log_metric('cls_precision', precision)
    mlflow.log_metric('cls_recall', recall)
    print(f'Classification - Accuracy: {accuracy:.4f}, F1: {f1:.4f}')

    class_report = classification_report(y_cls_test, y_cls_pred,
                                          target_names=position_labels, output_dict=True)
    report_path = os.path.join(ARTIFACT_DIR, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(class_report, f, indent=2)

    plot_feature_importance(best_reg, X.columns,
                            'Top 15 Feature Importance - Goals Regression',
                            'regressor_feature_importance.png')
    plot_feature_importance(best_cls, X.columns,
                            'Top 15 Feature Importance - Position Classification',
                            'classifier_feature_importance.png')
    plot_confusion_matrix(y_cls_test, y_cls_pred, position_labels,
                          'confusion_matrix_position.png')

    plot_residuals(y_reg_test, y_reg_pred, 'residual_plot.png')

    for f in os.listdir(ARTIFACT_DIR):
        fpath = os.path.join(ARTIFACT_DIR, f)
        if os.path.isfile(fpath):
            mlflow.log_artifact(fpath)

    mlflow.sklearn.log_model(best_reg, 'regressor_model',
                             input_example=X_test.iloc[:5])
    mlflow.sklearn.log_model(best_cls, 'classifier_model',
                             input_example=X_test.iloc[:5])

    run_id = run.info.run_id
    dagshub_url = f'https://dagshub.com/sandikoprastyo/Membangun_Model_SML/experiments/#/runs/{run_id}'
    print(f'\nDagsHub URL: {dagshub_url}')
    print(f'Run ID: {run_id}')
