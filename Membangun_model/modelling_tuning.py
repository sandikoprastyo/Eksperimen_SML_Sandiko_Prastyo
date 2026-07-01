import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import mlflow
import mlflow.sklearn
import os


DATA_PATH = 'fifa_dataset_preprocessing/fifa_data_clean.csv'
ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)

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
    print(f'Saved: {filename}')


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
    print(f'Saved: {filename}')


mlflow.sklearn.autolog(disable=True)

with mlflow.start_run(run_name='fifa_multi_output_tuning'):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }

    mlflow.log_param('model_type', 'RandomForestMultiOutput')
    mlflow.log_param('param_grid', str(param_grid))
    mlflow.log_param('test_size', 0.2)
    mlflow.log_param('random_state', 42)

    print('=== Regression: RandomForestRegressor + GridSearch ===')
    reg_base = RandomForestRegressor(random_state=42)
    reg_grid = GridSearchCV(reg_base, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    reg_grid.fit(X_train, y_reg_train)
    best_reg = reg_grid.best_estimator_

    for param, value in reg_grid.best_params_.items():
        mlflow.log_param(f'reg_{param}', value)
    mlflow.log_metric('reg_best_cv_score', -reg_grid.best_score_)

    y_reg_pred = best_reg.predict(X_test)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)

    mlflow.log_metric('reg_mse', mse)
    mlflow.log_metric('reg_mae', mae)
    mlflow.log_metric('reg_r2', r2)
    print(f'Regression - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

    print('\n=== Classification: RandomForestClassifier + GridSearch ===')
    cls_base = RandomForestClassifier(random_state=42)
    cls_grid = GridSearchCV(cls_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    cls_grid.fit(X_train, y_cls_train)
    best_cls = cls_grid.best_estimator_

    for param, value in cls_grid.best_params_.items():
        mlflow.log_param(f'cls_{param}', value)
    mlflow.log_metric('cls_best_cv_score', cls_grid.best_score_)

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
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')

    plot_feature_importance(best_reg, X.columns,
                            'Top 15 Feature Importance - Goals Regression',
                            'regressor_feature_importance.png')
    mlflow.log_artifact(os.path.join(ARTIFACT_DIR, 'regressor_feature_importance.png'))

    plot_feature_importance(best_cls, X.columns,
                            'Top 15 Feature Importance - Position Classification',
                            'classifier_feature_importance.png')
    mlflow.log_artifact(os.path.join(ARTIFACT_DIR, 'classifier_feature_importance.png'))

    plot_confusion_matrix(y_cls_test, y_cls_pred, position_labels,
                          'confusion_matrix_position.png')
    mlflow.log_artifact(os.path.join(ARTIFACT_DIR, 'confusion_matrix_position.png'))

    mlflow.sklearn.log_model(best_reg, 'regressor_model',
                             input_example=X_test.iloc[:5])
    mlflow.sklearn.log_model(best_cls, 'classifier_model',
                             input_example=X_test.iloc[:5])

    print(f'\nMLflow Tracking UI: http://localhost:5000')
    print(f'Run ID: {mlflow.active_run().info.run_id}')

mlflow.end_run()
