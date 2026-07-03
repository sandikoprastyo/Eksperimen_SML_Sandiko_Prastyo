import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


DATA_PATH = 'fifa_world_cup_2026_player_performance.csv'
OUTPUT_DIR = 'fifa_dataset_preprocessing'

ID_COLS = [
    'player_id', 'player_name', 'match_id', 'match_date',
    'stadium', 'city', 'club_name', 'nationality', 'team', 'opponent_team'
]

POSITION_MAPPING = {'Defender': 0, 'Forward': 1, 'Goalkeeper': 2, 'Midfielder': 3}
STAGE_MAPPING = {
    'Group Stage': 0, 'Round of 32': 1, 'Round of 16': 2,
    'Quarter Finals': 3, 'Semi Finals': 4,
    'Third Place Match': 5, 'Final': 6
}
RESULT_MAPPING = {'L': 0, 'D': 1, 'W': 2}
FOOT_MAPPING = {'Left': 0, 'Right': 1}


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f'Dataset loaded: {df.shape}')
    return df


def drop_id_columns(df):
    existing = [c for c in ID_COLS if c in df.columns]
    df_processed = df.drop(columns=existing)
    print(f'Shape after dropping ID columns: {df_processed.shape}')
    return df_processed


def encode_categorical(df):
    label_encoder = LabelEncoder()
    df['position_encoded'] = label_encoder.fit_transform(df['position'])
    df = df.drop(columns=['position'])
    print(f'Position encoded using LabelEncoder: {label_encoder.classes_}')

    if 'tournament_stage' in df.columns:
        df['tournament_stage'] = df['tournament_stage'].map(STAGE_MAPPING)
    if 'match_result' in df.columns:
        df['match_result'] = df['match_result'].map(RESULT_MAPPING)
    if 'preferred_foot' in df.columns:
        df['preferred_foot'] = df['preferred_foot'].map(FOOT_MAPPING)

    print(f'Shape after encoding: {df.shape}')
    return df, label_encoder


def impute_missing(df):
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    print(f'Missing values after imputation: {df_imputed.isnull().sum().sum()}')
    return df_imputed, imputer


def scale_features(df):
    target_reg = 'goals'
    target_cls = 'position_encoded'
    feature_cols = [c for c in df.columns if c not in [target_reg, target_cls]]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f'Features scaled using StandardScaler: {len(feature_cols)} features')
    print(f'Final shape: {df.shape}')
    return df, scaler


def save_artifacts(df, scaler, label_encoder, imputer, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'fifa_data_clean.csv')
    df.to_csv(csv_path, index=False)
    print(f'Cleaned dataset saved to: {csv_path}')

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    joblib.dump(imputer, os.path.join(output_dir, 'imputer.pkl'))
    print(f'Artifacts saved to: {output_dir}/')


def run_preprocessing(data_path=DATA_PATH, output_dir=OUTPUT_DIR):
    print('=' * 50)
    print('FIFA Dataset Preprocessing Pipeline')
    print('=' * 50)

    df = load_data(data_path)
    df = drop_id_columns(df)
    df, label_encoder = encode_categorical(df)
    df, imputer = impute_missing(df)
    df, scaler = scale_features(df)
    save_artifacts(df, scaler, label_encoder, imputer, output_dir)

    print('\nPreprocessing complete!')
    print(f'Output directory: {os.path.abspath(output_dir)}')
    print(f'Files: fifa_data_clean.csv, scaler.pkl, label_encoder.pkl, imputer.pkl')

    return df


if __name__ == '__main__':
    run_preprocessing()
