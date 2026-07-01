# Submission SML — Sandiko Prastyo

Sistem Machine Learning untuk **Player Scouting** berbasis FIFA World Cup 2026 Player Performance Dataset. Proyek ini mencakup end-to-end MLOps: eksperimen dataset, pembangunan model, CI/CD pipeline, serta monitoring dan logging.

---

## Identitas

| Item | Nilai |
|------|-------|
| **Nama** | Sandiko Prastyo |
| **Username Dicoding** | sandiko prastyo |
| **Docker Hub** | sandikoprastyo |
| **Python** | 3.12.7 |
| **MLflow** | 2.19.0 |

---

## Dataset

**Sumber**: [Kaggle — FIFA World Cup 2026 Player Performance](https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset)

- **File**: `fifa_world_cup_2026_player_performance.csv`
- **Ukuran**: 16 MB | 54.600 baris | 75 kolom
- **Target Multi-output**:
  - **Regresi**: `goals` — memprediksi jumlah gol pemain
  - **Klasifikasi**: `position` — mengklasifikasikan posisi (Defender/Forward/Midfielder/Goalkeeper)

---

## Target Penilaian

| Kriteria | Target | Level |
|----------|--------|-------|
| Kriteria 1 — Eksperimen Dataset | 4 pts | **Advanced** |
| Kriteria 2 — Model Building | 4 pts | **Advanced** |
| Kriteria 3 — Workflow CI | 4 pts | **Advanced** |
| Kriteria 4 — Monitoring & Logging | 4 pts | **Advanced** |
| **Total** | **16/16** | **Bintang 5 (A)** |

---

## Kriteria 1 — Eksperimen Dataset (Advanced)

**Repo GitHub**: [Eksperimen_SML_Sandiko_Prastyo](https://github.com/sandikoprastyo/Eksperimen_SML_Sandiko_Prastyo)

### Notebook: `preprocessing/Eksperimen_Sandiko_Prastyo.ipynb`

| Section | Isi |
|---------|-----|
| 1. Perkenalan Dataset | Sumber, deskripsi, tujuan eksperimen |
| 2. Import Library | pandas, numpy, matplotlib, seaborn, scikit-learn |
| 3. Memuat Dataset | Load CSV, info(), describe(), head(), shape |
| 4. EDA | Distribusi goals/position/age, korelasi, boxplot, scatter |
| 5. Data Preprocessing | Drop kolom irelevan, encoding position, scaling fitur, imputasi |

### Script Otomatis: `preprocessing/automate_Sandiko_Prastyo.py`

Fungsi preprocessing end-to-end:
- Load dataset
- Drop kolom tidak relevan (player_id, nama, tanggal, stadion, dll)
- Label encoding untuk position
- Handling missing values (median imputation)
- Standard scaling untuk semua fitur numerik
- Output: `fifa_data_clean.csv` + artifacts (scaler, encoder, imputer)

### GitHub Actions: `.github/workflows/preprocessing.yml`

Trigger: **Push** ke branch `main` pada folder `fifa_raw/` atau `preprocessing/`

Workflow:
1. Checkout repository
2. Setup Python 3.12
3. Install dependencies
4. Jalankan preprocessing
5. Upload artifact hasil preprocessing
6. Commit hasil preprocessing ke repo

---

## Kriteria 2 — Model Building (Advanced)

### Basic — `modelling.py`

- MLflow autolog untuk RandomForestRegressor
- Manual logging untuk RandomForestClassifier dengan metrics (accuracy, f1, precision, recall)
- Log model ke MLflow

### Skilled — `modelling_tuning.py`

- Hyperparameter tuning dengan GridSearchCV
- Manual logging: params (n_estimators, max_depth, min_samples_split) dan metrics (mse, mae, r2, accuracy, f1)
- 3 artifacts tambahan:
  - `regressor_feature_importance.png`
  - `classifier_feature_importance.png`
  - `confusion_matrix_position.png`

### Advanced — `modelling_dagshub.py`

- Remote MLflow tracking via DagsHub
- 5 artifacts tambahan (di luar autolog):
  1. Feature importance regresi
  2. Feature importance klasifikasi
  3. Confusion matrix
  4. Residual plot
  5. Classification report (JSON)

**Hasil Model**:

| Metrik | Value |
|--------|-------|
| Regresi MSE | 0.0368 |
| Regresi MAE | 0.0600 |
| Regresi R2 | 0.4174 |
| Klasifikasi Accuracy | 0.9642 |
| Klasifikasi F1 | 0.9642 |

**DagsHub**: [sandikoprastyo/Membangun_Model_SML](https://dagshub.com/sandikoprastyo/Membangun_Model_SML)

---

## Kriteria 3 — Workflow CI (Advanced)

**Repo GitHub**: [Workflow-CI](https://github.com/sandikoprastyo/Workflow-CI)

### Struktur MLProject

```
MLProject/
├── modelling.py
├── conda.yaml
├── MLProject
└── fifa_dataset_preprocessing/
```

### GitHub Actions: `.github/workflows/ci.yml`

**Basic**: Trigger push → Setup Python → Install deps → Run training → Log MLflow
**Skilled**: Upload artifacts ke GitHub Releases
**Advanced**: Build Docker image via `mlflow build-docker` → Push ke Docker Hub

**Docker Image**: `sandikoprastyo/fifa-player-scout`

---

## Kriteria 4 — Monitoring & Logging (Advanced)

### Serving

Model disajikan menggunakan FastAPI pada port 8000 dengan endpoint:
- `GET /health` — health check
- `POST /predict` — prediksi multi-output
- `GET /metrics` — metrik Prometheus

### Prometheus (10+ Metrics)

| Metrik | Tipe | Deskripsi |
|--------|------|-----------|
| `prediction_latency_seconds` | Histogram | Latency prediksi |
| `prediction_requests_total` | Counter | Total request |
| `prediction_errors_total` | Counter | Total error |
| `prediction_value` | Gauge | Nilai prediksi goals |
| `model_confidence` | Gauge | Confidence model |
| `goals_mean` | Gauge | Mean prediksi goals |
| `goals_std` | Gauge | Std prediksi goals |
| `model_memory_usage_bytes` | Gauge | Memory usage |
| `request_size_bytes` | Summary | Ukuran request |
| `predictions_by_position_total` | Counter | Prediksi per posisi |

### Grafana Dashboard

- **Nama Dashboard**: `sandiko prastyo - FIFA Player Performance Monitor`
- 11 panel mencakup semua metrik di atas
- Data source: Prometheus (localhost:8001)

### Alerting (3 Rules)

| Rule | Kondisi | Severity |
|------|---------|----------|
| High Latency | `latency > 0.5s` selama 1 menit | Warning |
| High Error Rate | `error rate > 0.1/s` selama 1 menit | Critical |
| Prediction Anomaly | Deviasi > 2 std dari rata-rata 1 jam | Warning |

---

## Cara Menjalankan

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r Membangun_model/requirements.txt
```

### 2. Eksperimen Dataset

```bash
# Buka notebook
jupyter notebook Eksperimen_SML_Sandiko_Prastyo/preprocessing/Eksperimen_Sandiko_Prastyo.ipynb

# Atau jalankan script otomatis
python Eksperimen_SML_Sandiko_Prastyo/preprocessing/automate_Sandiko_Prastyo.py
```

### 3. Training Model

```bash
cd Membangun_model

# Basic (autolog)
python modelling.py

# Tuning
python modelling_tuning.py

# Advanced (DagsHub)
MLFLOW_TRACKING_USERNAME=sandikoprastyo MLFLOW_TRACKING_PASSWORD=<token> python modelling_dagshub.py
```

### 4. MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5001
# Buka http://localhost:5001
```

### 5. Serving & Monitoring

```bash
# Start Prometheus exporter (port 8001)
python "Monitoring dan Logging/3.prometheus_exporter.py"

# Start serving (port 8000)
python "Monitoring dan Logging/model_serving.py"

# Test inference
python "Monitoring dan Logging/7.Inference.py"

# Prometheus: http://localhost:8001/metrics
# Grafana: http://localhost:3000 (admin/admin)
# API: http://localhost:8000/docs
```

---

## Struktur Final Submission

```
SMSML_Sandiko_Prastyo.zip
├── Eksperimen_SML_Sandiko_Prastyo.txt      → https://github.com/sandikoprastyo/Eksperimen_SML_Sandiko_Prastyo
├── Membangun_model/
│   ├── modelling.py                         # Basic: MLflow autolog
│   ├── modelling_tuning.py                  # Skilled: tuning + artifacts
│   ├── modelling_dagshub.py                 # Advanced: DagsHub remote
│   ├── fifa_dataset_preprocessing/          # Dataset siap latih
│   ├── screenshoot_dashboard.jpg            # MLflow UI dashboard
│   ├── screenshoot_artifak.jpg              # MLflow artifacts
│   ├── requirements.txt
│   └── DagsHub.txt                          → https://dagshub.com/sandikoprastyo/Membangun_Model_SML
├── Workflow-CI.txt                          → https://github.com/sandikoprastyo/Workflow-CI
└── Monitoring dan Logging/
    ├── 1.bukti_serving/                     # Screenshot serving
    ├── 2.prometheus.yml                     # Konfigurasi Prometheus
    ├── 3.prometheus_exporter.py             # Exporter metrik
    ├── 4.bukti monitoring Prometheus/       # 10+ metrik Prometheus
    ├── 5.bukti monitoring Grafana/          # Dashboard Grafana
    ├── 6.bukti alerting Grafana/            # 3 rules + notifikasi
    └── 7.Inference.py                       # Client inference
```

---

## Catatan Penting

1. **DagsHub**: Token disimpan sebagai environment variable
2. **Docker Hub**: Credentials disimpan di GitHub Secrets (`DOCKER_USERNAME`, `DOCKER_PASSWORD`)
3. **Grafana Dashboard**: Nama dashboard harus `sandiko prastyo` (sesuai username Dicoding)
4. **Dataset**: Ukuran 16 MB — gunakan Git LFS jika diperlukan
