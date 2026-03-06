# Membangun Model Machine Learning

Folder ini berfokus pada tahapan pembuatan, pelatihan, dan pelacakan model (Model Tracking) menggunakan MLflow, yang terintegrasi dengan DagsHub.

## Struktur dan File Utama

- `modelling.py`: Script utama untuk melatih model Baseline (RandomForestClassifier) menggunakan data hasil preprocessing. Metrik akurasi dan parameter model secara otomatis dilacak menggunakan `mlflow.sklearn.autolog()`.
- `modelling_tuning.py`: Script untuk melakukan hyperparameter tuning (jika diperlukan) dan melacak metrik tiap eksperimen.
- `breast_cancer_preprocessing/`: Folder sumber data (`train.csv` dan `test.csv`) yang digunakan untuk melatih model.
- `mlflow.db` & `mlartifacts/`: Penyimpanan lokal untuk rekaman hasil eksperimen MLflow (sqlite backend).

## Cara Menjalankan MLflow Logging

Model dilacak menggunakan MLflow Tracking Server lokal yang terhubung ke DagsHub untuk Remote Tracking. Beberapa artifak yang disimpan:

- `confusion_matrix.png`
- `feature_importance.csv`
- `classification_report.txt`
