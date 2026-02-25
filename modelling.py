import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    try:
        train_df = pd.read_csv('iris_preprocessing/train.csv')
        test_df = pd.read_csv('iris_preprocessing/test.csv')
    except Exception as e:
        print("Data tidak ditemukan di folder 'iris_preprocessing'. Pastikan dataset sudah disiapkan dari kriteria sebelumnya.")
        return

    X_train = train_df.drop('species', axis=1)
    y_train = train_df['species']
    X_test = test_df.drop('species', axis=1)
    y_test = test_df['species']

    mlflow.set_experiment("SML_Submission_Baseline")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Baseline_RandomForest"):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Baseline Accuracy: {acc}")

if __name__ == "__main__":
    main()
