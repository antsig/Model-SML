import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import os

def main():
    import dagshub
    dagshub.init(repo_owner='antsig',
                 repo_name='Model-SML',
                 mlflow=True)

    mlflow.set_tracking_uri("https://dagshub.com/antsig/Model-SML.mlflow")
    try:
        train_df = pd.read_csv('iris_preprocessing/train.csv')
        test_df = pd.read_csv('iris_preprocessing/test.csv')
    except Exception as e:
        print("Data tidak ditemukan.")
        return

    X_train = train_df.drop('species', axis=1)
    y_train = train_df['species']
    X_test = test_df.drop('species', axis=1)
    y_test = test_df['species']

    mlflow.set_experiment("SML_Submission_Tuning")

    with mlflow.start_run(run_name="Tuning_RandomForest"):
        # 1. Hyperparameter Tuning
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }
        
        clf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(clf, param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # 2. Prediction & Metrics
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # 2.5 Manual Logging: Parameters and Model (replacing autolog)
        mlflow.log_params(grid_search.best_params_)
        mlflow.sklearn.log_model(best_model, "tuned_rf_model")
        
        # 3. Manual Logging: Metrik tambahan uji set dan Artefak
        mlflow.log_metric("test_accuracy", acc)
        
        # 4. Artefak Tambahan 1: Confusion Matrix Plot
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close() # Best practice: menutup plot agar tidak membebani memori
        
        # 5. Artefak Tambahan 2: Feature Importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        fi_path = "feature_importance.csv"
        feature_importance.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)
        
        # 6. Artefak Tambahan 3 & Manual Metric: Classification Report
        report_dict = classification_report(y_test, preds, output_dict=True)
        report_str = classification_report(y_test, preds)
        
        # Log beberapa metrik manual (Macro Avg) sebagai best practice 
        mlflow.log_metric("macro_f1_score", report_dict['macro avg']['f1-score'])
        mlflow.log_metric("macro_precision", report_dict['macro avg']['precision'])
        
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_path)
        
        print(f"Tuning selesai. Best Accuracy: {acc}")
        print(f"Best Params: {grid_search.best_params_}")

if __name__ == "__main__":
    main()
