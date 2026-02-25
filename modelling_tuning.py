import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os

# Set DagsHub Tracking URI
import dagshub
dagshub.init(repo_owner='antsig',
             repo_name='Model-SML',
             mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/antsig/Model-SML.mlflow")

def main():
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

    # manual logging
    mlflow.autolog(disable=True)
    
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
        
        # 3. Manual Logging: Parameters & Metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        
        # 4. Manual Logging: Model
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        # 5. Artefak Tambahan 1: Confusion Matrix Plot
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # 6. Artefak Tambahan 2: Feature Importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        fi_path = "feature_importance.csv"
        feature_importance.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)
        
        print(f"Tuning selesai. Best Accuracy: {acc}")
        print(f"Best Params: {grid_search.best_params_}")

if __name__ == "__main__":
    main()
