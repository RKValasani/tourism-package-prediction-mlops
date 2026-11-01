import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# MLflow runs locally in GitHub Actions. The UI is started by the workflow.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("Tourism_Package_Prediction")

# Read train/test splits downloaded from the preceding workflow job.
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns").astype(int)
ytest = pd.read_csv("ytest.csv").squeeze("columns").astype(int)

numeric_features = Xtrain.select_dtypes(include=["number"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

classifier = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# A compact, explicit search space suitable for a CI workflow.
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample": [0.8],
    "classifier__colsample_bytree": [0.8]
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    refit=True,
    verbose=1
)

with mlflow.start_run(run_name="xgboost_grid_search") as parent_run:
    grid_search.fit(Xtrain, ytrain)

    # Log each tested parameter combination as a nested MLflow run.
    cv_results = grid_search.cv_results_
    for index, params in enumerate(cv_results["params"]):
        with mlflow.start_run(run_name=f"candidate_{index + 1}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", float(cv_results["mean_test_score"][index]))
            mlflow.log_metric("std_cv_f1", float(cv_results["std_test_score"][index]))

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", float(grid_search.best_score_))

    train_pred = best_model.predict(Xtrain)
    test_pred = best_model.predict(Xtest)
    test_probability = best_model.predict_proba(Xtest)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(ytrain, train_pred),
        "test_accuracy": accuracy_score(ytest, test_pred),
        "test_precision": precision_score(ytest, test_pred, zero_division=0),
        "test_recall": recall_score(ytest, test_pred, zero_division=0),
        "test_f1": f1_score(ytest, test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(ytest, test_probability)
    }
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

    print("Best parameters:", grid_search.best_params_)
    print("\nEvaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nClassification report:")
    print(classification_report(ytest, test_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(ytest, test_pred))

    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = (
        "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    )
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    with open("tourism_project/deployment/model_metadata.json", "w") as file:
        json.dump({
            "target": "ProdTaken",
            "features": Xtrain.columns.tolist(),
            "best_params": grid_search.best_params_,
            "metrics": {key: float(value) for key, value in metrics.items()}
        }, file, indent=2)

    mlflow.log_artifact(
        "tourism_project/deployment/model_metadata.json",
        artifact_path="model"
    )
    print(f"\nBest model saved to: {model_path}")
