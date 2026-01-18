import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)

DATA_DIR = "../data/processed"
CONFIG_PATH = "../configs/experiment.yaml"
RESULTS_DIR = "../results/tables"


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

datasets = config["datasets"]
cv_folds = config["cv_folds"]
cv_repeats = config["cv_repeats"]

CLASSIFICATION_MODEL = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

REGRESSION_MODEL = LinearRegression()

results = []

for dataset_name, dataset_info in datasets.items():
    task = dataset_info["task"]

    print(f"\nRunning repeated CV for: {dataset_name} ({task})")

    X = pd.read_csv(DATA_DIR + f"/{dataset_name}_features.csv")
    y = pd.read_csv(DATA_DIR + f"/{dataset_name}_target.csv").squeeze()

    if task == "classification":
        cv = RepeatedStratifiedKFold(
            n_splits=cv_folds,
            n_repeats=cv_repeats,
            random_state=42
        )
        model = CLASSIFICATION_MODEL

    else:  
        cv = RepeatedKFold(
            n_splits=cv_folds,
            n_repeats=cv_repeats,
            random_state=42
        )
        model = REGRESSION_MODEL

    fold_idx = 0

    for train_idx, test_idx in cv.split(X, y):
        fold_idx += 1

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            results.append({
                "dataset": dataset_name,
                "task": task,
                "fold": fold_idx,
                "accuracy": acc,
                "f1": f1,
                "roc_auc": auc
            })

        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                "dataset": dataset_name,
                "task": task,
                "fold": fold_idx,
                "mse": mse,
                "r2": r2
            })


results_df = pd.DataFrame(results)
output_path = RESULTS_DIR + "/repeated_cv_results.csv"
results_df.to_csv(output_path, index=False)

print("\nRepeated cross-validation completed.")
print(f"Results saved to: {output_path}")
