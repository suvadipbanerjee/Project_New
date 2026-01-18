import os
import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


DATA_DIR = "../data/processed"
RESULTS_DIR = "../results/tables"
CONFIG_PATH = "../configs/experiment.yaml"

os.makedirs(RESULTS_DIR, exist_ok=True)


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

TEST_SIZE = config["test_size"]
SEED = config["single_seed"]
DATASETS = config["datasets"]


CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "SVM": SVC(probability=True, random_state=SEED)
}

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=SEED),
    "SVR": SVR()
}


results = []

for dataset_name, dataset_cfg in DATASETS.items():
    task = dataset_cfg["task"]

    print(f"\nRunning single-seed for: {dataset_name} ({task})")

    X_path = os.path.join(DATA_DIR, f"{dataset_name}_features.csv")
    y_path = os.path.join(DATA_DIR, f"{dataset_name}_target.csv")

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    if task == "classification":
        models = CLASSIFICATION_MODELS

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                "dataset": dataset_name,
                "task": task,
                "model": model_name,
                "seed": SEED,
                "metric": "accuracy",
                "value": accuracy_score(y_test, y_pred)
            })

            results.append({
                "dataset": dataset_name,
                "task": task,
                "model": model_name,
                "seed": SEED,
                "metric": "f1",
                "value": f1_score(y_test, y_pred, average="binary")
            })

          
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                results.append({
                    "dataset": dataset_name,
                    "task": task,
                    "model": model_name,
                    "seed": SEED,
                    "metric": "roc_auc",
                    "value": roc_auc_score(y_test, y_prob)
                })

    else:  
        models = REGRESSION_MODELS

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                "dataset": dataset_name,
                "task": task,
                "model": model_name,
                "seed": SEED,
                "metric": "rmse",
                "value": np.sqrt(mean_squared_error(y_test, y_pred))
            })

            results.append({
                "dataset": dataset_name,
                "task": task,
                "model": model_name,
                "seed": SEED,
                "metric": "r2",
                "value": r2_score(y_test, y_pred)
            })


results_df = pd.DataFrame(results)
out_path = os.path.join(RESULTS_DIR, "single_seed.csv")
results_df.to_csv(out_path, index=False)

print(f"\n Single-seed results saved to {out_path}")
