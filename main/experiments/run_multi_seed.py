import os
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

with open("../configs/experiment.yaml", "r") as f:
    config = yaml.safe_load(f)

DATASETS = config["datasets"]
TEST_SIZE = config["test_size"]
N_SEEDS = config["multi_seeds"]

SEEDS = list(range(N_SEEDS))

PROCESSED_DIR = "../data/processed/"
RESULTS_DIR = "../results/tables/"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RESULTS_DIR, "multi_seed_results.csv")

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000)
}

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0)
}

results = []

for dataset_name, ds_cfg in DATASETS.items():
    task = ds_cfg["task"]
    print(f"\nRunning multi-seed: {dataset_name} ({task})")

    X_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_features.csv")
    y_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_target.csv")

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()

    models = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS

    for model_name, model in models.items():
        seed_metrics = []

        for seed in SEEDS:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=seed,
                stratify=y if task == "classification" else None
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task == "classification":
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                seed_metrics.append([acc, f1])

            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                seed_metrics.append([rmse, mae, r2])

        seed_metrics = np.array(seed_metrics)

        mean_vals = seed_metrics.mean(axis=0)
        std_vals = seed_metrics.std(axis=0)

        if task == "classification":
            results.extend([
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "metric": "accuracy",
                    "mean": mean_vals[0],
                    "std": std_vals[0]
                },
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "metric": "f1",
                    "mean": mean_vals[1],
                    "std": std_vals[1]
                }
            ])
        else:
            results.extend([
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "metric": "rmse",
                    "mean": mean_vals[0],
                    "std": std_vals[0]
                },
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "metric": "mae",
                    "mean": mean_vals[1],
                    "std": std_vals[1]
                },
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "metric": "r2",
                    "mean": mean_vals[2],
                    "std": std_vals[2]
                }
            ])

df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)

print("\nMulti-seed experiments completed.")
print(f"Saved to: {OUTPUT_FILE}")
