import pandas as pd

df = pd.read_csv("../results/tables/repeated_cv_results.csv")

metric_cols = ["accuracy", "f1", "roc_auc", "mse", "r2"]

agg = (
    df
    .melt(
        id_vars=["dataset", "task", "model"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="score"
    )
    .dropna()
    .groupby(["dataset", "task", "model", "metric"], as_index=False)
    .agg(
        mean=("score", "mean"),
        std=("score", "std")
    )
)

agg.to_csv("../results/tables/repeated_cv_aggregate.csv", index=False)
print("Saved repeated_cv_aggregate.csv")
