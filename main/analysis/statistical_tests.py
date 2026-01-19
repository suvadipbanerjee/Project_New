import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from pathlib import Path


BASE_DIR = Path("../results/tables")


def load_single_seed():
    df = pd.read_csv(BASE_DIR / "single_seed.csv")

    return df.rename(columns={"value": "score"})[
        ["dataset", "task", "model", "metric", "score"]
    ].assign(source="single_seed")


def load_multi_seed():
    df = pd.read_csv(BASE_DIR / "multi_seed_results.csv")

    return df.rename(columns={"mean": "score"})[
        ["dataset", "model", "metric", "score"]
    ].assign(source="multi_seed")


def load_repeated_cv_folds():
    df = pd.read_csv(BASE_DIR / "repeated_cv_results.csv")

    records = []
    metric_cols = ["mse", "r2", "accuracy", "f1", "roc_auc"]

    for _, row in df.iterrows():
        for m in metric_cols:
            if pd.notna(row.get(m)):
                records.append({
                    "dataset": row["dataset"],
                    "task": row["task"],
                    "model": row["model"],
                    "metric": m,
                    "fold": row["fold"],
                    "score": row[m],
                    "source": "repeated_cv"
                })

    return pd.DataFrame(records)


def load_repeated_cv_aggregate():
    df = pd.read_csv(BASE_DIR / "repeated_cv_aggregate.csv")

    return df.rename(columns={"mean": "score"})[
        ["dataset", "task", "model", "metric", "score"]
    ].assign(source="repeated_cv_mean")

def statistical_tests_repeated_cv(df):
    """
    Paired tests across repeated CV folds.
    """
    results = []

    for (dataset, task, model, metric), g in df.groupby(
        ["dataset", "task", "model", "metric"]
    ):
        scores = g.sort_values("fold")["score"].values

        if len(scores) < 2:
            continue  # cannot test

        # Compare against fold-wise mean baseline
        baseline = scores.mean()

        t_p = ttest_rel(scores, [baseline] * len(scores)).pvalue

        try:
            w_p = wilcoxon(scores - baseline).pvalue
        except ValueError:
            w_p = None

        results.append({
            "dataset": dataset,
            "task": task,
            "model": model,
            "metric": metric,
            "n_folds": len(scores),
            "ttest_p": t_p,
            "wilcoxon_p": w_p
        })

    return pd.DataFrame(results)


def descriptive_deltas(single, multi, cv_mean):
    merged = (
        single.merge(
            cv_mean,
            on=["dataset", "task", "model", "metric"],
            suffixes=("_single", "_cv"),
            how="inner"
        )
        .merge(
            multi,
            on=["dataset", "model", "metric"],
            how="left"
        )
    )

    merged["delta_single_vs_cv"] = (
        merged["score_single"] - merged["score_cv"]
    )

    merged["delta_multi_vs_cv"] = (
        merged["score"] - merged["score_cv"]
    )

    return merged[
        [
            "dataset", "task", "model", "metric",
            "score_single", "score_cv", "score",
            "delta_single_vs_cv", "delta_multi_vs_cv"
        ]
    ].rename(columns={"score": "score_multi"})

def main():
    single = load_single_seed()
    multi = load_multi_seed()
    cv_folds = load_repeated_cv_folds()
    cv_mean = load_repeated_cv_aggregate()
    stats = statistical_tests_repeated_cv(cv_folds)
    stats.to_csv("../results/tables/statistical_tests_repeated_cv.csv", index=False)

    deltas = descriptive_deltas(single, multi, cv_mean)
    deltas.to_csv("../results/tables/descriptive_comparison.csv", index=False)

    print(" Statistical tests saved to statistical_tests_repeated_cv.csv")
    print(" Descriptive deltas saved to descriptive_comparison.csv")


if __name__ == "__main__":
    main()
