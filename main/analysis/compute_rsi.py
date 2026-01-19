import pandas as pd

RESULTS_DIR = "../results/tables"
OUTPUT_PATH = RESULTS_DIR + "/rsi_scores.csv"


def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Relative Stability Index (RSI = std / |mean|)
    """

    required_cols = {"dataset", "model", "metric", "mean", "std"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, "
            f"found {list(df.columns)}"
        )

    rsi_df = df.copy()

    rsi_df["rsi"] = rsi_df["std"] / rsi_df["mean"].abs()

    cols = ["dataset", "model", "metric", "mean", "std", "rsi"]
    if "task" in rsi_df.columns:
        cols.insert(1, "task")

    return rsi_df[cols]


def main():
    input_path = RESULTS_DIR +"/multi_seed_results.csv"

    df = pd.read_csv(input_path)

    rsi_df = compute_rsi(df)

    rsi_df = rsi_df.sort_values(
        by=["dataset", "metric", "rsi"],
        ascending=[True, True, True]
    )

    rsi_df.to_csv(OUTPUT_PATH, index=False)

    print(" RSI computation complete")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nSample:")
    print(rsi_df.head())


if __name__ == "__main__":
    main()

