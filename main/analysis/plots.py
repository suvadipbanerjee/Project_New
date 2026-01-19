import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(BASE_DIR, "../results/tables")
FIG_DIR = os.path.join(BASE_DIR, "../results/figures")

os.makedirs(FIG_DIR, exist_ok=True)
sns.set(style="whitegrid", context="paper")


def plot_evaluation_bias():
    df = pd.read_csv(os.path.join(TABLE_DIR, "descriptive_comparison.csv"))

    for metric in df["metric"].unique():
        plt.figure(figsize=(7, 4))

        sub = df[df["metric"] == metric]

        for model in sub["model"].unique():
            m = sub[sub["model"] == model]

            plt.plot(
                ["Single seed", "Cross-validation", "Repeated CV"],
                [
                    m["score_single"].mean(),
                    m["score_cv"].mean(),
                    m["score_multi"].mean(),
                ],
                marker="o",
                linewidth=2,
                label=model
            )

        plt.title(f"Effect of evaluation protocol on {metric}")
        plt.ylabel(metric)
        plt.xlabel("Evaluation protocol")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(FIG_DIR, f"evaluation_bias_{metric}.png"),
            dpi=300
        )
        plt.close()

def plot_statistical_significance():
    df = pd.read_csv(
        os.path.join(TABLE_DIR, "statistical_tests_repeated_cv.csv")
    )

    for test_col in ["ttest_p", "wilcoxon_p"]:
        pivot = df.pivot_table(
            index="model",
            columns="dataset",
            values=test_col
        )

        plt.figure(figsize=(8, 4))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2g",
            cmap="coolwarm_r",
            center=0.05,
            cbar_kws={"label": "p-value"}
        )

        plt.title(f"Repeated CV statistical test: {test_col}")
        plt.tight_layout()

        plt.savefig(
            os.path.join(FIG_DIR, f"{test_col}_heatmap.png"),
            dpi=300
        )
        plt.close()

def plot_rsi_tradeoff():
    df = pd.read_csv(os.path.join(TABLE_DIR, "rsi_scores.csv"))

    plt.figure(figsize=(6, 4))

    sns.scatterplot(
        data=df,
        x="mean",
        y="rsi",
        hue="model",
        style="metric",
        s=90
    )

    plt.xlabel("Mean performance")
    plt.ylabel("RSI (lower = more stable)")
    plt.title("Performance vs reproducibility trade-off")
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIG_DIR, "performance_vs_rsi.png"),
        dpi=300
    )
    plt.close()

def main():
    print("Generating publication-quality figures...")
    plot_evaluation_bias()
    plot_statistical_significance()
    plot_rsi_tradeoff()
    print("Done. Figures saved to results/figures/")


if __name__ == "__main__":
    main()

