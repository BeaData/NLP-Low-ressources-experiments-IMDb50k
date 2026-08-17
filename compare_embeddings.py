#!/usr/bin/python3

"""
Compare V6 and V7 embedding experiments.

Experiments:
    V6.1 - Sentence Embeddings
    V6.2 - Sentence Embeddings + normalization
    V6.3 - Centroid classifier
    V7.1 - GloVe 100d + Mean Pooling
    V7.2 - GloVe 100d + TF-IDF Weighted Mean
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    "V6.1",
    "V6.2",
    "V6.3",
    "V7.1",
    "V7.2",
]


REQUIRED_COLUMNS = [
    "Experiment",
    "Train size",
    "Accuracy",
    "Macro F1",
]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_embedding_results(dataset_directory):
    """
    Load V6 and V7 embedding results from one dataset directory.

    Parameters
    ----------
    dataset_directory : str or Path
        Dataset directory containing the embedding result CSV files.

    Returns
    -------
    pandas.DataFrame
        Combined results.
    """

    dataset_directory = Path(dataset_directory)

    dataframes = []

    for experiment in EXPERIMENTS:

        filepath = (
            dataset_directory
            / f"results_{experiment.replace('.', '_')}_embeddings.csv"
        )

        if not filepath.exists():

            print(f"Missing file: {filepath}")
            continue

        print(f"Loading: {filepath}")

        dataframe = pd.read_csv(filepath)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:

            print(
                f"Missing columns in {filepath}: "
                f"{', '.join(missing_columns)}"
            )

            continue

        # The CSV already contains the correct experiment name.
        dataframes.append(
            dataframe[REQUIRED_COLUMNS].copy()
        )

    if not dataframes:

        return pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )

    return pd.concat(
        dataframes,
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

def display_results(results):
    """
    Display combined embedding results.
    """

    display_columns = [
        "Experiment",
        "Train size",
        "Accuracy",
        "Macro F1",
    ]

    print()
    print("=" * 80)
    print("COMBINED RESULTS")
    print("=" * 80)

    print(
        results[display_columns]
        .sort_values(
            [
                "Experiment",
                "Train size",
            ]
        )
        .round(4)
        .to_string(index=False)
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_embedding_comparison(
    results,
    dataset_name,
    dataset_directory,
):
    """
    Plot Macro F1 evolution for V6 and V7 experiments.

    One curve is produced for each experiment.
    """

    if results.empty:

        print("No results available.")
        return

    # -------------------------------------------------
    # Prepare plotting data
    # -------------------------------------------------

    # Y_AXIS_MIN = 0.5
    Y_AXIS_MIN = 0.6
    Y_AXIS_MAX = 1.0

    plot_data = (
        results[
            [
                "Experiment",
                "Train size",
                "Macro F1",
            ]
        ]
        .sort_values(
            [
                "Experiment",
                "Train size",
            ]
        )
    )

    # -------------------------------------------------
    # Create figure
    # -------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    # -------------------------------------------------
    # Plot one curve per experiment
    # -------------------------------------------------

    for experiment in EXPERIMENTS:

        subset = plot_data[
            plot_data["Experiment"] == experiment
        ]

        if subset.empty:
            continue

        ax.plot(
            subset["Train size"].to_numpy(),
            subset["Macro F1"].to_numpy(),
            marker="o",
            linewidth=2,
            alpha=0.8,
            label=experiment,
        )

    # -------------------------------------------------
    # Axes
    # -------------------------------------------------

    ax.set_title(
        f"{dataset_name} - Embeddings Performance Evolution",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Training set size",
        fontsize=12,
    )

    ax.set_ylabel(
        "Macro F1",
        fontsize=12,
    )

    ax.set_xscale("log")

    train_sizes = sorted(
        plot_data["Train size"].unique()
    )

    ax.set_xticks(train_sizes)

    ax.set_xticklabels(
        train_sizes
    )

    # ax.set_ylim(
    #     0,
    #     1,
    # )
    ax.set_ylim(
        Y_AXIS_MIN,
        Y_AXIS_MAX,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    # -------------------------------------------------
    # Legend
    # -------------------------------------------------

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    if handles:

        ax.legend(
            handles,
            labels,
            title="Experiment",
            framealpha=0.8,
        )

    else:

        print(
            "Warning: no curves were created for the plot."
        )

    # -------------------------------------------------
    # Save figure
    # -------------------------------------------------

    fig.tight_layout()

    figure_path = (
        dataset_directory
        / "embedding_methods_comparison.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    print()
    print(f"Saved: {figure_path}")

    plt.show()

    plt.close(fig)


def save_results_table(results, dataset_directory):
    """
    Save the combined embedding results as CSV and HTML.
    """

    display_columns = [
        "Experiment",
        "Train size",
        "Accuracy",
        "Macro F1",
    ]

    display_data = (
        results[display_columns]
        .sort_values(
            [
                "Experiment",
                "Train size",
            ]
        )
        .reset_index(drop=True)
    )

    csv_path = (
        dataset_directory
        / "results_embeddings_comparison.csv"
    )

    html_path = (
        dataset_directory
        / "results_embeddings_comparison.html"
    )

    display_data.to_csv(
        csv_path,
        index=False,
    )

    display_data.to_html(
        html_path,
        index=False,
    )

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {html_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python compare_embeddings.py <dataset_directory>"
        )

        return

    dataset_directory = Path(
        sys.argv[1]
    )

    if not dataset_directory.exists():

        print(
            f"Directory not found: "
            f"{dataset_directory}"
        )

        return

    dataset_name = (
        dataset_directory.name
    )

    print()
    print("=" * 80)
    print(
        f"EMBEDDING COMPARISON - "
        f"{dataset_name}"
    )
    print("=" * 80)

    # -------------------------------------------------
    # Load
    # -------------------------------------------------

    results = load_embedding_results(
        dataset_directory
    )

    if results.empty:

        print("No embedding results found.")
        return

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    save_results_table(
            results,
            dataset_directory,
        )

    # -------------------------------------------------
    # Check experiments
    # -------------------------------------------------

    print()
    print(
        "Experiments found:",
        results["Experiment"].unique()
    )

    # -------------------------------------------------
    # Display
    # -------------------------------------------------

    display_results(
        results
    )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    plot_embedding_comparison(
        results,
        dataset_name,
        dataset_directory,
    )


if __name__ == "__main__":
    main()
