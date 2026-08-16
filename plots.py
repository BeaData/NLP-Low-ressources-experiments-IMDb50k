#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrices(results):
    """
    Plot confusion matrices for each training set size.

    For each training size, the matrices are displayed
    two at a time in a one-row, two-column layout.

    This layout provides more space for datasets with
    many classes, such as DBpedia.
    """

    for dataset_size in sorted(
        results["Train size"].unique()
    ):

        subset = results[
            results["Train size"] == dataset_size
        ].reset_index(drop=True)

        # -------------------------------------------------
        # Display two confusion matrices per figure
        # -------------------------------------------------

        for start_index in range(0, len(subset), 2):

            current_subset = subset.iloc[
                start_index:start_index + 2
            ].reset_index(drop=True)

            fig = plt.figure(figsize=(16, 8.5))

            grid = fig.add_gridspec(
                1,
                3,
                width_ratios=[1, 0.42, 1],
            )

            left_ax = fig.add_subplot(grid[0, 0])
            legend_ax = fig.add_subplot(grid[0, 1])
            right_ax = fig.add_subplot(grid[0, 2])

            axes = [left_ax, right_ax]

            labels = {
                0: "Company",
                1: "Educational Institution",
                2: "Artist",
                3: "Athlete",
                4: "Office Holder",
                5: "Mean Of Transportation",
                6: "Building",
                7: "Natural Place",
                8: "Village",
                9: "Animal",
                10: "Plant",
                11: "Album",
                12: "Film",
                13: "Written Work",
            }

            legend_ax.axis("off")

            legend_text = "\n".join(
                [
                    f"{label}: {name}"
                    for label, name in labels.items()
                ]
            )

            legend_ax.text(
                0.5,
                0.5,
                legend_text,
                ha="center",
                va="center",
                fontsize=10,
                linespacing=1.5,
            )

            fig.suptitle(
                (f"Confusion Matrices - Training Size {dataset_size}"),
                fontsize=18,
                fontweight="bold",
            )

            for index, ax in enumerate(axes):

                if index >= len(current_subset):

                    ax.set_visible(False)
                    continue

                row = current_subset.iloc[index]

                ConfusionMatrixDisplay.from_predictions(
                    row["y_true"],
                    row["y_pred"],
                    ax=ax,
                    colorbar=False,
                    values_format="d",
                )

                title = (
                    f"{row['Vectorizer']}\n"
                    f"{row['Classifier']}"
                )

                if "Experiment" in row.index:

                    title += (f"\n({row['Experiment']})")

                ax.set_title(title, fontsize=14, pad=10)
                ax.tick_params(axis="both", labelsize=9)

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()


def plot_best_by_size(winners_df):
    """
    Plot the Macro F1 score of the best experiment
    at each training size.

    Different colors and marker shapes identify
    different winning configurations.

    Parameters
    ----------
    winners_df : pandas.DataFrame
        One winning experiment for each training size.
    """

    winners_df = (
        winners_df
        .sort_values(by="Train size")
        .reset_index(drop=True)
        .copy()
    )

    # Create one identifier for each winning configuration.
    winners_df["Configuration"] = (
        winners_df["Vectorizer"]
        + " + "
        + winners_df["Classifier"]
    )

    configurations = (
        winners_df["Configuration"]
        .unique()
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
        "P",
        "X",
        "v",
        "<",
        ">",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Connect the best scores in order of training size.
    ax.plot(
        winners_df["Train size"],
        winners_df["Macro F1"],
        linestyle="-",
        color="gray",
        linewidth=1.5,
        zorder=1,
    )

    # Plot each winning configuration with a different marker.
    for index, configuration in enumerate(configurations):

        configuration_data = winners_df[
            winners_df["Configuration"]
            == configuration
        ]

        ax.scatter(
            configuration_data["Train size"],
            configuration_data["Macro F1"],
            marker=markers[
                index % len(markers)
            ],
            s=100,
            label=configuration,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xticks(winners_df["Train size"])

    ax.set_xticklabels(
        winners_df["Train size"]
        .astype(str)
    )
    ax.set_xlabel("Training size")
    ax.set_ylabel("Macro F1")

    ax.set_title("Best Macro F1 Score by Training Size")

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    ax.legend(
        title="Winning configurations",
        loc="best",
    )

    plt.show()


def plot_winning_configurations(
    results_df,
    best_by_size_df,
):
    """
    Plot the performance trajectories of configurations
    that achieve the best overall result for at least one
    training set size.

    Each unique best-performing configuration is displayed
    once across all available training set sizes.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Complete results for all configurations and
        training set sizes.

    best_by_size_df : pandas.DataFrame
        Best overall result for each training set size.
    """

    config_cols = [
        col for col in [
            "Preprocessor",
            "Vectorizer",
            "Classifier",
        ]
        if col in best_by_size_df.columns
    ]

    unique_best = (
        best_by_size_df[config_cols]
        .drop_duplicates()
    )

    winning_results = results_df.merge(
        unique_best,
        on=config_cols,
        how="inner",
    )

    if winning_results.empty:
        print("No winning configurations to plot.")
        return

    if "Preprocessor" in winning_results.columns:
        winning_results["Configuration"] = (
            winning_results["Preprocessor"].fillna("None").astype(str)
            + " + "
            + winning_results["Vectorizer"].astype(str)
            + " + "
            + winning_results["Classifier"].astype(str)
        )
    else:
        winning_results["Configuration"] = (
            winning_results["Vectorizer"].astype(str)
            + " + "
            + winning_results["Classifier"].astype(str)
        )

    plt.figure(figsize=(11, 7))
    markers = ["o", "s", "^", "D", "v"]

    for index, (configuration, group) in enumerate(
        winning_results.groupby("Configuration")
    ):
        group = group.sort_values("Train size")

        plt.plot(
            group["Train size"],
            group["Macro F1"],
            marker=markers[index % len(markers)],
            linewidth=2,
            alpha=0.7,
            label=configuration,
        )

    plt.xscale("log")
    train_sizes = sorted(
        winning_results["Train size"].unique()
    )
    plt.xticks(train_sizes, train_sizes)
    plt.xlabel("Training set size")
    plt.ylabel("Macro F1")
    plt.title(
        "Performance Trajectories of "
        "Best-Performing Configurations"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(
        title="Configuration",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.show()


def plot_default_vs_gridsearch(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare the best Default and best GridSearchCV results
    for each training set size.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Complete V5 results containing both variants.

    Returns
    -------
    pandas.DataFrame
        Best Macro F1 per variant and per training size.
    """

    variants = ["Default", "GridSearchCV"]

    missing_variants = [
        variant
        for variant in variants
        if variant not in results_df["Variant"].values
    ]

    if missing_variants:
        print(
            f"Warning: missing variants: "
            f"{', '.join(missing_variants)}"
        )

    df = results_df[
        results_df["Variant"].isin(variants)
    ]

    if df.empty:
        print("No data to plot.")
        return pd.DataFrame()

    best_by_variant = (
        df.sort_values("Macro F1", ascending=False)
        .drop_duplicates(subset=["Train size", "Variant"])
        .pivot(
            index="Train size",
            columns="Variant",
            values="Macro F1",
        )
        .reindex(columns=variants)
        .sort_index()
    )

    best_by_variant["Best"] = best_by_variant.idxmax(axis=1)

    print()
    print("=" * 80)
    print("BEST MACRO F1: DEFAULT VS GRIDSEARCHCV")
    print("=" * 80)

    print(
        best_by_variant.round(4)
    )

    ax = best_by_variant.plot(
        kind="bar",
        figsize=(10, 5),
    )

    ax.set_title(
        "Best Macro F1 by Training Size: "
        "Default vs GridSearchCV"
    )
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Macro F1")
    ax.legend(title="Variant")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return best_by_variant


def plot_v6_confusion_matrices(results):
    """
    Plot confusion matrices separately for each V6 experiment.

    One figure is produced for each pair of confusion matrices.
    Matrices are displayed two at a time with a central legend
    containing the DBpedia class labels.

    Parameters
    ----------
    results : pandas.DataFrame
        Evaluation results containing:
        - Experiment
        - Train size
        - y_true
        - y_pred
    """

    for experiment in sorted(results["Experiment"].unique()):

        subset = results[
            results["Experiment"] == experiment
        ].sort_values("Train size").reset_index(drop=True)

        # -------------------------------------------------
        # Display two confusion matrices per figure
        # -------------------------------------------------

        for start_index in range(0, len(subset), 2):

            current_subset = subset.iloc[
                start_index:start_index + 2
            ].reset_index(drop=True)

            fig = plt.figure(figsize=(16, 8.5))

            grid = fig.add_gridspec(
                1,
                3,
                width_ratios=[1, 0.42, 1],
            )

            left_ax = fig.add_subplot(grid[0, 0])
            legend_ax = fig.add_subplot(grid[0, 1])
            right_ax = fig.add_subplot(grid[0, 2])

            axes = [left_ax, right_ax]

            # -------------------------------------------------
            # DBpedia class labels
            # -------------------------------------------------

            labels = {
                0: "Company",
                1: "Educational Institution",
                2: "Artist",
                3: "Athlete",
                4: "Office Holder",
                5: "Mean Of Transportation",
                6: "Building",
                7: "Natural Place",
                8: "Village",
                9: "Animal",
                10: "Plant",
                11: "Album",
                12: "Film",
                13: "Written Work",
            }

            legend_ax.axis(
                "off"
            )

            legend_text = "\n".join(
                [
                    f"{label}: {name}"
                    for label, name in labels.items()
                ]
            )

            legend_ax.text(
                0.5,
                0.5,
                legend_text,
                ha="center",
                va="center",
                fontsize=10,
                linespacing=1.5,
            )

            # -------------------------------------------------
            # Figure title
            # -------------------------------------------------

            fig.suptitle(
                (f"{experiment} - Confusion Matrices"),
                fontsize=18,
                fontweight="bold",
            )

            # -------------------------------------------------
            # Confusion matrices
            # -------------------------------------------------

            for index, ax in enumerate(axes):

                if index >= len(current_subset):

                    ax.set_visible(False)
                    continue

                row = current_subset.iloc[index]

                ConfusionMatrixDisplay.from_predictions(
                    row["y_true"],
                    row["y_pred"],
                    ax=ax,
                    colorbar=False,
                    values_format="d",
                )

                ax.set_title(
                    (f"Training size: {row['Train size']}"),
                    fontsize=14,
                    pad=10,
                )

                ax.tick_params(axis="both", labelsize=9)

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()


def plot_embedding_performance(results):
    """
    Plot Accuracy and Macro F1 evolution by training size
    for each embedding experiment.
    """

    if results.empty:
        print("No results available.")
        return

    required_columns = [
        "Experiment",
        "Train size",
        "Accuracy",
        "Macro F1",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        print(f"Missing columns: {', '.join(missing_columns)}")
        return

    plot_data = (results[required_columns].sort_values("Train size"))

    fig, ax = plt.subplots(figsize=(10, 6))

    for experiment in plot_data["Experiment"].unique():

        subset = plot_data[plot_data["Experiment"] == experiment]

        ax.plot(
            subset["Train size"],
            subset["Macro F1"],
            marker="o",
            linewidth=2,
            alpha=0.7,
            label=f"{experiment} Macro F1",
        )

    ax.set_title(
        "Embeddings Performance Evolution",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Training set size", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(plot_data["Train size"].unique())
    ax.set_xticklabels(plot_data["Train size"].unique())
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Experiment", framealpha=0.8)

    fig.tight_layout()
    plt.show()
