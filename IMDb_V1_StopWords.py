#!/usr/bin/python3

"""
V1 - Stop words experiment.

Compare Count/TF-IDF vectorizers with and without English stop words,
using MultinomialNB and LinearSVC, across multiple training sizes.

Winners are selected by Macro F1, then Accuracy, then inference time.
"""

from IMDb_framework import (
    TRAIN_SIZES,
    TEST_FILE,
    add_stage_metadata,
    generate_baseline_configs,
    run_experiments,
    save_stage_csv,
    select_winners,
)

from utils import load

from plots import (
    plot_confusion_matrices,
    plot_best_by_size,
    plot_winning_configurations,
)


DISPLAY_COLUMNS = [
    "Experiment",
    "Train size",
    "Base Vectorizer",
    "StopWords",
    "Preprocessor",
    "Variant",
    "Vectorizer",
    "Classifier",
    "Accuracy",
    "Macro F1",
    "Train time (s)",
    "Inference time (s)",
]


def main():
    """
    V1 main entry point.
    """

    vectorizer_specs = [
        {"name": "Count"},
        {"name": "Count + StopWords"},
        {"name": "TF-IDF"},
        {"name": "TF-IDF + StopWords"},
    ]

    classifier_specs = [
        {"name": "MultinomialNB"},
        {"name": "LinearSVC"},
    ]

    configs = generate_baseline_configs(
        vectorizer_specs=vectorizer_specs,
        classifier_specs=classifier_specs,
        train_sizes=TRAIN_SIZES,
    )

    test_df = load(TEST_FILE)

    if test_df is None:
        return

    results_df = run_experiments(configs, test_df)

    if results_df is None:
        return

    results_df = add_stage_metadata(
        results_df,
        experiment_name="V1",
    )

    # ------------------------------------------------------------------
    # V1 official winners: one per (Train size, Base Vectorizer,
    # StopWords) combination -> 4 winners per training size.
    # ------------------------------------------------------------------

    comparison_winners_df = select_winners(
        results_df,
        group_columns=[
            "Train size",
            "Base Vectorizer",
            "StopWords",
        ],
    )

    print("\n")
    print("=" * 80)
    print("WINNERS OF THE EXPERIMENTAL COMPARISONS")
    print("=" * 80)

    print(
        comparison_winners_df[DISPLAY_COLUMNS].round({
            "Accuracy": 3,
            "Macro F1": 3,
            "Train time (s)": 4,
            "Inference time (s)": 4,
        })
    )

    # ------------------------------------------------------------------
    # Best overall result per training size
    # ------------------------------------------------------------------

    best_by_size_df = select_winners(
        results_df,
        group_columns=["Train size"],
    )

    print("\n")
    print("=" * 80)
    print("BEST RESULT BY TRAINING SIZE")
    print("=" * 80)

    print(
        best_by_size_df[DISPLAY_COLUMNS].round({
            "Accuracy": 3,
            "Macro F1": 3,
            "Train time (s)": 4,
            "Inference time (s)": 4,
        })
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    plot_confusion_matrices(comparison_winners_df)
    plot_best_by_size(best_by_size_df)
    plot_winning_configurations(results_df, best_by_size_df)

    # ------------------------------------------------------------------
    # CSV exports
    # ------------------------------------------------------------------

    save_stage_csv(
        results_df,
        columns=DISPLAY_COLUMNS,
        path="results_V1.csv",
    )

    save_stage_csv(
        comparison_winners_df,
        columns=DISPLAY_COLUMNS,
        path="winners_V1.csv",
    )


if __name__ == "__main__":
    main()
