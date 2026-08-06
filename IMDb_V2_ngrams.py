#!/usr/bin/python3

"""
V2 - N-gram experiment.

For every unique winning configuration of V1, compare:
- Unigrams (baseline)
- Bigrams (challenger)

Each unique configuration is evaluated on every training size
so that performance trajectories are complete.

The official winners CSV keeps only the winners of the duels
defined line by line by winners_V1.csv.
"""

import pandas as pd

from IMDb_framework import (
    TRAIN_SIZES,
    TEST_FILE,
    add_stage_metadata,
    generate_challenge_configs,
    make_experiment,
    run_experiments,
    save_stage_csv,
    select_duel_winners,
    select_winners,
)

from utils import load, load_winners

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


def create_ngram_challenges(row, train_size):
    base_vectorizer = row["Vectorizer"]
    classifier = row["Classifier"]
    preprocessor = row.get("Preprocessor", "None")

    if preprocessor is None or (
        isinstance(preprocessor, float) and pd.isna(preprocessor)
    ):
        preprocessor = "None"

    return [
        make_experiment(
            train_size=train_size,
            vectorizer_name=base_vectorizer,
            classifier_name=classifier,
            variant="Unigrams",
            preprocessor_name=preprocessor,
            ngram_range=(1, 1),
        ),
        make_experiment(
            train_size=train_size,
            vectorizer_name=base_vectorizer + " + Bigrams",
            classifier_name=classifier,
            variant="Bigrams",
            preprocessor_name=preprocessor,
            ngram_range=(1, 2),
        ),
    ]


def main():
    """
    V2 main entry point.
    """

    # Load the official V1 winners.
    previous_winners = load_winners("winners_V1.csv")

    # Generate the full superset: every unique V1 configuration
    # on every training size, with unigrams and bigrams.
    configs = generate_challenge_configs(
        previous_winners_df=previous_winners,
        challenge_factory=create_ngram_challenges,
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
        experiment_name="V2",
    )

    # ------------------------------------------------------------------
    # Official V2 winners: one per V1 duel line.
    # This gives 4 winners per training size.
    # ------------------------------------------------------------------

    comparison_winners_df = select_duel_winners(
        results_df=results_df,
        previous_winners_df=previous_winners,
        challenge_factory=create_ngram_challenges,
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
        path="results_V2.csv",
    )

    save_stage_csv(
        comparison_winners_df,
        columns=DISPLAY_COLUMNS,
        path="winners_V2.csv",
    )


if __name__ == "__main__":
    main()
