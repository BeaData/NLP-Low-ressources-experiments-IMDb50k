#!/usr/bin/python3

"""
V3 - Stemming experiment.

For every winning configuration of V2, compare:
- Baseline (no stemming)
- Porter stemming
- Snowball stemming

Each unique configuration is evaluated on every training size
so that performance trajectories are complete.

The official winners CSV keeps only the winners of the duels
defined line by line by winners_V2.csv.
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

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


def make_preprocessor(
    stemmer_name: str,
    remove_stopwords: bool = False,
):
    """
    Return a sklearn-compatible callable that applies stemming.

    If remove_stopwords is True, English stop words are removed
    before stemming.
    """

    if stemmer_name == "porter":
        stemmer = PorterStemmer()
    elif stemmer_name == "snowball":
        stemmer = SnowballStemmer("english")
    else:
        return None

    def preprocess(text):
        words = text.split()

        if remove_stopwords:
            words = [
                word for word in words
                if word.lower() not in ENGLISH_STOP_WORDS
            ]

        return " ".join(
            stemmer.stem(word)
            for word in words
        )

    return preprocess


def get_ngram_range(vectorizer_name: str):
    """
    Infer ngram range from the vectorizer name.
    """

    if "Bigrams" in vectorizer_name:
        return (1, 2)

    return (1, 1)


def create_stemming_challenges(row, train_size):

    vectorizer = row["Vectorizer"]
    classifier = row["Classifier"]

    remove_stopwords = "StopWords" in vectorizer

    ngram_range = get_ngram_range(vectorizer)

    return [
        make_experiment(
            train_size=train_size,
            vectorizer_name=vectorizer,
            classifier_name=classifier,
            variant="Baseline",
            preprocessor_name=str(row.get("Preprocessor", "None")),
            preprocessor=None,
            ngram_range=ngram_range,
        ),
        make_experiment(
            train_size=train_size,
            vectorizer_name=vectorizer,
            classifier_name=classifier,
            variant="Porter",
            preprocessor_name="Porter",
            preprocessor=make_preprocessor(
                "porter",
                remove_stopwords=remove_stopwords,
            ),
            ngram_range=ngram_range,
        ),
        make_experiment(
            train_size=train_size,
            vectorizer_name=vectorizer,
            classifier_name=classifier,
            variant="Snowball",
            preprocessor_name="Snowball",
            preprocessor=make_preprocessor(
                "snowball",
                remove_stopwords=remove_stopwords,
            ),
            ngram_range=ngram_range,
        ),
    ]


def main():
    """
    V3 main entry point.
    """

    # Load the official V2 winners.
    previous_winners = load_winners("winners_V2.csv")

    # Generate the full superset: every unique V2 configuration
    # on every training size, with baseline / Porter / Snowball.
    configs = generate_challenge_configs(
        previous_winners_df=previous_winners,
        challenge_factory=create_stemming_challenges,
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
        experiment_name="V3",
    )

    # ------------------------------------------------------------------
    # Official V3 winners: one per V2 duel line.
    # ------------------------------------------------------------------

    comparison_winners_df = select_duel_winners(
        results_df=results_df,
        previous_winners_df=previous_winners,
        challenge_factory=create_stemming_challenges,
    )

    if comparison_winners_df.empty:
        print("Warning: no official winners were selected.")
    else:
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

    if not comparison_winners_df.empty:
        plot_confusion_matrices(comparison_winners_df)

    plot_best_by_size(best_by_size_df)

    plot_winning_configurations(
        results_df,
        best_by_size_df,
    )

    # ------------------------------------------------------------------
    # CSV exports
    # ------------------------------------------------------------------

    save_stage_csv(
        results_df,
        columns=DISPLAY_COLUMNS,
        path="results_V3.csv",
    )

    if not comparison_winners_df.empty:
        save_stage_csv(
            comparison_winners_df,
            columns=DISPLAY_COLUMNS,
            path="winners_V3.csv",
        )


if __name__ == "__main__":
    main()
