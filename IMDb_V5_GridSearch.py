#!/usr/bin/python3

"""
V5 - Hyperparameter optimization with GridSearchCV.

For each winning configuration of V4:
- Default: train with the default hyperparameters.
- GridSearchCV: optimize hyperparameters with 5-fold stratified CV.

The official winners CSV keeps the best variant (Default or GridSearchCV)
for each V4 duel line.
"""

import time
from functools import lru_cache

import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from IMDb_framework import (
    TRAIN_SIZES,
    TEST_FILE,
    ExperimentConfig,
    build_classifier,
    build_vectorizer,
    save_stage_csv,
    select_winners,
)

from utils import load, load_winners, prepare

from plots import (
    plot_confusion_matrices,
    plot_best_by_size,
    plot_default_vs_gridsearch,
)


# ---------------------------------------------------------------------------
# Runtime options
# ---------------------------------------------------------------------------

# Set to True to evaluate every unique V4 configuration on every training
# size (complete trajectories for plots). Much slower with GridSearchCV.
EXPAND_TO_ALL_SIZES = False

# Set to True to use the full original parameter grids.
# False uses a lighter grid to keep runtime acceptable on modest hardware.
REDUCED_GRID = True

PREVIOUS_WINNERS_FILE = "winners_V4.csv"


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


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def clean_preprocessor(value):
    """Convert missing / NaN preprocessor values to 'None'."""

    if value is None:
        return "None"

    if isinstance(value, float) and pd.isna(value):
        return "None"

    text = str(value).strip()

    if text in ("", "nan", "None", "<NA>", "NaT"):
        return "None"

    return text


def _remove_stopwords(words):
    """Filter out English stop words from a list of tokens."""

    return [
        word for word in words
        if word.lower() not in ENGLISH_STOP_WORDS
    ]


def _ensure_nltk_resources():
    """Download NLTK data if not already present."""

    import nltk

    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "taggers/averaged_perceptron_tagger_eng":
        "averaged_perceptron_tagger_eng",
        "tokenizers/punkt_tab": "punkt_tab",
    }

    for find_path, download_name in resources.items():
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


@lru_cache(maxsize=1)
def get_wordnet_lemmatizer():
    """Load the WordNet lemmatizer once."""

    _ensure_nltk_resources()

    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


@lru_cache(maxsize=None)
def make_wordnet_preprocessor(remove_stopwords: bool):
    """
    Return a sklearn-compatible callable that applies
    NLTK WordNet lemmatization with POS tagging.
    """

    from nltk import pos_tag
    from nltk.corpus import wordnet

    lemmatizer = get_wordnet_lemmatizer()

    def wordnet_pos(treebank_tag):
        """Map treebank POS tags to WordNet POS tags."""

        if treebank_tag.startswith("J"):
            return wordnet.ADJ

        if treebank_tag.startswith("V"):
            return wordnet.VERB

        if treebank_tag.startswith("R"):
            return wordnet.ADV

        return wordnet.NOUN

    def preprocess(text):
        words = text.split()

        if remove_stopwords:
            words = _remove_stopwords(words)

        tagged = pos_tag(words)

        return " ".join(
            lemmatizer.lemmatize(word, wordnet_pos(pos))
            for word, pos in tagged
        )

    return preprocess


@lru_cache(maxsize=1)
def get_spacy_nlp():
    """Load the spaCy English model once."""

    import spacy

    return spacy.load(
        "en_core_web_sm",
        disable=["parser", "ner"],
    )


@lru_cache(maxsize=None)
def make_spacy_preprocessor(remove_stopwords: bool):
    """
    Return a sklearn-compatible callable that applies
    spaCy lemmatization.
    """

    nlp = get_spacy_nlp()

    def preprocess(text):
        doc = nlp(text)

        return " ".join(
            token.lemma_
            for token in doc
            if not (remove_stopwords and token.is_stop)
        )

    return preprocess


def make_stemmer_preprocessor(stemmer_name: str, remove_stopwords: bool):
    """
    Return a sklearn-compatible callable that applies
    Porter or Snowball stemming.
    """

    if stemmer_name == "porter":
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()

    elif stemmer_name == "snowball":
        from nltk.stem.snowball import SnowballStemmer

        stemmer = SnowballStemmer("english")

    else:
        raise ValueError(
            f"Unknown stemmer '{stemmer_name}'. "
            "Expected 'porter' or 'snowball'."
        )

    def preprocess(text):
        words = text.split()

        if remove_stopwords:
            words = _remove_stopwords(words)

        return " ".join(stemmer.stem(word) for word in words)

    return preprocess


def make_preprocessor_from_name(
        preprocessor_name: str,
        remove_stopwords: bool
):
    """
    Rebuild a preprocessor callable from a stored name.
    """

    name = clean_preprocessor(preprocessor_name)

    if name == "Porter":
        return make_stemmer_preprocessor("porter", remove_stopwords)

    if name == "Snowball":
        return make_stemmer_preprocessor("snowball", remove_stopwords)

    if name == "WordNet":
        return make_wordnet_preprocessor(remove_stopwords)

    if name == "spaCy":
        return make_spacy_preprocessor(remove_stopwords)

    return None


# ---------------------------------------------------------------------------
# Grid and evaluation
# ---------------------------------------------------------------------------


def get_ngram_range(vectorizer_name: str):
    """Infer ngram range from the vectorizer name."""

    if "Bigrams" in vectorizer_name:
        return (1, 2)

    return (1, 1)


def get_param_grid(classifier_name: str):
    """
    Return the hyperparameter grid for a classifier.
    """

    if classifier_name == "MultinomialNB":

        if REDUCED_GRID:
            return {
                "classifier__alpha": [0.1, 1.0],
                "classifier__fit_prior": [True, False],
            }

        return {
            "classifier__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            "classifier__fit_prior": [True, False],
        }

    if classifier_name == "LinearSVC":

        if REDUCED_GRID:
            return {
                "classifier__C": [0.1, 1.0],
                "classifier__class_weight": [None, "balanced"],
            }

        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        }

    raise ValueError(f"No parameter grid for '{classifier_name}'.")


def build_v5_pipeline(config: ExperimentConfig) -> Pipeline:
    """
    Build a Pipeline from an ExperimentConfig.

    The preprocessor is already integrated into the vectorizer
    by build_vectorizer(), so it must NOT be a separate step.
    """

    return Pipeline(
        [
            ("vectorizer", build_vectorizer(config)),
            ("classifier", build_classifier(config)),
        ]
    )


def build_cv():
    """Return the cross-validation strategy used by GridSearchCV."""

    return StratifiedKFold(
        # n_splits=5,
        n_splits=3,
        shuffle=True,
        random_state=42,
    )


def evaluate_default(
    config: ExperimentConfig,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """
    Train and evaluate one experiment with default hyperparameters.
    """

    pipeline = build_v5_pipeline(config)

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = pipeline.predict(X_test)
    inference_time = time.perf_counter() - start

    return {
        "Train size": config.train_size,
        "Variant": "Default",
        "Preprocessor": config.preprocessor_name,
        "Vectorizer": config.vectorizer_name,
        "Classifier": config.classifier_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro F1": f1_score(y_test, predictions, average="macro"),
        "Train time (s)": train_time,
        "Inference time (s)": inference_time,
        "Best params": None,
        "CV Macro F1": None,
        "y_true": y_test,
        "y_pred": predictions,
    }


def evaluate_gridsearch(
    config: ExperimentConfig,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """
    Train and evaluate one experiment with GridSearchCV.
    """

    pipeline = build_v5_pipeline(config)
    param_grid = get_param_grid(config.classifier_name)
    cv = build_cv()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
    )

    start = time.perf_counter()
    grid_search.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = grid_search.predict(X_test)
    inference_time = time.perf_counter() - start

    print()
    print("Best parameters:")
    print(grid_search.best_params_)
    print()

    print("Best CV Macro F1:")
    print(grid_search.best_score_)

    return {
        "Train size": config.train_size,
        "Variant": "GridSearchCV",
        "Preprocessor": config.preprocessor_name,
        "Vectorizer": config.vectorizer_name,
        "Classifier": config.classifier_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro F1": f1_score(y_test, predictions, average="macro"),
        "Train time (s)": train_time,
        "Inference time (s)": inference_time,
        "Best params": str(grid_search.best_params_),
        "CV Macro F1": grid_search.best_score_,
        "y_true": y_test,
        "y_pred": predictions,
    }


# ---------------------------------------------------------------------------
# Configuration generation
# ---------------------------------------------------------------------------


def create_v5_challenges(row, train_size):
    """
    Create Default and GridSearchCV experiments for one
    winning configuration.
    """

    vectorizer_name = row["Vectorizer"]
    classifier_name = row["Classifier"]

    preprocessor_name = clean_preprocessor(
        row.get("Preprocessor", "None")
    )

    remove_stopwords = "StopWords" in vectorizer_name

    classifier_params = (
        {"max_iter": 2500}
        if classifier_name == "LinearSVC"
        else {}
    )

    common_kwargs = {
        "train_size": train_size,
        "vectorizer_name": vectorizer_name,
        "classifier_name": classifier_name,
        "preprocessor_name": preprocessor_name,
        "preprocessor": make_preprocessor_from_name(
            preprocessor_name,
            remove_stopwords,
        ),
        "ngram_range": get_ngram_range(vectorizer_name),
        "classifier_params": classifier_params,
    }

    return [
        ExperimentConfig(
            variant="Default",
            **common_kwargs,
        ),
        ExperimentConfig(
            variant="GridSearchCV",
            **common_kwargs,
        ),
    ]


def generate_v5_configs(previous_winners: pd.DataFrame):
    """
    Generate V5 experiment configurations.

    By default, only the exact winning sizes from V4 are used.
    If EXPAND_TO_ALL_SIZES is True, all unique configurations
    are evaluated on every training size.
    """

    configs = []

    if not EXPAND_TO_ALL_SIZES:
        for _, row in previous_winners.iterrows():
            configs.extend(
                create_v5_challenges(row, row["Train size"])
            )
        return configs

    unique_configs = previous_winners[
        ["Vectorizer", "Classifier", "Preprocessor"]
    ].drop_duplicates()

    for _, row in unique_configs.iterrows():
        for train_size in TRAIN_SIZES:
            configs.extend(
                create_v5_challenges(row, train_size)
            )

    return configs


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_v5_experiments(configs, test_df):
    """
    Run all V5 experiments, loading each train dataset once.
    """

    X_test, y_test = prepare(test_df)
    print(f"Fixed test set: {len(X_test)} articles")

    configs_by_size: dict[int, list[ExperimentConfig]] = {}

    for config in configs:
        configs_by_size.setdefault(config.train_size, []).append(config)

    results = []

    for train_size in sorted(configs_by_size):

        print("\n")
        print("#" * 80)
        print(f"DATASET SIZE : {train_size}")
        print("#" * 80)

        train = load(f"imdb_train_{train_size}.csv")

        if train is None:
            continue

        X_train, y_train = prepare(train)

        for config in configs_by_size[train_size]:

            print("\n")
            print(f"VARIANT      : {config.variant}")
            print(
                f"CONFIG       : {config.vectorizer_name} "
                f"+ {config.classifier_name} "
                f"+ {config.preprocessor_name}"
            )
            print("#" * 80)

            if config.variant == "Default":
                result = evaluate_default(
                    config,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
            else:
                result = evaluate_gridsearch(
                    config,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )

            results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metadata and winner selection
# ---------------------------------------------------------------------------


def add_v5_metadata(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the canonical metadata columns for V5.
    """

    df = results_df.copy()
    df["Experiment"] = "V5"

    def extract_base_vectorizer(name):
        if "TF-IDF" in name:
            return "TF-IDF"
        if "Count" in name:
            return "Count"
        return name

    df["Base Vectorizer"] = df["Vectorizer"].apply(
        extract_base_vectorizer
    )

    df["StopWords"] = df["Vectorizer"].apply(
        lambda name: (
            "With StopWords"
            if "StopWords" in name
            else "Without StopWords"
        )
    )

    if "Preprocessor" not in df.columns:
        df["Preprocessor"] = "None"

    df["Preprocessor"] = df["Preprocessor"].fillna("None")

    return df


def select_v5_winners(
    results_df: pd.DataFrame,
    previous_winners_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the official V5 winners.

    For each V4 winner line, compare Default vs GridSearchCV
    and keep the best result.
    """

    selected = []

    for _, previous_row in previous_winners_df.iterrows():

        train_size = previous_row["Train size"]
        vectorizer = previous_row["Vectorizer"]
        classifier = previous_row["Classifier"]
        preprocessor = clean_preprocessor(
            previous_row.get("Preprocessor", "None")
        )

        candidates = results_df[
            (results_df["Train size"] == train_size)
            & (results_df["Vectorizer"] == vectorizer)
            & (results_df["Classifier"] == classifier)
            & (
                results_df["Preprocessor"].fillna("None").astype(str)
                .str.strip()
                == preprocessor
            )
            & (results_df["Variant"].isin(["Default", "GridSearchCV"]))
        ]

        if candidates.empty:
            print(
                f"Warning: no candidates for {vectorizer} "
                f"+ {classifier} at size {train_size}"
            )
            continue

        winner = candidates.sort_values(
            by=[
                "Macro F1",
                "Accuracy",
                "Inference time (s)",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).iloc[0]

        selected.append(winner)

    return pd.DataFrame(selected)


# ---------------------------------------------------------------------------
# V5-specific summaries
# ---------------------------------------------------------------------------


def calculate_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Default and GridSearchCV results on the fixed test set.

    Macro F1 is the primary comparison metric. When Macro F1 values
    are identical, Accuracy and inference time are used as tie-breakers.
    """

    comparison_columns = [
        "Train size",
        "Base Vectorizer",
        "StopWords",
        "Preprocessor",
        "Vectorizer",
        "Classifier",
    ]

    comparison = results_df.pivot(
        index=comparison_columns,
        columns="Variant",
        values=[
            "Accuracy",
            "Macro F1",
            "Train time (s)",
            "Inference time (s)",
        ],
    )

    comparison.columns = [
        "_".join(str(column) for column in column_tuple)
        for column_tuple in comparison.columns
    ]

    comparison = comparison.reset_index()

    comparison["Delta Accuracy"] = (
        comparison["Accuracy_GridSearchCV"]
        - comparison["Accuracy_Default"]
    )

    comparison["Delta Macro F1"] = (
        comparison["Macro F1_GridSearchCV"]
        - comparison["Macro F1_Default"]
    )

    comparison["Delta Inference time (s)"] = (
        comparison["Inference time (s)_GridSearchCV"]
        - comparison["Inference time (s)_Default"]
    )

    comparison["Comparison result"] = "Identical Macro F1"
    comparison["Official winner"] = "Tie"

    for index, row in comparison.iterrows():

        # Primary criterion: Macro F1
        if row["Delta Macro F1"] > 0:
            comparison.loc[index, "Comparison result"] = (
                "GridSearchCV better Macro F1"
            )
            comparison.loc[index, "Official winner"] = (
                "GridSearchCV"
            )

        elif row["Delta Macro F1"] < 0:
            comparison.loc[index, "Comparison result"] = (
                "Default better Macro F1"
            )
            comparison.loc[index, "Official winner"] = "Default"

        # Macro F1 is identical: apply the official tie-breakers.
        else:

            # First tie-breaker: Accuracy
            if row["Delta Accuracy"] > 0:
                comparison.loc[index, "Official winner"] = (
                    "GridSearchCV"
                )

            elif row["Delta Accuracy"] < 0:
                comparison.loc[index, "Official winner"] = (
                    "Default"
                )

            # Second tie-breaker: lower inference time
            elif row["Delta Inference time (s)"] < 0:
                comparison.loc[index, "Official winner"] = (
                    "GridSearchCV"
                )

            elif row["Delta Inference time (s)"] > 0:
                comparison.loc[index, "Official winner"] = (
                    "Default"
                )

    return comparison


def print_v5_comparison(comparison_df: pd.DataFrame):
    """Display Default vs GridSearchCV comparison."""

    columns = [
        "Train size",
        "Preprocessor",
        "Vectorizer",
        "Classifier",
        "Accuracy_Default",
        "Accuracy_GridSearchCV",
        "Delta Accuracy",
        "Macro F1_Default",
        "Macro F1_GridSearchCV",
        "Delta Macro F1",
    ]

    print()
    print("=" * 100)
    print("V5 DEFAULT VS GRIDSEARCHCV")
    print("=" * 100)

    print(comparison_df[columns].round(4))


def summarize_v5_results(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize V5 results globally and by training dataset size.

    Macro F1 is the primary comparison metric. When Macro F1 values
    are identical, Accuracy and inference time determine the official
    winner. A complete tie is kept as "Tie".
    """

    total_runs = len(comparison_df)

    gridsearch_better_macro_f1 = (
        comparison_df["Comparison result"]
        == "GridSearchCV better Macro F1"
    ).sum()

    default_better_macro_f1 = (
        comparison_df["Comparison result"]
        == "Default better Macro F1"
    ).sum()

    identical_macro_f1 = (
        comparison_df["Comparison result"]
        == "Identical Macro F1"
    ).sum()

    gridsearch_official_wins = (
        comparison_df["Official winner"]
        == "GridSearchCV"
    ).sum()

    default_official_wins = (
        comparison_df["Official winner"]
        == "Default"
    ).sum()

    complete_ties = (
        comparison_df["Official winner"]
        == "Tie"
    ).sum()

    print()
    print("=" * 100)
    print("V5 RESULT COUNTS")
    print("=" * 100)

    print(
        "GridSearchCV better Macro F1 : "
        f"{gridsearch_better_macro_f1} / {total_runs}"
    )

    print(
        "Default better Macro F1      : "
        f"{default_better_macro_f1} / {total_runs}"
    )

    print(
        "Identical Macro F1           : "
        f"{identical_macro_f1} / {total_runs}"
    )

    print()

    print(
        "GridSearchCV official wins   : "
        f"{gridsearch_official_wins} / {total_runs}"
    )

    print(
        "Default official wins        : "
        f"{default_official_wins} / {total_runs}"
    )

    print(
        "Complete ties                : "
        f"{complete_ties} / {total_runs}"
    )

    summary_df = (
        comparison_df
        .groupby("Train size")
        .agg(
            Runs=(
                "Delta Macro F1",
                "count",
            ),
            Mean_Delta_Macro_F1=(
                "Delta Macro F1",
                "mean",
            ),
            Median_Delta_Macro_F1=(
                "Delta Macro F1",
                "median",
            ),
            GridSearchCV_Better_Macro_F1=(
                "Comparison result",
                lambda values: (
                    values
                    == "GridSearchCV better Macro F1"
                ).sum(),
            ),
            Default_Better_Macro_F1=(
                "Comparison result",
                lambda values: (
                    values
                    == "Default better Macro F1"
                ).sum(),
            ),
            Identical_Macro_F1=(
                "Comparison result",
                lambda values: (
                    values
                    == "Identical Macro F1"
                ).sum(),
            ),
            GridSearchCV_Official_Wins=(
                "Official winner",
                lambda values: (
                    values
                    == "GridSearchCV"
                ).sum(),
            ),
            Default_Official_Wins=(
                "Official winner",
                lambda values: (
                    values
                    == "Default"
                ).sum(),
            ),
            Complete_Ties=(
                "Official winner",
                lambda values: (
                    values
                    == "Tie"
                ).sum(),
            ),
        )
        .reset_index()
    )

    return summary_df


def print_v5_summary(
    summary_df: pd.DataFrame,
):
    """
    Display the V5 summary by training dataset size.
    """

    print()
    print("=" * 140)
    print("V5 SUMMARY BY DATASET SIZE")
    print("=" * 140)

    display_columns = [
        "Train size",
        "Runs",
        "Mean_Delta_Macro_F1",
        "Median_Delta_Macro_F1",
        "GridSearchCV_Better_Macro_F1",
        "Default_Better_Macro_F1",
        "Identical_Macro_F1",
        "GridSearchCV_Official_Wins",
        "Default_Official_Wins",
        "Complete_Ties",
    ]

    print(
        summary_df[display_columns].round({
            "Mean_Delta_Macro_F1": 4,
            "Median_Delta_Macro_F1": 4,
        })
    )

    print(
        summary_df[display_columns].round({
            "Mean_Delta_Macro_F1": 4,
            "Median_Delta_Macro_F1": 4,
        })
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    V5 main entry point.
    """

    previous_winners = load_winners(PREVIOUS_WINNERS_FILE)

    configs = generate_v5_configs(previous_winners)

    test_df = load(TEST_FILE)

    if test_df is None:
        return

    results_df = run_v5_experiments(configs, test_df)

    if results_df is None:
        return

    results_df = add_v5_metadata(results_df)

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("V5 GLOBAL SUMMARY")
    print("=" * 80)

    summary_columns = [
        "Train size",
        "Variant",
        "Preprocessor",
        "Vectorizer",
        "Classifier",
        "Accuracy",
        "Macro F1",
        "CV Macro F1",
        "Train time (s)",
        "Inference time (s)",
        "Best params",
    ]

    print(
        results_df[summary_columns].round({
            "Accuracy": 3,
            "Macro F1": 3,
            "CV Macro F1": 3,
            "Train time (s)": 4,
            "Inference time (s)": 4,
        })
    )

    # ------------------------------------------------------------------
    # Official V5 winners: one per V4 duel line.
    # ------------------------------------------------------------------

    comparison_winners_df = select_v5_winners(
        results_df,
        previous_winners,
    )

    if not comparison_winners_df.empty:
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
    else:
        print("Warning: no official winners were selected.")

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

    plot_default_vs_gridsearch(results_df)

    # ------------------------------------------------------------------
    # Default vs GridSearchCV comparison
    # ------------------------------------------------------------------

    comparison_df = calculate_deltas(results_df)

    print_v5_comparison(comparison_df)

    summary_df = summarize_v5_results(comparison_df)

    print_v5_summary(summary_df)

    # ------------------------------------------------------------------
    # CSV exports
    # ------------------------------------------------------------------

    save_stage_csv(
        results_df,
        columns=DISPLAY_COLUMNS + ["Best params", "CV Macro F1"],
        path="results_V5.csv",
    )

    if not comparison_winners_df.empty:
        save_stage_csv(
            comparison_winners_df,
            columns=DISPLAY_COLUMNS,
            path="winners_V5.csv",
        )

    comparison_df.to_csv("V5_comparison.csv", index=False)
    summary_df.to_csv("V5_summary.csv", index=False)


if __name__ == "__main__":
    main()
