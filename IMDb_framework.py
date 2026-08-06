#!/usr/bin/python3
"""
Shared framework for V1-V5 experiment stages.

Each stage follows the same workflow:
1. Load a fixed test set.
2. Build experiment configurations (baseline grid or challengers).
3. Run all configurations on every training size.
4. Add metadata.
5. Select stage-specific winners.
6. Export results and winners CSVs.
7. Display matrices, best-by-size plot, and winning-configuration plot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from utils import load, prepare

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

TRAIN_SIZES = [50, 200, 500, 2000, 10000]
TEST_FILE = "imdb_test.csv"
TRAIN_FILE_TEMPLATE = "imdb_train_{}.csv"

# ---------------------------------------------------------------------------
# Experiment description
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    Complete description of one classification pipeline.

    The pipeline is always:
        text -> vectorizer -> classifier
    """

    train_size: int
    vectorizer_name: str
    classifier_name: str
    variant: str
    preprocessor_name: str = "None"
    preprocessor: Any = None
    ngram_range: tuple[int, int] = (1, 1)
    vectorizer_params: dict = field(default_factory=dict)
    classifier_params: dict = field(default_factory=dict)


def make_experiment(
    train_size: int,
    vectorizer_name: str,
    classifier_name: str,
    variant: str,
    preprocessor_name: str = "None",
    preprocessor: Any = None,
    ngram_range: tuple[int, int] = (1, 1),
    vectorizer_params: dict | None = None,
    classifier_params: dict | None = None,
) -> ExperimentConfig:
    """
    Convenience factory for ExperimentConfig.
    """

    return ExperimentConfig(
        train_size=train_size,
        vectorizer_name=vectorizer_name,
        classifier_name=classifier_name,
        variant=variant,
        preprocessor_name=preprocessor_name,
        preprocessor=preprocessor,
        ngram_range=ngram_range,
        vectorizer_params=vectorizer_params or {},
        classifier_params=classifier_params or {},
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_vectorizer(config):
    name = config.vectorizer_name
    use_stopwords = "StopWords" in name

    params = dict(config.vectorizer_params)
    params["ngram_range"] = config.ngram_range

    # If a preprocessor (stemming) is provided,
    # do NOT pass stop_words to the vectorizer.
    # Stop-word removal is handled by the preprocessor.
    if use_stopwords and config.preprocessor is None:
        params["stop_words"] = "english"

    if config.preprocessor is not None:
        params["preprocessor"] = config.preprocessor

    if name.startswith("Count"):
        return CountVectorizer(**params)

    if name.startswith("TF-IDF"):
        return TfidfVectorizer(**params)

    raise ValueError(f"Unknown vectorizer '{name}'.")


def build_classifier(config: ExperimentConfig):
    """
    Rebuild a classifier from an ExperimentConfig.
    """

    name = config.classifier_name
    params = dict(config.classifier_params)

    if name == "MultinomialNB":
        return MultinomialNB(**params)

    if name == "LinearSVC":
        return LinearSVC(**params)

    raise ValueError(f"Unknown classifier '{name}'.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    config: ExperimentConfig,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """
    Train and evaluate one experiment.

    Returns metrics, metadata, and predictions.
    """

    pipeline = Pipeline(
        [
            ("vectorizer", build_vectorizer(config)),
            ("classifier", build_classifier(config)),
        ]
    )

    name = f"{config.vectorizer_name} + {config.classifier_name}"

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = pipeline.predict(X_test)
    inference_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Training time : {train_time:.4f} s")
    print(f"Inference time: {inference_time:.4f} s")
    print(f"Accuracy      : {accuracy:.3f}")
    print(f"Macro F1      : {macro_f1:.3f}\n")

    print(classification_report(y_test, predictions, digits=3))

    return {
        "Train size": config.train_size,
        "Variant": config.variant,
        "Preprocessor": config.preprocessor_name,
        "Vectorizer": config.vectorizer_name,
        "Classifier": config.classifier_name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Train time (s)": train_time,
        "Inference time (s)": inference_time,
        "y_true": y_test,
        "y_pred": predictions,
    }


def run_experiments(
    configs: Iterable[ExperimentConfig],
    test_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    Run all experiments grouped by training size.

    Loading is done once per training size, not once per config.
    """

    if test_df is None:
        test_df = load(TEST_FILE)

    if test_df is None:
        return None

    X_test, y_test = prepare(test_df)
    print(f"Fixed test set: {len(X_test)} articles")

    configs_by_size: dict[int, list[ExperimentConfig]] = {}

    for config in configs:
        configs_by_size.setdefault(config.train_size, []).append(config)

    results = []

    for size, size_configs in sorted(configs_by_size.items()):

        print("\n")
        print("#" * 80)
        print(f"DATASET SIZE : {size}")
        print("#" * 80)

        train = load(TRAIN_FILE_TEMPLATE.format(size))

        if train is None:
            print(f"Skip {size}: train file not found.")
            continue

        X_train, y_train = prepare(train)

        for config in size_configs:
            results.append(
                evaluate(
                    config,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def add_stage_metadata(
    results_df: pd.DataFrame,
    experiment_name: str,
) -> pd.DataFrame:
    """
    Add the canonical metadata columns shared by all stages.

    Columns:
    - Experiment
    - Base Vectorizer
    - StopWords
    - Preprocessor
    """

    df = results_df.copy()

    df["Experiment"] = experiment_name

    def extract_base_vectorizer(name: str) -> str:
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

    return df


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------


def select_winners(
    results_df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """
    Select the best experiment inside each comparison group.

    Ranking criteria:
    1) Highest Macro F1
    2) Highest Accuracy
    3) Lowest inference time
    """

    winners = []

    for _, group in results_df.groupby(group_columns):

        ranked = group.sort_values(
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
        )

        winners.append(ranked.iloc[0].copy())

    return pd.DataFrame(winners)


def _norm(value):
    """Normalize values for duel matching."""
    if value is None:
        return "None"
    if isinstance(value, float) and pd.isna(value):
        return "None"
    return str(value).strip()


def select_duel_winners(
    results_df: pd.DataFrame,
    previous_winners_df: pd.DataFrame,
    challenge_factory,
) -> pd.DataFrame:
    """
    Select the official winners of a challenger stage.

    Matching is done on Train size / Vectorizer / Classifier / Preprocessor
    with normalized strings so that None, NaN, and "None" are equivalent.
    """

    if results_df.empty:
        raise ValueError("results_df is empty.")

    results_df = results_df.copy()

    # Build a search key for every result row.
    preprocessor_col = (
        results_df["Preprocessor"]
        if "Preprocessor" in results_df.columns
        else pd.Series("None", index=results_df.index)
    )

    results_df["_match_key"] = list(zip(
        results_df["Train size"].map(_norm),
        results_df["Vectorizer"].map(_norm),
        results_df["Classifier"].map(_norm),
        preprocessor_col.map(_norm),
    ))

    selected = []

    for _, previous_row in previous_winners_df.iterrows():

        train_size = previous_row["Train size"]
        candidates = challenge_factory(previous_row, train_size)

        candidate_keys = set()
        for candidate in candidates:
            candidate_keys.add((
                _norm(candidate.train_size),
                _norm(candidate.vectorizer_name),
                _norm(candidate.classifier_name),
                _norm(candidate.preprocessor_name),
            ))

        duel_results = results_df[
            results_df["_match_key"].isin(candidate_keys)
        ]

        if duel_results.empty:
            # Fallback: if previous winners have no Preprocessor column,
            # match without it.
            fallback_keys = set()
            for candidate in candidates:
                fallback_keys.add((
                    _norm(candidate.train_size),
                    _norm(candidate.vectorizer_name),
                    _norm(candidate.classifier_name),
                ))

            fallback_results = results_df[
                results_df["_match_key"].apply(
                    lambda k: (k[0], k[1], k[2]) in fallback_keys
                )
            ]

            if not fallback_results.empty:
                duel_results = fallback_results
            else:
                print(
                    f"Warning: no match found for "
                    f"{previous_row['Vectorizer']} "
                    f"+ {previous_row['Classifier']} "
                    f"at size {train_size}"
                )
                continue

        winner = duel_results.sort_values(
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

        selected.append(winner.drop(columns=["_match_key"]))

    if not selected:
        return pd.DataFrame(
            columns=results_df.columns.drop("_match_key")
        )

    return pd.DataFrame(selected)


# ---------------------------------------------------------------------------
# Configuration generators
# ---------------------------------------------------------------------------


def generate_baseline_configs(
    vectorizer_specs: list[dict],
    classifier_specs: list[dict],
    train_sizes: list[int] | None = None,
) -> list[ExperimentConfig]:
    """
    Generate the initial V1 grid: every vectorizer x every classifier
    on every training size.
    """

    configs: list[ExperimentConfig] = []
    sizes = train_sizes or TRAIN_SIZES

    for train_size in sizes:
        for vectorizer_spec in vectorizer_specs:
            for classifier_spec in classifier_specs:

                configs.append(
                    make_experiment(
                        train_size=train_size,
                        vectorizer_name=vectorizer_spec["name"],
                        classifier_name=classifier_spec["name"],
                        variant=vectorizer_spec.get("variant", "Baseline"),
                        preprocessor_name=vectorizer_spec.get(
                            "preprocessor_name", "None"
                        ),
                        preprocessor=vectorizer_spec.get("preprocessor"),
                        ngram_range=vectorizer_spec.get(
                            "ngram_range", (1, 1)
                        ),
                        vectorizer_params=vectorizer_spec.get(
                            "params", {}
                        ),
                        classifier_params=classifier_spec.get(
                            "params", {}
                        ),
                    )
                )

    return configs


def generate_challenge_configs(
    previous_winners_df: pd.DataFrame,
    challenge_factory: Callable[[pd.Series, int], list[ExperimentConfig]],
    train_sizes: list[int] | None = None,
) -> list[ExperimentConfig]:
    """
    Generate the full superset used for plotting.

    This evaluates every unique previous configuration on every
    training size, not only on the sizes where it won.

    The official winners CSV is obtained separately with
    select_duel_winners().
    """

    configs: list[ExperimentConfig] = []
    sizes = sorted(train_sizes or TRAIN_SIZES)

    config_columns = ["Vectorizer", "Classifier"]

    if "Preprocessor" in previous_winners_df.columns:
        config_columns.append("Preprocessor")

    unique_configs = (
        previous_winners_df[config_columns]
        .drop_duplicates()
    )

    for _, base_row in unique_configs.iterrows():

        for train_size in sizes:

            configs.extend(
                challenge_factory(base_row, train_size)
            )

    return configs


# ---------------------------------------------------------------------------
# CSV export helper
# ---------------------------------------------------------------------------


def save_stage_csv(
    df: pd.DataFrame,
    columns: list[str],
    path: str,
) -> None:
    """
    Save a stage CSV with a fixed column order.
    """

    df[columns].to_csv(path, index=False)
