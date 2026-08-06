#!/usr/bin/python3

"""
Embedding framework for V6 experiments.

Experiments:
- V6.1 : Raw sentence embeddings + LinearSVC
- V6.2 : L2 normalized embeddings + LinearSVC
- V6.3 : Raw sentence embeddings + Nearest Centroid
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from utils import load, prepare

from plots import (
    plot_v6_confusion_matrices,
    plot_embedding_performance,
)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TEST_FILE = "imdb_test.csv"
TRAIN_SIZES = [50, 200, 500, 2000, 10000]


@dataclass
class EmbeddingConfig:
    """
    Configuration of one embedding experiment.
    """

    experiment: str
    variant: str
    normalization: str
    classifier: str


EXPERIMENTS = [

    EmbeddingConfig(
        experiment="V6.1",
        variant="Raw",
        normalization="None",
        classifier="LinearSVC",
    ),

    EmbeddingConfig(
        experiment="V6.2",
        variant="L2",
        normalization="L2",
        classifier="LinearSVC",
    ),

    EmbeddingConfig(
        experiment="V6.3",
        variant="Raw",
        normalization="None",
        classifier="Nearest Centroid",
    ),

]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_embedding_model():
    """
    Load sentence-transformer model once.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_texts(model, texts):
    """
    Encode texts into sentence embeddings.
    """

    start = time.perf_counter()

    embeddings = model.encode(
        texts.tolist(),
        show_progress_bar=True,
    )

    encoding_time = (time.perf_counter() - start)

    return embeddings, encoding_time

# ---------------------------------------------------------------------------
# Embedding transformations
# ---------------------------------------------------------------------------


def apply_normalization(embeddings, normalization_type):
    """
    Apply optional embedding normalization.

    Currently used only for V6.2.
    """

    if normalization_type == "L2":

        start = time.perf_counter()
        embeddings = normalize(embeddings, norm="l2")
        normalization_time = (time.perf_counter() - start)

        return embeddings, normalization_time

    return embeddings, 0.0


# ---------------------------------------------------------------------------
# Nearest Centroid classifier
# ---------------------------------------------------------------------------


class NearestCentroidClassifier:
    """
    Simple nearest centroid classifier.

    Each class is represented by the mean embedding
    of its training examples.
    """

    def __init__(self):

        self.classes_ = None
        self.centroids_ = None

    def fit(self, X, y):
        """
        Compute one centroid per class.
        """

        start = time.perf_counter()

        self.classes_ = np.unique(y)

        centroids = []

        for class_label in self.classes_:

            class_embeddings = X[y == class_label]

            centroid = (class_embeddings.mean(axis=0))
            centroids.append(centroid)

        self.centroids_ = np.array(centroids)

        train_time = (time.perf_counter() - start)

        return train_time

    def predict(self, X):
        """
        Assign each sample to the closest centroid.
        """

        distances = np.linalg.norm(
            X[:, np.newaxis, :]
            - self.centroids_[np.newaxis, :, :],
            axis=2,
        )

        closest = np.argmin(distances, axis=1)

        return self.classes_[closest]


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------


def build_classifier(classifier_name):
    """
    Create the requested classifier.
    """

    if classifier_name == "LinearSVC":
        return LinearSVC(max_iter=2500)

    if classifier_name == "Nearest Centroid":
        return NearestCentroidClassifier()

    raise ValueError(
        f"Unknown classifier: {classifier_name}"
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def train_and_predict(
    config,
    X_train_embeddings,
    y_train,
    X_test_embeddings,
):
    """
    Train one classifier and predict test embeddings.
    """

    classifier = build_classifier(config.classifier)

    train_start = time.perf_counter()

    if config.classifier == "Nearest Centroid":
        classifier_train_time = classifier.fit(
            X_train_embeddings,
            y_train,
        )

    else:
        classifier.fit(X_train_embeddings, y_train)
        classifier_train_time = (time.perf_counter() - train_start)

    inference_start = time.perf_counter()
    predictions = classifier.predict(X_test_embeddings)

    inference_time = (time.perf_counter() - inference_start)

    return (
        predictions,
        classifier_train_time,
        inference_time,
    )

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_embedding_experiments():
    """
    Run V6.1, V6.2 and V6.3 experiments.

    The embedding model is loaded once.
    The fixed test set is encoded once.
    """

    print()
    print("=" * 80)
    print("V6 SENTENCE EMBEDDING EXPERIMENTS")
    print("=" * 80)

    print()
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    model = load_embedding_model()

    results = []

    # -------------------------------------------------
    # Load fixed test set
    # -------------------------------------------------

    test = load(TEST_FILE)

    if test is None:
        return None

    X_test, y_test = prepare(test)

    print()
    print(f"Fixed test set: {len(X_test)} articles")
    print()

    print("Encoding fixed test set...")
    X_test_embeddings, test_encoding_time = (
        encode_texts(model, X_test)
    )

    print()
    print(f"Test encoding time: {test_encoding_time:.4f} s")

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------

    for train_size in TRAIN_SIZES:

        print()
        print("#" * 80)
        print(f"DATASET SIZE : {train_size}")
        print("#" * 80)

        train = load(f"imdb_train_{train_size}.csv")

        if train is None:
            continue

        X_train, y_train = prepare(train)

        print()
        print("Encoding train set...")

        X_train_embeddings, train_encoding_time = (
            encode_texts(model, X_train)
        )

        # -------------------------------------------------
        # Run V6.1 / V6.2 / V6.3
        # -------------------------------------------------

        for config in EXPERIMENTS:

            print()
            print(f"EXPERIMENT   : {config.experiment}")
            print(f"VARIANT      : {config.variant}")
            print(f"NORMALIZATION: {config.normalization}")

            train_embeddings, normalization_train_time = (
                apply_normalization(
                    X_train_embeddings,
                    config.normalization,
                )
            )

            test_embeddings, normalization_test_time = (
                apply_normalization(
                    X_test_embeddings,
                    config.normalization,
                )
            )

            predictions, classifier_train_time, inference_time = (
                train_and_predict(
                    config,
                    train_embeddings,
                    y_train,
                    test_embeddings,
                )
            )

            accuracy = accuracy_score(y_test, predictions)
            macro_f1 = f1_score(y_test, predictions, average="macro")

            results.append(
                {
                    "Experiment": config.experiment,
                    "Train size": train_size,

                    "Representation": "Sentence Embeddings",
                    "Representation details": EMBEDDING_MODEL_NAME,

                    "Normalization": config.normalization,

                    "Variant": config.variant,
                    "Classifier": config.classifier,

                    "Accuracy": accuracy,
                    "Macro F1": macro_f1,

                    "Train encoding time (s)": train_encoding_time,
                    "Test encoding time (s)": test_encoding_time,
                    "Normalization train time (s)": normalization_train_time,
                    "Normalization test time (s)": normalization_test_time,
                    "Classifier train time (s)": classifier_train_time,
                    "Inference time (s)": inference_time,

                    "y_true": y_test,
                    "y_pred": predictions,
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_embedding_results(results_df):
    """
    Save one CSV file per V6 experiment.
    """

    for experiment in results_df["Experiment"].unique():

        experiment_df = results_df[
            results_df["Experiment"] == experiment
        ]

        csv_columns = [
            column
            for column in experiment_df.columns
            if column not in [
                "y_true",
                "y_pred",
            ]
        ]

        filename = (
            f"results_{experiment.replace('.', '_')}"
            "_embeddings.csv"
        )

        experiment_df[
            csv_columns
        ].to_csv(
            filename,
            index=False,
        )

        print(f"Saved: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    results_df = run_embedding_experiments()

    if results_df is None:
        return

    print()
    print("=" * 80)
    print("V6 RESULTS")
    print("=" * 80)

    print(
        results_df[
            [
                "Experiment",
                "Train size",
                "Variant",
                "Classifier",
                "Accuracy",
                "Macro F1",
                "Classifier train time (s)",
                "Inference time (s)",
            ]
        ].round(4)
    )

    # -------------------------------------------------
    # Confusion matrices
    # -------------------------------------------------

    plot_v6_confusion_matrices(results_df)

    # -------------------------------------------------
    # Performance evolution plot
    # -------------------------------------------------

    plot_embedding_performance(results_df)

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    save_embedding_results(results_df)


if __name__ == "__main__":
    main()
