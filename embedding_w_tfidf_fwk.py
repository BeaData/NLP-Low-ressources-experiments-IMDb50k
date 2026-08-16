#!/usr/bin/python3

"""
Word embedding framework for V7 experiments.

V7.2:
    GloVe 100d + TF-IDF Weighted Mean + LinearSVC
"""

import re
import time

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC

from utils import load, prepare
from plots import (
    plot_v6_confusion_matrices,
    plot_embedding_performance,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

GLOVE_FILE = "../glove/glove.6B.100d.txt"
TEST_FILE = "dbpedia_test_14000.csv"
TRAIN_SIZES = [
    50,
    200,
    500,
    2000,
    10000,
]
EMBEDDING_DIMENSION = 100
EXPERIMENT = "V7.2"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text):
    """
    Tokenize text into lowercase alphabetic tokens.
    """

    return re.findall(r"[a-z]+", text.lower())


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(texts):
    """
    Build the vocabulary required by a collection of documents.
    """
    vocabulary = set()

    for text in texts:
        vocabulary.update(tokenize(text))

    return vocabulary


# ---------------------------------------------------------------------------
# IDF computation
# ---------------------------------------------------------------------------

def compute_idf(texts):
    """
    Compute IDF weights from a training corpus.

    IDF is computed only from the supplied documents.
    This prevents data leakage from the test set.

    Returns:
        dictionary mapping token -> IDF weight
    """

    document_count = len(texts)

    document_frequency = {}

    for text in texts:

        tokens = set(tokenize(text))

        for token in tokens:
            document_frequency[token] = (
                document_frequency.get(token, 0) + 1
            )

    idf = {}

    for token, df in document_frequency.items():

        idf[token] = np.log(
            (1 + document_count) / (1 + df)
        ) + 1.0

    return idf


# ---------------------------------------------------------------------------
# Document encoding
# ---------------------------------------------------------------------------

def encode_document_tfidf(
    text,
    vectors,
    idf,
    embedding_dim=100,
):
    """
    Encode one document using TF-IDF weighted mean pooling.

    OOV tokens are ignored.

    If all tokens are OOV, return a zero vector.
    """

    tokens = tokenize(text)

    if not tokens:
        return np.zeros(
            embedding_dim,
            dtype=np.float32,
        )

    # Term frequency
    term_frequency = {}

    for token in tokens:
        term_frequency[token] = (
            term_frequency.get(token, 0) + 1
        )

    weighted_vectors = []
    weights = []

    for token, tf in term_frequency.items():

        if token not in vectors:
            continue

        if token not in idf:
            continue

        weight = ((tf / len(tokens)) * idf[token])
        weighted_vectors.append(vectors[token] * weight)
        weights.append(weight)

    if not weighted_vectors:
        return np.zeros(embedding_dim, dtype=np.float32)

    return (
        np.sum(np.asarray(weighted_vectors), axis=0)
        / np.sum(weights)
    ).astype(np.float32)


def encode_documents_tfidf(
    texts,
    vectors,
    idf,
    embedding_dim=100,
):
    """
    Encode multiple documents using TF-IDF weighted mean pooling.

    Returns:
        embeddings
        number of documents with no known GloVe tokens
    """

    embeddings = []
    empty_documents = 0

    for text in texts:

        embedding = encode_document_tfidf(
            text,
            vectors,
            idf,
            embedding_dim=embedding_dim,
        )

        if not np.any(embedding):
            empty_documents += 1

        embeddings.append(embedding)

    return (
        np.asarray(embeddings, dtype=np.float32),
        empty_documents,
    )


# ---------------------------------------------------------------------------
# GloVe loading
# ---------------------------------------------------------------------------

def load_required_glove_vectors(
    filepath,
    required_words,
):
    """
    Load only GloVe vectors required by the vocabulary.

    Words not present in GloVe are ignored.
    """
    vectors = {}

    start = time.perf_counter()

    with open(filepath, "r", encoding="utf-8") as file:

        for line in file:
            parts = line.rstrip().split(" ")
            word = parts[0]

            if word not in required_words:
                continue

            vector = np.asarray(parts[1:], dtype=np.float32)
            vectors[word] = vector

            if len(vectors) == len(required_words):
                break

    load_time = time.perf_counter() - start

    return vectors, load_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print()
    print("=" * 80)
    print("V7.2 - GloVe 100d + TF-IDF Weighted Mean + LinearSVC")
    print("=" * 80)

    # -------------------------------------------------
    # Load fixed test set
    # -------------------------------------------------

    test = load(TEST_FILE)

    if test is None:
        return

    X_test, y_test = prepare(test)

    print()
    print(f"Test documents: {len(X_test):,}")

    # -------------------------------------------------
    # Load largest training set
    # -------------------------------------------------

    train = load(f"dbpedia_train_{max(TRAIN_SIZES)}.csv")

    if train is None:
        return

    X_train, y_train = prepare(train)

    print(f"Training documents: {len(X_train):,}")

    # -------------------------------------------------
    # Build global vocabulary
    # -------------------------------------------------

    print()
    print("Building vocabulary...")

    start = time.perf_counter()

    all_texts = pd.concat(
        [
            X_train,
            X_test,
        ],
        ignore_index=True,
    )

    vocabulary = build_vocabulary(all_texts)
    vocabulary_time = (time.perf_counter() - start)

    print(f"Vocabulary size: {len(vocabulary):,}")
    print(f"Vocabulary time: {vocabulary_time:.4f} s")

    # -------------------------------------------------
    # Load required GloVe vectors
    # -------------------------------------------------

    print()
    print("Loading required GloVe vectors...")

    vectors, glove_load_time = (
        load_required_glove_vectors(
            GLOVE_FILE,
            vocabulary,
        )
    )

    print(f"Vectors loaded: {len(vectors):,}")
    print(f"GloVe load time: {glove_load_time:.4f} s")

    # -------------------------------------------------
    # Vocabulary coverage
    # -------------------------------------------------

    known_words = set(vectors)

    oov_words = vocabulary - known_words

    print()
    print("=" * 80)
    print("VOCABULARY COVERAGE")
    print("=" * 80)

    print(f"Total vocabulary: {len(vocabulary):,}")
    print(f"Known vocabulary:  {len(known_words):,}")
    print(f"OOV vocabulary:    {len(oov_words):,}")

    if vocabulary:

        coverage = (len(known_words) / len(vocabulary) * 100)
        print(f"Coverage:          {coverage:.4f}%")

    # -------------------------------------------------
    # Run experiments
    # -------------------------------------------------

    results = []

    print()
    print("=" * 80)
    print("V7.2 CLASSIFICATION")
    print("=" * 80)

    for train_size in TRAIN_SIZES:

        print()
        print("#" * 80)
        print(f"TRAINING SIZE: {train_size}")
        print("#" * 80)

        train = load(f"dbpedia_train_{train_size}.csv")

        if train is None:
            continue

        X_train, y_train = prepare(train)

        # -------------------------------------------------
        # Compute IDF from training set only
        # -------------------------------------------------

        print()
        print("Computing IDF weights...")

        idf_start = time.perf_counter()
        idf = compute_idf(X_train)
        idf_time = (time.perf_counter() - idf_start)

        print(f"IDF vocabulary: {len(idf):,}")
        print(f"IDF computation time: {idf_time:.4f} s")

        # -------------------------------------------------
        # Encode training set
        # -------------------------------------------------

        print()
        print("Encoding training set...")

        encoding_start = time.perf_counter()

        X_train_embeddings, empty_train = (
            encode_documents_tfidf(
                X_train,
                vectors,
                idf,
                embedding_dim=EMBEDDING_DIMENSION,
            )
        )

        train_encoding_time = (time.perf_counter() - encoding_start)

        print(f"Embedding shape: {X_train_embeddings.shape}")
        print(f"Empty documents: {empty_train}")
        print(f"Train encoding time: {train_encoding_time:.4f} s")

        # -------------------------------------------------
        # Encode test set using training IDF
        # -------------------------------------------------

        print()
        print("Encoding test set...")

        test_encoding_start = (time.perf_counter())

        X_test_embeddings, empty_test = (
            encode_documents_tfidf(
                X_test, vectors, idf,
                embedding_dim=EMBEDDING_DIMENSION,
            )
        )

        test_encoding_time = (time.perf_counter()
            - test_encoding_start
        )

        print(f"Test embedding shape: {X_test_embeddings.shape}")
        print(f"Empty test documents: {empty_test}")
        print(f"Test encoding time: {test_encoding_time:.4f} s")

        # -------------------------------------------------
        # Train LinearSVC
        # -------------------------------------------------

        classifier = LinearSVC(max_iter=2500)
        train_start = time.perf_counter()

        classifier.fit(X_train_embeddings, y_train)
        classifier_train_time = (time.perf_counter() - train_start)

        # -------------------------------------------------
        # Inference
        # -------------------------------------------------

        inference_start = (time.perf_counter())
        predictions = classifier.predict(X_test_embeddings)
        inference_time = (time.perf_counter() - inference_start)

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        from sklearn.metrics import (accuracy_score, f1_score)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")

        print()
        print(f"Accuracy:              {accuracy:.4f}")
        print(f"Macro F1:              {macro_f1:.4f}")
        print(f"Classifier train time: {classifier_train_time:.4f} s")
        print(f"Inference time:        {inference_time:.4f} s")

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------

        results.append(
            {
                "Experiment": EXPERIMENT,
                "Train size": train_size,

                "Representation": ("Word Embeddings"),
                "Representation details": (
                    "GloVe 6B 100d + "
                    "TF-IDF Weighted Mean"
                ),
                "Preprocessor": "None",
                "Variant": ("TF-IDF Weighted Mean"),

                "Classifier": "LinearSVC",

                "Accuracy": accuracy,
                "Macro F1": macro_f1,

                "IDF computation time (s)": (idf_time),
                "Train encoding time (s)": (train_encoding_time),
                "Test encoding time (s)": (test_encoding_time),
                "GloVe load time (s)": (glove_load_time),
                "Classifier train time (s)": (classifier_train_time),
                "Inference time (s)": (inference_time),

                "y_true": y_test,
                "y_pred": predictions,
            }
        )

    # -------------------------------------------------
    # Results summary
    # -------------------------------------------------

    results_df = pd.DataFrame(results)

    print()
    print("=" * 80)
    print("V7.2 RESULTS")
    print("=" * 80)

    print(
        results_df[
            [
                "Experiment",
                "Train size",
                "Representation",
                "Representation details",
                "Classifier",
                "Accuracy",
                "Macro F1",
                "IDF computation time (s)",
                "Train encoding time (s)",
                "Test encoding time (s)",
                "Classifier train time (s)",
                "Inference time (s)",
            ]
        ].round(4)
    )

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    csv_columns = [
        column
        for column in results_df.columns
        if column not in [
            "y_true",
            "y_pred",
        ]
    ]

    results_df[csv_columns].to_csv(
        "results_V7_2_embeddings.csv",
        index=False,
    )

    print()
    print("Saved: results_V7_2_embeddings.csv")

    # -------------------------------------------------
    # Confusion matrices
    # -------------------------------------------------

    plot_v6_confusion_matrices(results_df)

    # -------------------------------------------------
    # Performance evolution
    # -------------------------------------------------

    plot_embedding_performance(results_df)


if __name__ == "__main__":
    main()

"""V7.2 est moins bon que V7.1 à tous les niveaux sur AGNews,
avec un écart particulièrement marqué à 50/200/500,
mais il se rapproche à 2 000 et 10 000."""
