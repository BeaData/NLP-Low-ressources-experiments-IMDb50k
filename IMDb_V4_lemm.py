#!/usr/bin/python3

"""
V4 - Lemmatization experiment.

For every winning configuration of V3, compare:
- Baseline (the V3 winning preprocessor, if any)
- WordNet lemmatizer (NLTK)
- spaCy lemmatizer

Each unique configuration is evaluated on every training size
so that performance trajectories are complete.

The official winners CSV keeps only the winners of the duels
defined line by line by winners_V3.csv.
"""

from functools import lru_cache

import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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


# ---------------------------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------------------------


def _ensure_nltk_resources():
    """Download NLTK data if not already present."""

    import nltk

    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "taggers/averaged_perceptron_tagger_eng":
        "averaged_perceptron_tagger_eng",
    }

    for find_path, download_name in resources.items():
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


# ---------------------------------------------------------------------------
# NLTK WordNet lemmatizer
# ---------------------------------------------------------------------------


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
            lemmatizer.lemmatize(
                word,
                wordnet_pos(pos),
            )
            for word, pos in tagged
        )

    return preprocess


# ---------------------------------------------------------------------------
# spaCy lemmatizer
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Previous-stage preprocessor reconstruction (stemming baseline)
# ---------------------------------------------------------------------------


def make_stemmer_preprocessor(stemmer_name: str, remove_stopwords: bool):
    """
    Rebuild the Porter or Snowball stemmer used in V3.
    """

    if stemmer_name == "porter":
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()

    elif stemmer_name == "snowball":
        from nltk.stem.snowball import SnowballStemmer

        stemmer = SnowballStemmer("english")

    else:
        raise ValueError(
            f"Unknown stemmer: {stemmer_name}"
        )

    def preprocess(text):
        words = text.split()

        if remove_stopwords:
            words = _remove_stopwords(words)

        return " ".join(stemmer.stem(word) for word in words)

    return preprocess


def make_previous_preprocessor(preprocessor_name: str, remove_stopwords: bool):
    """
    Rebuild the preprocessor of the V3 winning configuration.
    """

    name = clean_preprocessor(preprocessor_name)

    if name == "Porter":
        return make_stemmer_preprocessor("porter", remove_stopwords)

    if name == "Snowball":
        return make_stemmer_preprocessor("snowball", remove_stopwords)

    return None


# ---------------------------------------------------------------------------
# N-gram inference
# ---------------------------------------------------------------------------


def get_ngram_range(vectorizer_name: str):
    """Infer ngram range from the vectorizer name."""

    if "Bigrams" in vectorizer_name:
        return (1, 2)

    return (1, 1)


# ---------------------------------------------------------------------------
# Challenge factory
# ---------------------------------------------------------------------------


def create_lemmatization_challenges(row, train_size):
    """
    Create the baseline, WordNet, and spaCy experiments
    for one V3 winning configuration.
    """

    vectorizer = row["Vectorizer"]
    classifier = row["Classifier"]

    remove_stopwords = "StopWords" in vectorizer
    ngram_range = get_ngram_range(vectorizer)

    previous_preprocessor_name = clean_preprocessor(
        row.get("Preprocessor", "None")
    )

    previous_preprocessor = make_previous_preprocessor(
        previous_preprocessor_name,
        remove_stopwords,
    )

    return [
        # Baseline: the V3 winning configuration as-is.
        make_experiment(
            train_size=train_size,
            vectorizer_name=vectorizer,
            classifier_name=classifier,
            variant="Baseline",
            preprocessor_name=previous_preprocessor_name,
            preprocessor=previous_preprocessor,
            ngram_range=ngram_range,
        ),

        # Challenger: NLTK WordNet lemmatization.
        make_experiment(
            train_size=train_size,
            vectorizer_name=vectorizer,
            classifier_name=classifier,
            variant="WordNet",
            preprocessor_name="WordNet",
            preprocessor=make_wordnet_preprocessor(
                remove_stopwords,
            ),
            ngram_range=ngram_range,
        ),

        # Challenger: spaCy lemmatization.
        make_experiment(
            train_size=train_size,
            vectorizer_name=vectorizer,
            classifier_name=classifier,
            variant="spaCy",
            preprocessor_name="spaCy",
            preprocessor=make_spacy_preprocessor(
                remove_stopwords,
            ),
            ngram_range=ngram_range,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    V4 main entry point.
    """

    # Load the official V3 winners.
    previous_winners = load_winners("winners_V3.csv")

    # Generate the full superset: every unique V3 configuration
    # on every training size, with baseline / WordNet / spaCy.
    configs = generate_challenge_configs(
        previous_winners_df=previous_winners,
        challenge_factory=create_lemmatization_challenges,
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
        experiment_name="V4",
    )

    # ------------------------------------------------------------------
    # Official V4 winners: one per V3 duel line.
    # ------------------------------------------------------------------

    comparison_winners_df = select_duel_winners(
        results_df=results_df,
        previous_winners_df=previous_winners,
        challenge_factory=create_lemmatization_challenges,
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
        path="results_V4.csv",
    )

    if not comparison_winners_df.empty:
        save_stage_csv(
            comparison_winners_df,
            columns=DISPLAY_COLUMNS,
            path="winners_V4.csv",
        )


if __name__ == "__main__":
    main()
