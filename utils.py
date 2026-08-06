#!/usr/bin/python3

from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
import os
from io import StringIO

import pandas as pd


def load_csv(path: str) -> pd.DataFrame | None:  # type: ignore
    """
    Load a CSV dataset from the given path.

    The function checks that the path is valid,
    that the file exists and that it has a .csv extension.
    If an error occurs, the function returns None.

    Args:
        path (str): Path to the CSV file.

    Returns:
        The loaded dataset if successful, otherwise None.
    """
    if not isinstance(path, str):
        print("Error: path must be a string.")
        return None
    if not os.path.isfile(path):
        print(f"Error: file '{path}' does not exist.")
        return None
    if not path.lower().endswith(".csv"):
        print("Error: bad format, file is not a CSV.")
        return None

    try:
        return pd.read_csv(path)

    except Exception as e:
        print(f"Error: unable to read the file as a CSV dataset. {e}")
        return None


def load_url(url: str) -> pd.DataFrame | None:  # type: ignore
    """
    Load a CSV dataset from a URL.

    Args:
        url (str): URL to the CSV file.

    Returns:
        The loaded dataset if successful, otherwise None.
    """
    if not isinstance(url, str):
        print("Error: URL must be a string.")
        return None

    try:
        with urlopen(url, timeout=10) as response:
            content = response.read().decode('utf-8')
            return pd.read_csv(StringIO(content))

    except (URLError, HTTPError) as e:
        print(f"Error: unable to download from URL. {e}")
        return None

    except Exception as e:
        print(f"Error: unable to read the downloaded file as CSV. {e}")
        return None


def load(path: str) -> pd.DataFrame | None:  # type: ignore
    """
    Load a CSV dataset from either a local path or a URL.

    Args:
        path (str): Local path or URL to the CSV file.

    Returns:
        The loaded dataset if successful, otherwise None.
    """
    try:
        parsed = urlparse(path)
        is_url = all([parsed.scheme, parsed.netloc])

    except Exception:
        is_url = False

    return load_url(path) if is_url else load_csv(path)


def load_winners(csv_path: str) -> pd.DataFrame:
    """
    Load winners selected during the previous experiment stage.
    """
    winners = load(csv_path)

    if winners is None:
        raise FileNotFoundError(
            f"Unable to load '{csv_path}'."
        )

    print()
    print("=" * 80)
    print("WINNERS LOADED")
    print("=" * 80)

    display_columns = [
        column
        for column in [
            "Train size",
            "Experiment",
            "Preprocessor",
            "Vectorizer",
            "Classifier",
            "Accuracy",
            "Macro F1",
        ]
        if column in winners.columns
    ]

    print(
        winners[display_columns]
    )

    return winners


def prepare(
    df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    Build the text corpus and extract class labels.

    ```
    Supported formats:

    AG News:
        - Title
        - Description
        - Class Index

    DBpedia:
        - title
        - content
        - label

    Pre-combined format:
        - Text
        - Class Index
    """

    if (
        "Text" in df.columns
        and "Class Index" in df.columns
    ):

        text = (
            df["Text"]
            .fillna("")
            .astype(str)
        )

        labels = df["Class Index"]

    elif (
        "Title" in df.columns
        and "Description" in df.columns
        and "Class Index" in df.columns
    ):

        text = (
            df["Title"]
            .fillna("")
            .astype(str)
            + " "
            + df["Description"]
            .fillna("")
            .astype(str)
        )

        labels = df["Class Index"]

    elif (
        "title" in df.columns
        and "content" in df.columns
        and "label" in df.columns
    ):

        text = (
            df["title"]
            .fillna("")
            .astype(str)
            + " "
            + df["content"]
            .fillna("")
            .astype(str)
        )

        labels = df["label"]

    else:

        raise ValueError(
            "Dataset format not recognized. "
            "Expected one of the following formats: "
            "'Text' + 'Class Index', "
            "'Title' + 'Description' + "
            "'Class Index', or "
            "'title' + 'content' + 'label'."
        )

    if not isinstance(text, pd.Series):

        raise TypeError(
            "Text must be a pandas Series."
        )

    if not isinstance(labels, pd.Series):

        raise TypeError(
            "Labels must be a pandas Series."
        )

    return text, labels
