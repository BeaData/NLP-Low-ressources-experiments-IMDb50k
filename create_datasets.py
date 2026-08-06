#!/usr/bin/python3

from pathlib import Path
from utils import prepare
import pandas as pd


def load_imdb_directory(directory):
    """
    Load the IMDb dataset from a train/ or test/ directory.

    Returns
    -------
    pandas.DataFrame

    Columns
    -------
    Text
    Class Index

    Labels
    ------
    0 : Negative
    1 : Positive
    """

    rows = []

    label_mapping = {
        "neg": 0,
        "pos": 1,
    }

    directory = Path(directory)

    for sentiment, label in label_mapping.items():
        sentiment_dir = directory / sentiment

        for file in sorted(sentiment_dir.glob("*.txt")):
            text = file.read_text(encoding="utf-8")
            rows.append(
                {
                    "Text": text,
                    "Class Index": label,
                }
            )

    df = pd.DataFrame(rows)
    return df


def create_nested_datasets(
    df: pd.DataFrame,
    output_prefix: str,
    sizes: list[int],
    class_column: str
) -> None:
    """
    Create nested and approximately balanced datasets.

    Each class is shuffled once using a fixed random seed.
    Larger datasets contain all observations from smaller
    datasets.

    Example:
        50 ⊂ 200 ⊂ 500 ⊂ 2000 ⊂ 10000

    When a requested size is not divisible by the number
    of classes, the remaining observations are distributed
    across the first classes. The difference between class
    sizes is therefore at most one observation.
    """

    groups = {}

    # Shuffle each class once
    for label in sorted(df[class_column].unique()):

        groups[label] = (
            df[df[class_column] == label]
            .sample(
                frac=1,
                random_state=42
            )
            .reset_index(drop=True)
        )

    n_classes = len(groups)

    print(f"Number of classes: {n_classes}")
    print()

    # Check that every class contains enough observations
    largest_size = max(sizes)

    base = largest_size // n_classes
    remainder = largest_size % n_classes

    maximum_per_class = (
        base + 1
        if remainder > 0
        else base
    )

    for label, group in groups.items():

        if len(group) < maximum_per_class:

            raise ValueError(
                f"Class {label} contains only "
                f"{len(group)} observations, but "
                f"{maximum_per_class} are required."
            )

    for size in sizes:

        base = size // n_classes
        remainder = size % n_classes

        parts = []

        for i, label in enumerate(sorted(groups)):

            n_samples = base

            if i < remainder:
                n_samples += 1

            parts.append(
                groups[label].iloc[:n_samples]
            )

        subset = pd.concat(
            parts,
            ignore_index=True
        )

        # Shuffle the final dataset so that classes are mixed
        subset = (
            subset
            .sample(
                frac=1,
                random_state=42
            )
            .reset_index(drop=True)
        )

        filename = f"{output_prefix}_{size}.csv"

        subset.to_csv(
            filename,
            index=False
        )

        class_counts = (
            subset[class_column]
            .value_counts()
            .sort_index()
        )

        print(
            f"{filename:<28} "
            f"{len(subset):>6} articles"
        )

        print(
            f"Class distribution: "
            f"{class_counts.to_dict()}"
        )

        print()


def create_balanced_test_dataset(
    df: pd.DataFrame,
    output_filename: str,
    class_column: str,
    samples_per_class: int
) -> None:
    """
    Create a balanced test dataset.

    The same number of observations is sampled from
    every class.
    """

    parts = []

    for label in sorted(df[class_column].unique()):

        class_data = df[
            df[class_column] == label
        ]

        if len(class_data) < samples_per_class:

            raise ValueError(
                f"Class {label} contains only "
                f"{len(class_data)} observations, but "
                f"{samples_per_class} are required."
            )

        sampled_class = class_data.sample(
            n=samples_per_class,
            random_state=42
        )

        parts.append(sampled_class)

    test_subset = pd.concat(
        parts,
        ignore_index=True
    )

    # Shuffle the final test dataset
    test_subset = (
        test_subset
        .sample(
            frac=1,
            random_state=42
        )
        .reset_index(drop=True)
    )

    text, labels = prepare(test_subset)

    df_test = pd.DataFrame(
        {
            "Text": text,
            "Class Index": labels
        }
    )

    df_test.to_csv(
        output_filename,
        index=False
    )

    class_counts = (
        df_test["Class Index"]
        .value_counts()
        .sort_index()
    )

    print(
        f"{output_filename:<28} "
        f"{len(df_test):>6} articles"
    )

    print(
        f"Class distribution: "
        f"{class_counts.to_dict()}"
    )


def main():
    """
    Create nested IMDb training datasets and a
    balanced IMDb test dataset.

    Training sizes:
        50
        200
        500
        2000
        10000

    Test size:
        3800 observations
        1900 per class
    """

    train_sizes = [
        50,
        200,
        500,
        2000,
        10000,
    ]

    print("Creating nested IMDb training datasets")
    print()

    train = load_imdb_directory("train")

    create_nested_datasets(
        df=train,
        output_prefix="imdb_train",
        sizes=train_sizes,
        class_column="Class Index",
    )

    print()
    print("Creating balanced IMDb test dataset")
    print()

    test = load_imdb_directory("test")

    create_balanced_test_dataset(
        df=test,
        output_filename="imdb_test.csv",
        class_column="Class Index",
        samples_per_class=1900,
    )


if __name__ == "__main__":
    main()
