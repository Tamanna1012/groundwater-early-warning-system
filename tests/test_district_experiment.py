"""Tests for src/models/district_experiment.py.

The experiment's conclusion only means anything if its arms are fair, and
the one place that could quietly go wrong is the cardinality cap: it has to
decide which districts to keep using the training split alone. If test rows
leaked into that decision, the `capped` arm would be reading the future and
the comparison against `current` would be worthless.
"""

import pandas as pd

from district_experiment import MAX_CATEGORIES, WORST_KEYS, cap_categories


def test_cap_categories_keeps_the_most_common_training_districts():
    train = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 2)
    test = pd.Series(["A", "B", "C"])

    capped_train, capped_test = cap_categories(train, test)

    # Nothing is dropped here: 3 distinct values is well under the cap.
    assert set(capped_train) == {"A", "B", "C"}
    assert list(capped_test) == ["A", "B", "C"]


def test_cap_categories_folds_the_rarest_districts_into_one_bucket():
    # One more distinct district than the model can take as a categorical.
    train = pd.Series(sum(([f"D{i}"] * (MAX_CATEGORIES + 1 - i) for i in range(MAX_CATEGORIES + 1)), []))
    test = pd.Series([f"D{i}" for i in range(MAX_CATEGORIES + 1)])

    capped_train, capped_test = cap_categories(train, test)

    assert capped_train.nunique() <= MAX_CATEGORIES, "must fit inside HistGradientBoosting's limit"
    assert "OTHER" in set(capped_train)
    # D0 is the most frequent and must survive; the least frequent must not.
    assert "D0" in set(capped_train)
    assert f"D{MAX_CATEGORIES}" not in set(capped_train)


def test_cap_categories_decides_using_training_rows_only():
    """A district that appears only in test must become OTHER. If the keep
    set were built from both splits, the arm would be using test data to
    choose its own feature encoding."""
    train = pd.Series(["A"] * 10 + ["B"] * 5)
    test = pd.Series(["A", "B", "UNSEEN"])

    _, capped_test = cap_categories(train, test)

    assert list(capped_test) == ["A", "B", "OTHER"]


def test_cap_categories_does_not_let_a_test_only_district_displace_a_trained_one():
    """Even if the unseen district is overwhelmingly frequent in test, it
    cannot enter the keep set or push anything out of it."""
    train = pd.Series(["A"] * 3 + ["B"] * 2)
    test = pd.Series(["FLOOD"] * 1000 + ["A", "B"])

    capped_train, capped_test = cap_categories(train, test)

    assert set(capped_train) == {"A", "B"}
    assert set(capped_test) == {"OTHER", "A", "B"}
    assert "FLOOD" not in set(capped_test)


def test_worst_district_keys_are_state_qualified():
    """District names repeat across states, so every grouping in this
    experiment keys on STATE/District. A bare name here would silently merge
    two different districts."""
    assert "RJ/Sikar" in WORST_KEYS
    assert "Sikar" not in WORST_KEYS
    assert all("/" in key for key in WORST_KEYS)
    assert len(WORST_KEYS) == 10
