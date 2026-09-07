"""Data tests for Breast Cancer dataset."""

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame


def test_schema():
    """Dataset must contain 30 features plus target."""
    assert df.shape[1] == 31
    assert "target" in df.columns


def test_no_missing():
    """Dataset must not contain missing values."""
    assert df.isnull().sum().sum() == 0


def test_two_classes():
    """Breast Cancer dataset must contain 2 classes."""
    assert df["target"].nunique() == 2


def test_target_values():
    """Target values must be 0 and 1."""
    assert set(df["target"].unique()) == {0, 1}