"""Model quality tests for Breast Cancer dataset."""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_ACCURACY = 0.95
MIN_ROC_AUC = 0.98


X, y = load_breast_cancer(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=10.0,
                max_iter=10000,
                random_state=42,
            ),
        ),
    ]
)

pipe.fit(X_train, y_train)


def test_accuracy_gate():
    """Model accuracy must meet the minimum threshold."""
    predictions = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= MIN_ACCURACY, (
        f"accuracy {accuracy:.4f} is below {MIN_ACCURACY}"
    )


def test_roc_auc_gate():
    """Model ROC-AUC must meet the minimum threshold."""
    probabilities = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)

    assert roc_auc >= MIN_ROC_AUC, (
        f"ROC-AUC {roc_auc:.4f} is below {MIN_ROC_AUC}"
    )


def test_pipeline_has_scaler():
    """Preprocessing must be included in the model pipeline."""
    assert "scaler" in pipe.named_steps


def test_predict_shape():
    """Prediction output must have the correct shape."""
    predictions = pipe.predict(X_test.head(5))
    assert predictions.shape == (5,)