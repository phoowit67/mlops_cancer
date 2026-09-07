import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    """Load the registered cancer model and make predictions."""

    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    print(
        f"Loading model '{MODEL_NAME}' "
        f"with alias '@{MODEL_ALIAS}'..."
    )

    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )

    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(
            f"Please make sure a model version has "
            f"the alias '@{MODEL_ALIAS}' in MLflow UI."
        )
        return

    # Load original dataset
    data = load_breast_cancer(
        as_frame=True
    )

    X = data.data
    y = data.target
    target_names = data.target_names

    # First sample of malignant (target 0)
    malignant_index = y[y == 0].index[0]

    # First sample of benign (target 1)
    benign_index = y[y == 1].index[0]

    sample_data = X.loc[
        [malignant_index, benign_index]
    ]

    actual_labels = y.loc[
        [malignant_index, benign_index]
    ]

    predictions = model.predict(sample_data)

    print("-" * 50)

    for i, prediction in enumerate(predictions):

        actual = actual_labels.iloc[i]

        actual_name = target_names[actual]
        predicted_name = target_names[int(prediction)]

        result = (
            "Correct"
            if actual == prediction
            else "Incorrect"
        )

        print(
            f"Actual: {actual_name:<10} "
            f"Predicted: {predicted_name:<10} "
            f"Result: {result}"
        )

    print("-" * 50)


if __name__ == "__main__":
    load_and_predict()