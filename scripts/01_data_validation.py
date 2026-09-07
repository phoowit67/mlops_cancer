import mlflow
from sklearn.datasets import load_breast_cancer


def validate_data():
    """Load and validate the breast cancer dataset."""

    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")

        mlflow.set_tag("ml.step", "data_validation")

        # Load data
        cancer_data = load_breast_cancer(as_frame=True)
        df = cancer_data.frame

        # Explore dataset
        num_rows, num_cols = df.shape
        num_classes = df["target"].nunique()
        missing_values = df.isnull().sum().sum()
        class_balance = df["target"].value_counts(normalize=True).min()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Target names: {cancer_data.target_names}")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"Minimum class balance: {class_balance:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("class_balance", class_balance)

        mlflow.log_param("num_classes", num_classes)

        # Validation rules
        validation_status = "Success"

        if (
            missing_values > 0
            or num_classes != 2
            or class_balance < 0.2
        ):
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)

        print(f"Validation status: {validation_status}")

        # Stop pipeline if validation fails
    if validation_status == "Failed":
        print("Validation failed.")
        raise SystemExit(1)

    print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()