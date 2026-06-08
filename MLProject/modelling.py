import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

# ==========================================
# DagsHub Initialization
# ==========================================

if not os.getenv("CI"):
    try:
        import dagshub

        dagshub.init(
            repo_owner="Andrian206",
            repo_name="Eksperimen_SML_Rio-Andika-Andriansyah",
            mlflow=True
        )

        print("Dagshub initialized successfully")

    except Exception as e:
        print(f"Dagshub init error: {e}")

# ==========================================
# Load Dataset
# ==========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(
    current_dir,
    "student_performance_preprocessing"
)

X_train = pd.read_csv(
    os.path.join(dataset_dir, "X_train.csv")
)

y_train = pd.read_csv(
    os.path.join(dataset_dir, "y_train.csv")
).values.ravel()

X_test = pd.read_csv(
    os.path.join(dataset_dir, "X_test.csv")
)

y_test = pd.read_csv(
    os.path.join(dataset_dir, "y_test.csv")
).values.ravel()

# ==========================================
# Training
# ==========================================

with mlflow.start_run(
    run_name="LinearRegression_ManualLogging"
):

    print("Starting training...")

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ======================================
    # Metrics
    # ======================================

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(
        y_test,
        y_pred
    )

    # ======================================
    # Parameters
    # ======================================

    mlflow.log_param(
        "model_type",
        "LinearRegression"
    )

    mlflow.log_param(
        "fit_intercept",
        model.fit_intercept
    )

    mlflow.log_param(
        "positive",
        model.positive
    )

    # ======================================
    # Metrics Logging
    # ======================================

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)

    # ======================================
    # Model Logging
    # ======================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.head(5)
    )

    # ======================================
    # Artifact 1
    # ======================================

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")

    plot1 = "actual_vs_predicted.png"

    plt.savefig(plot1)
    plt.close()

    mlflow.log_artifact(plot1)

    # ======================================
    # Artifact 2
    # ======================================

    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals)

    plt.axhline(y=0)

    plt.xlabel("Predicted")
    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plot2 = "residual_plot.png"

    plt.savefig(plot2)
    plt.close()

    mlflow.log_artifact(plot2)

    # ======================================
    # Artifact 3
    # ======================================

    report_file = "evaluation_report.txt"

    with open(report_file, "w") as f:

        f.write(
            f"MSE : {mse}\n"
        )

        f.write(
            f"RMSE : {rmse}\n"
        )

        f.write(
            f"MAE : {mae}\n"
        )

        f.write(
            f"R2 : {r2}\n"
        )

        f.write(
            f"MAPE : {mape}\n"
        )

    mlflow.log_artifact(report_file)

    print("Training completed")

print("Run ended successfully")