import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
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

try:
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
dataset_dir = os.path.join(current_dir, "student_performance_preprocessing")

X_train = pd.read_csv(os.path.join(dataset_dir, "X_train.csv"))
y_train = pd.read_csv(
    os.path.join(dataset_dir, "y_train.csv")
).values.ravel()

X_test = pd.read_csv(os.path.join(dataset_dir, "X_test.csv"))
y_test = pd.read_csv(
    os.path.join(dataset_dir, "y_test.csv")
).values.ravel()

# ==========================================
# MLflow Run
# ==========================================

with mlflow.start_run(run_name="LinearRegression_ManualLogging"):

    print("Memulai pelatihan model...")

    # ======================================
    # Training
    # ======================================

    lr_model = LinearRegression()

    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    # ======================================
    # Evaluation
    # ======================================

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # ======================================
    # Parameters
    # ======================================

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("fit_intercept", lr_model.fit_intercept)
    mlflow.log_param("positive", lr_model.positive)

    # ======================================
    # Metrics
    # ======================================

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)

    # ======================================
    # Save Model
    # ======================================

    input_contoh = X_train.head(5) 

    mlflow.sklearn.log_model(
        sk_model=lr_model,
        artifact_path="model",
        input_example=input_contoh 
    )

    # ======================================
    # Artifact 1
    # Actual vs Predicted
    # ======================================

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")

    actual_pred_plot = "actual_vs_predicted.png"

    plt.savefig(actual_pred_plot)
    plt.close()

    mlflow.log_artifact(actual_pred_plot)

    # ======================================
    # Artifact 2
    # Residual Plot
    # ======================================

    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    residual_plot = "residual_plot.png"

    plt.savefig(residual_plot)
    plt.close()

    mlflow.log_artifact(residual_plot)

    # ======================================
    # Artifact 3
    # Evaluation Report
    # ======================================

    report_file = "evaluation_report.txt"

    with open(report_file, "w") as f:
        f.write("=== MODEL EVALUATION REPORT ===\n\n")
        f.write(f"MSE  : {mse}\n")
        f.write(f"RMSE : {rmse}\n")
        f.write(f"MAE  : {mae}\n")
        f.write(f"R2   : {r2}\n")
        f.write(f"MAPE : {mape}\n")

    mlflow.log_artifact(report_file)

    # ======================================
    # Console Output
    # ======================================

    print("\nPelatihan selesai!")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.4f} ({mape*100:.2f}%)")

print("Run ended successfully")
