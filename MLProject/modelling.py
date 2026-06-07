import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np

try:
    dagshub.init(repo_owner='Andrian206', repo_name='Eksperimen_SML_Rio-Andika-Andriansyah', mlflow=True)
    print("Dagshub initialized")
except Exception as e:
    print(f"Dagshub init error: {e}")

mlflow.set_experiment("Student_Performance_Base_Model")

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "student_performance_preprocessing")

X_train = pd.read_csv(os.path.join(dataset_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(dataset_dir, "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(dataset_dir, "X_test.csv"))
y_test = pd.read_csv(os.path.join(dataset_dir, "y_test.csv")).values.ravel()

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="LinearRegression_Autolog"):
    print("Memulai pelatihan model regresi linier...")
    
    mlflow.log_param("model_type", "LinearRegression")
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    y_pred = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    
    print(f"Pelatihan selesai!")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.4f} ({mape*100:.2f}%)")

print("Run ended successfully")