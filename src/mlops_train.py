import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import joblib
import mlflow
from urllib.parse import urlparse
from get_data import read_params
import json

# Evaluate model metrics
def eval_metrics(actual, predicted):
    # Compute Mean Squared Error and take the square root to get RMSE
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    # Calculate MAE and R^2
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    return rmse, mae, r2

# Train and evaluate model
def train_and_evaluate(config_path):
    config = read_params(config_path)

    # Extract configurations from the YAML file
    model_path = config["model_path"]
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    random_state = config["base"]["random_state"]
    target = config["base"]["target_col"]

    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]

    # Read training and testing datasets
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_features = train_data.drop(target, axis=1)
    test_features = test_data.drop(target, axis=1)

    train_target = pd.DataFrame(train_data[target])
    test_target = pd.DataFrame(test_data[target])

    # MLflow tracking configuration
    mlflow_config = config["mlflow_config"]
    mlflow.set_tracking_uri(mlflow_config["remote_server_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # Start the MLflow run
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        
        # Initialize and train the ElasticNet model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        lr.fit(train_features, train_target)

        # Make predictions
        predicted = lr.predict(test_features)

        # Evaluate metrics
        rmse, mae, r2 = eval_metrics(test_target, predicted)

        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Log model to MLflow
        tracking_uri_type = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_uri_type != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.log_model(lr, "model")

        # Save the model locally
        os.makedirs(model_path, exist_ok=True)
        model_file_path = os.path.join(model_path, "model.joblib")
        joblib.dump(lr, model_file_path)

        # Optionally save evaluation metrics and parameters
        # Save evaluation metrics to a JSON file (uncomment if desired)
        # scores = {"rmse": rmse, "mae": mae, "r2": r2}
        # with open(config["reports"]["score"], 'w') as scores_file:
        #     json.dump(scores, scores_file)

        # Save model parameters to a JSON file (uncomment if desired)
        # params = {"alpha": alpha, "l1_ratio": l1_ratio}
        # with open(config["reports"]["params"], 'w') as params_file:
        #     json.dump(params, params_file)

# Main function
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml", help="Path to the config file")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
