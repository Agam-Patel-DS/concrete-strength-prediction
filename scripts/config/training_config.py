from utils.common import read_yaml

params=read_yaml("configs/config.yaml")

import os

class TrainingConfig:
    """
    Configuration class for regression model training.
    """
    def __init__(self):
        # Define the models and hyperparameters for tuning
        self.models = {
            "RandomForest": {
                "model_class": "sklearn.ensemble.RandomForestRegressor",
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }
            },
            "GradientBoosting": {
                "model_class": "sklearn.ensemble.GradientBoostingRegressor",
                "params": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10]
                }
            }
        }
        # Specify paths for train/test data and saving models
        self.train_data_path = params["training"]["train_data_path"]
        self.test_data_path = params["training"]["test_data_path"]
        self.output_model_path = params["training"]["output_model_path"]
        
        # Ensure the model output directory exists
        os.makedirs(self.output_model_path, exist_ok=True)
        
        # MLflow logging configuration
        self.mlflow_tracking_uri = params["training"]["mlflow_tracking_uri"] # Replace with your DagsHub repo
        self.mlflow_experiment_name = params["training"]["mlflow_experiment_name"]
