import os
import importlib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import dagshub
from custom_logs import get_logger
logger = get_logger("training.log")

dagshub.init(repo_owner='Agam-Patel-DS', repo_name='kidney-disease-classification', mlflow=True)
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Agam-Patel-DS/kidney-disease-classification.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="Agam-Patel-DS"
os.environ["MLFLOW_TRACKING_PASSWORD"]="3d17a240490c70ca0d48f408ad47e0542b365a0a"

class TrainingPipeline:
    """
    Pipeline for training regression models with hyperparameter tuning, evaluation, and logging experiments to MLflow.
    """
    def __init__(self, config):
        self.config = config
        # Set up MLflow
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

    def load_data(self):
        """
        Load training and testing data from CSV files.
        """
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        # Split features and target
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        
        return X_train, X_test, y_train, y_test

    def train_and_log(self):
        """
        Train models, tune hyperparameters, evaluate performance, and log results to MLflow.
        """
        X_train, X_test, y_train, y_test = self.load_data()

        best_model_scores = []

        for model_name, model_info in self.config.models.items():
            model_class_path = model_info["model_class"]
            param_grid = model_info["params"]
            
            # Dynamically import the model class
            module_name, class_name = model_class_path.rsplit(".", 1)
            model_class = getattr(importlib.import_module(module_name), class_name)
            
            # Initialize the model and GridSearchCV
            model = model_class()
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
            
            # Train and perform hyperparameter tuning
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate the model
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save the best model
            model_save_path = os.path.join(self.config.output_model_path, f"{model_name}_best_model.pkl")
            joblib.dump(best_model, model_save_path)
            
            # Log to MLflow
            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("model_name", model_name)
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                mlflow.log_artifact(model_save_path)  # Log the model file
                
            # Collect the score for summary
            best_model_scores.append((model_name, grid_search.best_params_, mse, r2, model_save_path))
            
            logger.info(f"Model: {model_name}, MSE: {mse}, R2: {r2}")
        
        # Log the summary of all models
        summary_path = os.path.join(self.config.output_model_path, "model_summary.csv")
        pd.DataFrame(best_model_scores, columns=["Model", "Best Params", "MSE", "R2", "Model Path"]).to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)
        logger.info(f"Summary saved at: {summary_path}")
