# train.py
"""
Standalone training script for RedShield AI: Phoenix models.

This script is the entry point for the MLOps re-training pipeline.
It handles:
- Loading configuration via Hydra.
- Setting up an MLflow experiment run for tracking.
- Loading and preparing training data (simulated for this example).
- Instantiating and training a model (e.g., TCNN).
- Evaluating the model and logging metrics.
- Logging the trained model artifact to the MLflow registry.

To run:
`poetry run python train.py`
"""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import numpy as np

# Import the model definition from the core application logic
from core import TCNN

# --- System Setup ---
logger = logging.getLogger(__name__)

# --- Dummy Data Generation for Demonstration ---
def get_dummy_data(tcnn_params, num_samples=1000):
    """
    Generates dummy time-series data for training the TCNN.
    In a real MLOps pipeline, this function would be replaced with a call
    to a feature store or data lake.
    """
    input_seq_len = 168 # e.g., one week of hourly data
    X = torch.randn(num_samples, tcnn_params['input_size'], input_seq_len)
    y = torch.randn(num_samples, tcnn_params['output_size'])
    return X, y


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_training(config: DictConfig):
    """
    Executes the model training and logging pipeline.
    """
    # Convert Hydra config to a standard dictionary for easy use
    config_dict = OmegaConf.to_container(config, resolve=True)
    tcnn_params = config_dict.get('tcnn_params', {})
    model_name = config_dict.get("ml_models", {}).get("tcnn_name", "phoenix_tcnn")
    
    logger.info(f"Starting training run for model: {model_name}")
    logger.info(f"Model parameters: {tcnn_params}")

    # --- 1. MLflow Experiment Setup ---
    # Set the experiment name. If it doesn't exist, MLflow creates it.
    mlflow.set_experiment("Phoenix Model Training")
    
    with mlflow.start_run() as run:
        logger.info(f"MLflow run started. Run ID: {run.info.run_id}")
        
        # --- 2. Log Parameters ---
        # Log all TCNN parameters for full reproducibility.
        mlflow.log_params(tcnn_params)
        mlflow.log_param("training_data_samples", 1000) # Example of logging a data param
        
        # --- 3. Load Data ---
        # In a real pipeline, this would connect to a data source.
        # Here, we generate dummy data for demonstration.
        X_train, y_train = get_dummy_data(tcnn_params, 1000)
        X_val, y_val = get_dummy_data(tcnn_params, 200)
        
        # --- 4. Instantiate and Train Model ---
        model = TCNN(**tcnn_params)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        logger.info("Starting model training loop...")
        # Simplified training loop for demonstration
        for epoch in range(5): # Train for a few epochs
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch+1}, Training Loss: {loss.item():.4f}")
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # --- 5. Evaluate Model ---
        logger.info("Evaluating model on validation set...")
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        logger.info(f"Final Validation Loss: {val_loss.item():.4f}")
        
        # --- 6. Log Metrics ---
        # Log the final validation metric, which is crucial for model comparison.
        mlflow.log_metric("validation_loss", val_loss.item())
        
        # --- 7. Log Model Artifact to Registry ---
        logger.info("Logging model to MLflow Model Registry...")
        # This logs the model, its dependencies (conda.yaml), and a loader script.
        # It's registered under the name specified in the config.
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="tcnn_model", # Subfolder within the MLflow run
            registered_model_name=model_name
        )
        logger.info(f"Successfully logged and registered model as '{model_name}'.")

if __name__ == "__main__":
    run_training()
