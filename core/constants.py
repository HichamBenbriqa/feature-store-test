"""
Customer Loyalty Prediction System Constants

This module defines the configuration constants used throughout the customer loyalty
prediction system. It includes file paths, feature definitions, and model configurations
used by various components of the system.

Constants Categories:
    - Feature Store Configuration: Names and identifiers for Feature Store resources
    - Data File Paths: Locations of various data files used in the system
    - Model Configuration: Feature lists and model file locations
    - Training Configuration: Feature and target variable definitions

Note:
    All paths are relative to the project root directory.
    Ensure all referenced directories exist before running the system.

Usage:
    from core import constants

    # Access feature group name
    feature_group = constants.FEATURE_GROUP_NAME

    # Load data using defined paths
    raw_data = pd.read_csv(constants.RAW_DATA_PATH)

    # Get training features
    features = constants.TRAINING_FEATURES
"""

FEATURE_GROUP_NAME = "customer_purchase_features"

RAW_DATA_PATH = "data/test_task_data.csv"
HISTORICAL_DATA_PATH = "data/historical_data.csv"
INFERENCE_DATA_PATH = "data/inference_data.csv"

TRAINING_FEATURES = ["latest_purchase_value", "avg_purchase_value", "avg_loyalty_score"]
TARGET_VARIABLE = "latest_loyalty_score"

MODEL_PATH = "models/loyalty_predictor.pkl"
