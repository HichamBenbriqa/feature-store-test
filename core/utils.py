"""
AWS SageMaker Feature Store Utility Functions

This module provides utility functions for working with AWS SageMaker Feature Store,
S3, and machine learning model operations. It includes functionality for monitoring
Feature Store operations, managing data in S3, feature engineering, and model training.

The module provides utilities for:
- Logging
- Feature group creation monitoring
- Offline store data availability checking
- Feature value extraction
- Feature engineering from historical data
- Model training and persistence

Note:
    These utilities are designed to work with AWS services and require appropriate
    AWS credentials and permissions to function correctly.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import pickle
from time import sleep
import pandas as pd
import boto3
from sklearn.linear_model import LinearRegression
from core import constants


def setup_logger(name: str, log_level=logging.DEBUG) -> logging.Logger:
    """
    Create a logger with both file and console handlers.
    Args:
        name (str): Name of the logger, typically __name__ from the calling module
        log_level: Logging level, defaults to DEBUG
    Returns:
        logging.Logger: Configured logger instance
    """
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = logging.getLogger(name)

    logger.handlers.clear()

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=f"{logs_dir}/{name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


def wait_for_feature_group_creation_complete(feature_group, logger):
    """
    Monitor the creation status of a SageMaker Feature Group until completion.

    Args:
        feature_group: SageMaker Feature Group instance to monitor

    Raises:
        RuntimeError: If feature group creation fails

    Note:
        This function polls the feature group status every 5 seconds until
        creation is complete or fails.
    """
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        logger.info("Waiting for Feature Group Creation")
        sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")

    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    logger.info(f"FeatureGroup {feature_group.name} successfully created.")


def wait_for_offline_data_to_be_available(
    feature_store_manager, s3_bucket_name, logger
):
    """
    Monitor S3 bucket until offline store data becomes available.

    Args:
        feature_store_manager: Instance of FeatureStoreManager
        s3_bucket_name (str): Name of the S3 bucket

    Note:
        This function polls the S3 bucket every 60 seconds until
        the offline store data is available.
    """
    access_key = os.getenv("AWS_ACCESS_KEY")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=aws_region,
    )
    feature_group_resolved_output_s3_uri = (
        feature_store_manager.feature_group.describe()
        .get("OfflineStoreConfig")
        .get("S3StorageConfig")
        .get("ResolvedOutputS3Uri")
    )
    feature_group_s3_prefix = feature_group_resolved_output_s3_uri.replace(
        f"s3://{s3_bucket_name}/", ""
    )
    offline_store_contents = None

    while offline_store_contents is None:
        objects_in_bucket = s3_client.list_objects(
            Bucket=s3_bucket_name, Prefix=feature_group_s3_prefix
        )
        if "Contents" in objects_in_bucket and len(objects_in_bucket["Contents"]) > 1:
            offline_store_contents = objects_in_bucket["Contents"]
        else:
            logger.info("Waiting for data in offline store...\n")
            sleep(60)
    logger.info("Data available.")


def get_feature_value(record, feature_name):
    """
    Extract a specific feature value from a Feature Store record.

    Args:
        record (list): List of feature records from Feature Store
        feature_name (str): Name of the feature to extract

    Returns:
        str: Value of the requested feature

    Note:
        The function assumes the feature exists in the record. If the feature
        doesn't exist, it will raise an IndexError.
    """
    return str(
        list(filter(lambda r: r["FeatureName"] == feature_name, record))[0][
            "ValueAsString"
        ]
    )


def engineer_features(historical_df):
    """
    Calculate and engineer features from historical customer data.

    Args:
        historical_df (pandas.DataFrame): DataFrame containing historical customer data
            with columns:
            - customer_id
            - purchase_timestamp
            - purchase_value
            - loyalty_score

    Returns:
        pandas.DataFrame: Engineered features including:
            - customer_id
            - purchase_timestamp (formatted)
            - latest_purchase_value
            - avg_purchase_value
            - avg_loyalty_score
            - latest_loyalty_score

    Note:
        This function performs several transformations:
        1. Converts timestamps to datetime format
        2. Groups data by customer_id
        3. Calculates latest and average values for purchases and loyalty scores
        4. Formats timestamps for Feature Store compatibility
    """
    historical_df["purchase_timestamp"] = pd.to_datetime(
        historical_df["purchase_timestamp"]
    )

    customer_features = (
        historical_df.groupby("customer_id")
        .agg(
            {
                "purchase_timestamp": "max",
                "purchase_value": [
                    "last",
                    "mean",
                ],
                "loyalty_score": [
                    "last",
                    "mean",
                ],
            }
        )
        .reset_index()
    )

    customer_features.columns = [
        "customer_id",
        "purchase_timestamp",
        "latest_purchase_value",
        "avg_purchase_value",
        "avg_loyalty_score",
        "latest_loyalty_score",
    ]

    customer_features["purchase_timestamp"] = customer_features[
        "purchase_timestamp"
    ].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    return customer_features


def train_model(X, y):
    """
    Train a linear regression model for loyalty prediction.

    Args:
        X (numpy.ndarray): Feature matrix for training
        y (numpy.ndarray): Target values (loyalty scores)

    Returns:
        sklearn.linear_model.LinearRegression: Trained linear regression model

    Note:
        The model is automatically saved to 'models/loyalty_predictor.pkl'
        after training.

    Raises:
        FileNotFoundError: If the models directory doesn't exist
        PermissionError: If there are insufficient permissions to save the model
    """
    model = LinearRegression()
    model.fit(X, y)

    with open(constants.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model
