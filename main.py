"""
Customer Loyalty Prediction System - Main Entry Point

This module serves as the main entry point for the Customer Loyalty Prediction System.
It orchestrates the complete workflow from data preparation to model training and inference,
including AWS service setup and feature store operations.

The workflow includes:

1. Data preparation and feature engineering

2. Feature group creation and data ingestion

3. Model training with historical data

4. Real-time inference pipeline execution


Note:
    Ensure all required AWS credentials and permissions are properly configured
    before running this script.
"""

import os
import random
import boto3
import pandas as pd
from sagemaker.session import Session
# from dotenv import load_dotenv

from core import constants, utils
from core.feature_store_manager import FeatureStoreManager
from core.inference import RealTimeInference
import tests


# def setup_aws_services():
#     """
#     Initialize and configure AWS service clients and sessions.

#     Returns:
#         tuple: Contains:
#             - feature_store_session (Session): Configured SageMaker session
#             - feature_store_runtime: Feature store runtime client
#             - s3_bucket_name (str): Default S3 bucket name
#             - aws_role (str): AWS IAM role ARN
#             - aws_region (str): AWS region name

#     Note:
#         Requires appropriate AWS credentials and permissions to be configured.
#     """
#     load_dotenv()

#     aws_role = os.getenv("AWS_ROLE")
#     aws_region = boto3.Session().region_name

#     boto_session = boto3.Session(region_name=aws_region)
#     sagemaker_client = boto_session.client(
#         service_name="sagemaker", region_name=aws_region
#     )
#     feature_store_runtime = boto_session.client(  # pylint: disable=W0621
#         service_name="sagemaker-featurestore-runtime", region_name=aws_region
#     )
#     feature_store_session = Session(  # pylint: disable=W0621
#         boto_session=boto_session,
#         sagemaker_client=sagemaker_client,
#         sagemaker_featurestore_runtime_client=feature_store_runtime,
#     )

#     s3_bucket_name = feature_store_session.default_bucket()  # pylint: disable=W0621

#     return (
#         feature_store_session,
#         feature_store_runtime,
#         s3_bucket_name,
#         aws_role,
#         aws_region,
#     )


def setup_aws_services():
    """
    Initialize and configure AWS service clients and sessions.

    Returns:
        tuple: Contains:
            - feature_store_session (Session): Configured SageMaker session
            - feature_store_runtime: Feature store runtime client
            - s3_bucket_name (str): Default S3 bucket name
            - aws_role (str): AWS IAM role ARN
            - aws_region (str): AWS region name
    """
    aws_role = os.getenv("AWS_ROLE")
    access_key = os.getenv("AWS_ACCESS_KEY")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")

    if access_key and secret_key:
        boto_session = boto3.Session(
            aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
    else:
        boto_session = boto3.Session()

    sagemaker_client = boto_session.client(
        service_name="sagemaker", region_name=aws_region
    )

    feature_store_runtime = boto_session.client(  # pylint: disable=W0621
        service_name="sagemaker-featurestore-runtime", region_name=aws_region
    )

    feature_store_session = Session(  # pylint: disable=W0621
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=feature_store_runtime,
    )

    s3_bucket_name = feature_store_session.default_bucket()  # pylint: disable=W0621

    return (
        feature_store_session,
        feature_store_runtime,
        s3_bucket_name,
        aws_role,
        aws_region,
    )


if __name__ == "__main__":
    (
        feature_store_session,
        feature_store_runtime,
        s3_bucket_name,
        AWS_ROLE,
        AWS_REGION,
    ) = setup_aws_services()

    logger = utils.setup_logger(__name__)

    # 1. Data preparation
    historical_df = pd.read_csv(constants.HISTORICAL_DATA_PATH)
    engineered_df = utils.engineer_features(historical_df)
    tests.test_feature_engineering(historical_df, engineered_df)

    # 2. Create feature group and manager
    feature_store_manager = FeatureStoreManager(
        feature_store_session=feature_store_session,
        s3_bucket_name=s3_bucket_name,
        feature_store_runtime=feature_store_runtime,
        aws_role=AWS_ROLE,
        logger=logger,
    )
    feature_store_manager.create_feature_group()

    # 3. Ingest historical data
    feature_store_manager.ingest_features(engineered_df)
    utils.wait_for_offline_data_to_be_available(
        feature_store_manager=feature_store_manager,
        s3_bucket_name=s3_bucket_name,
        logger=logger,
    )

    # Test data insertion with random customer
    random_customer_id = random.choice(historical_df["customer_id"].unique())
    tests.test_insertion(
        feature_store_runtime=feature_store_runtime, customer_id=random_customer_id
    )

    # 4. Train model
    dataset = feature_store_manager.build_training_dataset()
    X = dataset[constants.TRAINING_FEATURES]
    y = dataset[constants.TARGET_VARIABLE]
    utils.train_model(X, y)

    # 5. Run inference
    inferencer = RealTimeInference(feature_store_manager, logger=logger)
    inferencer.inference_pipeline()
