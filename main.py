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
5. Validation testing of feature store operations

Dependencies:
    - boto3: AWS SDK for Python
    - pandas: Data manipulation and analysis
    - sagemaker: Amazon SageMaker Python SDK
    - python-dotenv: Environment variable management
    - core: Custom project modules

Environment Variables Required:
    - AWS_ROLE: AWS IAM role ARN with necessary permissions

Note:
    Ensure all required AWS credentials and permissions are properly configured
    before running this script.
"""

import os
import random
import boto3
import pandas as pd
from sagemaker.session import Session
from dotenv import load_dotenv

from core import constants, utils
from core.feature_store_manager import FeatureStoreManager
from core.inference import RealTimeInference
import tests


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

    Note:
        Requires appropriate AWS credentials and permissions to be configured.
    """
    load_dotenv()

    aws_role = os.getenv("AWS_ROLE")
    aws_region = boto3.Session().region_name

    boto_session = boto3.Session(region_name=aws_region)
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
    # Set up AWS services and configurations
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
        AWS_REGION=AWS_REGION,
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


# import os
# import random
# import boto3
# import pandas as pd
# from sagemaker.session import Session

# from dotenv import load_dotenv

# from core import constants, utils
# from core.feature_store_manager import FeatureStoreManager
# from core.inference import RealTimeInference
# import tests

# load_dotenv()

# AWS_ROLE = os.getenv("AWS_ROLE")
# AWS_REGION = boto3.Session().region_name


# boto_session = boto3.Session(region_name=AWS_REGION)
# sagemaker_client = boto_session.client(service_name="sagemaker", region_name=AWS_REGION)
# feature_store_runtime = boto_session.client(
#     service_name="sagemaker-featurestore-runtime", region_name=AWS_REGION
# )
# feature_store_session = Session(
#     boto_session=boto_session,
#     sagemaker_client=sagemaker_client,
#     sagemaker_featurestore_runtime_client=feature_store_runtime,
# )

# s3_bucket_name = feature_store_session.default_bucket()


# if __name__ == "__main__":
#     # running

#     # 1- Data preparation
#     historical_df = pd.read_csv(constants.HISTORICAL_DATA_PATH)
#     engineered_df = utils.engineer_features(historical_df)

#     tests.test_feature_engineering(historical_df, engineered_df)

#     # 2- Create feature group
#     feature_store_manager = FeatureStoreManager(
#         feature_store_session=feature_store_session,
#         s3_bucket_name=s3_bucket_name,
#         feature_store_runtime=feature_store_runtime,
#         aws_role=AWS_ROLE,
#     )
#     feature_store_manager.create_feature_group()

#     # 3- Ingest historical data
#     feature_store_manager.ingest_features(engineered_df)

#     utils.wait_for_offline_data_to_be_available(
#         feature_store_manager=feature_store_manager,
#         s3_bucket_name=s3_bucket_name,
#         AWS_REGION=AWS_REGION,
#     )

#     # Select a random customer ID
#     random_customer_id = random.choice(historical_df["customer_id"].unique())
#     tests.test_insertion(
#         feature_store_runtime=feature_store_runtime, customer_id=random_customer_id
#     )

#     # 4- Read historical data and train sample model
#     dataset = feature_store_manager.build_training_dataset()
#     X = dataset[constants.TRAINING_FEATURES]
#     y = dataset[constants.TARGET_VARIABLE]
#     utils.train_model(X, y)

#     # 5- Inference
#     inferencer = RealTimeInference(feature_store_manager)
#     inferencer.inference_pipeline()

#     # TESTING
#     df = inferencer.inference_data
#     # First, ensure the purchase_timestamp is in datetime format
#     df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"])

#     # Group by customer_id and get the index of the latest purchase for each customer
#     latest_purchase_indices = df.groupby("customer_id")["purchase_timestamp"].idxmax()

#     # Use these indices to select the rows with the latest purchase for each customer
#     latest_purchases = df.loc[latest_purchase_indices]

#     # Sort the result by customer_id for better readability
#     latest_purchases = latest_purchases.sort_values("customer_id")
#     combined_data = []
#     for _, row in latest_purchases.iterrows():
#         customer = str(row["customer_id"])
#         assert feature_store_manager.customer_features_exist(customer_id=customer)
#         d = feature_store_manager.get_latest_features(customer_id=customer)
#         assert d["customer_id"] == customer
#         assert d["latest_purchase_value"] == str(row["purchase_value"])
#         assert d["latest_loyalty_score"] == str(row["predictions"])

#         combined_row = {
#             "customer_id": customer,
#             "latest_purchase_timestamp": row["purchase_timestamp"],
#             "latest_purchase_value_df": row["purchase_value"],
#             "predicted_loyalty_score_df": row["predictions"],
#             "latest_purchase_value_fs": d["latest_purchase_value"],
#             "latest_loyalty_score_fs": d["latest_loyalty_score"],
#             "avg_purchase_value_fs": d["avg_purchase_value"],
#             "avg_loyalty_score_fs": d["avg_loyalty_score"],
#         }
#         combined_data.append(combined_row)

#     inspection_df = pd.DataFrame(combined_data)

#     # Save the DataFrame to a CSV file for manual inspection
#     inspection_df.to_csv("inspection_data.csv", index=False)