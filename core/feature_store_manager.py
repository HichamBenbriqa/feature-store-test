"""
SageMaker Feature Store Management Module

This module provides a comprehensive interface for managing customer features
in Amazon SageMaker Feature Store.

It handles both online and offline feature storage, feature ingestion,
and retrieval operations for customer purchase and loyalty data.

The module implements a FeatureStoreManager class that abstracts the complexity of:
- Creating and managing feature groups
- Ingesting new feature data
- Building training datasets
- Managing customer-specific feature operations
- Handling both batch and real-time feature updates
"""

import logging
from datetime import datetime
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)

from core import utils, constants


class FeatureStoreManager:
    """
    A manager class for handling Amazon SageMaker Feature Store operations.

    This class provides functionality to create and manage feature groups, ingest features,
    build training datasets, and handle customer-specific feature operations in both
    online and offline feature stores.

    Attributes:
        feature_store_session (Session): SageMaker session for Feature Store operations
        s3_bucket_name (str): Name of the S3 bucket for storing offline features
        feature_store_runtime: Runtime client for Feature Store operations
        aws_role (str): AWS IAM role ARN with necessary permissions
        feature_group (FeatureGroup): Feature group instance for storing customer features
        logger (logging.Logger): unified logger object
    """

    def __init__(
        self,
        feature_store_session: Session,
        s3_bucket_name: str,
        feature_store_runtime,
        aws_role: str,
        logger: logging.Logger,
    ):
        """
        Initialize the FeatureStoreManager with necessary AWS resources.

        Args:
            feature_store_session (Session): SageMaker session for Feature Store operations
            s3_bucket_name (str): Name of the S3 bucket for storing offline features
            feature_store_runtime: Runtime client for Feature Store operations
            aws_role (str): AWS IAM role ARN with necessary permissions
            logger (logging.Logger): unified logger object
        """
        self.feature_store_session = feature_store_session
        self.s3_bucket_name = s3_bucket_name
        self.feature_store_runtime = feature_store_runtime
        self.aws_role = aws_role
        self.feature_group = None
        self.logger = logger

    def create_feature_group(self):
        """
        Create a new feature group with predefined feature definitions.

        Creates a feature group with customer-related features including purchase history
        and loyalty scores. Enables both online and offline stores, with data stored in
        the specified S3 bucket location.

        Raises:
            Exception: If feature group creation fails
        """
        self.feature_group = FeatureGroup(
            name=constants.FEATURE_GROUP_NAME,
            feature_definitions=[
                FeatureDefinition("customer_id", FeatureTypeEnum.STRING),
                FeatureDefinition("purchase_timestamp", FeatureTypeEnum.STRING),
                FeatureDefinition("latest_purchase_value", FeatureTypeEnum.FRACTIONAL),
                FeatureDefinition("avg_purchase_value", FeatureTypeEnum.FRACTIONAL),
                FeatureDefinition("avg_loyalty_score", FeatureTypeEnum.FRACTIONAL),
                FeatureDefinition("latest_loyalty_score", FeatureTypeEnum.FRACTIONAL),
            ],
            sagemaker_session=self.feature_store_session,
        )

        self.feature_group.create(
            s3_uri=f"s3://{self.s3_bucket_name}/feature-store/customer_features",
            record_identifier_name="customer_id",
            event_time_feature_name="purchase_timestamp",
            role_arn=self.aws_role,
            enable_online_store=True,
        )

        utils.wait_for_feature_group_creation_complete(
            self.feature_group, logger=self.logger
        )

    def ingest_features(self, features_df):
        """
        Ingest features from a DataFrame into the Feature Store.

        Args:
            features_df (pandas.DataFrame): DataFrame containing feature values to ingest

        Note:
            The DataFrame must contain columns matching the feature definitions
            in the feature group.
        """
        self.feature_group.ingest(data_frame=features_df, max_workers=1, wait=True)

    def build_training_dataset(self):
        """
        Build a training dataset by querying the Feature Store using Athena.

        Returns:
            pandas.DataFrame: DataFrame containing the latest feature values for each customer

        Raises:
            Exception: If query execution fails or data retrieval encounters an error
        """
        try:
            customer_query = self.feature_group.athena_query()
            customer_table = customer_query.table_name

            query_string = f"""SELECT customer_id,
                                       latest_purchase_value,
                                       avg_purchase_value,
                                       avg_loyalty_score,
                                       latest_loyalty_score
                                FROM "{customer_table}"
                                ORDER BY customer_id, purchase_timestamp DESC"""

            customer_query.run(
                query_string=query_string,
                output_location=f"s3://{self.s3_bucket_name}/query_results/",
            )
            customer_query.wait()
            train_df = customer_query.as_dataframe()
            return train_df

        except Exception as e:
            self.logger.error(f"Error in build_training_dataset: {str(e)}")
            raise

    def customer_features_exist(self, customer_id):
        """
        Check if features exist for a given customer ID in the online store.

        Args:
            customer_id (str): The customer ID to check

        Returns:
            bool: True if customer features exist, False otherwise
        """
        response = self.feature_store_runtime.get_record(
            FeatureGroupName=constants.FEATURE_GROUP_NAME,
            RecordIdentifierValueAsString=customer_id,
        )
        if "Record" not in response:
            self.logger.warning(f"{customer_id} has no records in the feature store")
            return False
        return True

    def get_latest_features(self, customer_id):
        """
        Retrieve the latest feature values for a customer from the online store.

        Args:
            customer_id (str): The customer ID to retrieve features for

        Returns:
            dict: Dictionary containing the latest feature values for the customer
        """
        response = self.feature_store_runtime.get_record(
            FeatureGroupName="customer_purchase_features",
            RecordIdentifierValueAsString=customer_id,
        )

        record = response["Record"]

        customer_features = {
            "customer_id": utils.get_feature_value(record, "customer_id"),
            "purchase_timestamp": utils.get_feature_value(record, "purchase_timestamp"),
            "latest_purchase_value": utils.get_feature_value(
                record, "latest_purchase_value"
            ),
            "avg_purchase_value": utils.get_feature_value(record, "avg_purchase_value"),
            "avg_loyalty_score": utils.get_feature_value(record, "avg_loyalty_score"),
            "latest_loyalty_score": utils.get_feature_value(
                record, "latest_loyalty_score"
            ),
        }

        return customer_features

    def add_customer_features(self, customer_id, new_purchase_value, new_loyalty_score):
        """
        Add initial features for a new customer to the Feature Store.

        Args:
            customer_id (str): The customer ID to add features for
            new_purchase_value (float): The value of the customer's first purchase
            new_loyalty_score (float): The initial loyalty score for the customer

        Returns:
            dict: Dictionary containing the newly added feature values

        Note:
            For new customers, the average values are initialized to match
            the first purchase value and loyalty score.
        """
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        record = {
            "customer_id": str(customer_id),
            "purchase_timestamp": current_time,
            "latest_purchase_value": str(new_purchase_value),
            "avg_purchase_value": str(new_purchase_value),
            "avg_loyalty_score": str(new_loyalty_score),
            "latest_loyalty_score": str(new_loyalty_score),
        }

        self.feature_store_runtime.put_record(
            FeatureGroupName=self.feature_group.name,
            Record=[{"FeatureName": k, "ValueAsString": v} for k, v in record.items()],
        )

        return record

    def update_customer_features(
        self, customer_id, new_purchase_value, new_loyalty_score
    ):
        """
        Update existing customer features with new purchase data and loyalty score.

        Args:
            customer_id (str): The customer ID to update features for
            new_purchase_value (float): The value of the customer's latest purchase
            new_loyalty_score (float): The latest predicted loyalty score

        Returns:
            dict: Dictionary containing the updated feature values

        Note:
            Updates both latest values and recalculates averages based on
            previous values and new data.
        """
        current_features = self.get_latest_features(customer_id)

        old_avg_purchase = float(current_features["avg_purchase_value"])
        new_avg_purchase = (old_avg_purchase + new_purchase_value) / 2

        old_avg_loyalty = float(current_features["avg_loyalty_score"])
        new_avg_loyalty = (old_avg_loyalty + new_loyalty_score) / 2

        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        record = {
            "customer_id": str(customer_id),
            "purchase_timestamp": current_time,
            "latest_purchase_value": str(new_purchase_value),
            "avg_purchase_value": str(new_avg_purchase),
            "avg_loyalty_score": str(new_avg_loyalty),
            "latest_loyalty_score": str(new_loyalty_score),
        }

        self.feature_store_runtime.put_record(
            FeatureGroupName=self.feature_group.name,
            Record=[{"FeatureName": k, "ValueAsString": v} for k, v in record.items()],
        )

        return record
