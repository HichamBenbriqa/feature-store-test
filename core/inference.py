"""
Real-Time Customer Loyalty Prediction Module

This module implements real-time inference capabilities for predicting customer loyalty scores
based on purchase behavior and historical data. It integrates with SageMaker Feature Store
for feature management and provides both batch and single-event prediction capabilities.

The module handles:
- Loading and managing ML models for inference
- Processing real-time purchase events
- Retrieving and updating customer features
- Managing the complete inference pipeline for batch predictions
- Integration with Feature Store for persistent storage

"""

import logging
import pickle
import time
import random
import pandas as pd
from core import constants
from core.feature_store_manager import FeatureStoreManager


class RealTimeInference:
    """
    A class for handling real-time customer loyalty predictions and feature updates.

    This class manages the entire inference process, from loading the model and data
    to processing events and updating the feature store with new predictions.

    Attributes:
        feature_store_manager (FeatureStoreManager): Manager for feature store operations
        data_path (str): Path to the inference data file
        model_path (str): Path to the serialized model file
        inference_data (pandas.DataFrame): Loaded inference data
        model: Loaded prediction model
        logger (logging.Logger): unified logger object
    """

    def __init__(
        self,
        feature_store_manager: FeatureStoreManager,
        data_path: str = constants.INFERENCE_DATA_PATH,
        model_path: str = constants.MODEL_PATH,
        logger: logging.Logger = None,
    ):
        """
        Initialize the RealTimeInference instance.

        Args:
            feature_store_manager (FeatureStoreManager): Manager for feature store operations
            data_path (str, optional): Path to inference data.
                                       Defaults to constants.INFERENCE_DATA_PATH
            model_path (str, optional): Path to model file. Defaults to constants.MODEL_PATH

        Raises:
            FileNotFoundError: If model_path is invalid
            pickle.UnpicklingError: If model file is corrupted
        """
        self.feature_store_manager = feature_store_manager
        self.data_path = data_path
        self.inference_data = None
        self.logger = logger
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def get_inference_data(self):
        """
        Load and prepare inference data from the specified CSV file.

        Loads the data, converts timestamp columns to datetime format,
        and sorts records by timestamp in ascending order.

        Raises:
            FileNotFoundError: If data_path is invalid
            pd.errors.EmptyDataError: If CSV file is empty
        """
        self.inference_data = pd.read_csv(self.data_path)
        self.inference_data["purchase_timestamp"] = pd.to_datetime(
            self.inference_data["purchase_timestamp"]
        )
        self.inference_data = self.inference_data.sort_values(
            "purchase_timestamp", ascending=True
        )

    def enrich_event(self, customer_exists, event):
        """Enrich purchase event with historical features for prediction.

        Combines current purchase data with historical features if customer exists in feature store,
        otherwise uses default values:
            - avg_purchase_value=current purchase value
            - avg_loyalty_score=0

        Args:
            customer_exists (bool): Whether customer has existing feature store record
            event (dict): Purchase event containing 'customer_id' and 'purchase_value'

        Returns:
            dict: Features for prediction:
                - latest_purchase_value: Current purchase amount
                - avg_purchase_value: Historical average or current amount if new customer
                - avg_loyalty_score: Historical average or 0 if new customer
        """
        customer_id = event["customer_id"]
        purchase_value = float(event["purchase_value"])

        if customer_exists:
            customer_features = self.feature_store_manager.get_latest_features(
                customer_id
            )
            return {
                "latest_purchase_value": purchase_value,
                "avg_purchase_value": float(customer_features["avg_purchase_value"]),
                "avg_loyalty_score": float(customer_features["avg_loyalty_score"]),
            }

        return {
            "latest_purchase_value": purchase_value,
            "avg_purchase_value": purchase_value,
            "avg_loyalty_score": 0,
        }

    def predict(self, features):
        """
        Generate loyalty score prediction for given features.
        Args:
            features (dict): Feature dictionary containing:
                - latest_purchase_value (float)
                - avg_purchase_value (float)
                - avg_loyalty_score (float)
        Returns:
            float: Predicted loyalty score
        """
        # Convert features to DataFrame with named columns
        feature_df = pd.DataFrame(
            [
                [
                    features["latest_purchase_value"],
                    features["avg_purchase_value"],
                    features["avg_loyalty_score"],
                ]
            ],
            columns=[
                "latest_purchase_value",
                "avg_purchase_value",
                "avg_loyalty_score",
            ],
        )

        return self.model.predict(feature_df)[0]

    def insert_customer_features(
        self, customer_exists, customer_id, purchase_amount, prediction
    ):
        """Update or insert customer features in feature store.

        Either updates existing customer record or creates new one based on customer_exists flag.

        Args:
            customer_exists (bool): Whether customer has existing feature store record
            customer_id (str): Customer identifier
            purchase_amount (float): Current purchase value
            prediction (float): Predicted loyalty score to store as latest_loyalty_score
        """

        if customer_exists:
            self.logger.debug(f"Updating the records of the customer: {customer_id}")
            self.feature_store_manager.update_customer_features(
                customer_id,
                purchase_amount,
                prediction,  # Use predicted loyalty score as the new latest_loyalty_score
            )
        else:
            self.logger.debug(f"Inserting the records of the customer: {customer_id}")
            self.feature_store_manager.add_customer_features(
                customer_id, purchase_amount, prediction
            )

    def process_single_event(self, row):
        """
        Process a single event (purchase) for a customer,
        including feature enrichment and prediction.

        This function simulates network latency,
        checks if the customer exists in the feature store,
        enriches the event data, makes a prediction,
        and updates the feature store with the new data.

        Args:
            row (pandas.Series): A row from the inference data containing customer
                                 and purchase information.
                                 Expected to have 'customer_id', 'purchase_value',
                                 and 'purchase_timestamp' fields.
        """
        if random.random() < 0.2:  # 20% chance of high latency
            delay = random.uniform(1, 3)  # High delay
            self.logger.debug(f"Waiting for High delay: {delay}")
            time.sleep(delay)

        else:
            delay = random.uniform(0.1, 0.5)  # Normal delay
            self.logger.debug(f"Waiting for Normal delay: {delay}")
            time.sleep(delay)
        customer_exists = self.feature_store_manager.customer_features_exist(
            str(row["customer_id"])
        )
        event = {
            "customer_id": str(row["customer_id"]),
            "purchase_value": row["purchase_value"],
            "timestamp": row["purchase_timestamp"],
        }

        features = self.enrich_event(customer_exists, event)
        prediction = self.predict(features)

        self.insert_customer_features(
            customer_exists,
            str(row["customer_id"]),
            row["purchase_value"],
            prediction,
        )

    def inference_pipeline(self):
        """
        Execute real-time inference pipeline for streaming predictions.

        This method processes the inference data, simulating a real-time streaming environment.
        It handles network delays, potential message loss,
        and includes a retry mechanism for failed events.

        The pipeline performs the following steps:
        1. Loads and prepares the inference data.
        2. Iterates through each event (row) in the data.
        3. Simulates network issues and handles failed events.
        4. Processes each event using the `process_single_event` method.
        5. Attempts to retry any failed events.

        Returns:
            pandas.DataFrame: The processed inference data, potentially including new predictions
                              or updates made during the pipeline execution.

        Raises:
            Any exceptions not caught during individual event processing or retries will be propagated.
        """
        self.get_inference_data()

        failed_events = []  # Store failed events for retry
        self.logger.info(f"Total Events: {self.inference_data.shape}")
        for _, row in self.inference_data.iterrows():
            self.logger.info(f"Received event: {row['customer_id']} ")
            if random.random() < 0.05:
                self.logger.warning(
                    f"Network issue for customer {row['customer_id']} - Adding to retry queue"
                )
                failed_events.append(row)
                continue
            try:
                self.process_single_event(row)

            except Exception as e:  # pylint: disable=W0718
                self.logger.error(
                    f"Error processing event for customer {row['customer_id']}: {e}"
                )
                failed_events.append(row)

        if failed_events:
            self.logger.info(f"Retrying {len(failed_events)} failed events...")
            for row in failed_events:
                self.logger.info(f"Re-handling event: {row['customer_id']} ")
                try:
                    self.process_single_event(row)
                except Exception as e:  # pylint: disable=W0718
                    self.logger.error(
                        f"Failed to process event for customer {row['customer_id']} after retry: {e}"
                    )

        return self.inference_data

    # def inference_pipeline(self):
    #     """
    #     Execute the complete inference pipeline for batch predictions.

    #     The pipeline:
    #     1. Loads and prepares inference data
    #     2. Processes each purchase event
    #     3. Generates predictions for each event
    #     4. Updates the feature store with new predictions
    #     5. Adds predictions to the inference data

    #     Returns:
    #         pandas.DataFrame: Original inference data enriched with predictions

    #     Note:
    #         This method modifies the internal inference_data attribute by
    #         adding a 'predictions' column with the generated predictions.
    #     """
    #     self.get_inference_data()
    #     results = []

    #     for _, row in self.inference_data.iterrows():
    #         event = {
    #             "customer_id": str(row["customer_id"]),
    #             "purchase_value": row["purchase_value"],
    #         }
    #         features = self.process_event(event)
    #         prediction = self.predict(features)
    #         self.insert_customer_features(
    #             str(row["customer_id"]), row["purchase_value"], prediction
    #         )
    #         results.append(prediction)

    #     self.inference_data["predictions"] = results
    #     return self.inference_data


# """summary"""

# import pickle
# import pandas as pd
# from core import constants
# from core.feature_store_manager import FeatureStoreManager


# class RealTimeInference:
#     """summary"""

#     def __init__(
#         self,
#         feature_store_manager: FeatureStoreManager,
#         data_path: str = constants.INFERENCE_DATA_PATH,
#         model_path: str = constants.MODEL_PATH,
#     ):
#         self.feature_store_manager = feature_store_manager
#         self.data_path = data_path
#         self.infrence_data = None
#         with open(model_path, "rb") as f:
#             self.model = pickle.load(f)

#     def get_inference_data(self):
#         """summary"""
#         self.infrence_data = pd.read_csv(self.data_path)
#         self.infrence_data["purchase_timestamp"] = pd.to_datetime(
#             self.infrence_data["purchase_timestamp"]
#         )

#         self.infrence_data = self.infrence_data.sort_values(
#             "purchase_timestamp", ascending=True
#         )

#     def process_event(self, event):
#         """Process a single real-time event"""
#         customer_id = event["customer_id"]
#         purchase_amount = float(event["purchase_value"])

#         customer_exists = self.feature_store_manager.customer_features_exist(
#             customer_id
#         )

#         if customer_exists:
#             # Get historical features
#             customer_features = self.feature_store_manager.get_latest_features(
#                 customer_id
#             )
#             # in case customer already processed we enrich the data from feature store
#             return {
#                 "latest_purchase_value": purchase_amount,
#                 "avg_purchase_value": float(customer_features["avg_purchase_value"]),
#                 "avg_loyalty_score": float(customer_features["avg_loyalty_score"]),
#             }
#         return {
#             "latest_purchase_value": purchase_amount,
#             "avg_purchase_value": purchase_amount,
#             "avg_loyalty_score": 0,
#         }

#     def predict(self, features):
#         """summary"""
#         return self.model.predict(
#             [
#                 [
#                     features["latest_purchase_value"],
#                     features["avg_purchase_value"],
#                     features["avg_loyalty_score"],
#                 ]
#             ]
#         )[0]

#     def insert_customer_features(self, customer_id, purchase_amount, prediction):
#         """summary"""
#         # Update feature store with predicted loyalty score
#         customer_exists = self.feature_store_manager.customer_features_exist(
#             customer_id
#         )
#         if customer_exists:
#             self.feature_store_manager.update_customer_features(
#                 customer_id,
#                 purchase_amount,
#                 prediction,  # Use predicted loyalty score as the new latest_loyalty_score
#             )
#         else:
#             self.feature_store_manager.add_customer_features(
#                 customer_id, purchase_amount, prediction
#             )

#     def inference_pipeline(self):
#         """summary"""
#         self.get_inference_data()
#         results = []
#         for _, row in self.infrence_data.iterrows():
#             event = {
#                 "customer_id": str(row["customer_id"]),
#                 "purchase_value": row["purchase_value"],
#             }
#             features = self.process_event(event)
#             prediction = self.predict(features)
#             self.insert_customer_features(
#                 str(row["customer_id"]), row["purchase_value"], prediction
#             )
#             results.append(prediction)

#         self.infrence_data["predictions"] = results
