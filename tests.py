"""
Testing Module

This module contains custom validation tests (not unit tests) to verify the functionality
of the feature store implementation. These tests are designed to check:

- Feature engineering process
- Record insertion
"""


def test_insertion(feature_store_runtime, customer_id=100):
    """
    Test the insertion of a customer record into the feature store.

    This function attempts to retrieve a record for a given customer ID from the
    feature store. It then asserts that the response contains a 'Record' key,
    indicating successful retrieval and, by extension, successful insertion.

    Args:
        feature_store_runtime (boto3.client): A boto3 client for the SageMaker Feature Store runtime.
        customer_id (int, optional): The ID of the customer to test. Defaults to 100.

    Returns:
        None

    Raises:
        AssertionError: If the response does not contain a 'Record' key.
    """
    record_identifier_value = str(customer_id)
    print(f"Testing with {record_identifier_value}")
    response = feature_store_runtime.get_record(
        FeatureGroupName="customer_purchase_features",
        RecordIdentifierValueAsString=record_identifier_value,
    )
    assert "Record" in response


def test_feature_engineering(historical_df, engineered_df):
    """
    Test the feature engineering process by comparing the input historical dataframe
    with the output engineered dataframe.

    This function performs two tests:
    1. Checks if the engineered dataframe has the correct columns.
    2. Verifies if the number of unique customers in the engineered dataframe
       matches the number in the historical dataframe.

    Args:
        historical_df (pd.DataFrame): The original historical dataframe before feature engineering.
        engineered_df (pd.DataFrame): The resulting dataframe after feature engineering.

    Raises:
        AssertionError: If the engineered dataframe doesn't have the expected columns
                        or if the number of unique customers doesn't match.

    Returns:
        None
    """
    expected_columns = {
        "customer_id",
        "purchase_timestamp",
        "latest_purchase_value",
        "avg_purchase_value",
        "latest_loyalty_score",
        "avg_loyalty_score",
    }
    assert (
        set(engineered_df.columns) == expected_columns
    ), f"Missing or extra columns. Expected {expected_columns}, got {set(engineered_df.columns)}"

    assert (
        len(engineered_df) == len(historical_df["customer_id"].unique())
    ), f"Expected {len(historical_df['customer_id'].unique())} customers, got {len(engineered_df)}"
