import pandas as pd
import numpy as np
import boto3
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session
from core import utils, constants


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


def setup_aws_session():
    """
    Set up AWS and SageMaker session objects.

    Returns:
        tuple: (feature_group, featurestore_runtime) - Configured Feature Group and runtime client
    """
    boto_session = boto3.Session()
    sagemaker_client = boto_session.client(service_name="sagemaker")
    featurestore_runtime = boto_session.client(
        service_name="sagemaker-featurestore-runtime"
    )
    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )

    feature_group = FeatureGroup(
        name=constants.FEATURE_GROUP_NAME, sagemaker_session=feature_store_session
    )

    return feature_group, featurestore_runtime


def get_test_customers(csv_path, min_records=2, num_customers=10):
    """
    Get random customers from CSV that have multiple records.

    Args:
        csv_path (str): Path to CSV file
        min_records (int): Minimum number of records a customer should have
        num_customers (int): Number of customers to select

    Returns:
        list: Selected customer IDs
    """
    df = pd.read_csv(csv_path)
    customer_counts = df["customer_id"].value_counts()
    eligible_customers = customer_counts[customer_counts >= min_records].index.tolist()

    if len(eligible_customers) < num_customers:
        raise ValueError(f"Not enough customers with {min_records} or more records")

    return np.random.choice(eligible_customers, num_customers, replace=False).tolist()


def fetch_offline_store_data(feature_group):
    """
    Fetch all data from the offline feature store.

    Args:
        feature_group: SageMaker Feature Group object

    Returns:
        pd.DataFrame: Data from offline store
    """
    query = feature_group.athena_query()
    customer_table = query.table_name
    query.run(
        query_string=f"SELECT * FROM {customer_table}",
        output_location=f"s3://{feature_group.sagemaker_session.default_bucket()}/feature-store/test_queries",
    )
    query.wait()
    return query.as_dataframe()


def fetch_online_store_record(featurestore_runtime, customer_id):
    """
    Fetch a customer's record from the online feature store.

    Args:
        featurestore_runtime: SageMaker Feature Store runtime client
        customer_id: Customer ID to fetch

    Returns:
        dict: Customer's features from online store
    """
    response = featurestore_runtime.get_record(
        FeatureGroupName=constants.FEATURE_GROUP_NAME,
        RecordIdentifierValueAsString=str(customer_id),
    )
    record = response["Record"]

    return {
        "customer_id": utils.get_feature_value(record, "customer_id"),
        "purchase_timestamp": utils.get_feature_value(record, "purchase_timestamp"),
        "latest_purchase_value": float(
            utils.get_feature_value(record, "latest_purchase_value")
        ),
    }


def get_latest_csv_record(df, customer_id):
    """
    Get the latest record for a customer from CSV data.

    Args:
        df (pd.DataFrame): DataFrame containing customer records
        customer_id: Customer ID to look up

    Returns:
        pd.Series: Latest record for the customer
    """
    customer_df = df[df["customer_id"] == customer_id]
    return customer_df.sort_values("purchase_timestamp", ascending=False).iloc[0]


def test_offline_store():
    """
    Test data consistency in the offline feature store.
    Verifies that selected customers have at least 2 records.
    """
    csv_path = "data/inference_data.csv"
    feature_group, _ = setup_aws_session()

    # Get test customers and offline data
    test_customers = get_test_customers(csv_path)
    print(f"Testing offline store for customers: {test_customers}")

    offline_df = fetch_offline_store_data(feature_group)

    # Check each customer's records
    for customer_id in test_customers:
        customer_records = offline_df[offline_df["customer_id"] == customer_id]
        assert (
            len(customer_records) >= 2
        ), f"Customer {customer_id} has fewer than 2 records"

    print("✅ All customers have required number of records in offline store")


def test_online_store():
    """
    Test data consistency between online feature store and CSV data.
    Compares purchase values and timestamps for selected customers.
    """
    csv_path = "data/inference_data.csv"
    _, featurestore_runtime = setup_aws_session()

    # Get test customers and CSV data
    test_customers = get_test_customers(csv_path)
    print(f"Testing online store for customers: {test_customers}")

    df = pd.read_csv(csv_path)

    # Compare records for each customer
    for customer_id in test_customers:
        csv_record = get_latest_csv_record(df, customer_id)
        online_record = fetch_online_store_record(featurestore_runtime, customer_id)

        csv_purchase = float(csv_record["purchase_value"])
        online_purchase = online_record["latest_purchase_value"]

        print(f"\nCustomer: {customer_id}")
        print(f"CSV Purchase Value: {csv_purchase}")
        print(f"Online Purchase Value: {online_purchase}")
        print(f"CSV Timestamp: {csv_record['purchase_timestamp']}")
        print(f"Online Timestamp: {online_record['purchase_timestamp']}")

        if abs(csv_purchase - online_purchase) > 0.0001:
            print("❌ Values don't match!")
        else:
            print("✅ Values match!")
