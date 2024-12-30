# Pipedrive test: SageMaker Feature Store

## Overview

This project demonstrates the implementation of a Feature store using AWS SageMaker for managing and serving features to machine learning models. The solution focuses on predicting customer loyalty scores based on their purchase history and leverages AWS SageMaker Feature store for efficient feature management and retrieval.


## Assumptions and design decisions


### Assumptions:

1. The company aims to implement **an ML solution that assigns loyalty scores to customers based on their purchasing behavior, considering both current and historical purchase values.**

2. The existing loyalty scores in the data are based on heuristics, and the company wants to transition to an ML-based approach.

3. The company decided to implement a feature store to store and consume **features that are diffuclt to compute in real-time (e.g average purchase value)**

4. As an ML platform engineer, I have access to the company's historical transactional data in CSV format. My responsibilities are:

    - **Feature engineering**: Developing features from historical data and training a regression model to predict loyalty scores for future customers.
  
    - **Feature storage and serving**: Implementing a feature store solution to reliably store and update engineered feature values for inference. **Specifically, features that are computation-heavy and require the past transactions of a customer to compute them**
  
    - **Real-time Inference**: Processing real-time purchase events and assigning loyalty scores using the trained model and feature store, based on:

        - Latest purchase value
        - Average past purchase values
        - Average historical loyalty scores

### Design Decisions

1. **Data Splitting**
    - Following assumption #1, the data is split 70/30
    - 70% represents historical data
    - 30% simulates real-time purchase events

2. **Feature Engineering**
  - Data is grouped by customer ID to compute:
    - Purchase average
    - Average loyalty score
  - The features are:
    - latest_purchase_value
    - average_purchase_value
    - average_loyalty_score
  - Objective: predict latest_loyalty_score

3. **Feature Storage and Serving**
  - AWS SageMaker Feature Store is selected for its:
    - Scaling capabilities
    - Integration with other AWS services (e.g., Kinesis)
  - Both offline and online storage are enabled:
    - Offline storage: Accessed via Athena queries for historical data
    - Online storage: Accessed via get_record() API for latest feature values
  - Historical data supports training while latest feature values enrich purchase events for prediction.

4. **Real-time Inference**
  - Simulates real-time environment using 30% of test data
  - Incorporates realistic production scenarios:
    - Latency-induced delays
    - 5% event loss due to network issues
    - Lost events are recovered and reprocessed
  - Event processing workflow:
    1. Enriches events with historical feature values when available, otherwise uses defaults (RealTimeInference.enrich_event())
    2. Assigns loyalty score using enriched features
    3. Updates customer feature values:
       - Existing customers: feature_store_manager.update_customer_features()
       - New customers: feature_store_manager.add_customer_features()

4. The overall data flow is as follows:

![Alt text](<Feature Store Test.png>)

- Historical data is loaded and undergoes feature engineering
- Engineered features are ingested into the AWS SageMaker Feature store
- The Feature Store is used to retrieve historical features for model training
- Real-time events are simulated, latest features are retrieved for real-time inference
- Feature store is updated with new data

## Key scripts

1. `main.py`: Implements the entire workflow
2. `core/feature_store_manager.py`: Manages interactions with the AWS SageMaker Feature Store
3. `core/inference.py`: Handles real-time inference
4. `core/utils.py`: Contains utility functions for data processing and AWS interactions
5. `core/constants.py`: Stores constant values used across the project
6. `tests.py`: Includes test functions used during developpement.

## Setup and Installation

1. Clone the repository
2. Create folder `/data` and copy the csv files: `historical_data.csv` and `inference_data.csv` to it.
3. Install required dependencies: `poetry install`
4. Set environment variables:

```bash
AWS_ACCESS_KEY=<your AWS access key>
AWS_SECRET_ACCESS_KEY=<your AWS secret access key>
AWS_DEFAULT_REGION=<your AWS region>
AWS_ROLE=<your AWS SageMaker full access role>
```

Run the main script to execute the entire workflow:

```bash
poetry run python main.py
```

**I created a Dockerfile to make it easy to run the project and see workflow in action:**

```bash
docker build -t pipedrive-test .
```

```bash
docker run \
    -e AWS_ACCESS_KEY=<your AWS access key> \
    -e AWS_SECRET_ACCESS_KEY=<your AWS secret access key> \
    -e AWS_DEFAULT_REGION=<your AWS region> \
    -e AWS_ROLE=<your AWS SageMaker full access role> \
    pipedrive-test:latest
```
