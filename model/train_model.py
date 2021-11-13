# Script to train machine learning model.
import os
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def run(data_local_path):
    # Add the necessary imports for the starter code.

    # Add code to load in the data.

    if os.path.isfile(data_local_path):
        logger.info(f"Loading data file {data_local_path}")
        data = pd.read_csv(data_local_path)
    else:
        logger.info(f'Data file {data_local_path} not found')
        exit()

    # Optional enhancement, use K-fold cross validation instead
    # of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
                                            test,
                                            categorical_features=cat_features,
                                            label="salary",
                                            training=False,
                                            encoder=encoder,
                                            lb=lb
                                        )

    # Train and save a model.
    logger.info("Model training started")
    model = train_model(X_train, y_train)

    logger.info("Saving model and encoders")
    model_path = 'model/saved_model.pkl'
    pickle.dump(model, open(model_path, 'wb'))

    encoder_path = 'model/saved_encoder.pkl'
    pickle.dump(encoder, open(encoder_path, 'wb'))

    lb_path = 'model/saved_lb.pkl'
    pickle.dump(lb, open(lb_path, 'wb'))

    # Test model
    y_pred = inference(model, X_test)
    y_pred = y_pred.round()

    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall: {recall:.2f}")
    logger.info(f"F1: {fbeta:.2f}")


if __name__ == "__main__":

    data_local_path = 'data/raw-census.csv'
    run(data_local_path)
