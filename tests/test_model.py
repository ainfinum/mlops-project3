import pandas as pd
import pytest
import os
import numpy as np
import pickle
from model.ml.data import process_data
from model.ml.model import compute_model_metrics, inference
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


TEST_DATA_PATH = "data/raw-census.csv"
MODEL_PATH = "model/saved_models/saved_model.pkl"
ENCODER_PATH = "model/saved_models/saved_encoder.pkl"
LB_PATH = "model/saved_models/saved_lb.pkl"


@pytest.fixture
def data():
    """Load some test data."""

    if os.path.isfile(TEST_DATA_PATH):
        logger.info(f"Loading data file {TEST_DATA_PATH}")
        data = pd.read_csv(TEST_DATA_PATH, nrows=200)
    else:
        logger.info(f"Data file {TEST_DATA_PATH} not found")
        exit()

    return data


@pytest.fixture
def cat_features():
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
    return cat_features


@pytest.fixture
def model():
    return pickle.load(open(MODEL_PATH, "rb"))


@pytest.fixture
def encoder():
    return pickle.load(open(ENCODER_PATH, "rb"))


@pytest.fixture
def lb():
    return pickle.load(open(LB_PATH, "rb"))


def test_process_data(data, cat_features):

    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    assert (
        X_train.shape[0] == data.shape[0]
        ), "Wrong number of rows in source data"

    assert (
        X_train.shape[1] > data.shape[1]
    ), "Wrong number of features in processed data"

    assert (
        y_train.shape[0] == data.shape[0]
    ), "Wrong shape of y_train rows after processing data"


def test_compute_model_metrics():

    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    preds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision > 0.9
    assert recall > 0.6
    assert fbeta > 0.6


def test_inference(model, encoder, lb, data, cat_features):
    """Test model inference"""

    X_test, y_test, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label="salary",
        training=False,
    )

    y_pred = inference(model, X_test)

    assert y_pred.shape[0] == X_test.shape[0], "Wrong predictions shape"
    pred_average = np.average(y_pred)
    assert (
        1 >= pred_average >= 0
    ), "Prediction average of {pred_average} is not between 0 and 1"


# if __name__ == "__main__":

# test_inference(model(), encoder(), lb(), data(), cat_features())
# test_compute_model_metrics()
# test_process_data(data(), cat_features())
