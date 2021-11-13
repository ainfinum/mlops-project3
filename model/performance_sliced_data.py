import pandas as pd
import pytest
import os
import numpy as np
import pickle
from ml.data import process_data
from ml.model import compute_model_metrics, inference
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


DATA_PATH = 'data/raw-census.csv'
MODEL_PATH = 'model/saved_models/saved_model.pkl'
ENCODER_PATH = 'model/saved_models/saved_encoder.pkl'
LB_PATH = 'model/saved_models/saved_lb.pkl'


def slice_data(df, feature):
    """Function for calculating descriptive stats on slices of the Iris dataset."""
    for cls in df["class"].unique():
        df_temp = df[df["class"] == cls]
        mean = df_temp[feature].mean()
        stddev = df_temp[feature].std()
        print(f"Class: {cls}")
        print(f"{feature} mean: {mean:.4f}")
        print(f"{feature} stddev: {stddev:.4f}")
    print()


def model_performance_on_sliced_data(feature, model, encoder, lb, data):

    

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

    for cls in data[feature].unique():
        data_temp = data[data[feature] == cls]

        #print(data_temp.head())
        #print(data_temp.info())

        X_test, y_test, encoder, lb = process_data(
            data_temp, categorical_features=cat_features, encoder=encoder, lb=lb, label="salary", training=False
        )
        
        y_pred = inference(model, X_test)
        y_pred = y_pred.round()
        #y_0 = y_pred[y_pred==0]
        #y_1 = y_pred[y_pred==1]
        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        logger.info(f"--- Model metrics for sliced data by '{feature}:{cls}'")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1: {fbeta:.4f}")


if __name__ == "__main__":

    model = pickle.load(open(MODEL_PATH, 'rb'))
    encoder = pickle.load(open(ENCODER_PATH, 'rb'))
    lb = pickle.load(open(LB_PATH, 'rb'))

    if os.path.isfile(DATA_PATH):
        logger.info(f"Loadind data file {DATA_PATH}")
        data = pd.read_csv(DATA_PATH)
    else:
        logger.info(f'Data file {DATA_PATH} not found')
        exit()

    feature = "workclass"
    model_performance_on_sliced_data(feature, model, encoder, lb, data)
