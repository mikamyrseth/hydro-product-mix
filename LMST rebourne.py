from typing import List
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  # to plot the time series plot
from sklearn import metrics  # for the evaluation
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import math
from keras.callbacks import EarlyStopping
import arff
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


def get_unique_customer_ids(data: DataFrame) -> List[int]:
    return data['CUSTOMER_ID'].unique()


def get_unique_products(data: DataFrame) -> List[int]:
    return data['PRODUCT_ID'].unique()


def get_all_rows_with_customer(data: DataFrame, customer_id: int) -> List:
    return data[data["CUSTOMER_ID"] == customer_id]


def get_all_unique_dates(data: DataFrame) -> List:
    return data['DATE']


def get_all_rows_with_date(data: DataFrame, date: int) -> List:
    return data[data['DATE'] == date]

# Helper functions


def norm(x, trains_stats):
    return (x - train_stats['mean']) / train_stats['std']


def format_output(data):
    values = []
    for column in data:
        column = np.array(column)
        values.append(column)
    return tuple(values)


def build_model(length, data):
    outputs = []
    dense = []
    # Define model layers.
    input_layer = Input(shape=(len(train.columns),))
    first_dense = Dense(units='128', activation='relu')(input_layer)
    dense.append(first_dense)
    for i in range(length):
        new_output = Dense(units='1', name=str(data.columns[i]))(dense[-1])
        new_dense = Dense(units='128', activation='relu')(dense[-1])
        dense.append(new_dense)
        outputs.append(new_output)
        print()
    model = Model(inputs=input_layer, outputs=outputs)
    return model


if __name__ == "__main__":

    # Get data
    data = pd.read_csv("Output.csv", decimal=",")

    # Sort by date
    data.sort_values(by=['DATE'])

    # Convert categorical variables to numbers
    for i in data.select_dtypes('object').columns:
        le = LabelEncoder().fit(data[i])
        data[i] = le.transform(data[i])

    unique_customer_ids = get_unique_customer_ids(data)
    unique_product_ids = get_unique_products(data)
    print(f"Number of unique cutomers: {len(unique_customer_ids)}")
    print(f"Number of unique products: {len(unique_product_ids)}")
    for customer_id in unique_customer_ids:
        customer_data = get_all_rows_with_customer(data, customer_id)
        unique_products_in_customers_scope = get_unique_products(customer_data)
        print(
            f"Number of unique products ordered by {customer_id}: {len(unique_products_in_customers_scope)}")

        unique_dates = get_all_unique_dates(customer_data)

        # Consolidate lines with same date

        # Create dict with product mixes
        product_mixes = {}
        for date in unique_dates:
            product_mix = {}
            data_points = get_all_rows_with_date(customer_data, date)
            for _, data_point in data_points.iterrows():
                product_id = data_point['PRODUCT_ID']
                product_fraction = data_point['PRODUCT_FRACTION']
                product_mix[product_id] = product_fraction
            product_mixes[date] = product_mix

        # Create new dataframe
        consolidated_customer_data = pd.DataFrame(
            product_mixes).transpose().fillna(0)
        print(consolidated_customer_data)

        X, y = consolidated_customer_data.values[:,
                                                 :-1], consolidated_customer_data.values[:, -1]

        print("X:")
        print(X)
        print("y:")
        print(y)

        train, test = train_test_split(
            consolidated_customer_data, test_size=0.2, random_state=1)
        train, val = train_test_split(train, test_size=0.2, random_state=1)

        # Get PRICE and PTRATIO as the 2 outputs and format them as np
        # arrays
        # PTRATIO - pupil-teacher ratio by town
        train_stats = train.describe()
        for column_name in train_stats.columns:
            train_stats.pop(column_name)
        train_stats = train_stats.transpose()
        train_Y = format_output(train)
        test_Y = format_output(test)
        val_Y = format_output(val)

        norm_train_X = np.array(norm(train, train_stats))
        norm_test_X = np.array(norm(test, train_stats))
        norm_val_X = np.array(norm(val, train_stats))

        print("Jøss")

        model = build_model(91, consolidated_customer_data)
        # Specify the optimizer, and compile the model with loss functions for both outputs
        optimizer = tf.keras.optimizers.SGD(lr=0.001)
        loss = {}
        _metrics = {}
        for column_name in consolidated_customer_data.columns:
            loss[column_name] = 'mse'
            _metrics[column_name] = tf.keras.metrics.RootMeanSquaredError()
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=_metrics
                      )

        # Train the model for 100 epochs
        history = model.fit(norm_train_X, train_Y,
                            epochs=6, batch_size=10, validation_data=(norm_test_X, test_Y)
                            )

        break
