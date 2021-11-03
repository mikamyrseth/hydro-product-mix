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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor


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


def get_consolidated_data_by_date(data):
    consolidated_data_by_month = {}
    for index, row in data.iterrows():
        print(f"Processing row {index}")
        date = row['DATE']
        product_id = row['PRODUCT_ID']
        fraction = row['PRODUCT_FRACTION']
        if date in consolidated_data_by_month:
            mixes = consolidated_data_by_month[date]
            if product_id in mixes:
                mix = mixes[product_id]
                mix['fraction'] = mix['fraction'] + fraction
                mix['count'] = mix['count'] + 1
            else:
                mix = {}
                mix['fraction'] = fraction
                mix['count'] = 1
                mixes[product_id] = mix
        else:
            mixes = {}
            mix = {}
            mix['fraction'] = fraction
            mix['count'] = 1
            mixes[product_id] = mix
            consolidated_data_by_month[date] = mixes
    # Flatten
    for date, mixes in consolidated_data_by_month.items():
        for product_id, mix in mixes.items():
            fraction_sum = mix['fraction']
            count = mix['count']
            mixes[product_id] = fraction_sum/count

    df = pd.DataFrame(consolidated_data_by_month).transpose()


def machine_learning_magic(df, model):
    X, y = df.values[:, :-1], df.values[:, -1]
    # ensure all data are floating point values
    X = X.astype('float32')
    # encode strings to integer
    y = LabelEncoder().fit_transform(y)
    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15)
    print("shapes")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model.fit(X_train, y_train, epochs=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)


if __name__ == "__main__":

    # Get data
    data = pd.read_csv("Output.csv", decimal=",")

    # Sort by date
    data.sort_values(by=['DATE'])

    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(70),
        # tf.keras.layers.Dense(70),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dense(3933),
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    # Convert categorical variables to numbers
    for i in data.select_dtypes('object').columns:
        le = LabelEncoder().fit(data[i])
        data[i] = le.transform(data[i])

    unique_customer_ids = get_unique_customer_ids(data)
    unique_product_ids = get_unique_products(data)
    print(f"Number of unique cutomers: {len(unique_customer_ids)}")
    print(f"Number of unique products: {len(unique_product_ids)}")

    customer_mixes = []
    count = 1
    for customer_id in unique_customer_ids:
        print(f"Processing customer {count} of {len(unique_customer_ids)}")
        customer_data = get_all_rows_with_customer(data, customer_id)
        unique_products_in_customers_scope = get_unique_products(data)
        # print(
        #     f"Number of unique products ordered by {customer_id}: {len(unique_products_in_customers_scope)}")

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
            keys = product_mix.keys()
            missing_products = [
                product for product in unique_products_in_customers_scope if product not in keys]
            for global_product_id in missing_products:
                product_mix[global_product_id] = 0

            product_mixes[date] = product_mix

        customer_mixes.append(product_mixes)
        count += 1

        # Create new dataframe
        consolidated_customer_data = pd.DataFrame(
            product_mixes).transpose()

        print("Using df:")
        print(consolidated_customer_data)

        # https://stackoverflow.com/questions/39263002/calling-fit-multiple-times-in-keras

        machine_learning_magic(consolidated_customer_data, model)
