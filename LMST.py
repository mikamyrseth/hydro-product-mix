from typing import List
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  # to plot the time series plot
from sklearn import metrics  # for the evaluation
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf


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


# Kok https://analyticsindiamag.com/how-to-do-multivariate-time-series-forecasting-using-lstm/
def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])
        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)


if __name__ == "__main__":

    # Get data
    data = pd.read_csv("Output.csv", decimal=",")

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

        X_scaler = MinMaxScaler()
        Y_scaler = MinMaxScaler()
        X_data = X_scaler.fit_transform(
            data[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description', 'traffic_volume']])
        Y_data = Y_scaler.fit_transform(data)

        break
