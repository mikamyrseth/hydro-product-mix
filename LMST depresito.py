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
import plotly.graph_objects as go
import pandas
from prophet import Prophet
import time
from prophet.plot import plot_plotly, plot_components_plotly


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


if __name__ == "__main__":

    # Get data
    data = pd.read_csv("Output.csv", decimal=",")

    # Sort by date
    data.sort_values(by=['DATE'])

    # Convert categorical variables to numbers
    # for i in data.select_dtypes('object').columns:
    #     le = LabelEncoder().fit(data[i])
    #     data[i] = le.transform(data[i])

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

        for column_name in consolidated_customer_data.columns

        # Prep data.
        model_input = filtered_df[["DATE", "PRODUCT_FRACTION"]]
        model_input = model_input.rename(
            columns={'DATE': 'ds', 'PRODUCT_FRACTION': 'y'})
        print(model_input.head(69))

        # Fit model.
        m = Prophet()
        m.fit(model_input)

        # Create prediction.
        future = m.make_future_dataframe(31, 'M')
        forecast = m.predict(future)
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        fig1 = m.plot(forecast)
        fig1.show()
        plot_plotly(m, forecast).show()
        plot_components_plotly(m, forecast).show()

        break
