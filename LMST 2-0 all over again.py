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

# kok https://www.relataly.com/time-series-forecasting-multi-step-regression-using-neural-networks-with-multiple-outputs-in-python/5800/#h-step-3-train-the-multi-output-neural-network-model


# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, input_sequence_length time steps per sample, and f features
def partition_dataset(input_sequence_length, output_sequence_length, data,):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(input_sequence_length, data_len - output_sequence_length):
        # contains input_sequence_length values 0-input_sequence_length * columns
        x.append(data.iloc[i-input_sequence_length:i, :])
        # contains the prediction values for validation (3rd column = Close),  for single-step prediction
        y.append(data.iloc[i:i + output_sequence_length, 5])

    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y


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

        # Predict customer.

        input_sequence_length = 77
        output_sequence_length = 10

        consolidated_customer_data.to_csv(f"{customer_id}.csv")

        arff.dump(f"{customer_id}.arff", consolidated_customer_data.values,
                  relation='relation name', names=consolidated_customer_data.columns)

        df_train = consolidated_customer_data.copy()
        np_scaled = consolidated_customer_data.copy()

        train_data_length = math.ceil(np_scaled.shape[0] * 0.8)

        # Create the training and test data
        train_data = np_scaled[0:train_data_length]
        test_data = np_scaled[train_data_length - input_sequence_length:]

        # Configure the neural network model
        model = Sequential()
        n_output_neurons = 91

        x_train, y_train = partition_dataset(
            input_sequence_length, output_sequence_length, train_data)
        x_test, y_test = partition_dataset(
            input_sequence_length, output_sequence_length, test_data)

        # Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
        n_input_neurons = x_train.shape[1] * x_train.shape[2]
        print(n_input_neurons, x_train.shape[1], x_train.shape[2])
        model.add(LSTM(n_input_neurons, return_sequences=True,
                       input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(n_input_neurons, return_sequences=False))
        model.add(Dense(20))
        model.add(Dense(n_output_neurons))

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Training the model
        epochs = 10
        batch_size = 16
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test)
                            )

        # Plot training & validation loss values
        fig, ax = plt.subplots(figsize=(10, 5), sharex=True)
        plt.plot(history.history["loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
        plt.legend(["Train", "Test"], loc="upper left")
        plt.grid()
        plt.show()

        break
