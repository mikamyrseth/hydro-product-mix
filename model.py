# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import pandas as pd
from pandas import DataFrame

from data.raw import Column

if __name__ == "__main__":
    # Get data _____________________________________________________________________________________
    data = pd.read_csv("./data/raw.csv", sep=";")

    # Clean data ___________________________________________________________________________________

    # Remove customer with ID 3793871 because their product mixes were corrupted. They summed to
    # 2 instead of 1.
    # data = data[data[Column.CUSTOMER_ID] != 3793871]

    # Pre-process data _____________________________________________________________________________

    # Aggregate product mixes
    # prod_mix_columns = [Column.CUSTOMER_ID, Column.YEAR, Column.MONTH, ]
    prod_mix_key = [Column.AREA, Column.PRODUCT_TYPE_ID, Column.PRODUCT_CATEGORY, Column.CUSTOMER_ID, Column.YEAR, Column.MONTH]
    # data.sort_values(by=prod_mix_columns, inplace=True)

    # https://stackoverflow.com/questions/44348426/pandas-groupby-custom-function-to-each-series
    # https://stackoverflow.com/questions/17679089/pandas-dataframe-groupby-two-columns-and-get-counts
    # https://stackoverflow.com/questions/43172970/python-pandas-groupby-aggregate-on-multiple-columns-then-pivot
    test: DataFrame = data.groupby(prod_mix_key)[Column.PRODUCT_FRACTION].sum().reset_index()
    test = test[1.1 < test[Column.PRODUCT_FRACTION]]
    test = test.groupby(Column.CUSTOMER_ID).size().reset_index()
    test.to_csv("test.csv", sep=";")

    # # One-hot encoding of certain columns
    # for column in [Column.AREA, Column.PRODUCT_TYPE_ID, Column.PRODUCT_CATEGORY]:
    #     one_hot = pd.get_dummies(data[column], prefix=column)
    #     data.drop(columns=[column], inplace=True)
    #     data = one_hot.merge(data, left_index=True, right_index=True)
    #
    # data.to_csv("test.csv", sep=";")

    # model = keras.Sequential()
    #
    # model.add(keras.Input(batch_shape=(None, None, 28)))
    # model.add(layers.LSTM(256, return_sequences=True))
    # model.add(layers.LSTM(256))
    # model.add(layers.Dense(2))
    #
    # print(model.summary())
