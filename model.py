# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import pandas as pd

from data.raw import Column

if __name__ == "__main__":
    # Get data
    data = pd.read_csv("./data/raw.csv", sep=";")

    prod_mix_columns = [f"{Column.CUSTOMER_ID}", f"{Column.YEAR}", f"{Column.MONTH}", ]
    data.sort_values(by=prod_mix_columns,
                     ascending=[True, True, True], inplace=True)

    test = data
    test["COUNT"] = 1
    test = data.groupby(prod_mix_columns)["COUNT"].sum().reset_index()
    test.to_csv("test.csv", sep=";")

    # for column in [Column.AREA, Column.PRODUCT_TYPE_ID, Column.PRODUCT_CATEGORY]:
    #     one_hot = pd.get_dummies(data[f"{column}"], prefix=f"{column}")
    #     data.drop(columns=[f"{column}"], inplace=True)
    #     data = one_hot.merge(data, left_index=True, right_index=True)
    #
    # data.to_csv("test.csv", sep=";")

    # model = keras.Sequential()
    #
    # model.add(keras.Input(batch_shape=(None, None, 100)))
    # model.add(layers.LSTM(256, return_sequences=True))
    # model.add(layers.LSTM(256))
    # model.add(layers.Dense(4))
    #
    # print(model.summary())
