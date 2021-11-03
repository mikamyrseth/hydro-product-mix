import numpy as np
import pandas as pd
from pandas import DataFrame
from tensorflow import keras
from tensorflow.keras import layers

from data.raw import Column

PROD_MIX_KEY = [Column.AREA, Column.PRODUCT_CATEGORY, Column.PRODUCT_TYPE_ID, Column.CUSTOMER_ID,
                Column.YEAR, Column.MONTH]


def clean_data() -> None:
    """ Clean data.

    Load and clean data. Save cleaned data in own file.

    Quite expensive.

    :return: None.
    """
    data = pd.read_csv("./data/cleaned.csv", sep=";")

    # Remove product mixes summing to 2
    # Find sum of product fraction in each product mix
    local = data.groupby(PROD_MIX_KEY)[Column.PRODUCT_FRACTION].sum().reset_index()
    # Find invalid product fraction
    local = local[1.1 < local[Column.PRODUCT_FRACTION]]
    # Filter out from data based on local
    for _, row in local.iterrows():
        ting = data.loc[
            (data[Column.AREA] == row[Column.AREA]) &
            (data[Column.PRODUCT_CATEGORY] == row[Column.PRODUCT_CATEGORY]) &
            (data[Column.PRODUCT_TYPE_ID] == row[Column.PRODUCT_TYPE_ID]) &
            (data[Column.CUSTOMER_ID] == row[Column.CUSTOMER_ID]) &
            (data[Column.YEAR] == row[Column.YEAR]) &
            (data[Column.MONTH] == row[Column.MONTH])
            ]

        for index, _ in ting.iterrows():
            data.loc[index, Column.PRODUCT_FRACTION] /= 2

    # Save cleaned data
    data.to_csv("./data/cleaned.csv", sep=";")


def create_model(data: DataFrame):
    model = keras.Sequential([
        layers.Input(batch_shape=(None, None, 3934)),
        layers.LSTM(256, return_sequences=True),
        layers.LSTM(256),
        layers.Dense(3934),
    ])

    # Fit

    return model


def main():
    # clean_data()

    data = pd.read_csv("./data/cleaned.csv", sep=";")
    data = preprocess(data)
    model = create_model(data)

    print(model.summary())


def preprocess(data: DataFrame) -> DataFrame:
    """ Pre-process data.

    Data should already be cleaned.

    :param data: DataFrame with data to preprocess.
    :return: DataFrame with preprocessed data.
    """

    sequence_map: dict[tuple, DataFrame] = dict()

    def ensure_sequence_key(product_mix: DataFrame):
        # print(product_mix.items()[0][Column.AREA])
        key = (product_mix[Column.AREA].iloc[0], product_mix[Column.PRODUCT_TYPE_ID].iloc[0],
               product_mix[Column.PRODUCT_CATEGORY].iloc[0],
               product_mix[Column.CUSTOMER_ID].iloc[0])
        print(key)
        if key not in sequence_map:
            d = DataFrame(np.zeros(shape=(104, 3934)), dtype=pd.SparseDtype("float", 0))
            d[0] = pd.arrays.SparseArray(np.ones(104, ), dtype=pd.SparseDtype("float", 1))
            sequence_map[key] = d
            print(d.memory_usage())
            print(d.dtypes)
            print(d.sparse.density)

    def a(product_mix: DataFrame):
        ensure_sequence_key(product_mix)

        for index, row in product_mix.iterrows():
            print(row[Column.PRODUCT_ID])
            print(row[Column.PRODUCT_FRACTION])
        exit(0)

    data.sort_values(by=PROD_MIX_KEY, inplace=True)
    data.groupby(PROD_MIX_KEY).apply(a)


if __name__ == "__main__":
    main()
    exit(0)
