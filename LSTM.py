from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
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

    # Encode product IDs
    le = LabelEncoder().fit(data[Column.PRODUCT_ID])
    data[Column.PRODUCT_ID] = le.transform(data[Column.PRODUCT_ID])
    data[Column.PRODUCT_ID] += 1

    # Store product mixes in customer history blocks

    customer_history_map: dict[tuple, DataFrame] = dict()

    def generate_customer_key(product_mix: DataFrame) -> tuple:
        return (product_mix[Column.AREA].iloc[0], product_mix[Column.PRODUCT_TYPE_ID].iloc[0],
                product_mix[Column.PRODUCT_CATEGORY].iloc[0],
                product_mix[Column.CUSTOMER_ID].iloc[0])

    def get_product_mix_index(product_mix: DataFrame) -> int:
        # (1, 2015) is 26 element (index 25)
        # (M-1) + 12(Y-2015) + 25
        # M + 12Y - 1 - 12*2015 + 25
        return product_mix[Column.MONTH].iloc[0] + 12 * product_mix[Column.YEAR].iloc[0] - 24156

    last_key: Optional[tuple] = None

    def prepare_customer_history(product_mix: DataFrame) -> tuple:
        """ Ensure customer, with area, product type and category has a DataFrame in the history to
        store product mixes in.

        It is assumed that all product mixes with the same consumer will appear in a row. The
        moment a new key appears, the old customer history will be converted to an immutable format.

        :param product_mix: The product mix which customer information should be extracted from.
        :return: Key where customer product mixes are stored in history.
        """
        key = generate_customer_key(product_mix)
        if key not in customer_history_map:
            # Create new DataFrame
            d = DataFrame(np.zeros(shape=(104, 3934)), dtype="float32")
            d[0] = 1
            customer_history_map[key] = d

            nonlocal last_key
            # Clean up previous DataFrame
            if last_key is not None:
                customer_history_map[last_key] = customer_history_map[last_key].astype()
            last_key = key
        return key

    def save_product_mix(product_mix: DataFrame) -> None:
        key = prepare_customer_history(product_mix)
        product_mix_index = get_product_mix_index(product_mix)
        for index, row in product_mix.iterrows():
            product_id = row[Column.PRODUCT_ID]
            sparse_data_type = customer_history_map[key].dtypes[product_id]
            customer_history_map[key][product_id] = customer_history_map[key][product_id].sparse.to_dense()
            customer_history_map[key].loc[product_mix_index, product_id] = row[Column.PRODUCT_FRACTION]
            customer_history_map[key][product_id] = customer_history_map[key][product_id].astype(sparse_data_type)

    data.sort_values(by=PROD_MIX_KEY, inplace=True)
    data.groupby(PROD_MIX_KEY).apply(save_product_mix)


if __name__ == "__main__":
    main()
    exit(0)
