import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pandas import DataFrame
from scipy.sparse import dok_matrix
from sklearn.preprocessing import LabelEncoder

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

    data = data.groupby(
        [Column.AREA, Column.PRODUCT_TYPE_ID, Column.PRODUCT_CATEGORY, Column.CUSTOMER_ID,
         Column.YEAR, Column.MONTH, Column.PRODUCT_ID]
    )[Column.PRODUCT_FRACTION].agg("sum")

    # Save cleaned data
    data.to_csv("./data/cleaned.csv", sep=";")


def create_model():
    model = keras.Sequential([
        keras.layers.Input(batch_shape=(None, None, 3934)),
        keras.layers.LSTM(1024, return_sequences=True),
        keras.layers.LSTM(1024),
        keras.layers.Dense(3934),
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["accuracy"],
    )

    return model


def fit_model(data, model):

    model.fit(data.batch(690), epochs=5)

    return model


def main():
    # clean_data()

    data = pd.read_csv("./data/cleaned.csv", sep=";")
    dataset = preprocess(data)
    model = create_model()
    model = fit_model(dataset, model)
    model.save('models/model_2')

    # model = tf.keras.models.load_model('models/model_1')
    # for x, y in dataset.skip(800).take(1):
    #     np.savetxt("x", x.numpy(), "%.6f")
    #     np.savetxt("y", y.numpy(), "%.6f")
    #     np.savetxt("p", tf.reshape(model.predict(tf.reshape(x, (1, 26, 3934))), (3934,)), "%.6f")


def preprocess(data: DataFrame):
    """ Pre-process data.

    Data should already be cleaned.

    :param data: DataFrame with data to preprocess.
    :return: Sparse matrices containing Xs and Ys.
    """

    # Encode product IDs
    le = LabelEncoder().fit(data[Column.PRODUCT_ID])
    data[Column.PRODUCT_ID] = le.transform(data[Column.PRODUCT_ID])
    data[Column.PRODUCT_ID] += 1

    # Store product mixes in customer history blocks

    customer_history_map: dict[tuple, dok_matrix] = dict()

    def generate_customer_key(product_mix: DataFrame) -> tuple:
        return (product_mix[Column.AREA].iloc[0], product_mix[Column.PRODUCT_TYPE_ID].iloc[0],
                product_mix[Column.PRODUCT_CATEGORY].iloc[0],
                product_mix[Column.CUSTOMER_ID].iloc[0])

    def get_product_mix_index(product_mix: DataFrame) -> int:
        # (1, 2015) is 27 element (index 26)
        # (M-1) + 12(Y-2015) + 26
        # M + 12Y - 1 - 12*2015 + 26
        return product_mix[Column.MONTH].iloc[0] + 12 * product_mix[Column.YEAR].iloc[0] - 24155

    def prepare_customer_history(product_mix: DataFrame) -> tuple:
        """ Ensure customer, with area, product type and category has a sparse matrix in the
        history to store product mixes in.

        :param product_mix: The product mix which customer information should be extracted from.
        :return: Key where customer product mixes are stored in history.
        """
        key = generate_customer_key(product_mix)
        if key not in customer_history_map:
            # Create new DataFrame
            d = dok_matrix((105, 3934), dtype="float32")
            d[:, 0] = 1
            customer_history_map[key] = d
        return key

    def save_product_mix(product_mix: DataFrame) -> None:
        key = prepare_customer_history(product_mix)
        product_mix_index = get_product_mix_index(product_mix)
        customer_history_map[key][product_mix_index, 0] = 0
        for index, row in product_mix.iterrows():
            product_id = row[Column.PRODUCT_ID]
            customer_history_map[key][product_mix_index, product_id] = row[Column.PRODUCT_FRACTION]

    print("Customer history construction started")

    data.sort_values(by=PROD_MIX_KEY, inplace=True)
    data.groupby(PROD_MIX_KEY).apply(save_product_mix)
    print("\tCustomer records completely traversed")

    def data_generator():
        histories = customer_history_map.values()
        for history in histories:
            dense = history.toarray()
            for i in range(26, 105):
                yield dense[i - 26:i, :], dense[i, :]

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(26, 3934), dtype=tf.float32),
            tf.TensorSpec(shape=(3934,), dtype=tf.float32),
        )
    )

    return dataset


if __name__ == "__main__":
    main()
    exit(0)
