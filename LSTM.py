import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pandas import DataFrame
from scipy.sparse import dok_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.backend as kb
import time

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
    local = data.groupby(PROD_MIX_KEY)[
        Column.PRODUCT_FRACTION].sum().reset_index()
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


def clean_prediction(predictions: np.ndarray) -> np.ndarray:
    """
    Clean predictions.

    :param predictions:
        A NumPy NDArray with shape (None, 3934) containing predictions to be cleaned.
    :return:
        The cleaned predictions.
    """

    # Copy empty product
    empty = predictions[:, 0]

    # Remove small and negative values.
    local = np.where(predictions <= 0.039, 0, predictions)

    if empty > 0.5:
        # Reset empty product
        local[:, 0] = 0

        # Normalize rows
        row_sums = local.sum(axis=1)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        local = local / row_sums[:, np.newaxis]

        # Insert empty product
        local[:, 0] = empty

        return local
    else:
        local = local*0
        # Insert empty product
        local[:, 0] = 1
        return local


def create_model():
    model = keras.Sequential([
        keras.layers.Input(batch_shape=(None, None, 3934)),
        keras.layers.LSTM(1024, return_sequences=True),
        keras.layers.LSTM(1024),
        keras.layers.Dense(3934),
    ])

    def custom_loss(y_actual, y_pred):
        # mask = K.less(y_pred, y_actual)
        custom_loss = kb.square(y_actual-y_pred)
        squared_again = kb.square(custom_loss)
        squared_again = kb.square(squared_again)
        squared_again = kb.square(squared_again)

    model.compile(
        optimizer="adam",
        loss='mse',
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

    # model = create_model()
    # model = fit_model(dataset, model)
    # model.save('models/model_1')

    model = tf.keras.models.load_model(
        'models/model_2', compile=False)

    start = time.time()
    print("hello")

    count = 0

    # Predict all and save
    # for x, y in dataset:
    #     np.savetxt("x.txt", x.numpy(), "%.6f")
    #     np.savetxt("y.txt", y.numpy(), "%.6f")
    #     np.savetxt(f"predictions/p_{count}.txt", tf.reshape(model.predict(
    #         tf.reshape(x, (1, 79, 3934))), (3934,)), "%.6f")

    #     count += 1

    # Predict all and save, but remove empty product
    empty_products = []
    x_empty_products = []
    y_empty_products = []
    for x, y in dataset:
        prediction = model.predict(tf.reshape(x, (1, 79, 3934)))

        cleaned_prediction = clean_prediction(prediction).reshape((3934,))

        prediction = prediction.reshape((3934,))

        last_x = x[-1]
        x_empty_product = last_x[0]
        x_empty_products.append(x_empty_product)
        rest_of_x = last_x[1:]
        np.savetxt(f"xdata/x_{count}.txt", rest_of_x, "%.6f")

        y_empty_product = y[0]
        y_empty_products.append(y_empty_product)
        rest_of_y = y[1:]
        np.savetxt(f"ydata/y_{count}.txt", rest_of_y, "%.6f")

        empty_product = prediction[0]
        rest_of_prediction = prediction[1:]
        rest_of_clean_prediction = cleaned_prediction[1:]
        empty_products.append(empty_product)

        np.savetxt(
            f"clean_predictions_3/p_{count}.txt", rest_of_clean_prediction, "%.6f")
        np.savetxt(
            f"raw_predictions_3/p_{count}.txt", rest_of_prediction, "%.6f")

        count += 1

    end = time.time()
    print(end - start)
    np.savetxt(f"empty_products.txt", empty_products, "%.6f")
    np.savetxt(f"x_empty_products.txt", x_empty_products, "%.6f")
    np.savetxt(f"y_empty_products.txt", y_empty_products, "%.6f")
    model_path = "models/model_2"

    # model = create_model()
    # model = fit_model(dataset, model)
    # model.save(model_path)

    # print("Calculating prediction")
    # model = tf.keras.models.load_model(model_path)
    # for x, y in dataset.skip(3).take(1):
    #     np.savetxt(f"x", x.numpy(), "%.6f")
    #     np.savetxt(f"y", y.numpy(), "%.6f")
    #     p = model.predict(tf.reshape(x, (1, 26, 3934)))
    #     p = clean_prediction(p).reshape((3934,))
    #     np.savetxt(f"p", p, "%.6f")


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
            customer_history_map[key][product_mix_index,
                                      product_id] = row[Column.PRODUCT_FRACTION]

    print("Customer history construction started")

    data.sort_values(by=PROD_MIX_KEY, inplace=True)
    data.groupby(PROD_MIX_KEY).apply(save_product_mix)
    print("\tCustomer records completely traversed")

    def data_generator():
        histories = customer_history_map.values()
        for history in histories:
            dense = history.toarray()
            yield dense[25:104, :], dense[104, :]

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(79, 3934), dtype=tf.float32),
            tf.TensorSpec(shape=(3934,), dtype=tf.float32),
        )
    )

    return dataset


if __name__ == "__main__":
    main()
    exit(0)
