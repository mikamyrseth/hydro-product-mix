from typing import List
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  # to plot the time series plot
from sklearn import metrics  # for the evaluation
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf


def get_customer_ids(data: DataFrame) -> List[int]:
    return data['CUSTOMER_ID'].unique()


def get_all_rows_with_customer(data: DataFrame, customer_id: int) -> List:
    return data[data["CUSTOMER_ID"] == customer_id]


if __name__ == "__main__":
    data = pd.read_csv("Output.csv", decimal=",")
    customer_ids = get_customer_ids(data)
    for customer_id in customer_ids:
        customer_data = get_all_rows_with_customer(data, customer_id)
