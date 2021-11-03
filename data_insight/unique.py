import pandas as pd
from pandas import DataFrame

from data.raw import Column


def print_unique_values_by_columns(data: DataFrame):
    product_mix_key = [Column.AREA, Column.PRODUCT_CATEGORY, Column.PRODUCT_TYPE_ID, Column.CUSTOMER_ID, Column.MONTH, Column.YEAR]
    local_data: DataFrame = data.drop(columns=[c for c in data.columns if c not in product_mix_key])
    local_data.drop_duplicates(inplace=True)

    with open("unique.csv", "w") as file:
        file.write("TYPE;COUNT_OF_UNIQUE\n")
        file.write(f"ROWS;{len(data)}\n")
        for col in data.columns:
            file.write(f"{col};{len(data[col].unique())}\n")
        file.write(f"PRODUCT_MIXES;{len(local_data)}")


def main():
    data = pd.read_csv("../data/raw.csv", sep=";")

    print_unique_values_by_columns(data)

    exit(0)


if __name__ == "__main__":
    main()
