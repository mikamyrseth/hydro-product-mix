import pandas as pd
from pandas import DataFrame

from data.raw import Column


def analyse_products_by(column: Column, data: DataFrame) -> None:
    """
    Find properties of products when viewed in relation to supplied column.

    The results will be written into various files.

    :param column: The column to analyse products in relation to.
    :param data: The dataset.
    :return: None.
    """
    included = [column, Column.PRODUCT_ID]
    drop_columns = [f"{c}" for c in Column if c not in included]
    local_data: DataFrame = data.drop(columns=drop_columns)

    product_map: dict[str, set[str]] = dict()
    for index, row in local_data.iterrows():
        discriminator = str(row[column])
        product = row[Column.PRODUCT_ID]
        if discriminator in product_map:
            product_map[discriminator].add(product)
        else:
            product_map[discriminator] = {product}

    with open(f"product_intersections_by_{column}.csv", "w") as file:
        file.write("CURRENT_VALUE;" + ";".join(product_map) + "\n")
        for current_key, current_value in product_map.items():
            file.write(f"{current_key};" + ";".join([str(len(current_value.intersection(value))) for value in product_map.values()]) + "\n")


def analyse_products_by_columns(data: DataFrame):
    analyse_products_by(Column.AREA, data,)
    analyse_products_by(Column.PRODUCT_TYPE_ID, data,)
    analyse_products_by(Column.PRODUCT_CATEGORY, data,)


def main():
    data = pd.read_csv("../data/raw.csv", sep=";")

    analyse_products_by_columns(data)

    exit(0)


if __name__ == "__main__":
    main()
