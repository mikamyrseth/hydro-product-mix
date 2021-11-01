from preprocessing.to_absolute_month import to_absolute_month

# with open("../raw.csv") as indata, open("../data/combined_month.csv", "w") as outdata:
#     outdata.write("AREA;PRODUCT_TYPE_ID;PRODUCT_CATEGORY;CUSTOMER_ID;PRODUCT_ID;PRODUCT_FRACTION"
#                   ";MONTHS_SINCE_JAN_2015\n")
#     for index, line in enumerate(indata):
#         if index == 0:
#             continue
#         line_data = line.split(";")
#         absolute_month = to_absolute_month(int(line_data[6]), int(line_data[7]))
#         outdata.write(f"{line_data[0]};{line_data[1]};{line_data[2]};{line_data[3]};{line_data[4]};{line_data[5]};{absolute_month}\n")

