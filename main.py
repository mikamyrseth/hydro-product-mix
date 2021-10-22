import plotly.graph_objects as go
import pandas
from prophet import Prophet
import time
from prophet.plot import plot_plotly, plot_components_plotly

# Prep data.
dataFrame = pandas.read_csv("Output.csv", decimal=",")
filtered_df = dataFrame[dataFrame["CUSTOMER_ID"] == 622116]
filtered_df = filtered_df[filtered_df["PRODUCT_ID"] == 71614]
model_input = filtered_df[["DATE", "PRODUCT_FRACTION"]]
model_input = model_input.rename(
    columns={'DATE': 'ds', 'PRODUCT_FRACTION': 'y'})
print(model_input.head(69))

# Fit model.
m = Prophet()
m.fit(model_input)

# Create prediction.
future = m.make_future_dataframe(31, 'M')
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
fig1 = m.plot(forecast)
fig1.show()
plot_plotly(m, forecast).show()
plot_components_plotly(m, forecast).show()
