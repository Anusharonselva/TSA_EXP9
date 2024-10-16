# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 
### Developed by: ANUSHARON.S
### Registration no.: 212222240010

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np

date_range = pd.date_range(start='2020-01-01', periods=1000, freq='D')

temperature_values = np.random.randint(15, 30, size=len(date_range))

weather_df = pd.DataFrame({'date': date_range, 'temperature': temperature_values})

weather_df.to_csv('weather_data.csv', index=False)
import pandas as pd

df = pd.read_csv('/content/weather_data.csv', parse_dates=['date'], index_col='date')
print(df.head())
print(df.describe())
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12, 6))
plt.plot(df['temperature'])  # Replace with the actual column name
plt.title('Time Series Plot of Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()

def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

adf_test(df['temperature'])  # Replace with the actual column name

plot_acf(df['temperature'], lags=30)
plot_pacf(df['temperature'], lags=30)
plt.show()
df['temperature_diff'] = df['temperature'].diff()
df['temperature_diff'].dropna(inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df['temperature_diff'])
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Differenced Temperature')
plt.show()

adf_test(df['temperature_diff'])
from statsmodels.tsa.arima.model import ARIMA

p, d, q = 1, 1, 1  # Replace with actual values from ACF/PACF analysis
model = ARIMA(df['temperature'], order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())
forecast = model_fit.forecast(steps=10)  # Forecast for next 10 days
print(forecast)

plt.figure(figsize=(12, 6))
plt.plot(df['temperature'], label='Original Data')
plt.plot(forecast.index, forecast, color='red', label='Forecast', marker='o')
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
from pmdarima import auto_arima

auto_model = auto_arima(df['temperature'], seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())
from sklearn.metrics import mean_squared_error

train_size = int(len(df) * 0.8)
train, test = df['temperature'][:train_size], df['temperature'][train_size:]

model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

predictions = model_fit.forecast(steps=len(test))
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Data')
plt.plot(test.index, predictions, color='red', label='Predicted Data')
plt.title('Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
```
### OUTPUT:
![Screenshot 2024-10-16 145325](https://github.com/user-attachments/assets/39a04194-0b47-4e85-be2b-bf49126b980a)
![Screenshot 2024-10-16 145338](https://github.com/user-attachments/assets/8b629756-38a4-4972-a316-ee74e3f69a1e)
![Screenshot 2024-10-16 145348](https://github.com/user-attachments/assets/813607f8-6a98-42b4-b23d-813dc0762520)
![Screenshot 2024-10-16 145402](https://github.com/user-attachments/assets/907bb654-de38-42c1-bea8-5e570f0310c6)
![Screenshot 2024-10-16 145414](https://github.com/user-attachments/assets/0351fac0-9ff5-4802-ae74-973ddd337f94)
![Screenshot 2024-10-16 145426](https://github.com/user-attachments/assets/57c3d324-da94-499d-87a3-89a156d7d0dc)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
