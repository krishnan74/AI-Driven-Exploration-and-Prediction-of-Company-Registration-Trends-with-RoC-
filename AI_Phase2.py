import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv("Data_Gov_Tamil_Nadu.csv")

# Data preprocessing for classification
# - Handle missing values
# - Encode categorical variables
# - Feature engineering for classification (if needed)

# Split the data into features (X) and the target (y)
X = data.drop(columns=["COMPANY_STATUS"])
y = data["COMPANY_STATUS"]

# Split the data into training and testing sets for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier for classification
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions for classification
y_pred = rf_classifier.predict(X_test)

# Evaluate the classification model
classification_accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

# Data preprocessing for time series forecasting
# Assuming you have a date column (DATE_OF_REGISTRATION)
data["DATE_OF_REGISTRATION"] = pd.to_datetime(data["DATE_OF_REGISTRATION"])
time_series_data = data.groupby("DATE_OF_REGISTRATION").size().reset_index(name="count")
time_series_data.set_index("DATE_OF_REGISTRATION", inplace=True)

# Fit an ARIMA model for time series forecasting
arima_model = ARIMA(time_series_data, order=(5, 1, 0))  # You can adjust the order based on your data
arima_model_fit = arima_model.fit(disp=0)

# Forecast future registration trends
forecast_periods = 12  # Adjustment of the number of forecast periods
forecast, stderr, conf_int = arima_model_fit.forecast(steps=forecast_periods)

# Plot the time series and forecast for time series forecasting
plt.figure(figsize=(12, 6))
plt.plot(time_series_data, label="Actual")
plt.plot(
    pd.date_range(
        start=time_series_data.index[-1], periods=forecast_periods, closed="right"
    ),
    forecast,
    label="Forecast",
    color="red",
)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Registration Count")
plt.title("Company Registration Trends")
plt.show()

# Print classification results and forecast for time series
print("Classification Results:")
print(f"Accuracy: {classification_accuracy}")
print(classification_report)

print("\nTime Series Forecast:")
print(f"Forecasted values: {forecast}")
