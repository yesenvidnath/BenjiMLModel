import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# File paths
DATA_DIR = "app/data/"
MODEL_DIR = "app/models/"
MODEL_PATH = os.path.join(MODEL_DIR, "forecasting_model.pth")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Load normalized data
users_df = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
incomes_df = pd.read_csv(os.path.join(DATA_DIR, "incomes.csv"))
expenses_df = pd.read_csv(os.path.join(DATA_DIR, "expenses.csv"))

# Preprocess data
def preprocess_data(incomes_df, expenses_df):
    # Aggregate total income per user
    income_agg = incomes_df.groupby('user_ID').agg(total_income=('amount', 'sum')).reset_index()

    # Aggregate total expenses per user per category
    expense_agg = expenses_df.groupby(['user_ID', 'category_name']).agg(total_expense=('amount', 'sum')).reset_index()

    # Pivot expense data to create one column per category
    expense_pivot = expense_agg.pivot(index='user_ID', columns='category_name', values='total_expense').fillna(0)

    # Merge income and expense data
    data = income_agg.merge(expense_pivot, on='user_ID', how='left').fillna(0)

    # Calculate net savings or loss
    data['net_savings'] = data['total_income'] - data.iloc[:, 2:].sum(axis=1)

    return data

def time_series_preparation(expenses_df):
    # Aggregate daily expenses
    expenses_df['expense_date'] = pd.to_datetime(expenses_df['expense_date'])
    daily_expenses = expenses_df.groupby('expense_date').agg(total_expense=('amount', 'sum')).reset_index()

    return daily_expenses

# Prepare data
data = preprocess_data(incomes_df, expenses_df)
daily_expenses = time_series_preparation(expenses_df)

# Prepare features for regression
X = data.drop(columns=['user_ID', 'net_savings'])
y = (data['net_savings'] > 0).astype(int)  # 1 if saving, 0 if losing

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define the neural network model
class ForecastingNN(nn.Module):
    def __init__(self, input_size):
        super(ForecastingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize neural network
input_size = X_train.shape[1]
model = ForecastingNN(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the neural network
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the neural network
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred >= 0.5).float()
    accuracy = (y_pred_class == y_test).sum() / y_test.size(0)
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')

# Save the neural network model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Time-series forecasting with ARIMA
arima_model = ARIMA(daily_expenses['total_expense'], order=(5, 1, 0))
arima_result = arima_model.fit()
print("ARIMA Model Summary:")
print(arima_result.summary())

# Forecasting with ARIMA
arima_forecast = arima_result.forecast(steps=30)  # Forecast for the next 30 days
print("ARIMA Forecast:")
print(arima_forecast)

# Time-series forecasting with Prophet
prophet_data = daily_expenses.rename(columns={'expense_date': 'ds', 'total_expense': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)
print("Prophet Forecast:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

# Visualize ARIMA and Prophet Forecasts
plt.figure(figsize=(12, 6))
plt.plot(daily_expenses['expense_date'], daily_expenses['total_expense'], label='Historical Data')
plt.plot(future['ds'], forecast['yhat'], label='Prophet Forecast')
plt.axhline(arima_forecast.mean(), color='r', linestyle='--', label='ARIMA Forecast Avg')
plt.legend()
plt.title("Spending Forecast")
plt.xlabel("Date")
plt.ylabel("Total Expense ($)")
plt.grid()
plt.show()
