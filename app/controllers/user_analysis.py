import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import json
import os
from app.threads.user_thread import manage_user_thread, load_thread_data, save_thread_data

# Paths
MODEL_PATH = "app/models/forecasting_model.pth"
DATA_PATH = "app/data/users.json"

# Load the trained model
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

# Load model
input_size = 12  # Adjust based on training features
model = ForecastingNN(input_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Load user data
with open(DATA_PATH, "r") as f:
    sample_data = json.load(f)

scaler = StandardScaler()  # To match training normalization

# Preprocess user data
def preprocess_user_data(user):
    total_income = sum(item["amount"] for item in user["incomes"])
    category_expenses = {}
    for expense in user["expenses"]:
        category = expense["category_name"]
        category_expenses[category] = category_expenses.get(category, 0) + expense["amount"]

    # Prepare input features
    categories = sorted(category_expenses.keys())
    features = [total_income] + [category_expenses.get(category, 0) for category in categories]
    return features, total_income, sum(category_expenses.values())

# Forecast future spending
def forecast_spending(total_income, total_expense):
    weekly_expense = total_expense / 7
    monthly_expense = weekly_expense * 30  # Approximate
    yearly_expense = weekly_expense * 365  # Approximate

    forecast_message = (
        f"If you continue spending at the current rate:\n"
        f"- Weekly spending: ${weekly_expense:.2f}\n"
        f"- Monthly spending: ${monthly_expense:.2f}\n"
        f"- Yearly spending: ${yearly_expense:.2f}\n"
    )

    if total_income > yearly_expense:
        forecast_message += "You are on track to save money annually. Great work!"
    else:
        forecast_message += "Warning: Expenses may exceed your income. Consider revising spending habits."

    return forecast_message, weekly_expense, monthly_expense, yearly_expense

# Generate chart data
def generate_chart_data(weekly_expense):
    weekly_data = [weekly_expense] * 7
    monthly_data = [weekly_expense * i for i in range(1, 31)]
    yearly_data = [weekly_expense * i for i in range(1, 366)]

    return {
        "weekly": weekly_data,
        "monthly": monthly_data[:30],
        "yearly": yearly_data[:365],
    }

# Analyze user
def analyze_user(user_id):
    # Find the user
    user_data = next((user for user in sample_data if user["user_ID"] == user_id), None)
    if not user_data:
        return {"status": "error", "message": f"No data found for User ID {user_id}."}

    # Manage or fetch the user's thread
    encrypted_filename, thread_file = manage_user_thread(user_id, user_data)

    # Preprocess and analyze
    features, total_income, total_expense = preprocess_user_data(user_data)

    # Normalize features
    features = scaler.fit_transform([features])
    features = torch.tensor(features, dtype=torch.float32)

    # Predict using the model
    with torch.no_grad():
        prediction = model(features).item()

    # Generate user-specific insights
    if prediction >= 0.7:
        insights = "You are saving at a high rate. Excellent financial discipline!"
    elif 0.5 <= prediction < 0.7:
        insights = "You are maintaining balanced finances with a modest saving trend."
    else:
        insights = "You are overspending. Consider reducing unnecessary expenses."

    # Forecast spending and generate chart data
    forecast_message, weekly_expense, monthly_expense, yearly_expense = forecast_spending(total_income, total_expense)
    chart_data = generate_chart_data(weekly_expense)

    # Save analysis data to the user's thread file
    thread_data = {
        "user_id": user_id,
        "insights": insights,
        "forecast": forecast_message,
        "chart_data": chart_data,
    }
    save_thread_data(encrypted_filename, thread_data)

    return {
        "status": "success",
        "user_id": user_id,
        "insights": insights,
        "forecast": forecast_message,
        "chart_data": chart_data,
        "thread_file": thread_file,
    }
