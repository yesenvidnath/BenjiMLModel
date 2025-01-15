import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import json
import os
import datetime
from app.threads.user_thread import manage_user_thread, save_thread_data

# Paths
MODEL_PATH = "app/models/forecasting_model.pth"
DATA_PATH = "app/data/users.json"  # Added .json back

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

scaler = StandardScaler()  # To match training normalization

# Define upper and lower limits for income and expenses
INCOME_LIMIT = 1e6
EXPENSE_LIMIT = 1e5

# Preprocess user data
def preprocess_user_data(user, expected_input_size=12):
    total_income = sum(float(item["amount"]) for item in user.get("incomes", []) if item["amount"] is not None)
    total_income = min(total_income, INCOME_LIMIT)

    category_expenses = [
        min(float(expense["amount"]), EXPENSE_LIMIT) for expense in user.get("expenses", []) if expense["amount"] is not None
    ]

    features = [total_income] + category_expenses

    if len(features) < expected_input_size:
        features += [0] * (expected_input_size - len(features))
    elif len(features) > expected_input_size:
        features = features[:expected_input_size]

    total_expense = sum(category_expenses)

    return features, total_income, total_expense

# Analyze user
def analyze_user(user_id):
    with open(DATA_PATH, "r") as f:
        sample_data = json.load(f)

    user_data = next((user for user in sample_data if user["user_ID"] == user_id), None)
    if not user_data:
        return {"status": "error", "message": f"No data found for User ID {user_id}."}

    encrypted_filename, thread_file = manage_user_thread(user_id, user_data)
    features, total_income, total_expense = preprocess_user_data(user_data)

    if len(features) != input_size:
        return {"status": "error", "message": f"Feature vector size mismatch: expected {input_size}, got {len(features)}."}

    # Fit the scaler
    scaler.fit([features])
    features = scaler.transform([features])
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(features).item()

    if not (0 <= prediction <= 1):
        return {"status": "error", "message": f"Prediction value out of range: {prediction}"}

    # Calculate percentages and generate insights
    saving_percentage = (total_income - total_expense) / total_income * 100 if total_income != 0 else 0
    spending_percentage = total_expense / total_income * 100 if total_income != 0 else 0

    # Generate forecast for weekly, monthly, and yearly expenses
    weekly_expense = prediction * total_income / 52 if total_income != 0 else 0  # Adjust prediction to weekly scale
    monthly_expense = weekly_expense * 4.345
    yearly_expense = weekly_expense * 52

    if total_income == 0 and total_expense == 0:
        insights = "No income or expense data available."
        forecasting_message = "There were no incomes or expenses recorded."
    elif total_income == 0:
        insights = "No income data available."
        forecasting_message = "There were no incomes recorded."
    elif total_expense == 0:
        insights = "No expense data available."
        forecasting_message = "There were no expenses recorded."
    else:
        if saving_percentage > 50:
            insights = f"You are saving at a high rate of {saving_percentage:.2f}%. Excellent financial discipline! Your spending for the next week would be {weekly_expense:.2f}, and for the next month would be {monthly_expense:.2f}. For the entire year, it would be {yearly_expense:.2f}. Keep up the great work!"
            forecasting_message = "Your financial path shines bright with robust savings. Keep up the great work!"
        elif 20 <= saving_percentage <= 50:
            insights = f"You are maintaining balanced finances with a modest saving trend of {saving_percentage:.2f}%. Your spending for the next week would be {weekly_expense:.2f}, and for the next month would be {monthly_expense:.2f}. For the entire year, it would be {yearly_expense:.2f}. Continue this steady path to prosperity."
            forecasting_message = "You are treading a balanced financial journey. Continue this steady path to prosperity."
        else:
            insights = f"You are overspending at a rate of {spending_percentage:.2f}%. Consider reducing unnecessary expenses. Your spending for the next week would be {weekly_expense:.2f}, and for the next month would be {monthly_expense:.2f}. For the entire year, it would be {yearly_expense:.2f}. It's wise to curb unnecessary expenditures to secure your future."
            forecasting_message = "Beware of your financial course. It's wise to curb unnecessary expenditures to secure your future."

    chart_data = generate_chart_data(weekly_expense)

    analysis_data = {
        "user_id": user_id,
        "thread_id": encrypted_filename,
        "insights": insights,
        "forecast": {
            "total_income": total_income,
            "total_expense": total_expense,
            "weekly_expense": weekly_expense,
            "monthly_expense": monthly_expense,
            "yearly_expense": yearly_expense,
        },
        "chart_data": chart_data,
        "analysis_date": datetime.datetime.now().isoformat(),
        "forecasting_message": forecasting_message,
        "saving_percentage": f"{saving_percentage:.2f}%",
        "spending_percentage": f"{spending_percentage:.2f}%"
    }
    save_thread_data(encrypted_filename, analysis_data)

    return {
        "user_id": user_id,
        "thread_id": encrypted_filename,
        "insights": insights,
        "forecast": analysis_data["forecast"],
        "chart_data": analysis_data["chart_data"],
        "forecasting_message": forecasting_message,
        "saving_percentage": f"{saving_percentage:.2f}%",
        "spending_percentage": f"{spending_percentage:.2f}%"
    }

def generate_chart_data(weekly_expense):
    weekly_data = [weekly_expense] * 7
    monthly_data = [weekly_expense * (i + 1) for i in range(5)]  # 5 weeks to represent months
    yearly_data = [weekly_expense * (i + 1) for i in range(12)]  # 12 months in a year

    chart_data = {
        "weekly": {
            "x": [f"Day {i+1}" for i in range(7)],  # Days 1-7
            "y": weekly_data,
        },
        "monthly": {
            "x": [f"Week {i}" for i in range(5)],  # Weeks 0-4
            "y": monthly_data,
        },
        "yearly": {
            "x": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],  # Months of the year
            "y": yearly_data,
        },
    }

    return chart_data
