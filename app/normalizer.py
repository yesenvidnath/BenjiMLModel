import pandas as pd
import json
import os

# Set paths for input and output
INPUT_FILE = 'app/data/user_data.json'
OUTPUT_DIR = 'app/data/'

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the JSON file
with open(INPUT_FILE, 'r') as f:
    user_data = json.load(f)

# Normalize the data
users = []
incomes = []
expenses = []

for user in user_data:
    # Extract user details
    user_id = user['user_ID']
    users.append({
        'user_ID': user_id,
        'user_name': user['user_name'],
        'user_email': user['user_email']
    })

    # Extract income details
    for income in user['incomes']:
        incomes.append({
            'user_ID': user_id,
            'source_name': income['source_name'],
            'amount': income['amount'],
            'frequency': income['frequency'],
            'description': income['description']
        })

    # Extract expense details
    for expense in user['expenses']:
        expenses.append({
            'user_ID': user_id,
            'expense_id': expense['expense_id'],
            'expense_date': expense['expense_date'],
            'reason_id': expense['reason_id'],
            'amount': expense['amount'],
            'description': expense['description'],
            'reason_text': expense['reason_text'],
            'category_name': expense['category_name']
        })

# Convert to DataFrames
users_df = pd.DataFrame(users)
incomes_df = pd.DataFrame(incomes)
expenses_df = pd.DataFrame(expenses)

# Save normalized data to CSV files
users_csv = os.path.join(OUTPUT_DIR, 'users.csv')
incomes_csv = os.path.join(OUTPUT_DIR, 'incomes.csv')
expenses_csv = os.path.join(OUTPUT_DIR, 'expenses.csv')

users_df.to_csv(users_csv, index=False)
incomes_df.to_csv(incomes_csv, index=False)
expenses_df.to_csv(expenses_csv, index=False)

# Display normalized data and save confirmation
print("Users Data (normalized):")
print(users_df.head())
print("\nIncomes Data (normalized):")
print(incomes_df.head())
print("\nExpenses Data (normalized):")
print(expenses_df.head())

print(f"\nNormalized files have been saved to '{OUTPUT_DIR}':")
print(f"- {users_csv}")
print(f"- {incomes_csv}")
print(f"- {expenses_csv}")
