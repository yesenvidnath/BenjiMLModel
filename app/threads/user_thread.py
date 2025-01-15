import os
import json
from hashlib import sha256
import datetime  # Ensure datetime is imported

# Dynamic dataset path
DATA_PATH = None

# Path to thread-specific data directory
THREAD_DATA_PATH = "app/threads/data/"
os.makedirs(THREAD_DATA_PATH, exist_ok=True)

def set_data_path(dataset_variable):
    global DATA_PATH
    DATA_PATH = f"app/data/{dataset_variable}"

# Generate a unique encrypted filename for user-specific threads
def generate_encrypted_filename(user_id):
    return sha256(str(user_id).encode()).hexdigest() + ".json"

# Manage user thread
def manage_user_thread(user_id, user_data):
    encrypted_filename = generate_encrypted_filename(user_id)
    thread_file = os.path.join(THREAD_DATA_PATH, encrypted_filename)

    if not os.path.exists(thread_file):
        # Initialize thread data with history
        with open(thread_file, "w") as f:
            thread_data = {
                "user_id": user_id,
                "history": []  # History list for tracking analysis
            }
            json.dump(thread_data, f, indent=2)

    return encrypted_filename, thread_file

# Load user thread data
def load_thread_data(encrypted_filename):
    thread_file = os.path.join(THREAD_DATA_PATH, encrypted_filename)
    if os.path.exists(thread_file):
        with open(thread_file, "r") as f:
            return json.load(f)
    return None

# Save user thread data with timestamp
def save_thread_data(encrypted_filename, new_data):
    thread_file = os.path.join(THREAD_DATA_PATH, encrypted_filename)
    if os.path.exists(thread_file):
        with open(thread_file, "r") as f:
            thread_data = json.load(f)
    else:
        thread_data = {"user_id": new_data["user_id"], "history": []}

    # Append the new data with timestamp to the history
    new_data_with_timestamp = {
        "timestamp": datetime.datetime.now().isoformat(),  # Fix here
        "data": new_data
    }
    thread_data["history"].append(new_data_with_timestamp)

    # Save the updated thread data
    with open(thread_file, "w") as f:
        json.dump(thread_data, f, indent=2)
