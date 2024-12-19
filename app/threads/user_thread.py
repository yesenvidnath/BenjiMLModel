import os
import json
from hashlib import sha256

# Path to the threads data directory
THREAD_DATA_PATH = "app/threads/data/"
os.makedirs(THREAD_DATA_PATH, exist_ok=True)

# Generate a unique encrypted filename based on the user ID
def generate_encrypted_filename(user_id):
    return sha256(str(user_id).encode()).hexdigest() + ".json"

# Manage user thread (create if it doesn't exist)
def manage_user_thread(user_id, user_data):
    # Generate file path for the user
    encrypted_filename = generate_encrypted_filename(user_id)
    thread_file = os.path.join(THREAD_DATA_PATH, encrypted_filename)

    # Create the file if it doesn't exist
    if not os.path.exists(thread_file):
        with open(thread_file, "w") as f:
            json.dump(user_data, f, indent=2)

    return encrypted_filename, thread_file

# Load thread data for a user
def load_thread_data(encrypted_filename):
    thread_file = os.path.join(THREAD_DATA_PATH, encrypted_filename)
    if os.path.exists(thread_file):
        with open(thread_file, "r") as f:
            return json.load(f)
    return None

# Save updated thread data for a user
def save_thread_data(encrypted_filename, data):
    thread_file = os.path.join(THREAD_DATA_PATH, encrypted_filename)
    with open(thread_file, "w") as f:
        json.dump(data, f, indent=2)
