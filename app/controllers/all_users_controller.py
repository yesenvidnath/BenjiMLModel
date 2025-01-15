import json
import os
import datetime
from app.threads.user_thread import manage_user_thread, save_thread_data, set_data_path
from app.controllers.user_analysis import analyze_user

# Paths
DATA_PATH = "app/data/users.json"
SENT_DATA_PATH = "app/data/SentData/"
LOG_PATH = "app/logs/fetching_log.txt"

os.makedirs(SENT_DATA_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def fetch_all_users(new_data):
    try:
        # Function to clean and convert amount fields to numeric
        def clean_amount_fields(data):
            if isinstance(data, list):
                for item in data:
                    clean_amount_fields(item)
            elif isinstance(data, dict):
                for key, value in data.items():
                    if key == "amount" and isinstance(value, str):
                        try:
                            data[key] = float(value)
                        except ValueError:
                            data[key] = 0.0  # Handle invalid numeric strings gracefully
                    else:
                        clean_amount_fields(value)

        # Clean the new data
        clean_amount_fields(new_data)

        # Save incoming data to SENT_DATA_PATH
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"user_data_{timestamp}.json"
        dataset_path = os.path.join(SENT_DATA_PATH, dataset_name)
        with open(dataset_path, "w") as f:
            json.dump(new_data, f, indent=2)

        # Load existing user_data.json
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        # Create a map of existing user IDs for quick lookup
        existing_user_ids = {user["user_ID"]: user for user in existing_data}

        # Update existing data with new data or add new users
        for user in new_data:
            if user["user_ID"] in existing_user_ids:
                # Update the existing user's data
                existing_user_ids[user["user_ID"]].update(user)
            else:
                # Add the new user
                existing_data.append(user)

        # Save the updated user_data.json
        with open(DATA_PATH, "w") as f:
            json.dump(existing_data, f, indent=2)

        # Set the dynamic data path for threads
        set_data_path(dataset_name)

        # Process each user
        result_data = []
        log_data = []
        for user in new_data:
            user_id = user["user_ID"]

            # Create or update thread
            encrypted_filename, thread_file = manage_user_thread(user_id, user)

            # Perform analysis
            user_analysis = analyze_user(user_id)

            # Save thread data
            save_thread_data(encrypted_filename, user_analysis)

            # Collect results
            result_data.append(user_analysis)
            log_data.append(
                {
                    "user_id": user_id,
                    "thread_file": thread_file,
                    "processed_at": datetime.datetime.now().isoformat(),
                }
            )

        # Log the operation
        with open(LOG_PATH, "a") as log_file:
            log_file.write(f"Operation Date: {datetime.datetime.now().isoformat()}\n")
            for entry in log_data:
                log_file.write(
                    f"User ID: {entry['user_id']}, Thread: {entry['thread_file']}, Processed At: {entry['processed_at']}\n"
                )
            log_file.write("\n")

        return {"status": "success", "data": result_data}

    except Exception as e:
        return {"status": "error", "message": str(e)}
