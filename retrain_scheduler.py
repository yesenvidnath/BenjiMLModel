import schedule
import time
import subprocess
import logging
from datetime import datetime

# Set up logging
LOG_FILE = "app/logs/retrain_log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths to scripts
NORMALIZER_SCRIPT = "app/normalizer.py"
RETRAINER_SCRIPT = "app/retrainer.py"

# Function to run the normalizer
def run_normalizer():
    logging.info("Starting normalizer process...")
    try:
        subprocess.run(["python", NORMALIZER_SCRIPT], check=True)
        logging.info("Normalizer process completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Normalizer process failed: {e}")
    except Exception as ex:
        logging.error(f"Unexpected error in normalizer: {str(ex)}")

# Function to run the retrainer
def run_retrainer():
    logging.info("Starting retrainer process...")
    try:
        retrain_result = subprocess.run(["python", RETRAINER_SCRIPT], check=True, capture_output=True, text=True, timeout=300)
        logging.info("Retrainer process completed successfully.")
        
        # Debugging logs for subprocess output
        logging.info(f"Retrainer stdout: {retrain_result.stdout}")
        logging.info(f"Retrainer stderr: {retrain_result.stderr}")
        
        # Extract accuracy from the retrainer output
        output_lines = retrain_result.stdout.split("\n")
        for line in output_lines:
            if "Accuracy" in line:
                logging.info(f"Model Accuracy: {line.strip()}")
                break
    except subprocess.CalledProcessError as e:
        logging.error(f"Retrainer process failed: {e}")
    except Exception as ex:
        logging.error(f"Unexpected error in retrainer: {str(ex)}")

# Full retrain process
def full_retrain_process():
    logging.info("Starting full retrain process...")
    try:
        run_normalizer()
        logging.info("Normalizer completed. Proceeding to retrainer.")
        run_retrainer()
        logging.info("Full retrain process completed.\n" + "-"*50)

        # Schedule the next execution after this process completes
        logging.info("Scheduling next retrain process in 24 hours.")
        schedule.every(24).hours.do(full_retrain_process)
    except Exception as e:
        logging.error(f"Error during retrain process: {str(e)}")

# Trigger the retrain process immediately
logging.info("Triggering initial retrain process...")
full_retrain_process()

# Run the scheduler loop
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
