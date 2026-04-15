import pandas as pd # Used to prepare the data table for multiple customers
import joblib # Used to load the industrial-strength brain (.joblib)
import os # Used to handle file paths consistently

def run_multi_test():
    """
    This script simulates a batch of 10 different customers.
    It demonstrates how the Atomic Engine handles multiple rows of raw data at once.
    """
    # PATH SETUP
    # This logic finds the 'models' folder regardless of where you run the script from.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(current_dir, '..', 'models', 'subscriber_factory.joblib')

    # 1. LOAD THE BRAIN
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("Waking up the brain for a batch simulation...")
    brain = joblib.load(MODEL_PATH)
    print("Brain is online.\n")

    # 2. DEFINE 10 UNIQUE CUSTOMERS
    # We include a wide variety of ages, scores, and cities to see the model's logic.
    test_data = {
        'Age': [45.0, 22.0, 35.0, 68.0, 19.0, 52.0, 29.0, 41.0, 33.0, 25.0],
        'Fare': [120.50, 15.0, 55.0, 200.0, 8.0, 95.0, 45.0, 110.0, 75.0, 12.0],
        'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'City': ['London', 'New York', 'Paris', 'Tokyo', 'Berlin', 'Sydney', 'Mumbai', 'Toronto', 'Dubai', 'Seoul'],
        'Score': [88.0, 45.0, 72.0, 95.0, 30.0, 82.0, 65.0, 89.0, 77.0, 40.0],
        'IsActive': [1, 0, 1, 1, 0, 1, 1, 1, 0, 0]
    }

    # Wrap the raw data into a Pandas DataFrame table.
    # This keeps the column names intact, which avoids the 'UserWarning' mess.
    df_batch = pd.DataFrame(test_data)

    print(f"--- Customer Batch (Count: {len(df_batch)}) ---")
    print(df_batch)

    # 3. BATCH PREDICTION
    # Instead of predicting one by one, we feed the whole table into the engine.
    predictions = brain.predict(df_batch)
    probabilities = brain.predict_proba(df_batch)

    print("\n--- Model Predictions ---")
    # We loop through the results to print a clean report for each customer.
    for i in range(len(df_batch)):
        # We find the 'Confidence' by looking at the highest probability for that row.
        confidence = max(probabilities[i]) * 100
        result = predictions[i]
        
        print(f"Customer {i+1:02}: {result:<10} | Confidence: {confidence: >5.1f}%")

if __name__ == "__main__":
    run_multi_test()
