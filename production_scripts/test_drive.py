import pandas as pd # Used to prepare the customer data in a table format
import joblib # Used to load the saved 'Brain' (subscriber_factory.joblib)
import os # Used to check for file existence

def test_drive():
    """
    This script acts as a 'Simulation' of a real customer using our model. 
    It proves and validates that the Atomic Engine can handle raw data instantly.
    """
    MODEL_PATH = '../models/subscriber_factory.joblib'
    
    # 1. LOAD THE SAVED BRAIN
    # Here's the thing: we don't need to load the cleaner, the scaler, or the brain separately.
    # The .joblib file contains the entire ATOMIC engine in one piece.
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find the model file at {MODEL_PATH}")
        print("Please run the master_pipeline.py first to create the brain!")
        return

    print("Loading the industrial brain... Please wait.")
    brain = joblib.load(MODEL_PATH)
    print("Success: The brain is online and ready to predict.")

    # 2. CREATE AN IMAGINARY CUSTOMER
    # We must provide the features in the exact order the model was trained on.
    # The order is: ['Age', 'Fare', 'Gender', 'City', 'Score', 'IsActive']
    new_customer = {
        'Age': [45.0],
        'Fare': [120.50],
        'Gender': ['Female'],
        'City': ['London'],
        'Score': [88.0],
        'IsActive': [1]
    }

    # Convert to a DataFrame so the column names are preserved.
    # This removes the 'UserWarning' by confirming to the brain that these are the correct features.
    df_customer = pd.DataFrame(new_customer)
    
    print("\n--- Simulation Target ---")
    print(df_customer)

    # 3. ASK THE BRAIN FOR THE ANSWER
    # Because we are using a DataFrame with names, the warnings will disappear.
    prediction = brain.predict(df_customer)
    
    # We can also ask for the 'Probability' - how sure the model is.
    # This shows the percentage of trees that voted for each type.
    probs = brain.predict_proba(df_customer)
    prob_score = max(probs[0]) * 100

    print("\n--- Brain Response ---")
    print(f"Predicted Subscription Tier: {prediction[0]}")
    print(f"Confidence Level: {prob_score:.2f}%")

if __name__ == "__main__":
    test_drive()
