import pandas as pd # The standard library for data tables (DataFrames)
import numpy as np # Used for advanced math and handling 'NaN' (Not a Number) missing values
import joblib # The library used for 'Persistence' - saving the model to a file
from sklearn import set_config # Used to change the global settings of Scikit-Learn

# 1. 🐼 PANDAS MODE CONFIGURATION
# 'transform_output="pandas"' is a modern industry setting.
# By default, Scikit-Learn output looks like a list of raw numbers (Numpy).
# This tells the system: "Keep everything in a Table format so I can see column names!"
set_config(transform_output="pandas")

def load_data(path):
    """
    This function acts as the 'Docking Bay' for our data.
    It brings the raw CSV into our Python environment.
    """
    print(f"📂 Loading Data from: {path}")
    
    # read_csv: Converts the external file into a Python DataFrame
    df = pd.read_csv(path)
    
    # WE SPLIT THE DATA HERE
    # X (Features): The 'Inputs' (Age, Fare, Gender, etc.)
    # We drop 'PassengerId' because it's just a unique identifier - it doesn't help the AI learn patterns!
    # We drop 'Subscription' because that is the 'Actual Answer' we are trying to predict.
    # errors='ignore': This is a 'Safety Switch'. It ensures the code doesn't crash 
    # even if these columns are already missing from the table.
    X = df.drop(columns=['PassengerId', 'Subscription'], errors='ignore')
    
    # y (Target): The 'Answer Key' - the specific thing the model is trying to master.
    # Here, we set it to the 'Subscription' column (Free, Basic, Premium).
    y = df['Subscription']
    
    return X, y

def main():
    # SETTING THE PATHS
    DATA_PATH = 'data/messy_ml_data.csv'
    
    # EXECUTION START
    # We unpack the result of load_data into X and y
    X, y = load_data(DATA_PATH)
    
    print("✅ Step 1 Complete: Data loaded and features separated.")
    print(f"📊 Features found: {list(X.columns)}")

# This ensures the code only runs if we specifically execute THIS script.
if __name__ == "__main__":
    main()
