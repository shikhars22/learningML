import pandas as pd # The standard library for data tables (DataFrames)
import numpy as np # Used for advanced math and handling 'NaN' (Not a Number) missing values
import joblib # The library used for 'Persistence' - saving the model to the hard disk
from sklearn import set_config # Used to change the global settings of Scikit-Learn
from sklearn.pipeline import Pipeline # An 'Assembly Line' that strings multiple steps together
from sklearn.compose import ColumnTransformer # The 'Switchboard' that routes data to different paths
from sklearn.impute import SimpleImputer # The 'Data Filler' that handles missing values (NaN)
from sklearn.preprocessing import StandardScaler, OneHotEncoder # The 'Fairness' and 'Translation' tools
from sklearn.ensemble import RandomForestClassifier # The 'Brain' - our core machine learning model
from imblearn.pipeline import Pipeline as ImbPipeline # A smarter 'Assembly Line' that can handle SMOTE
from imblearn.over_sampling import SMOTE # The 'Synthetic Generator' that balances our data classes

# 1. PANDAS MODE CONFIGURATION
# 'transform_output="pandas"' is a modern industry setting.
# By default, Scikit-Learn output looks like a list of raw numbers (Numpy).
# This tells the system: "Keep everything in a Table format so I can see column names!"
set_config(transform_output="pandas")

def clean_data_manually(df):
    """
    Standardizes the raw messy data before it hits the automated pipeline.
    This handles the inconsistent labels like 'M' vs 'Male' and 'NONE' vs 'Free'.
    """
    # ---------------------------------------------------------
    # 2. DROP DUPLICATES (THE SEMANTIC WAY)
    # ---------------------------------------------------------
    # We check all columns except PassengerId to find identical people.
    search_cols = df.columns.difference(['PassengerId'])
    df = df.drop_duplicates(subset=search_cols, keep='first').reset_index(drop=True)

    # 3. STANDARDIZE TEXT
    # We strip extra spaces from the labels so ' Male' and 'Male' are the same.
    for col in ['Gender', 'City', 'Subscription', 'IsActive']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 4. DATA MAPPING: ISACTIVE
    # We consolidate all 'Yes/No', '1/0', and 'True/False' variations into simple numbers.
    active_map = {'True': 1, '1': 1, '1.0': 1, 'Yes': 1, 'False': 0, '0': 0, '0.0': 0, 'No': 0, 'none': 0, 'nan': 0}
    df['IsActive'] = df['IsActive'].map(active_map).fillna(0).astype(int)

    # 5. DATA MAPPING: GENDER
    # Standardizes variations like M, female, F, and male into clean labels.
    gender_map = {'F': 'Female', 'female': 'Female', 'M': 'Male', 'male': 'Male'}
    df['Gender'] = df['Gender'].map(gender_map).fillna(df['Gender'])

    # 6. DATA MAPPING: SUBSCRIPTION (The Target)
    # This is our critical 'Answer' column. We merge all synonyms into 3 clean categories.
    sub_map = {
        'FREE': 'Free', 'Free': 'Free', 'NONE': 'Free', 'none': 'Free',
        'BASIC': 'Basic', 'Basic': 'Basic', 
        'PREMIUM': 'Premium', 'Premium': 'Premium'
    }
    df['Subscription'] = df['Subscription'].map(sub_map).fillna('Free')

    # 7. HANDLE OUTLIERS (Important for clean math)
    # We remove extreme values in Age and Score so the 'Brain' doesn't get distracted by weird data.
    def remove_outliers(data, column):
        if column not in data.columns:
            return data
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        return data[(data[column] >= (Q1 - 1.5 * IQR)) & (data[column] <= (Q3 + 1.5 * IQR))]

    df = remove_outliers(df, 'Age')
    df = remove_outliers(df, 'Score')
    
    return df.reset_index(drop=True)

def load_data(path):
    """
    This function acts as the 'Docking Bay' for our data.
    It brings the raw CSV in and applies our manual cleaning logic.
    """
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    
    # Apply the manual cleaning patterns discovered in our research.
    df = clean_data_manually(df)
    
    # WE SPLIT THE DATA HERE
    # X (Features): The 'Inputs' (Age, Fare, Gender, etc.)
    # We drop columns that are too complex or have zero predictive power like Email or Phone.
    drop_cols = ['PassengerId', 'Subscription', 'JoinDate', 'Email', 'Phone', 'Remarks', 'DeptCode', 'DiscountCode']
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # y (Target): The 'Answer Key' - the specific thing the model is trying to master.
    y = df['Subscription']
    
    return X, y

def build_switchboard(X):
    """
    This function builds the 'Automated Sorting Machinary'.
    It separates Numbers from Text and cleans them using different assembly lines.
    """
    # 1. FUTURE-PROOF DETECTION
    # We ask Pandas to identify which columns are Numbers and which are Categories (Text).
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object', 'string', 'str']).columns.tolist()
    
    print(f"Switchboard identified {len(numeric_columns)} numbers and {len(categorical_columns)} text columns.")

    # 2. THE NUMERICAL PATH (The Assembly Line for Numbers)
    numeric_path = Pipeline(steps=[
        # 'median' strategy fills a gap with the middle value of the whole column.
        ('gap_filler', SimpleImputer(strategy='median')),
        # 'StandardScaler' ensures all numbers are on the same fair scale (e.g., -3 to +3).
        ('scaler', StandardScaler())
    ])

    # 3. THE CATEGORICAL PATH (The Assembly Line for Text)
    categorical_path = Pipeline(steps=[
        # If a category is missing, we label it as 'unknown' rather than guessing.
        ('gap_filler', SimpleImputer(strategy='constant', fill_value='unknown')),
        # 'OneHotEncoder' translates words into 0s and 1s so the script can do math.
        ('translator', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 4. FINAL SORTING (The ColumnTransformer)
    # This acts as the Master Controller that moves the data through the paths we built.
    return ColumnTransformer(
        transformers=[
            ('num_route', numeric_path, numeric_columns),
            ('cat_route', categorical_path, categorical_columns)
        ],
        # verbose_feature_names_out=False: Keeps our column names simple (e.g., 'Age' staying as 'Age').
        verbose_feature_names_out=False
    )

def build_atomic_engine(switchboard):
    """
    This function assembles the 'Atomic Engine'.
    It bundles the Switchboard, the Data Balancer (SMOTE), and the Brain together.
    """
    # THE MASTER ENGINE
    # We use ImbPipeline because it is designed to ONLY run SMOTE during training.
    return ImbPipeline(steps=[
        # Step 1: Clean and Sort the data automatically.
        ('sorting_office', switchboard),
        
        # Step 2: Balance the data (Fixing the imbalanced classes).
        # k_neighbors=1: A safety setting to prevent crashing on very small datasets.
        ('balancer', SMOTE(random_state=42, k_neighbors=1)),
        
        # Step 3: THE BRAIN (Random Forest)
        # n_estimators=100: We use 100 'Decision Trees' to vote on the final answer.
        # random_state=42: Ensures the 'Brain' always learns the same way each time.
        ('brain', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

def main():
    # SETTING THE PATHS
    DATA_PATH = 'data/messy_ml_data.csv'
    
    # 1. LOADING AND CLEANING
    # We unpack the result into our inputs (X) and target (y).
    X, y = load_data(DATA_PATH)
    print(f"Features found: {list(X.columns)}")
    print("Step 1 Complete: Data cleaned and features separated.")
    
    # 2. BUILDING THE SWITCHBOARD (Automated Cleaning)
    # This prepares the machinery that will clean the data when we fit the model.
    switchboard = build_switchboard(X)
    print("Step 2 Complete: Switchboard (ColumnTransformer) is ready to sort data.")

    # 3. BUILDING THE ATOMIC ENGINE
    # We combine the Switchboard with our Balancer and our Brain into one Master Object.
    master_factory = build_atomic_engine(switchboard)
    print("Step 3 Complete: Atomic Engine is assembled and ready to learn.")

    # 4. THE LEARNING PHASE (Fitting)
    # This is the 'Fit' button. It sends the raw data through the sorting office and trains the Brain.
    print("The Brain is now learning from the data... Please wait.")
    master_factory.fit(X, y)
    print("Step 4 Complete: The model is now fully trained and intelligent!")

    # 5. ATOMIC PERSISTENCE (Saving)
    # We save the ENTIRE engine (Cleaner + Balancer + Brain) as a single file.
    MODEL_SAVE_PATH = 'models/subscriber_factory.joblib'
    joblib.dump(master_factory, MODEL_SAVE_PATH)
    print(f"Step 5 Complete: Factory saved permanently to: {MODEL_SAVE_PATH}")
    print("PRODUCTION READY: You can now use this file for instant predictions!")

# This ensures the code only runs if we specifically execute THIS script.
if __name__ == "__main__":
    main()
