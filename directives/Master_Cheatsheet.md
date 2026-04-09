# Data Preprocessing & Cleaning Master Cheatsheet 🚀

This document summarizes all the concepts, strategies, and code implementations covered in our Machine Learning data preparation journey. 

---

## 1. The Pro Pipeline Order 🏆
The order in which you clean data is critical. Getting it wrong can lead to errors or biased models.
1. **Drop Duplicates** (Remove redundant rows first to save processing time).
2. **Handle Inconsistencies** (Standardize text like "Male" vs "male").
3. **Impute Missing Values** (Scalers and outlier math require complete numerical arrays).
4. **Handle Outliers** (Remove or cap extreme values so they don't distort scaling).
5. **Feature Encoding** (Convert text categories to numbers).
6. **Feature Scaling** (Level the numerical playing field).

---

## 2. Handling Duplicates 👯

### The Concept
There are two types of duplicates:
*   **Exact Clones**: Every single column matches (including the ID).
*   **Semantic Duplicates**: The information matches, but the system generated a new ID (e.g., `PassengerId`).

### The Code
```python
import pandas as pd

# Load data
dataset = pd.read_csv('../data/messy_ml_data.csv')

# Check Exact Duplicates
exact_dups = dataset.duplicated().sum()

# Check Semantic Duplicates (Ignore the ID!)
search_cols = dataset.columns.difference(['PassengerId'])
semantic_dups = dataset.duplicated(subset=search_cols).sum()

# Remove Semantic Duplicates (Keeps the first occurrence)
dataset_clean = dataset.drop_duplicates(subset=search_cols, keep='first')
```

---

## 3. Handling Missing Values (Imputation) 🕳️

### The Concept
Machine learning models cannot process `NaN` (Not a Number) values. We must fill them intelligently.
*   **Mean**: The average. Use only if the data is a perfect bell curve with NO outliers.
*   **Median**: The middle value. **Always use this if outliers are present**, as it resists getting pulled by extreme numbers.
*   **Mode**: The most frequent value. Used for categorical/text data.

### The Code
```python
# 1. Fill Numerical columns with the Median (Safest approach)
dataset_clean['Age'] = dataset_clean['Age'].fillna(dataset_clean['Age'].median())
dataset_clean['Score'] = dataset_clean['Score'].fillna(dataset_clean['Score'].median())
dataset_clean['Fare'] = dataset_clean['Fare'].fillna(dataset_clean['Fare'].median())

# 2. Fill Categorical columns with the Mode
mode_value = dataset_clean['Subscription'].mode()[0]
dataset_clean['Subscription'] = dataset_clean['Subscription'].fillna(mode_value)
```

---

## 4. Handling Outliers 🌪️

### The Concept
Outliers are extreme values (like a 200-year-old person) that can ruin average calculations and ruin scaling algorithms.

**Technique 1: IQR (Interquartile Range)**
Best for skewed data or when you don't know the distribution. Cuts off the top and bottom tails.
*   $IQR = Q3 - Q1$
*   Limits: $Q1 - 1.5 * IQR$ to $Q3 + 1.5 * IQR$

**Technique 2: Z-Score (SciPy)**
Best for "Normal/Gaussian" (Bell Curve) data. Calculates how many Standard Deviations a point is from the Mean.
*   Rule: Anything with an absolute Z-score > 3 is an outlier.

### The Code (IQR Method)
```python
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Keep only normal data
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Sequentially remove outliers
dataset_clean = remove_outliers_iqr(dataset_clean, 'Age')
dataset_clean = remove_outliers_iqr(dataset_clean, 'Score')
```

---

## 5. Detecting Skewness 📉

### The Concept
Skewness dictates which Feature Scaling method you should use.
*   **Normal / Symmetrical (Score ~0)**: Use Standardization.
*   **Heavily Skewed (Score > 1 or < -1)**: Use Min-Max Scaling or Log Transformation.

### The Code
```python
# Check numerical skewness
print(dataset_clean['Age'].skew())
print(dataset_clean['Fare'].skew())
```

---

## 6. Feature Encoding 🔠➡️🔢

### The Concept
ML models only understand math. We have to turn text categories (like "Male/Female" or "Basic/Premium") into numbers.
*   **One-Hot Encoding**: For nominal data (no order). Creates new binary columns (0s and 1s).
*   **Ordinal Encoding**: For ordinal data (ranked order, e.g., Basic -> Premium -> Gold). Maps to 0, 1, 2.

### The Code
```python
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# 1. One-Hot Encoding (Pandas get_dummies is usually easiest)
# drop_first=True prevents the "Dummy Variable Trap"
dataset_clean = pd.get_dummies(dataset_clean, columns=['Gender'], drop_first=True)

# 2. Ordinal Encoding
encoder = OrdinalEncoder(categories=[['Basic', 'Premium', 'Gold']])
dataset_clean[['Subscription_Encoded']] = encoder.fit_transform(dataset_clean[['Subscription']])
```

---

## 7. Feature Scaling ⚖️

### The Concept
Scaling stops columns with large numbers (like Salary: $100k) from unfairly dominating columns with small numbers (like Kids: 3).

*   **Standardization (Z-Score)**: Centers data at Mean=0, StdDev=1. Best for bell curves and general algorithms (Logistic Regression, SVM).
*   **Normalization (Min-Max)**: Squashes data between 0 and 1. Best for Skewed data or Neural Networks. **Must remove outliers first!**

### The Code
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Use .copy() to prevent memory reference warnings!
dataset_final = dataset_clean.copy()
dataset_final = dataset_final.reset_index(drop=True)

# Normalization (0 to 1) for highly skewed data
minmax = MinMaxScaler()
dataset_final[['Age']] = minmax.fit_transform(dataset_final[['Age']])

# Standardization (Mean=0, Std=1) for normal data
standard = StandardScaler()
dataset_final[['Score']] = standard.fit_transform(dataset_final[['Score']])
```
