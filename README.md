# Machine Learning Data Cleaning & Preprocessing 🚀

This repository is dedicated to learning and practicing the essential steps of data cleaning and feature engineering for Machine Learning using Python, Pandas, and Scikit-Learn.

## 📂 Project Structure

```text
learningML/
├── data/               # Raw and processed datasets (ignored by Git)
├── directives/         # Detailed theory and strategy guides
├── notebooks/          # Step-by-step Jupyter Notebooks for practice
├── output/             # Visualization plots and analysis results
├── ml_book/            # Documentation website source (Jupyter Book)
└── execution/          # Internal scripts and utilities
```

## 🧠 Learning Journey

### 1. Data Cleaning
- Handling duplicates and inconsistent string casing.
- Standardizing categorical values (e.g., Gender mapping).

### 2. Missing Values
- Strategic filling of null values using **Mean**, **Median**, and **Mode**.
- Understanding why the **Median** is safer when outliers are present.

### 3. Encoding
- **One-Hot Encoding**: Used for nominal data like 'Gender' and 'IsActive'.
- **Ordinal Encoding**: Used for ordered data like 'Subscription' tiers.

### 4. Outlier Handling
- Identification using **IQR (Interquartile Range)** and **Z-Scores**.
- Visualizing outliers with Box Plots and Distribution curves.
- Strategies for **Trimming** vs. **Capping**.

### 5. Feature Scaling
- **Standardization**: Centering data at 0 with 1 standard deviation.
- **Normalization**: Squashing data into a 0 to 1 range.
- Importance of the "Order of Operations" (Impute -> Clean Outliers -> Scale).

## 🛠️ Tech Stack
- **Python 3.x**
- **Pandas**: Data manipulation
- **Numpy**: Mathematical operations
- **Scikit-Learn**: Preprocessing and Scaling
- **Seaborn/Matplotlib**: Data Visualization

---
*Created as part of the Advanced ML Practice series.*
