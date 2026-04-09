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

> **Looking for code examples?** Check out the comprehensive [Master Cheatsheet](directives/Master_Cheatsheet.md) covering all these concepts!

### 1. Data Cleaning
- **Handling Duplicates**: Understanding the critical difference between **Exact Clones** `dataset.duplicated()` and **Semantic Duplicates** `dataset.duplicated(subset=...)` (where identical people have different IDs).
- Standardizing categorical values (e.g., gender mapping) and fixing inconsistent string casing.

### 2. Missing Values (Imputation)
- Strategic filling of null values using **Mean**, **Median**, and **Mode**.
- Understanding why the **Median** is mathematically safer when outliers are present.

### 3. Handling Outliers
- Identification using **IQR (Interquartile Range)** for unknown distributions and **Z-Scores** (SciPy) for normal distributions.
- Visualizing outliers with Box Plots, clipping vs. trimming strategies.

### 4. Detecting Skewness
- Assessing whether data has a positive/negative skew using visual (KDE plots) and statistical methods (`.skew()`).
- Using Skewness to determine the correct scaling algorithm.

### 5. Feature Encoding
- **One-Hot Encoding**: Used for nominal data like 'Gender' and 'IsActive'.
- **Ordinal Encoding**: Used for ordered data like 'Subscription' tiers.

### 6. Feature Scaling
- **Standardization (StandardScaler)**: Centering data at 0 with 1 standard deviation. Best for normally distributed data.
- **Normalization (MinMaxScaler)**: Squashing data into a 0 to 1 range. Best for skewed data.
- The vital importance of the **Order of Operations** (Impute -> Clean Outliers -> Scale).

## 🛠️ Tech Stack
- **Python 3.x**
- **Pandas**: Data manipulation
- **Numpy**: Mathematical operations
- **Scikit-Learn**: Preprocessing and Scaling
- **Seaborn/Matplotlib**: Data Visualization

---
*Created as part of the Advanced ML Practice series.*
