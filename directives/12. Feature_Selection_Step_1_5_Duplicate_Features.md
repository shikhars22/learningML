## 👯 Feature Selection Step 1.5: Duplicate Features

Before moving to the statistical math of correlation, we must ensure there aren't any **identical "twin" columns** hiding in our data under different names.

---

### 1. The Concept: Identical Twins
A **Duplicate Feature** occurs when two columns have the exact same values for every single row in the training set.

*   **Why it happens**: Poorly planned One-Hot encoding, merging multiple datasets with overlapping information, or simple manual error.
*   **The Problem**: It adds zero value to the model but doubles the memory usage and can cause mathematical instability in some models.
*   **The Action**: Find them and keep only one.

---

### 2. Difference between Variance, Duplicates, and Correlation

| Step | Goal | Analogy |
| :--- | :--- | :--- |
| **Step 1: Variance** | Find columns that don't change. | A light that is always "off." |
| **Step 1.5: Duplicates** | Find different columns that are identical. | Two lights connected to the same switch. |
| **Step 2: Correlation** | Find columns that are very similar. | A light and a dimmer—they do similar things. |

---

### 3. Practical Implementation

Since we are looking for identical columns, we can't just use `drop_duplicates()` (which looks at rows). We have to **Transpose** the data (flip rows and columns) so that columns become rows, and then we check for duplicates.

#### The Automated Way
We analyze the columns in the training set, identify which ones are unique, and then subset both sets.

```python
# 1. Transpose the data (Flipping columns to rows)
# We use .T to flip it
x_train_T = x_train_fs1.T

# 2. Find duplicate Rows (which were our columns!)
# .duplicated() will return True for the second twin
duplicated_feat_mask = x_train_T.duplicated()

# 3. Get the list of column names to DROP
features_to_drop = x_train_T[duplicated_feat_mask].index.values

print(f"Duplicate columns found: {features_to_drop}")

# 4. Filter both sets
x_train_fs1_5 = x_train_fs1.drop(columns=features_to_drop)
x_test_fs1_5 = x_test_fs1.drop(columns=features_to_drop)

print(f"Final feature count: {x_train_fs1_5.shape[1]}")
```

---

### 4. Why ONLY on X_train?
> [!IMPORTANT]
> Just like Variance and Scaling, we decide which columns are duplicates based **ONLY on the Training set**. We then apply that same "Drop List" to the Testing set to keep them synchronized.
