# FunctionTransformer: Custom Logical Wrapping 🛠️

In Scikit-Learn, the **`FunctionTransformer`** is a bridge that allows you to take any custom Python function and turn it into an official Scikit-Learn "Transformer."

---

## 🧠 The Concept: Why use it?
Usually, we use pre-built tools like `StandardScaler` or `OneHotEncoder`. But real-world data science often requires custom math, such as **Log Transformations**, **Square Roots**, or **Unit Conversions**.

By wrapping your custom function in a `FunctionTransformer`, you ensure it can be included in a Scikit-Learn **Pipeline**, making your entire workflow automated and reproducible.

---

## 🛠️ Implementation Example

The most common use case is for **Log Transformation** to handle skewed data:

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# 1. Define your custom logic
# np.log1p (log + 1) is safer for data containing zeroes
def log_transform(x):
    return np.log1p(x)

# 2. Wrap it in a FunctionTransformer
transformer = FunctionTransformer(log_transform)

# 3. Use it just like a normal Scaler
dataset_clean['Fare_Logged'] = transformer.fit_transform(dataset_clean[['Fare']])
```

---

## 🎨 When to use it?

### 1. Log Transformation
To "squash" heavily skewed data (like Salary or Fares) so the model can see patterns more clearly.

### 2. Custom Mathematical Scaling
If you need to divide a column by a specific value (e.g., converting "Minutes" into "Hours") or square a feature ($Age \times Age$) to capture non-linear trends.

### 3. Cleaning Operations
Any custom cleaning step (like stripping whitespace or complex string mapping) that you want to perform consistently across training and test data.

---

## ⚖️ Comparison: Standard vs. Custom

| Feature | Standard Transformer | FunctionTransformer |
| :--- | :--- | :--- |
| **Logic** | Pre-built (Standard, Min-Max) | **Your Custom Python Code** |
| **Goal** | General Scaling | **Specific Math Operations** |
| **Pipeline Integration** | Yes | **Yes (The primary benefit)** |

> [!TIP]
> Use `np.log1p()` instead of `np.log()` inside your transformer. If your data contains even a single zero, a normal log will return `-inf` and crash your model, but `log1p` handles it safely!
