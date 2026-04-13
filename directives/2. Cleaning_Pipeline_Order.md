# The ML Pipeline: Order of Operations 🏆

A common question in data science is: **"Does the order of cleaning matter?"** The answer is **Yes.** Following the wrong order can result in biased models or code errors.

---

## 📅 The Ideal Pipeline
1.  **Imputation** (Fill Missing Values)
2.  **Outlier Handling** (Capping or Removing)
3.  **Feature Scaling** (Standardization/Normalization)

---

## 🧠 Why This Order?

### 1. Why Impute First?
Most Scaling algorithms (like `StandardScaler` or `MinMaxScaler`) are purely mathematical. They **cannot handle `NaN` values**. If you try to calculate a Mean or a Range while there are nulls, the computer will crash or return an error. You must have a complete set of numbers first.

### 2. Why Handle Outliers Before Scaling?
Scaling is **extremely sensitive** to extreme values.

*   **Normalization Trap**: If you have an Age column with `[20, 30, 40]` and one outlier of `200`, the 200 becomes the **1.0**. This "squashes" the 20, 30, and 40 into a tiny, indistinguishable range (e.g., `0.10, 0.15, 0.20`). Your model loses its ability to see the differences between normal people.
*   **Standardization Trap**: Outliers pull the **Mean ($\mu$)** and **Standard Deviation ($\sigma$)** toward them. This creates a "distorted" center, making the scaling inaccurate for the 99% of your "normal" data.

---

## ⚖️ The Exception: Robust Scaling
If you absolutely **cannot** remove your outliers (e.g., they represent valid but rare events), use the **`RobustScaler`** in Scikit-Learn.
- It uses the **Median** and **IQR** instead of Mean and Std Dev.
- It is designed to scale your data while "ignoring" the mathematical blast radius of the outliers.

---

## 🧪 Summary Table

| Step | Action | Why? |
| :--- | :--- | :--- |
| **1. Impute** | Fill NaNs | Scalers need complete numerical arrays to work. |
| **2. Outliers** | Remove/Cap | Prevents the scaled range from getting "distorted." |
| **3. Scaling** | Scale | Final step to prepare data for the ML model. |
