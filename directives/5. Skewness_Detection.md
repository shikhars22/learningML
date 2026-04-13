# Detecting and Measuring Skewness 📉

Identifying skewness is a critical step in data cleaning. It tells you whether your data is "natural" (Normal) or "biased" (Skewed), which dictates which scaling strategy you should choose.

---

## 1. The Visual Way (The "Eye Test")
The most intuitive way to spot skewness is by plotting a **Histogram** with a **KDE (Kernel Density Estimate)** curve.

- **Symmetrical (Normal)**: The "mountain" is perfectly centered.
- **Positive Skew (Right-Skewed)**: The peak is on the left; the "long tail" points to the right (common in Salary/Fare data).
- **Negative Skew (Left-Skewed)**: The peak is on the right; the "long tail" points to the left (common in Retirement age).

---

## 2. The Statistical Way (Pearson's Skewness)
Using the `.skew()` method in Pandas provides a definitive mathematical score.

### 📐 How to Read the Score:
| Score Range | Interpretation | Best Scaler Recommendation |
| :--- | :--- | :--- |
| **-0.5 to 0.5** | **Fairly Symmetrical** | `StandardScaler` (Stable) |
| **-1.0 to -0.5** or **0.5 to 1.0** | **Moderately Skewed** | `MinMaxScaler` or `StandardScaler` |
| **< -1.0** or **> 1.0** | **Heavily Skewed** | **Log Transformation** or `PowerTransformer` |

---

## 3. The "Mean vs. Median" Rule
A quick "back of the napkin" way to detect skewness without a plot:

1.  **Mean > Median**: The average is being pulled up by a right-hand tail (**Positive Skew**).
2.  **Mean < Median**: The average is being pulled down by a left-hand tail (**Negative Skew**).
3.  **Mean ≈ Median**: The data is balanced and **Symmetrical**.

---

## 🧪 Choosing the Right Treatment

- **For Symmetric Data**: Stick with **Standardization** (Z-scores).
- **For Heavily Skewed Data**: Apply a **Log Transformation** first to "squash" the tail, then scale. This makes it easier for the model to see the patterns in the "normal" part of the data.

> [!NOTE]
> Heavily skewed data often contains "valid outliers" (like a millionaire in a low-income neighborhood). Deleting these might lose important information, so **Log Scaling** is often a better choice than trimming!
