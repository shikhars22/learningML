## 🔗 Feature Selection Step 2: Correlation Analysis (Multi-collinearity)

Now that we've removed columns that don't change or are exact duplicates, we need to find features that are **redundant**—meaning they tell almost the exact same story as another feature.

---

### 1. The Behind-the-Scenes: The Math of $r$
The number you see in a correlation matrix (between -1.0 and 1.0) is called the **Pearson Correlation Coefficient**.

#### The Formula
$$ r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}} $$

#### The Intuition
*   **Numerator (Top)**: Measures **Covariance** (How much X and Y move **together**). If they both jump up at the same time, this number becomes positive.
*   **Denominator (Bottom)**: Measures **Total Spread** (How much they jump around on their own).
*   **The Result**: By dividing "Shared Movement" by "Total Movement," we "squeeze" the result into a fixed window of **-1.0 to 1.0** regardless of the scale of your data.

---

### 2. The Concept: Multi-collinearity
**Multi-collinearity** occurs when two or more independent features are highly correlated with each other.

*   **The Problem**: If two features (like `Height_in_Inches` and `Height_in_CM`) are 100% correlated, the model doesn't know which one to "blame" for the prediction. This makes the model's coefficients unstable and hard to interpret.
*   **The Goal**: Maintain a lean dataset where every feature provides **unique** information.

---

### 2. The Pearson Correlation Coefficient ($r$)
We use a value between **-1.0** and **1.0** to measure the relationship:
*   **1.0**: Perfect Positive Correlation (They move identical directions).
*   **-1.0**: Perfect Negative Correlation (They move in exactly opposite directions).
*   **0.0**: No relationship at all.

**The Threshold**: In professional ML, we usually look for correlations above **0.85 or 0.90**. If two features are that close, we drop one of them.

---

### 3. Practical Implementation: The Heatmap
The best way to "see" correlation is a **Heatmap**. This is a visual grid where the numbers are replaced by colors so you can spot the "Red Zones" (high correlation) at a glance.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Calculate the Correlation Matrix
# .corr() calculates the Pearson correlation coefficient between every pair of columns.
# It returns a square table where both the rows and columns are your feature names.
# A '1.0' on the diagonal means every column is 100% correlated with itself!
corrmat = x_train_fs1_5.corr()

# 2. Plot the Heatmap
# plt.figure sets the size of the drawing 'canvas' so the labels aren't cramped.
plt.figure(figsize=(12,10))

# sns.heatmap creates the visual grid.
# annot=True: This writes the actual correlation number inside each square.
# cmap='RdYlGn': This sets the color palette to Red-Yellow-Green.
# Green = Positive correlation, Red = Negative correlation.
# fmt=".2f": This rounds the displayed numbers to 2 decimal places.
sns.heatmap(corrmat, annot=True, cmap='RdYlGn', fmt=".2f")

# plt.show() actually renders the image in your notebook.
plt.show()
```

---

### 4. The Automated Removal Script
Doing this manually for 50 columns is impossible. We use a script to find columns that are too similar and mark them for deletion.

```python
# Definition of a custom function to find highly correlated features
def correlation_filter(data, threshold):
    # We use a set() to store column names we want to drop.
    # A set is better than a list because it automatically prevents duplicate names.
    col_corr = set() 
    
    # Generate the correlation matrix again inside the function
    corr_matrix = data.corr()
    
    # We use a 'Nested Loop' (a loop inside a loop) to compare every column to every other column.
    # i represents the current row in the matrix, j represents the current column.
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # We only look at the 'Lower Triangle' of the matrix (where j < i).
            # Why? Because the matrix is symmetrical. Correlation(A,B) is the same as Correlation(B,A).
            # Checking both would be redundant and waste time.
            
            # abs() calculates the Absolute Value. 
            # We care about -0.9 just as much as +0.9 because both mean redundancy!
            if abs(corr_matrix.iloc[i, j]) > threshold:
                
                # If the correlation is higher than our limit (e.g. 0.85),
                # we grab the name of that column and add it to our drop set.
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                
    return col_corr

# 1. Run the filter with a 0.85 threshold.
# This means if two columns are 85% similar, we will drop one of them.
to_drop = correlation_filter(x_train_fs1_5, 0.85)

# 2. Print the results so you can see if anything was found
print(f"Redundant columns to drop: {to_drop}")

# 3. Drop the redundant columns from both Train and Test sets
# This creates our Step 2 version of the variables (_fs2)
x_train_fs2 = x_train_fs1_5.drop(columns=to_drop)
x_test_fs2 = x_test_fs1_5.drop(columns=to_drop)

print(f"Final feature count after redundancy check: {x_train_fs2.shape[1]}")
```

---

### 5. Why ONLY on X_train?
> [!IMPORTANT]
> This is a statistical measurement. Your "Testing Set" might have weird, accidental correlations because it is small. We only trust the patterns found in the **Training Set**.
