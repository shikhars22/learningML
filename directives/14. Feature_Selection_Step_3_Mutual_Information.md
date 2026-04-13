## 🧠 Feature Selection Step 3: Mutual Information (Relevance)

After removing dead weight and redundancy, we now need to find which features actually "care" about the target variable. We use **Mutual Information** to rank features by their importance.

---

### 1. The Concept: Information Gain
**Mutual Information (MI)** measures the "dependency" between two variables. It asks: *"If I know the value of this feature, how much does my uncertainty about the target decrease?"*

*   **Higher Value**: Significant relationship. The feature is a strong "signal."
*   **0.0 Value**: No relationship. The feature is complete "noise" for this specific target.

### 2. Why Mutual Information?
Unlike the Correlation we did in Step 2, Mutual Information can capture **Non-Linear** relationships.
*   **Correlation**: Only sees straight lines.
*   **MI**: Sees complex patterns, curves, and clusters.

---

### 3. The "Ingredient" Analogy
Imagine you are making a specific dish (the target).
1.  **Step 1 & 2** were about throwing away rotten food (constants) and extra packs of the same spice (duplicates).
2.  **Step 3** is about tasting every remaining ingredient to see if it actually belongs in this specific dish. (A battery is unique and well-made, but it has **Zero Mutual Information** with a good soup!)

---

### 4. Practical Implementation

We use `mutual_info_classif` because our target (`Subscription`) is a classification problem.

```python
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt

# 1. Calculate the Mutual Information
# We use x_train_fs2 (the output from our last correlation step)
# random_state is used because the algorithm has a stochastic (random) element
mutual_info = mutual_info_classif(x_train_fs2, y_train, random_state=42)

# 2. Map the results back to the column names
# We create a Series where the index is our Feature Names
mi_series = pd.Series(mutual_info, index=x_train_fs2.columns)

# 3. Sort for Plotting
# For horizontal bars (barh), Matplotlib plots from the bottom up.
# To get the highest score at the TOP, we sort ascending here.
mi_plot_data = mi_series.sort_values(ascending=True)

# 4. Plotting the 'Signal'
plt.figure(figsize=(10, 8))
# barh creates a horizontal bar chart
mi_plot_data.plot(kind='barh', color='teal')
plt.title("Step 3: Feature Relevance Ranking (Highest at Top)")
plt.xlabel("Mutual Information Score")
plt.show()

# 5. Review the 'Top Candidates'
# We still print the head of the original descending series to see the top 10
print("🏆 The High-Signal Features:")
print(mi_series.head(10))
```

---

### 5. Why ONLY on X_train?
> [!IMPORTANT]
> Just like all selection steps, we calculate relevance based **only on the training data**. We do not want the test set to influence which features we choose to keep.
