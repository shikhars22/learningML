# Understanding Normalization (Min-Max Scaling) 📏

While **Standardization** (which we just did) centers data around zero, **Normalization** is about "squashing" your data into a fixed, predictable range—usually **0 to 1**.

---

## 🧠 The "Rubber Band" Concept
Imagine each of your columns is a rubber band of a different length:
- **Age** is a short rubber band (0 to 100).
- **Salary** is a massive rubber band (0 to 500,000).

Normalization takes both rubber bands and stretches (or shrinks) them so they are both **exactly 10 centimeters long**. 

- The **Minimum** value in your data becomes **0**.
- The **Maximum** value in your data becomes **1**.
- Everything else is placed proportionally in between.

---

## ⚖️ When do you use Normalization instead of Standardization?

### 1. When you don't know the distribution
If your data does **not** follow a Bell Curve (Normal Distribution), Normalization is often better. It doesn't care about means or standard deviations; it only cares about the boundaries (Min and Max).

### 2. Deep Learning & Image Processing
Neural Networks almost always prefer data between 0 and 1. For example, image pixels are naturally 0 (black) to 255 (white). We "normalize" them to 0.0–1.0 so the math inside the brain of the AI stays stable.

### 3. Specific Algorithm Requirements
Models like **KNN (K-Nearest Neighbors)** and **Artificial Neural Networks** are very sensitive to the range of data. Normalization ensures that "distance" calculations are fair across all features.

---

## ⚠️ The "Outlier" Warning (Crucial!)
Normalization is **extremely weak** against outliers. 
- If you have ages 20, 30, 40 and one outlier of **1,000**:
- The 1,000 becomes **1.0**.
- The 20, 30, and 40 will all be squashed to something like **0.02, 0.03, and 0.04**.

The model can no longer "see" the difference between a 20-year-old and a 40-year-old because they are now too close together at the bottom of the scale. 

> [!IMPORTANT]
> This is why we **must** remove outliers before using a Normalization (Min-Max) scaler!

---

## 🧪 Summary: The 0-to-1 Rule
- **Standardization**: Centers data at 0 (can go negative). Best for Bell Curves.
- **Normalization**: Squashes data between 0 and 1 (always positive). Best for Neural Networks and non-Gaussian data.
