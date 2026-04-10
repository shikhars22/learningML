# 🐼 Pandas Selection & Axis Cheatsheet

Understanding the difference between **Series** and **DataFrames**, and the direction of the **Axis**, is the foundation of data manipulation in Python.

---

## 1. Single vs. Double Brackets

| Syntax | Returns | Dimensionality | Best For... |
| :--- | :--- | :--- | :--- |
| `df['Age']` | **Series** | 1D (List) | Simple math (`.mean()`), status checks. |
| `df[['Age']]` | **DataFrame** | 2D (Table) | **Scikit-Learn Tools** (Scaler, Imputer). |
| `df[['A', 'B']]` | **DataFrame** | 2D (Table) | Selecting a subset of multiple columns. |

> [!IMPORTANT]
> **Scikit-Learn Transformers** (like `StandardScaler`) strictly require a 2D object. If you use single brackets, your code will likely crash.

---

## 2. Understanding `axis` (Direction)

When a command asks for an `axis`, it's asking for the **direction** of the operation.

### `axis=0` (The Rows)
*   **Analogy**: The "Ground" or floor.
*   **Direction**: Moves **Down** through the rows.
*   **Use Case**: `df.drop(5, axis=0)` drops Row index 5.

### `axis=1` (The Columns)
*   **Analogy**: The "Pillars" of a building.
*   **Direction**: Moves **Across** through the column names.
*   **Use Case**: `df.drop('Email', axis=1)` drops the Email column.

---

## 3. Quick Visual Guide

```text
       axis=1 (Across / Columns)
          ------>
  
   |   Col_A | Col_B | Col_C |
   |---------|-------|-------|
   | row_0   |  val  |  val  |  | axis=0
   | row_1   |  val  |  val  |  | (Down / Rows)
   | row_2   |  val  |  val  |  v
```

---

## 4. The "Counting vs. Dropping" Rule

A common point of confusion is when to use `axis=1` vs `axis=0` for functions like `.drop()` vs `.nunique()`/`.sum()`.

| Operation Type | Example | Axis | Visual Why... |
| :--- | :--- | :--- | :--- |
| **Removing/Dropping** | `.drop(cols, axis=1)` | **axis=1** | You are cutting **Across** the vertical pillars. |
| **Counting/Math** | `.nunique()` | **axis=0** | You are looking **Down** a pillar to count its parts. |

> [!TIP]
> **Most Math is `axis=0` by default.** You don't need to specify it for `.sum()`, `.mean()`, or `.nunique()` unless you specifically want to do math across a single row.

---

## 5. Why we scale like this:
```python
# RIGHT SIDE: Double brackets make it a 2D table for the Scaler
# LEFT SIDE: Double brackets ensure it slots correctly back into the main DF
df_final[['Fare']] = scaler.fit_transform(df_final[['Fare']])
```
