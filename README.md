# Advanced Machine Learning Pipeline: From Raw Data to Production 🚀

This repository is a comprehensive journey through the modern Machine Learning lifecycle. We transitioned from handling "Messy Data" to building a professional, balanced, and production-ready predictive pipeline.

## 📂 Project Structure

```text
learningML/
├── data/               # Raw and processed datasets (CSV)
├── directives/         # 30+ detailed guides on ML strategy & theory
├── notebooks/          # Experimental Jupyter Notebooks
├── output/             # Visualization plots and precision reports
├── models/             # [NEW] Saved model 'brains' (.joblib)
└── master_pipeline.py  # [WIP] Production-ready factory script
```

## 🧠 The Journey: Milestones & Mastery

### 1. Robust Pre-processing
- **The "Full Clean"**: [Cleaning Pipeline Order](directives/2. Cleaning_Pipeline_Order.md), missing value imputation, and [Skewness Detection](directives/5. Skewness_Detection.md).
- **Strategic Scaling**: Using [StandardScaler](directives/1. Feature_Scaling.md), [MinMaxScaler](directives/3. Normalization.md), and [Advanced Scaling](directives/4. Advanced_Scaling_Techniques.md).
- **Safe Encoding**: Proper management of One-Hot and Ordinal encoding using [FunctionTransformers](directives/7. FunctionTransformer_Guide.md).
- **Cheatsheet**: See the [Master Pre-Processing Cheatsheet](directives/6. Pre-Processing_Master_Cheatsheet.md).

### 2. The "Elite Team" (Feature Selection)
- **Filtering Noise**: Used [Variance Thresholds](directives/11. Feature_Selection_Step_1_Variance.md), [Duplicate Removal](directives/12. Feature_Selection_Step_1_5_Duplicate_Features.md), and [Correlation analysis](directives/13. Feature_Selection_Step_2_Correlation_Analysis.md).
- **Relevance**: [Mutual Information](directives/14. Feature_Selection_Step_3_Mutual_Information.md), [Statistical Selection](directives/15. Feature_Selection_Step_4_Statistical_Selection.md), and [Strategy for K](directives/16. Feature_Selection_Strategy_Selecting_K.md).
- **Advanced Selection**: [Recursive Feature Elimination](directives/17. Feature_Selection_Step_5_Recursive_Feature_Elimination.md), [RFECV Automation](directives/18. Feature_Selection_Step_6_RFECV_Automation.md), and [Individual vs Teamwork](directives/19. Feature_Selection_Individual_vs_Teamwork.md).
- **Sequential Methods**: [Forward vs Backward Elimination](directives/29. Feature_Selection_Forward_vs_Backward.md).

### 3. Model Training & Diagnostics
- **Baseline Models**: Established [Logistic Regression](directives/21. Model_Training_Step_1_Logistic_Regression.md) to diagnose "Majority Class Bias."
- **Forest Algorithms**: Transitioned to [Random Forest](directives/24. Model_Training_Step_2_Random_Forest.md) and [The Low Signal Paradox](directives/30. Feature_Selection_The_Low_Signal_Paradox.md).
- **Deep Metrics**: [The Quality Report](directives/22. Model_Evaluation_The_Quality_Report.md), [The F1-Score Balance](directives/23. Model_Evaluation_The_F1_Score_Balance.md), and [Cross-Validation](directives/20. Cross_Validation_The_Stress_Test.md).

### 4. Advanced Optimization
- **Hyperparameter Tuning**: Using [GridSearchCV](directives/25. Model_Training_Step_3_Hyperparameter_Tuning.md) to find the perfect "Coach" for the model.
- **SMOTE**: [Handling Imbalance with SMOTE](directives/26. Handling_Imbalance_Step_1_SMOTE.md) to fix model "Blindness."
- **Feature Engineering**: [Interaction Features](directives/27. The_Final_Frontier_Feature_Engineering_Interactions.md) pushing the data to the limit.

### 5. Production & Engineering
- **The Data Ceiling**: [The Final Resolution](directives/28. The_Final_Resolution_Squeezing_No_Blood_from_a_Stone.md) for low-signal data.
- **Model Persistence**: [The Permanent Brain](directives/31. Model_Persistence_The_Permanent_Brain.md) for instant loading.
- **Pipeline Engineering**: [Production-Ready Code](directives/32. Master_Pipeline_Production_Ready_Code.md) for automation.

## 🛠️ Tech Stack
- **Python 3.11**
- **Scikit-Learn**: The engine for training and pipelines
- **MLxtend**: Advanced feature selection (Forward/Backward)
- **Imbalanced-Learn**: SMOTE implementation
- **Pandas / Numpy**: Data manipulation
- **Seaborn / Matplotlib**: Visual forensics

---
*Developed as a training ground for professional ML Engineering.*
