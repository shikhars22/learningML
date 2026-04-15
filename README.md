# Advanced Machine Learning Pipeline: From Raw Data to Production

This repository is a comprehensive journey through the modern Machine Learning lifecycle. We transitioned from handling "Messy Data" to building a professional, balanced, and production-ready predictive pipeline.

## 📂 Project Structure

```text
learningML/
├── data/               # Raw and processed datasets (CSV)
├── directives/         # 30+ detailed guides on ML strategy & theory
├── notebooks/          # Experimental Jupyter Notebooks
├── models/             # Saved model 'brains' (.joblib)
├── production_scripts/ # [NEW] Deployment-ready scripts
│   ├── master_pipeline.py  # The industrial factory (Train + Prep)
│   ├── test_drive.py       # Single-passenger simulation
│   └── test_drive_multiple_input.py # 10-customer batch test
└── README.md
```

## 🧠 The Industrial Engine
Our final system uses an **Atomic Pipeline** architecture. This means the data cleaning, the math (scaling), and the brain (model) are all bundled into a single `.joblib` file.

### Key Features
- **Automatic Cleaning**: Uses `ColumnTransformer` to route data based on its type.
- **SMOTE Protection**: Balanced training that automatically turns off during real-world predictions.
- **Human Mapping**: Custom logic to handle messy real-world labels (`M` vs `Male`, etc.).

## 🤖 Automation (The Monday Schedule)
In the industry, we don't run these scripts manually. We schedule the `master_pipeline.py` to run every Monday morning using a system trigger. This ensures the model is always trained on the latest customer data.

### On Windows (Task Scheduler)
To automate this on your machine:
1. Open **Task Scheduler**.
2. Click **Create Basic Task** and name it `ML_Retrain_Monday`.
3. Set the Trigger to **Weekly** and select **Monday**.
4. Set the Action to **Start a Program**.
5. In the 'Program/script' box, type: `python`.
6. In the 'Add arguments' box, type the full path to your script:
   `C:\...\learningML\production_scripts\master_pipeline.py`

### On Linux/Mac (Cron)
Add this line to your crontab (`crontab -e`):
```bash
0 9 * * 1 /usr/bin/python3 /path/to/learningML/production_scripts/master_pipeline.py
```
*This runs the pipeline every Monday at 9:00 AM.*

## 🛠️ Tech Stack
- **Python 3.11**
- **Scikit-Learn**: The engine for training and pipelines.
- **Imbalanced-Learn**: SMOTE implementation for class balance.
- **Pandas / Numpy**: Data manipulation.
- **Joblib**: Model persistence and industrial loading.

---
*Developed as a training ground for professional ML Engineering.*
