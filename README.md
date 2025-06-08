# Multistage Anomaly Detection Pipeline

A robust Python pipeline for detecting and classifying anomalies in tabular time series data. The workflow integrates PCA-based dimensionality reduction, statistical filtering, and supervised machine learning models (CatBoost, LightGBM, Random Forest) for anomaly scoring and classification.

## Features

- **PCA Analysis:** Visualizes and reduces dimensionality of input features.
- **Tier-1 Statistical Filtering:** Identifies interesting and faulty points using Z-score thresholds and delta checks.
- **Tier-2 Anomaly Scoring:** Uses sliding windows and regression models (CatBoost, LightGBM) to assign anomaly scores.
- **Anomaly Classification:** Random Forest classifier (with hyperparameter optimization) distinguishes between normal and anomalous points.
- **Feature Engineering:** Includes squared and cubic terms for improved model performance.
- **Visualization:** Scree plots, biplots, heatmaps, and ROC curves for interpretability.
- **Flexible Scoring:** Supports both Euclidean and MAE-based anomaly scoring.

## Getting Started

### Prerequisites

- Python 3.7+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- catboost
- lightgbm

 ## Installation Install the required Python packages using pip:
 
```bash pip install pandas numpy matplotlib seaborn scikit-learn catboost lightgbm ```

### Data

Place your input CSV file (e.g., `230.csv`) in the project directory. The file should include an `Anomaly` column for supervised classification.

### Usage

1. **PCA Analysis**
   - Run the PCA section to visualize data structure and correlations.

2. **Tier-1 Filtering**
   - Filters out uninformative or faulty data points using Z-score logic.

3. **Tier-2 Anomaly Scoring**
   - Applies sliding window regression with CatBoost and LightGBM to compute anomaly scores.

4. **Anomaly Classification**
   - Classifies anomalies using a Random Forest (with optional hyperparameter tuning via GridSearchCV or RandomizedSearchCV).

5. **Evaluation**
   - Visualizes results and prints classification metrics.



