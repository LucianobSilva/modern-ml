# Tuning Notebook Documentation

This document provides a detailed explanation of the steps and code blocks in the `6_tuning.ipynb` notebook. The notebook focuses on tuning a machine learning model using various techniques and libraries.

## Libraries Used

### Python Libraries

- **os**: Provides functions for interacting with the operating system.
- **gc**: Provides an interface to the garbage collector.
- **pandas**: A powerful data manipulation and analysis library.
- **numpy**: A library for numerical computations.
- **catboost**: A gradient boosting library for categorical features.
- **sklearn**: A machine learning library with various tools for model training and evaluation.
- **matplotlib**: A plotting library for creating static, animated, and interactive visualizations.
- **seaborn**: A statistical data visualization library based on matplotlib.
- **optuna**: An optimization framework for hyperparameter tuning.
- **shap**: A library for interpreting machine learning models.
- **json**: Provides functions for working with JSON data.
- **joblib**: A library for serializing Python objects.

### Custom Utilities

- **global_code.util**: Contains custom utility functions for memory reduction, metric reporting, and plotting.

## Notebook Steps

### 1. Import Libraries and Set Up Environment

```python
import os
import gc

os.chdir('../../')
```

- **os.chdir('../../')**: Changes the current working directory to the parent directory.

### 2. Enable Auto-reload

```python
%load_ext autoreload
%autoreload 2
```

- **%load_ext autoreload**: Loads the autoreload extension.
- **%autoreload 2**: Automatically reloads modules before executing code.

### 3. Import Required Libraries

```python
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from global_code.util import reduce_mem_usage
import matplotlib.pyplot as plt
import seaborn as sns
from global_code.util import reduce_mem_usage, clf_metric_report, compute_and_plot_permutation_importance, plot_pr_calib_curve, plot_dis_probs, plot_shap_values
import optuna
import shap
import json
import joblib

sns.set(style='whitegrid')
```

- **pandas**: Used for data manipulation and analysis.
- **numpy**: Used for numerical computations.
- **catboost**: Used for training the CatBoost model.
- **sklearn**: Used for model evaluation and splitting data.
- **global_code.util**: Custom utility functions.
- **matplotlib** and **seaborn**: Used for data visualization.
- **optuna**: Used for hyperparameter tuning.
- **shap**: Used for model interpretation.
- **json** and **joblib**: Used for saving and loading data.

### 4. Read Data

```python
train_df = pd.read_parquet('./week_1/data/processed/train_df_v2.parquet')
validation_df = pd.read_parquet('./week_1/data/processed/validation_df_v2.parquet')
calibration_df = pd.read_parquet('./week_1/data/processed/calibration_df_v2.parquet')
```

- **pd.read_parquet**: Reads data from Parquet files.

### 5. Define Categorical Features

```python
cat_features = ['country', 'broad_job_category']
train_df[cat_features].dtypes
```

- **cat_features**: List of categorical features.

### 6. Split Data

```python
target = 'churn_420'

# Input variables and Target dataframes
X_train, y_train = train_df.drop(target, axis=1), train_df.loc[:, target]
X_validation, y_validation= validation_df.drop(target, axis=1), validation_df.loc[:, target]
X_calibration, y_calibration= calibration_df.drop(target, axis=1), calibration_df.loc[:, target]

# Freeing memory
train_df = None
calibration_df = None
validation_df = None
gc.collect()

print('Train Shape: ', X_train.shape, y_train.shape)
print('Validation shape: ', X_validation.shape, y_validation.shape)
print('Calibration shape: ', X_calibration.shape, y_calibration.shape)
```

- **target**: The target variable for prediction.
- **X_train, y_train**: Training data and labels.
- **X_validation, y_validation**: Validation data and labels.
- **X_calibration, y_calibration**: Calibration data and labels.
- **gc.collect()**: Frees up memory.

### 7. Load Selected Features

```python
selected_features = None
with open('./week_1/model/selected_features_list.json', 'r') as f:
    selected_features = json.load(f)
print(selected_features)
```

- **selected_features**: List of selected features for the model.

### 8. Manual Tuning

```python
## TODO Manual Tunning
```

- Placeholder for manual tuning steps.

### 9. Optuna Hyperparameter Tuning

Optuna is an automatic hyperparameter optimization framework designed to find the best hyperparameters for machine learning models. It uses a technique called Bayesian optimization to efficiently search the hyperparameter space.

### 10. Define Objective Function for Optuna

The objective function is the function that Optuna will optimize. It takes a trial object as input and returns a score that Optuna will try to maximize or minimize.

```python
def objective(trial):
    params = {
        'iterations': trial.suggest_categorical('iterations', [50, 100, 150, 200, 300, 500, 750, 1000]),
        'max_depth': trial.suggest_categorical('depth', [4, 6, 8, 10, 12, 14]),
        'colsample_bylevel': trial.suggest_categorical('colsample_bylevel', [0.5, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', 'SqrtBalanced']),
        'cat_features': cat_features,
        'verbose': 0
    }

    model = CatBoostClassifier(**params, eval_metric='PRAUC:use_weights=false')
    tscv = TimeSeriesSplit(n_splits=4)
    avg_precision_scores = []

    for train_index, val_index in tscv.split(X_train[selected_features]):
        X_train_fold, X_val_fold = X_train[selected_features].iloc[train_index], X_train[selected_features].iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), early_stopping_rounds=200)
        y_pred_fold = model.predict_proba(X_val_fold)[:, 1]
        avg_precision_scores.append(average_precision_score(y_val_fold, y_pred_fold))

    return np.mean(avg_precision_scores)
```

- **trial.suggest_categorical**: Suggests a value for a categorical hyperparameter.
- **trial.suggest_float**: Suggests a value for a continuous hyperparameter.
- **CatBoostClassifier**: Initializes the CatBoost model with the suggested hyperparameters.
- **TimeSeriesSplit**: Splits the data into training and validation sets for time series data.
- **average_precision_score**: Computes the average precision score for the validation set.

### 11. Create and Optimize Study

Create an Optuna study and optimize the objective function.

```python
study = optuna.create_study(direction='maximize')
print('Tuning the model...')
study.optimize(objective, timeout=60*20, n_trials=10)

best_params = study.best_params
print(f'Best parameters: {best_params}')
```

- **optuna.create_study**: Creates a new Optuna study.
- **study.optimize**: Optimizes the objective function.
- **timeout**: The maximum time (in seconds) to run the optimization.
- **n_trials**: The number of trials to run.

### 12. Save Best Parameters

```python
print('Saving the best parameters to a JSON file...\n ', json.dumps(best_params, indent=4))
best_params_path = './week_1/model/best_params.json'
with open(best_params_path, 'w') as f:
    json.dump(best_params, f, indent=4)
```

- **json.dump**: Saves the best parameters to a JSON file.

### 13. Retrain Model with Best Parameters

```python
best_params = None
if not best_params: 
    best_params = {
        "iterations": 750,
        "depth": 6,
        "colsample_bylevel": 1.0,
        "subsample": 0.7,
        "learning_rate": 0.007854446903112454,
        "auto_class_weights": "SqrtBalanced"
    }

model_tunned = CatBoostClassifier(**best_params, eval_metric='PRAUC:use_weights=false', cat_features=cat_features, random_state=125)
model_tunned.fit(X_train[selected_features], y_train, eval_set=(X_validation[selected_features], y_validation))

y_pred_tunned = model_tunned.predict_proba(X_validation[selected_features])[:, 1]
```

- **model_tunned**: Retrains the model with the best parameters.

### 14. Evaluate the Model

```python
print("Evaluating the model...")
clf_metric_report(y_pred_tunned, y_validation)
```

- **clf_metric_report**: Evaluates the model using various metrics.

### 15. Plot PR Curve and Calibration Curve

```python
plot_pr_calib_curve(y_pred_tunned, y_validation)
```

- **plot_pr_calib_curve**: Plots the Precision-Recall and Calibration curves.

### 16. Plot Distribution of Predicted Probabilities

```python
plot_dis_probs(y_pred_tunned, y_validation)
```

- **plot_dis_probs**: Plots the distribution of predicted probabilities.

### 17. Plot SHAP Values

```python
shape_explainer = shap.Explainer(model_tunned)
shape_values = shape_explainer(X_validation[selected_features])
plot_shap_values(shape_values, X_validation[selected_features], y_validation)
```

- **shap.Explainer**: Creates a SHAP explainer for the model.
- **plot_shap_values**: Plots the SHAP values.

### 18. Compare Models

```python
baseline_model_path = './week_1/model/baseline_model.joblib'
feat_selection_model_path = './week_1/model/feat_selection_model.joblib'

baseline_model = joblib.load(baseline_model_path)
feat_selection_model = joblib.load(feat_selection_model_path)

y_pred_baseline = baseline_model.predict_proba(X_validation)[:, 1]
y_pred_feat_selection = feat_selection_model.predict_proba(X_validation[selected_features])[:, 1]

print("Baseline Model Metrics:")
clf_metric_report(y_pred_baseline, y_validation)

print("\nFeature Selection Model Metrics:")
clf_metric_report(y_pred_feat_selection, y_validation)

print("\nTuned Model Metrics:")
clf_metric_report(y_pred_tunned, y_validation)
```

- **joblib.load**: Loads the baseline and feature selection models.
- **clf_metric_report**: Compares the models using various metrics.

### 19. Save Tuned Model

```python
tunned_model_path = './week_1/model/tunned_model.joblib'
joblib.dump(model_tunned, tunned_model_path)

print(f"Baseline model saved to: {tunned_model_path}")
```

- **joblib.dump**: Saves the tuned model to a file.

## Functions and Metrics

### clf_metric_report

- **ROC AUC**: Measures the area under the ROC curve.
- **Brier Score**: Measures the mean squared difference between predicted probabilities and actual outcomes.
- **Average Precision**: Measures the area under the Precision-Recall curve.
- **Log Loss**: Measures the performance of a classification model with probabilities.

### plot_pr_calib_curve

- **Precision-Recall Curve**: Plots precision vs. recall.
- **Calibration Curve**: Plots predicted probabilities vs. actual probabilities.

### plot_dis_probs

- **Distribution of Predicted Probabilities**: Plots the distribution of predicted probabilities.

### plot_shap_values

- **SHAP Values**: Plots the SHAP values for model interpretation.

