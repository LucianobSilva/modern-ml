# Machine Learning in Churn Case
- The bank, facing increasing competition, is concerned about customer churn but lacks both a formal churn definition and mature data practices. As a newly hired data scientist,    
    you’re responsible for creating a clear churn definition, implementing it, and using data to accurately predict churn—an effort previously unaddressed with only initial, informal behavioral rules providing limited insights.

- Binary Classification Problem:
    Churn = 1
    No Churn = 1


## 1. First look Data
- Read All files or databases historical (raw)
- Simple data describe
  - Handle missing values
  - Feature scaling (Normalization, Standardization)
  - Outlier detection


## 2. Churn Definition (Data Centric)
- Read All files or databases historical (raw)
- Merging Train Data and Test data (Full dataset)
- Define the Target and data splits 
  - Business knowledge??
- Define period splits (Train, Validation, Test)
- Created full dataset in (processed)


## 3. Feature Engineering
- Read full dataset (processed)
- Feature creation (e.g., days_between, customer_ag,converts some boolean columns to integers )
- Created Categories 
- Window Features
- Lifetime Features
- Target Definition
- Created new dataset considering the steps of Data Engineering (processed)


## 4. Sample date Split
- Read full dataset considering the steps of Data Engineering  (processed)
- Created range Target splits definition
- Features Loading (Necessary for model)
- Save partition split files (Train, Validation and Test)


## 5. Feature Selection
- Read all split files (Train, Validation and Test)
- Split Data (Drop targets)
- Train a Vanilla Base CatBoost Model
  - Train Model
  - Model evaluation (Metrics report (ROC AUC, Brier Score,        Average Precision, and Log Loss))
  - Precision- Recall Curve and Calibration Curve
  - SHAP Values
- Feature Selection with Boruta
  - Select important features
  - Save Select Features
- Train a CatBoost Model with Selected Features
  - Train Model
  - Model evaluation (Metrics report (ROC AUC, Brier Score,        Average Precision, and Log Loss))
  - Precision- Recall Curve and Calibration Curve
  - SHAP Values
- Comparison
  - Compares the performance of the baseline model and the model with selected features.
- Save Models
  - Base Model and Model with Selection features 

## 6. Tuning
- Read all split files (Train, Validation and Test) 
- Manual Tuning
- Perform Optuna Hyperparameter Tuning
  - Hyperparameter tuning for the CatBoost model.
- Process (Saves the best hyperparameters)
- Retrain the model with the Best Parameters
  - Load Best Hyperparameters
  - Train Model
  - Model evaluation (Metrics report (ROC AUC, Brier Score,        Average Precision, and Log Loss))
  - Plot Distribution
  - Precision- Recall Curve and Calibration Curve
  - SHAP Values

- Comparing the models
  - Baseline Model Metrics
  - Feature Selection Model Metrics
  - Tuned Model Metrics
- Save the model trained with selected features


## 7. Calibration
- Read all split files (Train, calibration, test)
- load baseline and feature selection models. 
- Calibrate model using (Platt scaling)
  - Sigmoid and isotonic regression
  - Compute metrics
- Calibrate the model using Venn-Abers
  - Compute metrics
- Compare Models

