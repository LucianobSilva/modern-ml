Objective

Create a Jupyter Notebook (to be saved in week_2/notebooks/2_quantile_regression) that analyzes and models Singapore Housing & Development Board (HDB) resale flat prices using the dataset located at week_2/data/raw/portugal_listings.csv. The notebook should perform Exploratory Data Analysis (EDA), feature engineering, inflation adjustment, window feature creation, data splitting, and model training. Finally, it should compare different quantile regression approaches (CatBoost MultiQuantile and Conformalized Quantile Regression with MAPIE) using MLflow for experiment tracking.

Dataset Description

The primary dataset (referred to as portugal_listings.csv in the path, though it contains Singapore flat transactions) has the following columns:

Title	Column	Description
Month	month	The month when the transaction occurred (e.g., YYYY-MM).
Town	town	The residential area where the flat is located.
Flat Type	flat_type	Classification of the flat based on the number of rooms (e.g., 3 ROOM, 4 ROOM).
Block	block	The block number of the flat.
Street Name	street_name	The street where the flat is located.
Storey Range	storey_range	The range of floors where the flat is situated (e.g., 04 TO 06).
Floor Area Sqm	floor_area_sqm	The total interior space of the flat in square meters.
Flat Model	flat_model	The design model of the flat (e.g., Model A, Improved).
Lease Commence Date	lease_commence_date	The year the lease started (e.g., 1985).
Remaining Lease	remaining_lease	Remaining lease term in a string format (e.g., 61 years 04 months).
Resale Price	resale_price	The transaction price of the flat in Singapore dollars.

Link to dataset source: Resale flat prices (data.gov.sg)

Additional dataset: You will also use a second file, week_2/data/raw/sg_price_index.csv, which contains two columns:
	•	quarter (e.g., 2009-Q1, 2013-Q2)
	•	index (a normalized price index, 100 at 2009-Q1)

Background: Housing & Development Board (HDB) in Singapore
	•	The HDB is Singapore’s public housing authority, planning and developing housing estates.
	•	Established on February 1, 1960, to resolve a housing crisis.
	•	Has built over 1 million flats across 24 towns, housing ~80% of Singapore’s resident population.

Tasks

1. Exploratory Data Analysis (EDA)
	1.	Import the dataset week_2/data/raw/portugal_listings.csv.
	2.	Inspect the dataset:
	•	Display dataset size (number of rows and columns).
	•	Show dataset info (column types, non-null counts, etc.).
	•	Summarize numerical features (e.g., mean, median, std) and categorical features (value counts).
	3.	Data type conversion: Cast the month column to a DateTime type.
	4.	Identify date range: Determine the minimum and maximum values of month.
	5.	Price trend plot:
	•	Group by month and compute the average resale_price.
	•	Plot the trend of average resale price over time.
	6.	Check missing values:
	•	Count missing values in each column.

5.	Correct for inflation
Load week_2/data/raw/sg_price_index.csv containing:
	•	quarter
	•	index (price index, normalized to 100 at 2009-Q1)
Goal: Use the price index to address upward trends in historical data from 2017-01-01 to 2023-12-31 and to extrapolate from 2024-01-01 to 2024-06-01.
	•	Correct training data for inflation
	•	Adjust historical prices to the 2023-12-31 equivalent using the price index.
	•	Create a new column corrected_price.
	•	Incorporate trend into predictions
	•	Fit a quadratic regression on the index over the training period (2017-01-01 to 2023-12-31) to capture the trend.
	•	Forecast the index (call it predicted_index) for the validation period (2024-01-01 to 2024-06-01).
	•	Create a new column predicted_index.
	•	Plot the actual index vs. the predicted index.

1. Feature Engineering
	1.	Target creation
	•	Create price_per_sqm = resale_price / floor_area_sqm.
	2.	List of original features
	•	Numerical: floor_area_sqm
	•	Categorical: flat_type, town, block, storey_range, flat_model
	3.	Date features
	•	From month, extract the month (1–12) and quarter (Q1–Q4) into separate features.
	4.	Age features
	•	Convert remaining_lease from strings (e.g., 61 years 04 months) to a numeric representation in years. Examples:
	•	61 years 04 months → 61.333… years
	•	61 years → 61.0 years
	•	Create lease_age by calculating month - lease_commence_date (in years). (Assume the difference at the monthly granularity.)
	6.	Window features
	1.	Create new columns to capture location detail:
	•	town_street = town + “_” + street_name
	•	town_street_block = town + “” + street_name + “” + block
	2.	Build time-based window features (3-month, 6-month, and 1-month windows) with a 1-month shift (shift(-1)) to avoid leakage:
	•	For each grouping level (town, town_street, town_street_block):
	•	Calculate the rolling average, std, max, min of corrected_price_per_sqm
	•	prior_{window}_{group_col}_avg_price_per_sqm
	•	prior_{window}_{group_col}_std_price_per_sqm
	•	prior_{window}_{group_col}_max_price_per_sqm
	•	prior_{window}_{group_col}_min_price_per_sqm
	•	Also add prior_{window}_month_count, the count of transactions in that window.
	7.	Feature lists
	•	Create a list of engineered features.
	•	Create a final feature list by combining engineered_features + original_features.

2. Splitting the Data: Train, Validation, and Calibration
	1.	Date-based split:
	•	Train: from 2020-01-01 until 2023-12-31 (or 2024-01-01, depending on your cutoff)
	•	Validation + Calibration: from 2024-01-01 to 2024-07-01
	•	Split this period 50/50 for validation vs. calibration.
	2.	Check dimensions:
	•	Report the number of rows in each subset.
	3.	Create X and y:
	•	X_train, y_train, X_val, y_val, X_calib, y_calib
	•	Use corrected_price as the target for modeling.

3. Train and Compare Models

Use MLflow to track experiments. You will conduct four main experiments:
	1.	Experiment 1: MultiQuantile Regression (CatBoost)

from catboost import CatBoostRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

# Quantile levels
quantile_levels = [0.1, 0.5, 0.9]
quantile_str = str(quantile_levels).replace('[','').replace(']','')

# Model parameters
params = {
    'iterations': 1000,
    'depth': 6,
}

# Initialize the CatBoostRegressor
model = CatBoostRegressor(
    loss_function=f'MultiQuantile:alpha={quantile_str}',
    **params,
)

# Fit
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_val)

# Coverage metrics
lower_bound = predictions[:, 0]
median = predictions[:, 1]
upper_bound = predictions[:, 2]

rmse = np.sqrt(mean_squared_error(y_val, median))
coverage = regression_coverage_score(y_val, lower_bound, upper_bound)
width = regression_mean_width_score(y_val, lower_bound, upper_bound)


	2.	Experiment 2: Inflation-Corrected Predictions
	•	Using the predicted_index, adjust the CatBoost predictions (lower_bound, median, upper_bound) for the validation period.
	•	Create new columns lower_bound_corrected, median_corrected, upper_bound_corrected.
	3.	Experiment 3: Conformalized Quantile Regression (CQR) with MAPIE

from mapie.regression import MapieQuantileRegressor

# Train individual CatBoost models for each alpha
list_estimators_cqr = []
for alpha_ in quantile_levels:
    estimator_ = CatBoostRegressor(
        loss_function=f'Quantile:alpha={alpha_}',
        **params,
    )
    estimator_.fit(X_train, y_train)
    list_estimators_cqr.append(estimator_)

# Conformal using MAPIE in 'prefit' mode
mapie_reg = MapieQuantileRegressor(
    list_estimators_cqr, 
    cv="prefit", 
    alpha=0.2
)
mapie_reg.fit(X_calib, y_calib)

y_pred_cqr, y_pis_cqr = mapie_reg.predict(X_val)
lower_bound_cqr = y_pis_cqr[:, 0]
median_cqr = y_pred_cqr
upper_bound_cqr = y_pis_cqr[:, 1]


	4.	Experiment 4: Inflation-Corrected Conformal Predictions
	•	Similar to Experiment 2 but apply inflation correction to (lower_bound_cqr, median_cqr, upper_bound_cqr).

For all experiments:
	•	Compute RMSE on the median predictions (actual vs. median).
	•	Compute interval coverage (using regression_coverage_score or MAPIE’s coverage metrics).
	•	Compute interval width (using regression_mean_width_score or MAPIE’s width metrics).
	•	Plot the prediction intervals vs. true labels on the validation set.
	•	X-axis: date (month)
	•	Y-axis: corrected_price (actual) and predicted intervals.
	•	Log all metrics, models, and charts to MLflow.

Final Instruction
* Add an short explanation for each step and sections (do not number the sections), add short comments to explain the code
* Save the completed Jupyter Notebook in the week_2/notebooks/2_quantile_regression directory.