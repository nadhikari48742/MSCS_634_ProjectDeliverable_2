# MSCS 634 – Project Deliverable 2
## Regression Modeling and Performance Evaluation

### Project Overview

This deliverable applies regression modeling to the California Housing dataset to predict median house values. The work includes data inspection, feature engineering, model training, test-set evaluation, five-fold cross-validation, and visual comparison of model performance.

### Dataset

- **Dataset:** California Housing
- **Source:** `fetch_california_housing` from scikit-learn
- **Number of records:** 20,640
- **Original predictor attributes:** 8
- **Target variable:** `median_house_value`
- **Target units:** The target is expressed in units of $100,000. For example, a target value of 2.0 represents $200,000.

The original predictors are median income (`MedInc`), house age (`HouseAge`), average rooms (`AveRooms`), average bedrooms (`AveBedrms`), population (`Population`), average occupancy (`AveOccup`), latitude, and longitude.

### Data Preparation and Exploration

The notebook loads the dataset into a Pandas DataFrame and checks its dimensions, data types, descriptive statistics, missing values, and duplicate records. The dataset contained 20,640 records and 9 columns after the target column was added. The displayed data-quality checks found no missing values.

Exploratory analysis includes:
- A histogram of the target variable.
- A correlation heatmap.
- Correlations between each original predictor and the target.

The correlation analysis showed that median income had the strongest positive correlation with median house value among the original predictors (approximately **0.688**). The target distribution also showed a noticeable concentration at the upper value of 5.0, which is a characteristic of the dataset's capped target values and should be kept in mind when interpreting predictions.

### Feature Engineering

Four additional columns were created:

1. `rooms_per_household` – uses the average rooms value.
2. `bedrooms_per_room` – average bedrooms divided by average rooms.
3. `population_per_household` – population divided by average occupancy.
4. `income_per_room` – median income divided by average rooms.

The engineered dataset contained 20,640 records and 13 columns, including the target. The modeling input contained 12 predictor columns.

### Modeling Approach

The data was divided into:
- **80% training data:** 16,512 records
- **20% testing data:** 4,128 records

A `StandardScaler` was included in each scikit-learn pipeline so that scaling was learned from the training data during model fitting. Three regression models were trained:

- **Linear Regression** – baseline regression model.
- **Ridge Regression** – linear regression with L2 regularization (`alpha=1.0`).
- **Lasso Regression** – linear regression with L1 regularization (`alpha=0.001`).

### Test-Set Evaluation

The models were evaluated on the held-out test set using R-squared, mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE).

| Model | R-squared | MSE | RMSE | MAE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.6306 | 0.4840 | 0.6957 | 0.5029 |
| Ridge Regression | 0.6306 | 0.4841 | 0.6957 | 0.5029 |
| Lasso Regression | 0.6293 | 0.4858 | 0.6970 | 0.5033 |

Linear Regression and Ridge Regression performed almost identically on the test set. Linear Regression had the marginally lowest displayed MSE and tied Ridge Regression on the displayed RMSE and MAE. Lasso Regression was slightly behind on these test-set measures.

### Five-Fold Cross-Validation

Five-fold cross-validation was performed using only the training data. The table reports the mean and standard deviation of R-squared and RMSE across the five validation folds.

| Model | Mean CV R-squared | Std. CV R-squared | Mean CV RMSE | Std. CV RMSE | Mean CV MAE |
|---|---:|---:|---:|---:|---:|
| Lasso Regression | 0.6482 | 0.0137 | 0.6855 | 0.0129 | 0.4961 |
| Ridge Regression | 0.6480 | 0.0140 | 0.6857 | 0.0129 | 0.4960 |
| Linear Regression | 0.6480 | 0.0140 | 0.6857 | 0.0129 | 0.4960 |

Lasso Regression had the lowest mean cross-validation RMSE and the highest mean cross-validation R-squared, although its advantage over Ridge and Linear Regression was very small. This suggests that all three models had broadly similar validation performance.

### Visualizations and Diagnostics

The notebook generates:
- Target-variable distribution histogram.
- Correlation heatmap.
- Test-set R-squared and RMSE comparison charts.
- Five-fold cross-validation R-squared and RMSE comparison charts.
- Actual-versus-predicted plot for the best test-set model.
- Residuals-versus-predictions plot and residual distribution.
- Feature-coefficient table and coefficient bar chart.

The actual-versus-predicted and residual plots provide additional ways to inspect prediction errors. The residual plot shows that errors are not perfectly uniform across the prediction range, so the linear models do not capture every pattern in the data. The target's upper cap also affects the interpretation of predictions near the maximum target value.

### Conclusion

The three regression models produced similar results. On the held-out test set, Linear Regression and Ridge Regression were effectively tied, with Linear Regression having the lowest displayed MSE. In five-fold cross-validation, Lasso Regression performed marginally better based on mean RMSE and mean R-squared.

Because the differences are small, the results do not establish a large practical advantage for one model. The comparison shows that regularization did not materially change performance for this feature set and parameter choices. Further work could test additional feature transformations, tune Ridge and Lasso regularization strengths, and evaluate nonlinear regression methods.

### Challenges and Considerations

- **Feature scale differences:** Predictors use different units and ranges. A scaler was placed inside each model pipeline to standardize features without fitting the scaler on the held-out test data.
- **Division in engineered features:** Engineered ratios can create invalid values if a denominator is zero. The notebook protects divisions and checks for infinite or missing values.
- **Similar model performance:** The three models have very close scores. The results are therefore reported with their small differences rather than claiming a substantial winner.
- **Capped target values:** The target distribution has a visible concentration at 5.0. This cap can limit how well regression predictions represent the most expensive areas.

### Repository Contents

Recommended repository structure:

```text
MSCS_634_ProjectDeliverable_2/
├── Residency_Day_2_Regression_Modeling.ipynb
├── README.md
├── regression_test_results.csv
└── regression_cross_validation_results.csv
```

The two CSV result files are generated when the notebook is run. Include them in the repository if they were downloaded from the Colab session.

### How to Run

1. Open Google Colab.
2. Select **File → Upload notebook**.
3. Upload `Residency_Day_2_Regression_Modeling.ipynb`.
4. Select **Runtime → Run all**.
5. Allow the dataset to be fetched by scikit-learn.
6. Review the evaluation tables and plots.
7. Download the generated CSV files if you want to include them in the repository.

### References

Scikit-learn developers. (n.d.). *California housing dataset*. Scikit-learn documentation. https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

Scikit-learn developers. (n.d.). *Linear models*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/linear_model.html

Scikit-learn developers. (n.d.). *Cross-validation: Evaluating estimator performance*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/cross_validation.html
