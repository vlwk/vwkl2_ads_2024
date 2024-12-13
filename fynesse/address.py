# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrix
import statsmodels.api as sm

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV, RidgeCV
from scipy.optimize import LinearConstraint, minimize

# final project

# task 1

def estimate_students(latitude: float, longitude: float, W, scaler, model) -> float:
    """
    Args:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.
    W
    scaler
    model: any nssec area model

    Returns:
    float: Estimated share of students in that area (value between 0 and 1).
    """
    point = Point(longitude, latitude) 
    point_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
    point_gdf = point_gdf.to_crs("EPSG:27700")
    result = gpd.sjoin(point_gdf, gdf_boundaries_2021_nssec_with_counts, how="left", predicate="within")
    cols = prepare_features(tags_nssec_flattened, gdf_boundaries_2021_nssec_with_counts, "_by_area").columns
    custom_X = pd.DataFrame(result[cols])
    custom_X_scaled = scaler.transform(custom_X)
    custom_X_scaled_with_const = np.insert(custom_X_scaled, 0, 1.0)
    custom_X_transformed = np.dot(custom_X_scaled_with_const, W.T)
    custom_X_transformed_with_const = np.insert(custom_X_transformed, 0, 1.0)
    df_custom_X_transformed_with_const = pd.DataFrame(custom_X_transformed_with_const.reshape(1, -1), columns=list(model.feature_names_in_))
    y_pred, _, _ = pred(df_custom_X_transformed_with_const, [result['L15_percentage'].iloc[0]], model)
    return result['L15_percentage'].iloc[0], y_pred[0]

def estimate_health(latitude: float, longitude: float, W, scaler, model) -> float:
    """
    Args:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.

    Returns:
    float: Estimated share of students in that area (value between 0 and 1).
    """
    point = Point(longitude, latitude)  
    point_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
    point_gdf = point_gdf.to_crs("EPSG:27700")
    result = gpd.sjoin(point_gdf, gdf_boundaries_2021_health_with_counts, how="left", predicate="within")
    cols = prepare_features(tags_health_flattened, gdf_boundaries_2021_health_with_counts, "_by_area").columns
    custom_X = pd.DataFrame(result[cols])
    custom_X_scaled = scaler.transform(custom_X)
    custom_X_scaled_with_const = np.insert(custom_X_scaled, 0, 1.0)
    custom_X_transformed = np.dot(custom_X_scaled_with_const, W.T)
    custom_X_transformed_with_const = np.insert(custom_X_transformed, 0, 1.0)
    df_custom_X_transformed_with_const = pd.DataFrame(custom_X_transformed_with_const.reshape(1, -1), columns=list(model.feature_names_in_))
    y_pred, _, _ = pred(df_custom_X_transformed_with_const, [result['very_good_percentage'].iloc[0]], model)
    return result['very_good_percentage'].iloc[0], y_pred[0]


# task 2

def optimise_new_features_with_pca_indices(
    old_features_unscaled, old_feature_indices, new_feature_indices, scaler, W, model, bounds_unscaled, constraints
):
    beta = model.coef_
    feature_names = scaler.feature_names_in_
    
    def prepare_full_features(old_features, new_features):
        """Helper function to construct and scale the full feature vector."""
        full_features_unscaled = np.zeros(len(feature_names))
        full_features_unscaled[old_feature_indices] = old_features
        full_features_unscaled[new_feature_indices] = new_features
        full_features_unscaled_df = pd.DataFrame([full_features_unscaled], columns=feature_names)
        full_features_scaled = scaler.transform(full_features_unscaled_df)[0]
        full_features_scaled_with_const = np.insert(full_features_scaled, 0, 1.0)
        return full_features_scaled_with_const

    def objective(new_features_unscaled):
        full_features_scaled_with_const = prepare_full_features(old_features_unscaled, new_features_unscaled)
        transformed_features = np.dot(full_features_scaled_with_const, W.T)  # Map to PCA space
        transformed_features_with_const = np.insert(transformed_features, 0, 1.0)
        return -np.dot(transformed_features_with_const, beta)  # Negative to maximise

    # Initial guess for new features (middle of bounds)
    x0 = np.array([(low + high) / 2 for low, high in bounds_unscaled])

    # Perform optimisation
    result = minimize(objective, x0, bounds=bounds_unscaled, constraints=constraints, method='SLSQP')

    if result.success:
        # Optimised new features
        optimal_new_features = result.x

        # Calculate the maximised y
        full_features_scaled_with_const = prepare_full_features(old_features_unscaled, optimal_new_features)
        transformed_features = np.dot(full_features_scaled_with_const, W.T)
        transformed_features_with_const = np.insert(transformed_features, 0, 1.0)
        maximised_y = np.dot(transformed_features_with_const, beta)

        return optimal_new_features, maximised_y
    else:
        raise ValueError("Optimisation failed:", result.message)


def get_predictions(latitude: float, longitude: float):
    point = Point(longitude, latitude) 
    point_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
    point_gdf = point_gdf.to_crs("EPSG:27700")
    result = gpd.sjoin(point_gdf, df_final, how="left", predicate="within")
    return result


# Prac 3 Question 1

def q1_cambridge(age_df):

  # Assuming age_df is already loaded into the environment
  # Summing population counts across all locations by age
  y = age_df.loc['Cambridge'].values  # Response (counts of population)
  x = np.arange(len(y))  # Age range as a 1D array

  # --- Separate Design Matrices ---
  # 0-18: Linear Trend with Intercept
  x_0_18 = x[x <= 18]
  y_0_18 = y[x <= 18]
  linear_trend_0_18 = np.column_stack([np.ones_like(x_0_18), x_0_18])  # Add intercept for 0-18

  # 19-100: Polynomial + Splines + RBF with Intercept
  x_19_100 = x[x > 18]
  y_19_100 = y[x > 18]

  # Polynomial basis functions (up to degree 3)
  polynomial_basis_19_100 = np.column_stack([np.ones_like(x_19_100), x_19_100, x_19_100**2, x_19_100**3])

  # Splines (cubic splines with knots at specific ages)
  splines_basis_19_100 = dmatrix("bs(x, knots=(35, 50, 75), degree=3, include_intercept=True)", {"x": x_19_100})

  # Radial Basis Functions (RBF)
  def rbf_kernel(x, centres, length_scale=10.0):
      """Generate RBF basis functions."""
      return np.exp(-((x[:, None] - centres[None, :]) ** 2) / (2 * length_scale ** 2))

  rbf_centres = np.linspace(20, 100, 5)  # Centres for RBFs
  rbf_basis_19_100 = rbf_kernel(x_19_100, rbf_centres)

  # Combine design matrix for 19-100
  design_matrix_19_100 = np.hstack([polynomial_basis_19_100, splines_basis_19_100, rbf_basis_19_100])

  # --- Fit GLM Models ---
  # Poisson GLM for 0-18
  m_poisson_0_18 = sm.GLM(y_0_18, linear_trend_0_18, family=sm.families.Poisson())
  m_poisson_0_18_results = m_poisson_0_18.fit()

  # Gaussian GLM for 0-18
  m_gaussian_0_18 = sm.GLM(y_0_18, linear_trend_0_18, family=sm.families.Gaussian())
  m_gaussian_0_18_results = m_gaussian_0_18.fit()

  # Poisson GLM for 19-100
  m_poisson_19_100 = sm.GLM(y_19_100, design_matrix_19_100, family=sm.families.Poisson())
  m_poisson_19_100_results = m_poisson_19_100.fit()

  # Gaussian GLM for 19-100
  m_gaussian_19_100 = sm.GLM(y_19_100, design_matrix_19_100, family=sm.families.Gaussian())
  m_gaussian_19_100_results = m_gaussian_19_100.fit()

  # --- Predictions ---
  # Predict for 0-18
  x_pred_0_18 = np.arange(0, 19)
  linear_trend_pred_0_18 = np.column_stack([np.ones_like(x_pred_0_18), x_pred_0_18])
  y_pred_poisson_0_18 = m_poisson_0_18_results.get_prediction(linear_trend_pred_0_18).summary_frame(alpha=0.05)
  y_pred_gaussian_0_18 = m_gaussian_0_18_results.get_prediction(linear_trend_pred_0_18).summary_frame(alpha=0.05)

  # Predict for 19-100
  x_pred_19_100 = np.arange(19, 101)
  polynomial_basis_pred_19_100 = np.column_stack([np.ones_like(x_pred_19_100), x_pred_19_100, x_pred_19_100**2, x_pred_19_100**3])
  splines_basis_pred_19_100 = dmatrix("bs(x_pred_19_100, knots=(35, 50, 75), degree=3, include_intercept=True)",
                                      {"x_pred_19_100": x_pred_19_100})
  rbf_basis_pred_19_100 = rbf_kernel(x_pred_19_100, rbf_centres)
  design_matrix_pred_19_100 = np.hstack([polynomial_basis_pred_19_100, splines_basis_pred_19_100, rbf_basis_pred_19_100])
  y_pred_poisson_19_100 = m_poisson_19_100_results.get_prediction(design_matrix_pred_19_100).summary_frame(alpha=0.05)
  y_pred_gaussian_19_100 = m_gaussian_19_100_results.get_prediction(design_matrix_pred_19_100).summary_frame(alpha=0.05)

  # --- Plot Results ---
  fig, ax = plt.subplots(figsize=(12, 6))

  # Scatter plot of actual data
  ax.scatter(x, y, marker='o', color='blue', edgecolor='black', s=50, alpha=0.75, label='Observed data')

  # Poisson fit for 0-18
  ax.plot(x_pred_0_18, y_pred_poisson_0_18['mean'], color='red', linewidth=2, label='Poisson Fit (0-18)')
  ax.fill_between(x_pred_0_18, y_pred_poisson_0_18['mean_ci_lower'], y_pred_poisson_0_18['mean_ci_upper'],
                  color='red', alpha=0.2)

  # Poisson fit for 19-100
  ax.plot(x_pred_19_100, y_pred_poisson_19_100['mean'], color='red', linewidth=2, linestyle='--', label='Poisson Fit (19-100)')
  ax.fill_between(x_pred_19_100, y_pred_poisson_19_100['mean_ci_lower'], y_pred_poisson_19_100['mean_ci_upper'],
                  color='red', alpha=0.2)

  # Gaussian fit for 0-18
  ax.plot(x_pred_0_18, y_pred_gaussian_0_18['mean'], color='cyan', linewidth=2, label='Gaussian Fit (0-18)')
  ax.fill_between(x_pred_0_18, y_pred_gaussian_0_18['mean_ci_lower'], y_pred_gaussian_0_18['mean_ci_upper'],
                  color='cyan', alpha=0.2)

  # Gaussian fit for 19-100
  ax.plot(x_pred_19_100, y_pred_gaussian_19_100['mean'], color='cyan', linewidth=2, linestyle='--', label='Gaussian Fit (19-100)')
  ax.fill_between(x_pred_19_100, y_pred_gaussian_19_100['mean_ci_lower'], y_pred_gaussian_19_100['mean_ci_upper'],
                  color='cyan', alpha=0.2)

  # Legend and labels
  ax.legend()
  ax.set_title("Separate Fits for Ages 0-18 and 19-100 (Poisson and Gaussian)")
  ax.set_xlabel("Age")
  ax.set_ylabel("Summed Population")
  plt.tight_layout()
  plt.show()

# Prac 3 Question 1

def q1_wholeuk(age_df):

  # Assuming age_df is already loaded into the environment
  # Summing population counts across all locations by age
  y = age_df.sum(axis=0).values  # Response (counts of population)
  x = np.arange(len(y))  # Age range as a 1D array

  # --- Build Design Matrix (Basis Functions) ---
  # Polynomial basis functions (up to degree 3)
  polynomial_basis = np.column_stack([x**i for i in range(4)])

  # Splines (cubic splines with knots at specific ages)
  # splines_basis = dmatrix("bs(x, knots=(20, 40, 60, 80), degree=3, include_intercept=False)", {"x": x})
  splines_basis = dmatrix("bs(x, knots=(15, 35, 50, 75), degree=3, include_intercept=False)", {"x": x})

  # Radial Basis Functions (RBF)
  def rbf_kernel(x, centres, length_scale=10.0):
      """Generate RBF basis functions."""
      return np.exp(-((x[:, None] - centres[None, :]) ** 2) / (2 * length_scale ** 2))

  rbf_centres = np.linspace(0, 100, 5)  # Centres for RBFs (e.g., 5 evenly spaced points)
  rbf_basis = rbf_kernel(x, rbf_centres)

  # Combine all basis functions into a single design matrix
  design_matrix = np.hstack([polynomial_basis, splines_basis, rbf_basis])

  # --- Fit GLM Models ---
  # Poisson GLM
  m_poisson = sm.GLM(y, design_matrix, family=sm.families.Poisson())
  m_poisson_results = m_poisson.fit()

  # Gaussian GLM
  m_gaussian = sm.GLM(y, design_matrix, family=sm.families.Gaussian())
  m_gaussian_results = m_gaussian.fit()

  # --- Predictions ---
  x_pred = np.arange(100)  # Predict for the full age range
  # Reconstruct the design matrix for predictions
  polynomial_basis_pred = np.column_stack([x_pred**i for i in range(4)])
  splines_basis_pred = dmatrix("bs(x_pred, knots=(15, 35, 50, 75), degree=3, include_intercept=False)", {"x_pred": x_pred})
  rbf_basis_pred = rbf_kernel(x_pred, rbf_centres)
  design_matrix_pred = np.hstack([polynomial_basis_pred, splines_basis_pred, rbf_basis_pred])

  # Predict using Poisson GLM
  y_pred_poisson = m_poisson_results.get_prediction(design_matrix_pred).summary_frame(alpha=0.05)

  # Predict using Gaussian GLM
  y_pred_gaussian = m_gaussian_results.get_prediction(design_matrix_pred).summary_frame(alpha=0.05)

  # --- Plot Results ---
  fig, ax = plt.subplots(figsize=(12, 6))

  # Scatter plot of actual data
  ax.scatter(x, y, marker='o', color='blue', edgecolor='black', s=50, alpha=0.75, label='Observed data')

  # Poisson fit
  ax.plot(x_pred, y_pred_poisson['mean'], color='red', linewidth=2, label='Poisson Fit')
  ax.fill_between(x_pred, y_pred_poisson['mean_ci_lower'], y_pred_poisson['mean_ci_upper'],
                  color='red', alpha=0.2, label='Poisson 95% CI')

  # Gaussian fit
  ax.plot(x_pred, y_pred_gaussian['mean'], color='cyan', linewidth=2, label='Gaussian Fit')
  ax.fill_between(x_pred, y_pred_gaussian['mean_ci_lower'], y_pred_gaussian['mean_ci_upper'],
                  color='cyan', alpha=0.2, label='Gaussian 95% CI')

  # Legend and labels
  ax.legend()
  ax.set_title("Age Distribution Fitted with Poisson and Gaussian GLM (Design Matrix)")
  ax.set_xlabel("Age")
  ax.set_ylabel("Summed Population")
  plt.tight_layout()
  plt.show()

# Prac 3 Question 4

def linreg_onefeature_oneage(ts062_df, xcol, ycol):

  # Generate data
  # x = ts062_df['col9_proportion']
  # y = ts062_df[21]
  x = ts062_df[xcol]
  y = ts062_df[ycol]

  # Add a constant column to include the intercept
  X = sm.add_constant(x)  # This adds an intercept (a column of ones)

  # Fit a simple linear model
  m_linear = sm.OLS(y, X)
  results = m_linear.fit()

  # Predictions
  x_pred = np.linspace(0, 0.3, 200)  # Range of x values for prediction
  X_pred = sm.add_constant(x_pred)  # Add intercept to prediction matrix

  y_pred_linear = results.get_prediction(X_pred).summary_frame(alpha=0.05)

  # Plotting
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111)
  ax.scatter(x, y, label="Data", zorder=2)

  # Linear model predictions
  ax.plot(x_pred, y_pred_linear['mean'], color='red', linestyle='--', label="Mean Prediction", zorder=1)
  ax.plot(x_pred, y_pred_linear['obs_ci_lower'], color='red', linestyle='-', label="Confidence Interval", zorder=1)
  ax.plot(x_pred, y_pred_linear['obs_ci_upper'], color='red', linestyle='-', zorder=1)
  ax.fill_between(
      x_pred,
      y_pred_linear['obs_ci_lower'],
      y_pred_linear['obs_ci_upper'],
      color='red',
      alpha=0.3,
      zorder=1
  )

  ax.set_ylim(bottom=0)
  ax.set_xlabel("col9_proportion")
  ax.set_ylabel("Target Variable")
  ax.legend()
  plt.tight_layout()
  plt.show()

  # Print summary of the model
  print(results.summary())

# Prac 3 Question 5

def linreg_allfeatures_oneage(ts062_df, ycol):

  # Generate data
  y = ts062_df[ycol]

  # Design matrix with all 9 features and add intercept
  design = ts062_df[[f'col{i}_proportion' for i in range(1, 10)]].values
  design_with_intercept = sm.add_constant(design)  # Adds a column of ones for the intercept

  # Fit the OLS model
  m_linear_basis = sm.OLS(y, design_with_intercept)
  results_basis = m_linear_basis.fit()

  # Predictions
  x_pred = ts062_df['col9_proportion'].values
  y_pred_linear_basis = results_basis.get_prediction(design_with_intercept).summary_frame(alpha=0.05)

  # Plotting
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111)

  # Scatter original data
  ax.scatter(x_pred, y, label="Actual Data", alpha=0.7, zorder=2)

  # Basis model predictions
  ax.plot(x_pred, y_pred_linear_basis['mean'], color='cyan', linestyle='--', label="Model Predictions", zorder=1)
  ax.fill_between(
      x_pred,
      y_pred_linear_basis['obs_ci_lower'],
      y_pred_linear_basis['obs_ci_upper'],
      color='cyan', alpha=0.3, label="Confidence Interval"
  )

  # Plot formatting
  ax.set_xlabel('col9_proportion')
  ax.set_ylabel('Target Variable')
  ax.legend()
  plt.tight_layout()
  plt.show()

  # Model summary
  print(results_basis.summary())

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV, RidgeCV
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Prac 3 Question 7

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV, RidgeCV
import statsmodels.api as sm
import matplotlib.pyplot as plt

def kfoldcrossval(ts062_df, ycol):

  # Data preparation
  X = ts062_df[[f'col{i}_proportion' for i in range(1, 10)]].values
  y = ts062_df[21].values

  # Values of k to test
  k_values = [2, 5, 10, 20]

  coefficients_ols = {k: [] for k in k_values}
  coefficients_lasso = {k: [] for k in k_values}
  coefficients_ridge = {k: [] for k in k_values}

  # Dictionaries to store results
  train_mse_results_ols = []
  test_mse_results_ols = []
  overfitting_metric_mse_ols = []
  train_r2_results_ols = []
  test_r2_results_ols = []
  overfitting_metric_r2_ols = []

  train_mse_results_lasso = []
  test_mse_results_lasso = []
  overfitting_metric_mse_lasso = []
  train_r2_results_lasso = []
  test_r2_results_lasso = []
  overfitting_metric_r2_lasso = []

  train_mse_results_ridge = []
  test_mse_results_ridge = []
  overfitting_metric_mse_ridge = []
  train_r2_results_ridge = []
  test_r2_results_ridge = []
  overfitting_metric_r2_ridge = []

  # Perform k-fold cross-validation for different k values
  for k in k_values:
      kf = KFold(n_splits=k, shuffle=True, random_state=42)

      train_mse_list_ols, test_mse_list_ols, train_r2_list_ols, test_r2_list_ols = [], [], [], []
      train_mse_list_lasso, test_mse_list_lasso, train_r2_list_lasso, test_r2_list_lasso = [], [], [], []
      train_mse_list_ridge, test_mse_list_ridge, train_r2_list_ridge, test_r2_list_ridge = [], [], [], []

      for train_index, test_index in kf.split(X):
          X_train, X_test = X[train_index], X[test_index]
          y_train, y_test = y[train_index], y[test_index]

          # Add intercept to design matrix
          X_train_with_intercept = sm.add_constant(X_train)
          X_test_with_intercept = sm.add_constant(X_test)

          # OLS Model
          model_ols = sm.OLS(y_train, X_train_with_intercept).fit()
          y_train_pred_ols = model_ols.predict(X_train_with_intercept)
          y_test_pred_ols = model_ols.predict(X_test_with_intercept)
          coefficients_ols[k].append(model_ols.params)

          train_mse_list_ols.append(mean_squared_error(y_train, y_train_pred_ols))
          test_mse_list_ols.append(mean_squared_error(y_test, y_test_pred_ols))
          train_r2_list_ols.append(r2_score(y_train, y_train_pred_ols))
          test_r2_list_ols.append(r2_score(y_test, y_test_pred_ols))
          

          # Lasso Model
          model_lasso = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=10000, random_state=42).fit(X_train_with_intercept, y_train)
          y_train_pred_lasso = model_lasso.predict(X_train_with_intercept)
          y_test_pred_lasso = model_lasso.predict(X_test_with_intercept)
          coefficients_lasso[k].append(model_lasso.coef_)

          train_mse_list_lasso.append(mean_squared_error(y_train, y_train_pred_lasso))
          test_mse_list_lasso.append(mean_squared_error(y_test, y_test_pred_lasso))
          train_r2_list_lasso.append(r2_score(y_train, y_train_pred_lasso))
          test_r2_list_lasso.append(r2_score(y_test, y_test_pred_lasso))

          # Ridge Model
          model_ridge = RidgeCV(alphas=np.logspace(-4, 1, 50), cv=5).fit(X_train_with_intercept, y_train)
          y_train_pred_ridge = model_ridge.predict(X_train_with_intercept)
          y_test_pred_ridge = model_ridge.predict(X_test_with_intercept)
          coefficients_ridge[k].append(model_ridge.coef_)

          train_mse_list_ridge.append(mean_squared_error(y_train, y_train_pred_ridge))
          test_mse_list_ridge.append(mean_squared_error(y_test, y_test_pred_ridge))
          train_r2_list_ridge.append(r2_score(y_train, y_train_pred_ridge))
          test_r2_list_ridge.append(r2_score(y_test, y_test_pred_ridge))

      # Store average results for each k
      train_mse_results_ols.append(np.mean(train_mse_list_ols))
      test_mse_results_ols.append(np.mean(test_mse_list_ols))
      overfitting_metric_mse_ols.append(np.mean(test_mse_list_ols) - np.mean(train_mse_list_ols))
      train_r2_results_ols.append(np.mean(train_r2_list_ols))
      test_r2_results_ols.append(np.mean(test_r2_list_ols))
      overfitting_metric_r2_ols.append(np.mean(train_r2_list_ols) - np.mean(test_r2_list_ols))

      train_mse_results_lasso.append(np.mean(train_mse_list_lasso))
      test_mse_results_lasso.append(np.mean(test_mse_list_lasso))
      overfitting_metric_mse_lasso.append(np.mean(test_mse_list_lasso) - np.mean(train_mse_list_lasso))
      train_r2_results_lasso.append(np.mean(train_r2_list_lasso))
      test_r2_results_lasso.append(np.mean(test_r2_list_lasso))
      overfitting_metric_r2_lasso.append(np.mean(train_r2_list_lasso) - np.mean(test_r2_list_lasso))

      train_mse_results_ridge.append(np.mean(train_mse_list_ridge))
      test_mse_results_ridge.append(np.mean(test_mse_list_ridge))
      overfitting_metric_mse_ridge.append(np.mean(test_mse_list_ridge) - np.mean(train_mse_list_ridge))
      train_r2_results_ridge.append(np.mean(train_r2_list_ridge))
      test_r2_results_ridge.append(np.mean(test_r2_list_ridge))
      overfitting_metric_r2_ridge.append(np.mean(train_r2_list_ridge) - np.mean(test_r2_list_ridge))

  # Plot Overfitting Metric (MSE) vs. k for all models
  plt.figure(figsize=(10, 5))
  plt.plot(k_values, overfitting_metric_mse_ols, marker='o', label='Overfitting (MSE) OLS', linestyle='--', color='red')
  plt.plot(k_values, overfitting_metric_mse_lasso, marker='x', label='Overfitting (MSE) Lasso', linestyle='--', color='green')
  plt.plot(k_values, overfitting_metric_mse_ridge, marker='s', label='Overfitting (MSE) Ridge', linestyle='--', color='blue')
  plt.title('Overfitting Metric (MSE) for OLS, Lasso, and Ridge')
  plt.xlabel('Number of Folds (k)')
  plt.ylabel('Overfitting Metric (Test MSE - Train MSE)')
  plt.legend()
  plt.grid(True)
  plt.show()

  # Plot Overfitting Metric (R^2) vs. k for all models
  plt.figure(figsize=(10, 5))
  plt.plot(k_values, overfitting_metric_r2_ols, marker='o', label='Overfitting (R²) OLS', linestyle='--', color='red')
  plt.plot(k_values, overfitting_metric_r2_lasso, marker='x', label='Overfitting (R²) Lasso', linestyle='--', color='green')
  plt.plot(k_values, overfitting_metric_r2_ridge, marker='s', label='Overfitting (R²) Ridge', linestyle='--', color='blue')
  plt.title('Overfitting Metric (R²) for OLS, Lasso, and Ridge')
  plt.xlabel('Number of Folds (k)')
  plt.ylabel('Overfitting Metric (Train R² - Test R²)')
  plt.legend()
  plt.grid(True)
  plt.show()

  # Plot MSE vs. k for all models
  plt.figure(figsize=(10, 5))
  plt.plot(k_values, train_mse_results_ols, marker='o', label='Train MSE (OLS)', linestyle='--', color='red')
  plt.plot(k_values, test_mse_results_ols, marker='o', label='Test MSE (OLS)', linestyle='-', color='red')
  plt.plot(k_values, train_mse_results_lasso, marker='x', label='Train MSE (Lasso)', linestyle='--', color='green')
  plt.plot(k_values, test_mse_results_lasso, marker='x', label='Test MSE (Lasso)', linestyle='-', color='green')
  plt.plot(k_values, train_mse_results_ridge, marker='s', label='Train MSE (Ridge)', linestyle='--', color='blue')
  plt.plot(k_values, test_mse_results_ridge, marker='s', label='Test MSE (Ridge)', linestyle='-', color='blue')
  plt.title('Cross-Validation: Train vs. Test MSE for OLS, Lasso, and Ridge')
  plt.xlabel('Number of Folds (k)')
  plt.ylabel('Mean Squared Error (MSE)')
  plt.legend()
  plt.grid(True)
  plt.show()


  # Plot R² vs. k for all models
  plt.figure(figsize=(10, 5))
  plt.plot(k_values, train_r2_results_ols, marker='o', label='Train R² (OLS)', linestyle='--', color='red')
  plt.plot(k_values, test_r2_results_ols, marker='o', label='Test R² (OLS)', linestyle='-', color='red')
  plt.plot(k_values, train_r2_results_lasso, marker='x', label='Train R² (Lasso)', linestyle='--', color='green')
  plt.plot(k_values, test_r2_results_lasso, marker='x', label='Test R² (Lasso)', linestyle='-', color='green')
  plt.plot(k_values, train_r2_results_ridge, marker='s', label='Train R² (Ridge)', linestyle='--', color='blue')
  plt.plot(k_values, test_r2_results_ridge, marker='s', label='Test R² (Ridge)', linestyle='-', color='blue')
  plt.title('Cross-Validation: Train vs. Test R² for OLS, Lasso, and Ridge')
  plt.xlabel('Number of Folds (k)')
  plt.ylabel('R²')
  plt.legend()
  plt.grid(True)
  plt.show()

  # --- Plot Coefficients for Each k ---
  for k in k_values:
      fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

      # OLS Coefficients
      ax[0].plot(np.array(coefficients_ols[k]).T, marker='o')
      ax[0].set_title(f'OLS Coefficients (k={k})')
      ax[0].set_xlabel('Coefficient Index')
      ax[0].set_ylabel('Coefficient Value')

      # Lasso Coefficients
      ax[1].plot(np.array(coefficients_lasso[k]).T, marker='x')
      ax[1].set_title(f'Lasso Coefficients (k={k})')
      ax[1].set_xlabel('Coefficient Index')

      # Ridge Coefficients
      ax[2].plot(np.array(coefficients_ridge[k]).T, marker='s')
      ax[2].set_title(f'Ridge Coefficients (k={k})')
      ax[2].set_xlabel('Coefficient Index')

      fig.suptitle(f'Coefficient Paths for k={k}')
      plt.tight_layout()
      plt.show()


# Practical 3 Question 8


# Use this box for any code you need

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

def ridgecoeff_age(ts062_df):

  # Data preparation
  X = ts062_df[[f'col{i}_proportion' for i in range(1, 10)]].values

  X = sm.add_constant(X)

  # Ridge model with cross-validation for optimal alpha
  ridge_cv = RidgeCV(alphas=np.logspace(-4, 1, 50), cv=5)


  # Store coefficients for each age column
  coefficients_by_age_ols = []
  coefficients_by_age_ridge = []

  # Iterate over all age columns (1 to 99)
  for age in range(0, 100):  # Assuming columns are named as integers 1 to 99
      y = ts062_df[age].values  # Target variable for current age
      model_ols = sm.OLS(y, X).fit()
      ridge_cv.fit(X, y)  # Fit Ridge model
      coefficients_by_age_ridge.append(ridge_cv.coef_)  # Store coefficients
      coefficients_by_age_ols.append(model_ols.params)

  # Convert to numpy array for easier plotting
  coefficients_by_age_ols = np.array(coefficients_by_age_ols)
  coefficients_by_age_ridge = np.array(coefficients_by_age_ridge)

  # Plot coefficients for each feature across all ages
  plt.figure(figsize=(12, 8))
  for feature_idx in range(1, X.shape[1]):  # Iterate over features
      plt.plot(range(0, 100), coefficients_by_age_ridge[:, feature_idx], label=f'Feature col{feature_idx}_proportion')

  # Formatting the plot
  plt.title('Ridge Coefficients for Each Feature Across Ages')
  plt.xlabel('Age')
  plt.ylabel('Coefficient Value')
  plt.legend(title="Features", bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True)
  plt.tight_layout()
  plt.show()



  # Plot coefficients for each feature across all ages
  plt.figure(figsize=(12, 8))
  for feature_idx in range(1, X.shape[1]):  # Iterate over features
      plt.plot(range(0, 100), coefficients_by_age_ols[:, feature_idx], label=f'Feature col{feature_idx}_proportion')

  # Formatting the plot
  plt.title('OLS Coefficients for Each Feature Across Ages')
  plt.xlabel('Age')
  plt.ylabel('Coefficient Value')
  plt.legend(title="Features", bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True)
  plt.tight_layout()
  plt.show()

  return coefficients_by_age_ols, coefficients_by_age_ridge

# Prac 3 Question 9

def compare_age_distributions(age_df, ts062_df, coefficients_by_age_ols, coefficients_by_age_ridge, locations=None):
    """
    Compare actual and predicted age distributions using OLS and Ridge coefficients.
    
    Parameters:
    - age_df: DataFrame containing actual age distributions for locations.
    - ts062_df: DataFrame containing features for locations.
    - coefficients_by_age_ols: Array of OLS coefficients for each age.
    - coefficients_by_age_ridge: Array of Ridge coefficients for each age.
    - locations: List of location indices to process. If None, all locations in age_df are used.
    
    Returns:
    - results: List of dictionaries containing location, actual, and predicted distributions (OLS & Ridge).
    """
    if locations is None:
        locations = age_df.index

    results = []

    for location in locations:
      if (location == 'Cambridge'):
        # Extract feature values for the current location from ts062_df
        location_features = ts062_df.loc[ts062_df['geography'] == location, [f'col{i}_proportion' for i in range(1, 10)]].values.flatten()


        if location_features.size == 0:
            print(f"Warning: Location {location} not found in ts062_df. Skipping...")
            continue

        location_features_with_intercept = np.insert(location_features, 0, 1)

        # Extract actual age distribution for the location
        actual_age_distribution = age_df.loc[location].values

        # Predict age distribution using OLS coefficients
        predicted_age_distribution_ols = []
        for age in range(0, 100):
            predicted_value = np.dot(location_features_with_intercept, coefficients_by_age_ols[age])
            predicted_age_distribution_ols.append(predicted_value)
        predicted_age_distribution_ols = np.array(predicted_age_distribution_ols)

        # Predict age distribution using Ridge coefficients
        predicted_age_distribution_ridge = []
        for age in range(0, 100):
            predicted_value = np.dot(location_features_with_intercept, coefficients_by_age_ridge[age])
            predicted_age_distribution_ridge.append(predicted_value)
        predicted_age_distribution_ridge = np.array(predicted_age_distribution_ridge)

        # Shift predictions to remove negatives
        # min_val_ols = np.min(predicted_age_distribution_ols)
        # if min_val_ols < 0:
        #     predicted_age_distribution_ols += abs(min_val_ols)

        min_val_ridge = np.min(predicted_age_distribution_ridge)
        if min_val_ridge < 0:
            predicted_age_distribution_ridge += abs(min_val_ridge)

        # Normalise predictions to match the total population
        total_population_actual = np.sum(actual_age_distribution)
        predicted_age_distribution_ols = predicted_age_distribution_ols * (total_population_actual / np.sum(predicted_age_distribution_ols))
        predicted_age_distribution_ridge = predicted_age_distribution_ridge * (total_population_actual / np.sum(predicted_age_distribution_ridge))

        # Store results for analysis or plotting
        results.append({
            "location": location,
            "actual": actual_age_distribution,
            "predicted_ols": predicted_age_distribution_ols,
            "predicted_ridge": predicted_age_distribution_ridge
        })

        # Optional: Print progress
        print(f"Processed location: {location}")

    # Plot actual vs predicted for each location
    for result in results:
        location = result["location"]
        actual = result["actual"]
        predicted_ols = result["predicted_ols"]
        predicted_ridge = result["predicted_ridge"]

        plt.figure(figsize=(8, 5))
        plt.plot(range(0, 100), actual, label='Actual Age Distribution', marker='o')
        plt.plot(range(0, 100), predicted_ols, label='Predicted Age Distribution (OLS)', linestyle='--')
        plt.plot(range(0, 100), predicted_ridge, label='Predicted Age Distribution (Ridge)', linestyle='-.')
        plt.title(f'Actual vs Predicted Age Distribution for {location}')
        plt.xlabel('Age')
        plt.ylabel('Number of People')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results
