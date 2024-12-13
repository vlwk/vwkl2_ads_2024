from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import osmnx as ox
from fuzzywuzzy import fuzz
import ast
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV, RidgeCV
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import KFold

# final project 

def add_geo_from_lat_long(df):
    # turns a points df into a gdf with a geo col 
    df2 = df.copy()
    transformer = Transformer.from_crs("epsg:4326", "epsg:27700", always_xy=True)
    bng_e, bng_n = transformer.transform(df2['longitude'].values, df2['latitude'].values)
    df2['geometry'] = [Point(e, n) for e, n in zip(bng_e, bng_n)]
    gdf = gpd.GeoDataFrame(df2, geometry='geometry',crs="EPSG:27700")
    return gdf

def tags_to_dict(df):
    df2 = df.copy()
    df2['tags_dict'] = df2['tags'].apply(ast.literal_eval)
    return df2

def is_valid_literal(val):
    try:
        ast.literal_eval(val)
        return True
    except (ValueError, SyntaxError):
        return False

def normalise_dict(gdf, tag_dict):
    # takes a gdf and a tags dict {k:v} and makes each k or k_v a col
    chunk_size = 50000
    total_rows = gdf.shape[0]
    chunks = range(0, total_rows, chunk_size)
    tags_dfs = []
    print("Starting batch processing...")
    for start in tqdm(chunks, desc="Processing chunks"):
        end = min(start + chunk_size, total_rows)
        chunk = gdf.iloc[start:end]
        tags_df = pd.json_normalize(chunk["tags_dict"])
        tags_df = tags_df.reindex(columns=list(tag_dict.keys()), fill_value=None)
        tags_dfs.append(tags_df)
    tags_df = pd.concat(tags_dfs, ignore_index=True)
    expanded_gdf = pd.concat([gdf, tags_df], axis=1)
    return expanded_gdf

def add_counts_to_boundaries(boundaries, points, tags, lvl): # gdf is for gdf boundaries
    ret = boundaries.copy()
    num_pts = points.groupby(lvl).size().reset_index(name="num_pts")
    ret = ret.merge(num_pts, on=lvl, how="left")
    ret["num_pts"] = ret["num_pts"].fillna(0).astype(int) 
    for f, v in tags.items():
        if (len(v) == 0):
            filtered_pts = points[~points[f].isna()]
            point_counts = filtered_pts.groupby(lvl).size().reset_index(name=f+"_count")
            ret = ret.merge(point_counts, on=lvl, how="left")
            ret[f + "_count"] = ret[f + "_count"].fillna(0).astype(int)
            ret[f + "_by_area"] = (ret[f + "_count"] / ret["area"]).fillna(0) * 100
        else:
            for value in v:
                filtered_pts = points[points[f] == value]
                point_counts = filtered_pts.groupby(lvl).size().reset_index(name=f+ "_" + value + "_count")
                ret = ret.merge(point_counts, on=lvl, how="left")
                ret[f + "_" + value + "_count"] = ret[f + "_" + value + "_count"].fillna(0).astype(int)
                ret[f + "_" + value + "_by_area"] = (ret[f + "_" + value + "_count"] / ret["area"]).fillna(0) * 100
    return ret

def basic_y_against_x(feature, x_suff, y_label, df):
    

    x = df[feature + x_suff]
    y = df[y_label]
    
    # Add a constant column to include the intercept
    X = sm.add_constant(x)  
    
    # Fit a simple linear model
    m_linear = sm.OLS(y, X)
    results = m_linear.fit()
    
    # Predictions
    x_pred = np.linspace(0, x.max(), 200)  # Range of x values for prediction
    X_pred = sm.add_constant(x_pred)  # Add intercept to prediction matrix
    
    y_pred_linear = results.get_prediction(X_pred).summary_frame(alpha=0.05)
    
    # Plotting
    fig = plt.figure(figsize=(6, 4))
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
    ax.set_xlabel(feature + x_suff)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Print summary of the model
    print(results.rsquared)



def get_ols_model(df_x, df_y):
    ols_model = sm.OLS(df_y, df_x)
    ols_model_fitted = ols_model.fit()
    print(ols_model_fitted.summary())
    return ols_model_fitted

def get_lasso_model(df_x, df_y):
    lasso_model = LassoCV(cv=5, random_state=42).fit(df_x, df_y)
    print("Optimal alpha (regularisation strength):", lasso_model.alpha_)
    return lasso_model

def get_ridge_model(df_x, df_y):
    ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(df_x, df_y)
    print("Optimal alpha (regularisation strength):", ridge_model.alpha_)
    return ridge_model

def pred(df_x, df_y, model):
    y_pred = model.predict(df_x)
    mse = mean_squared_error(df_y, y_pred)
    r2 = r2_score(df_y, y_pred)
    return y_pred, mse, r2

def plot_pred_vs_actual_index(y_actual, y_pred, label):
    fig = plt.figure(figsize = (6, 4))
    ax = fig.add_subplot(111)
    ax.scatter(np.arange(len(y_actual)), y_actual, label = "Actual", color = "black", zorder = 2)
    ax.plot(
        np.arange(len(y_pred)),
        y_pred,
        label = "Predicted",
        color = "blue",
        linestyle = "--",
        zorder = 1,
    )
    ax.set_xlabel("Index")
    ax.set_ylabel(label)
    ax.legend()
    plt.tight_layout()
    plt.show()

def scale_x(X_combined_df):
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined_df)
    X_combined_scaled_df = pd.DataFrame(X_combined_scaled, columns=X_combined_df.columns)
    X_combined_scaled_with_int_df = sm.add_constant(X_combined_scaled_df)
    return X_combined_scaled_with_int_df, scaler

def pca_transform_with_int(x_scaled):
    
    pca = PCA()
    X_pca = pca.fit_transform(x_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Principal Components')
    plt.grid()
    plt.show()
    
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Number of components to retain 95% variance: {n_components}")
    
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(x_scaled)

    X_reduced_with_int = sm.add_constant(X_reduced)
    column_labels = ['const'] + [f'x{i}' for i in range(X_reduced.shape[1])]

    X_reduced_df = pd.DataFrame(X_reduced_with_int, columns=column_labels)

    loadings = pd.DataFrame(
        pca.components_,
        columns=x_scaled.columns, 
        index=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    print("Feature Loadings (PCA Coefficients):")
    print(loadings)
    
    plt.figure(figsize=(12, 8))


    for i in range(loadings.shape[0]):
        plt.figure(figsize=(10, 6))
        plt.bar(loadings.columns, loadings.iloc[i], color='blue')
        plt.xlabel('Features')
        plt.ylabel('PCA Coefficient (Loading)')
        plt.title(f'PCA Loadings for PC{i+1}')
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()



    return X_reduced_df, pca.components_

def reg(y_label, X_pca, df):
    
    ols_model_fitted = get_ols_model(X_pca, df[y_label])
    y_pred, mse, r2 = pred(X_pca, df[y_label], ols_model_fitted)
    plot_pred_vs_actual_index(df[y_label], y_pred, y_label)
    print("mean squared error OLS: ", mse)
    
    lasso_model_fitted = get_lasso_model(X_pca, df[y_label])
    y_pred, mse, r2 = pred(X_pca, df[y_label], lasso_model_fitted)
    plot_pred_vs_actual_index(df[y_label], y_pred, y_label)
    print("mean squared error Lasso: ", mse)
    print("r-squared value Lasso: ", r2)
    print("coeffs Lasso: ", lasso_model_fitted.coef_)
    
    ridge_model_fitted = get_ridge_model(X_pca, df[y_label])
    y_pred, mse, r2 = pred(X_pca, df[y_label], ridge_model_fitted)
    plot_pred_vs_actual_index(df[y_label], y_pred, y_label)
    print("mean squared error Ridge: ", mse)
    print("r-squared value Ridge: ", r2)
    print("coeffs Ridge: ", ridge_model_fitted.coef_)

    return ols_model_fitted, lasso_model_fitted, ridge_model_fitted




# task 1 specific
def prepare_features(tags_flattened, df, x_suff):
    X_combined = []
    feature_names = []
    for feature in tags_flattened:
        x = df[feature + x_suff]
        X_combined.append(x) 
        feature_names.extend([feature + x_suff])
    X_combined_df = pd.DataFrame(np.column_stack(X_combined), columns=feature_names)
    return X_combined_df




# task 2 specific

def prepare_features_pt2(tags_flattened, df):
    X_combined = []
    feature_names = []
    for feature in tags_flattened:
        x_2011 = df[feature + "_count_2011"]
        x_2021 = df[feature + "_count_2021"]

        x_change = x_2021 - x_2011
        
        X_combined.append(x_2011)
        # X_combined.append(x_2021)

        X_combined.append(x_change)
        
        # feature_names.extend([feature + "_count_2011", feature + "_count_2021"])

        feature_names.extend([feature + "_count_2011", feature + "_2011_to_2021"])
    
    X_combined_df = pd.DataFrame(np.column_stack(X_combined), columns=feature_names)
    return X_combined_df

def merge_2011_and_2021(df1, df2):
    omit_cols = ['LAD24CD', 'LAD24NM', 'LAD24NMW', 'BNG_E', 'BNG_N', 'LONG', 'LAT',
       'geometry', 'area', 'average_price_2011', 'average_price_2021',
       'average_price_change', 'average_price_change_pct', 'ea_pct_2011',
       'ea_2011', 'ea_pct_2021', 'ea_2021', 'ea_change']
    rename_mapping_df1 = {col: f"{col}_2011" for col in df1.columns if col not in omit_cols}
    rename_mapping_df2 = {col: f"{col}_2021" for col in df2.columns if col not in omit_cols}
    df1_renamed = df1.rename(columns=rename_mapping_df1)
    df2_renamed = df2.rename(columns=rename_mapping_df2)
    df_combined = pd.merge(df1_renamed, df2_renamed, on=['LAD24CD','ea_change','average_price_change','ea_2011','ea_2021','average_price_2011','average_price_2021'], how='inner')
    return df_combined



def kfold_cross_validation(X, y, model_function, k_values):
    train_minus_test_mse = []

    for k in k_values:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        train_mse, test_mse = [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = model_function(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_mse.append(mean_squared_error(y_train, y_train_pred))
            test_mse.append(mean_squared_error(y_test, y_test_pred))

        avg_train_mse = np.mean(train_mse)
        avg_test_mse = np.mean(test_mse)
        train_minus_test_mse.append(avg_test_mse - avg_train_mse)

    return train_minus_test_mse

def plot_kfold_results(k_values, ols_results, lasso_results, ridge_results):
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, ols_results, label="OLS", marker="o")
    plt.plot(k_values, lasso_results, label="Lasso", marker="o")
    plt.plot(k_values, ridge_results, label="Ridge", marker="o")
    plt.xlabel("Number of Folds (k)")
    plt.ylabel("Test MSE - Train MSE")
    plt.title("Cross-Validation Results")
    plt.legend()
    plt.grid()
    plt.show()





# Practical 2 Exercise 1

def count_osm_features(locations_dict: dict, tags: dict) -> pd.DataFrame:
  class InsufficientResponseError(Exception):
    pass
  data = []
  for location, (latitude, longitude) in locations_dict.items():
    try:
      poi_counts = count_pois_near_coordinates(latitude, longitude, tags)
    except InsufficientResponseError:
      poi_counts = {tag: 0 for tag in tags}  # Fill with 0 for missing data
    poi_counts['location'] = location
    data.append(poi_counts)
  df_featurecount_location = pd.DataFrame(data)
  return df_featurecount_location

# Practical 2 Exercise 2

def plot_elbow_curve(df, max_k=10):
    """
    Plot the elbow curve to determine the optimal number of clusters.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for clustering.
        max_k (int): Maximum number of clusters to test for the elbow method. Default is 10.
    """
    # Step 1: Prepare the data by removing the location column for clustering
    X = df.drop(columns=['location'])

    # Step 2: Standardise the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Calculate SSE for each number of clusters
    sse = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k), sse, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of squared distances")
    plt.title("Elbow Method for Optimal k")
    plt.show()

def perform_kmeans_clustering(df, optimal_k):
    """
    Perform K-means clustering on a DataFrame with a specified number of clusters.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for clustering.
        optimal_k (int): The number of clusters to use in K-means clustering.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'cluster' column.
    """
    # Prepare the data by removing the location column for clustering
    X = df.drop(columns=['location'])

    # Standardise the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Return the DataFrame with clusters
    return df[['location', 'cluster'] + list(X.columns)]

# Practical 2 Exercise 4

def plot_feature_distance_heatmap(df, feature_columns, metric='euclidean'):
    """
    Plot a heatmap of the feature-based distance matrix for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for analysis.
        feature_columns (list): List of columns to use for distance calculations.
        location_column (str): Column to use as labels in the distance matrix. Default is 'location'.
        metric (str): Distance metric to use (e.g., 'euclidean'). Default is 'euclidean'.

    Returns:
        pd.DataFrame: A DataFrame representing the distance matrix with location names as index and columns.
    """

    # Step 1: Extract the features and normalize them
    features = df[feature_columns]
    scaler = StandardScaler()
    features_normalised = scaler.fit_transform(features)

    # Step 2: Calculate the pairwise distance matrix
    distance_matrix = pdist(features_normalised, metric=metric)
    distance_matrix_square = squareform(distance_matrix)

    # Step 3: Convert to DataFrame for easier interpretation
    distance_df = pd.DataFrame(distance_matrix_square, index=df['location'], columns=df['location'])

    # Step 4: Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_df, cmap="viridis", annot=True, fmt=".2f", cbar=True)
    plt.title("Feature-Based Distance Matrix of Locations")
    plt.show()

    return distance_df

# Practical 2 Exercise 5

def plot_feature_correlation_heatmap(df, feature_columns):
    """
    Plot a heatmap of the correlation matrix for specified features in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        feature_columns (list): List of columns to include in the correlation matrix.
        cmap (str): Color map for the heatmap. Default is 'coolwarm'.

    Returns:
        pd.DataFrame: The correlation matrix DataFrame.
    """

    # Calculate the correlation matrix for the specified features
    correlation_matrix = df[feature_columns].corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.show()

# Practical 2 Exercise 8

def plot_buildings(latitude, longitude, tags, distance_km, place_name):
  box_width = distance_km / 2.2 * 0.02  # Adjust based on approximation for 1km x 1km area
  box_height = distance_km / 2.2 * 0.02
  north = latitude + box_height / 2
  south = latitude - box_width / 2
  west = longitude - box_width / 2
  east = longitude + box_width / 2
  graph = ox.graph_from_bbox(north, south, east, west)
  pois = ox.geometries_from_bbox(north, south, east, west, tags)
  pois_with_address = pois[(pois['addr:housenumber'].notnull()) & (pois['addr:street'].notnull())]
  pois_without_address = pois[(pois['addr:housenumber'].isnull()) | (pois['addr:street'].isnull())]

# Retrieve nodes and edges
  nodes, edges = ox.graph_to_gdfs(graph)

# Get place boundary related to the place name as a geodataframe
  area = ox.geocode_to_gdf(place_name)

  fig, ax = plt.subplots()

  # Plot the footprint
  area.plot(ax=ax, facecolor="white")

    # Plot street edges
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  # Plot all POIs 
  pois_with_address.plot(ax=ax, color="blue", alpha=0.7, markersize=10,label="with address")
  pois_without_address.plot(ax=ax, color="purple", alpha=0.7, markersize=10,label="without address")
  plt.tight_layout()


# Practical 2 Exercise 9

# Define a function to calculate similarity score
def calculate_similarity(row):
    housenumber_similarity = fuzz.token_sort_ratio(str(row['addr:housenumber']).lower(), str(row['primary_addressable_object_name']).lower())
    street_similarity = fuzz.token_sort_ratio(str(row['addr:street']).lower(), str(row['street']).lower())
    # Combine the similarities (e.g., average, or other weighted approach)
    return (housenumber_similarity + street_similarity) / 2

def find_top_similar_addresses(pois_df, df_q7, threshold=70):
    # Perform a cross join
    df_cross = pois_df.assign(key=1).merge(df_q7.assign(key=1), on='key', suffixes=('_pois', '_q7')).drop('key', axis=1)
    
    # Calculate similarity and filter by threshold
    df_cross['similarity_score'] = df_cross.apply(calculate_similarity, axis=1)
    df_cross = df_cross[df_cross['similarity_score'] >= threshold]
    
    # Sort by similarity score in descending order
    df_cross = df_cross.sort_values(by='similarity_score', ascending=False)
    
    # Drop duplicates from pois_df side to ensure each entry in pois_df has at most one match
    df_similar = df_cross.drop_duplicates(subset=['addr:housenumber', 'addr:street'])
    
    # Select relevant columns
    df_similar = df_similar[['addr:housenumber', 'addr:street', 'primary_addressable_object_name', 
                             'secondary_addressable_object_name', 'street', 'price', 'geometry_area', 'similarity_score']]
    
    return df_similar

# Practical 2 Exercise 10

def get_correlation_price_area(df):
    correlation = df['price'].corr(df['geometry_area'])
    return correlation

def plot_price_vs_area(df):
  plt.figure(figsize=(8, 6))
  sns.regplot(x='geometry_area', y='price', data=df, scatter_kws={'s': 10}, line_kws={'color': 'red'})
  plt.title('Price vs Area of Property')
  plt.xlabel('Geometry Area (m²)')
  plt.ylabel('Price')
  plt.show()

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def plot_price_vs_area_outliers_removed(df):
    # Remove outliers from both 'geometry_area' and 'price'
    df_no_outliers = remove_outliers_iqr(df, 'geometry_area')
    df_no_outliers = remove_outliers_iqr(df_no_outliers, 'price')

    # Plot without outliers
    plt.figure(figsize=(8, 6))
    sns.regplot(x='geometry_area', y='price', data=df_no_outliers, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    plt.title('Price vs Area of Property (Outliers Removed)')
    plt.xlabel('Geometry Area (m²)')
    plt.ylabel('Price')
    plt.show()

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError



# Practical 3 Exercise 1

def rbf_kernel(x, centres, length_scale=10.0):
    """Generate RBF basis functions."""
    return np.exp(-((x[:, None] - centres[None, :]) ** 2) / (2 * length_scale ** 2))

# Practical 3 Exercise 4



def augment_df_with_norm_age(norm_age_df, ts062_df):
  ts062_df = ts062_df.merge(norm_age_df[21], left_on='geography', right_index=True, how='left')
  for i in range(0, 100):
    if (i != 21 and (i not in ts062_df.columns)):
      ts062_df = ts062_df.merge(norm_age_df[i], left_on='geography', right_index=True, how='left')
  colname_l15 = ts062_df.columns[12]
  print(colname_l15)
  for i in range(4, 13):
    ts062_df['col' + str(i-3) + '_proportion'] = ts062_df[ts062_df.columns[i]] / ts062_df[ts062_df.columns[3]]
  return ts062_df



