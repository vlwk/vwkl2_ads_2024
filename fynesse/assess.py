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

