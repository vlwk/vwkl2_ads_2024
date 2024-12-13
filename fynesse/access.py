from .config import *
import requests
import pymysql
import time
import csv
import pandas as pd
import osmnx as ox

import zipfile
import io
import os
"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

# final project Task 1



def is_valid_wkt(wkt):
    try:
        loads(wkt)
        return True
    except WKTReadingError:
        return False


class NodeCollectorHandler(osm.SimpleHandler):
    """First pass: Collect all node locations."""
    def __init__(self):
        super().__init__()
        # self.all_node_locations = {}  # Dictionary to store all nodes with their coordinates
        self.tagged_nodes = []  # To store nodes with at least one tag
        self.cnt = 0

    def node(self, n):
        # Store all nodes for reference
        # self.all_node_locations[n.id] = (n.location.lat, n.location.lon)

        # Store tagged nodes
        if n.tags:
            self.tagged_nodes.append({
                'id': n.id,
                'lat': n.location.lat,
                'lon': n.location.lon,
                'tags': dict(n.tags)
            })
            self.cnt += 1
            if (self.cnt % 1000 == 0):
                print(str(self.cnt) + " nodes added")

def process_osm_file_with_one_pass(osm_file, nodes_csv):
    """
    Process an OSM PBF file using a two-pass approach:
    1. First pass: Collect all node locations and tagged nodes.
    2. Second pass: Process ways and calculate average coordinates.

    Parameters:
        osm_file (str): Path to the OSM PBF file.
        nodes_csv (str): Path to save the tagged nodes as a CSV file.
        ways_csv (str): Path to save the ways with average coordinates as a CSV file.
    """
    print(f"Processing file: {osm_file}")

    # First pass: Collect all nodes
    print("First pass: Collecting nodes...")
    node_handler = NodeCollectorHandler()
    node_handler.apply_file(osm_file)

    # Save tagged nodes to CSV
    tagged_nodes_df = pd.DataFrame(node_handler.tagged_nodes)
    tagged_nodes_df.to_csv(nodes_csv, index=False)
    print(f"Tagged Nodes saved to {nodes_csv}")










def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

def hello_world():
  print("Hello from the data science library!")

def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  print('Data stored for year: ' + str(year))
  conn.commit()

# Practical 2 Exercise 7

def select_houses_from_database(conn, latitude, longitude, distance_km = 1.0, date_from='2020-01-01'):
    """
    Retrieve house data from the database within a specified radius around a latitude and longitude.

    Args:
        conn: Database connection object created using fynesse library.
        latitude (float): Central latitude for the search area.
        longitude (float): Central longitude for the search area.
        distance_km (float): Radius in km around the latitude and longitude to search.
        date (str): Minimum date of transfer to filter the records. Default is '2020-01-01'.

    Returns:
        pd.DataFrame: DataFrame containing house data within the specified area and date range.
    """

    box_width = distance_km / 2.2 * 0.02 # 2.2 km = 0.02 box units
    box_height = distance_km / 2.2 * 0.02

    # Create a cursor object
    cur = conn.cursor()

    # Execute the query
    cur.execute(
        f"""
        SELECT * FROM prices_coordinates_data_2 
        WHERE latitude BETWEEN {latitude - box_width / 2} AND {latitude + box_width / 2}
        AND longitude BETWEEN {longitude - box_height / 2} AND {longitude + box_height / 2}
        AND date_of_transfer >= '{date_from}';
        """
    )

    # Fetch all results and column names
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Close the cursor
    cur.close()

    return df

# Practical 2 Exercise 8

def return_pois_near_coordinates_full_addr(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> pd.DataFrame:
    """
    Retrieve Points of Interest (POIs) near a given pair of coordinates within a specified distance.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'building': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.

    Returns:
        pd.DataFrame: A DataFrame of POIs with relevant attributes for each tag.
    """

    # Define the bounding box
    box_width = distance_km / 2.2 * 0.02  # Adjust based on approximation for 1km x 1km area
    box_height = distance_km / 2.2 * 0.02
    north = latitude + box_height / 2
    south = latitude - box_width / 2
    west = longitude - box_width / 2
    east = longitude + box_width / 2

    # Fetch the POIs within the bounding box
    pois = ox.geometries_from_bbox(north, south, east, west, tags)

    # Convert to DataFrame
    pois_df = pd.DataFrame(pois)

    # Filter the DataFrame for relevant columns (if available)
    if 'addr:housenumber' in pois_df.columns and 'addr:street' in pois_df.columns:
        pois_df = pois_df.dropna(subset=['addr:housenumber', 'addr:street'])
        pois_df = pois_df[['addr:housenumber', 'addr:street', 'addr:postcode', 'geometry']]
    else:
        pois_df = pois_df.reset_index(drop=True)

    pois_df['geometry_area'] = pois_df['geometry'].apply(lambda geom: geom.area)

    return pois_df



# Practical 3 Exercise 1

def download_census_data(code, base_dir=''):
  url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
  extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(extract_dir)

  print(f"Files extracted to: {extract_dir}")

def load_census_data(code, level='msoa'):
  return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')



