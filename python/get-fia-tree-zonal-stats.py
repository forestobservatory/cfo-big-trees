"""Computes the mean, variance, skewness and kurtosis of tree height and canopy cover for a 1 ha buffer around fuzzed FIA plot centers"""

import logging
import os
import sys

import cfo
import geopandas as gpd
import numpy as np
import rasterio as rio
from scipy import stats
from tqdm import tqdm

# set the file paths
this_dir, this_script = os.path.split(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(this_dir), "data")
fia_path = os.path.join(data_dir, "fia_sierra_plots.shp")
output_shp = os.path.join(data_dir, "fia-with-cfo-stats.shp")
output_csv = f"{output_shp[:-4]}.csv"

# cloud data paths
ch_path = "gs://cfo-public/vegetation/California-Vegetation-CanopyHeight-2020-Spring-00003m.tif"
cc_path = "gs://cfo-public/vegetation/California-Vegetation-CanopyCover-2020-Spring-00003m.tif"
lf_path = "gs://cfo-public/vegetation/California-Vegetation-LadderFuelDensity-2020-Summer-00010m.tif"
lc_path = "gs://cfo-public/vegetation/California-Vegetation-CanopyLayerCount-2020-Summer-00010m.tif"

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s %(levelname)s %(name)s [%(funcName)s] | %(message)s"),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info(f"Running {this_script}")

# authenticate with the cfo api for access to the cloud data
forest = cfo.api()
forest.authenticate()

# read the vector and set the buffer radius
buffer_size = 500
fia = gpd.read_file(fia_path)
buffer = fia.buffer(buffer_size).to_frame("geometry")
# logging setup
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s %(levelname)s %(name)s [%(funcName)s] | %(message)s"),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# create a function to get stats from each buffer
def buffer_stats(row, src):

    # read the data based on the bounds of the buffer
    xmin, ymin, xmax, ymax = row.geometry.bounds
    window = rio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
    data = src.read(1, window=window, masked=True)

    # compute the stats
    mn = data.mean()
    va = data.std() ** 2
    sk = stats.skew(data, axis=None, nan_policy="omit")
    ku = stats.kurtosis(data, axis=None, nan_policy="omit")
    mx = data.max()
    p5 = np.percentile(data, 5)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)
    p95 = np.percentile(data, 95)

    return mn, va, sk, ku, mx, p5, p25, p75, p95


# loop through all the inputs
paths = [ch_path, cc_path, lf_path, lc_path]
labels = ["CH", "CC", "LF", "LC"]
stat_names = ["MN", "VA", "SK", "KU", "MX", "05", "25", "75", "95"]
dfs = [fia]

logger.info("Extracting data from each covariate")

# run the stats on each dataset
for path, label in tqdm(zip(paths, labels), total=len(paths), desc="Covariate"):
    with rio.open(path, "r") as src:
        df = buffer.apply(buffer_stats, axis=1, result_type="expand", src=src)
        df.columns = [f"{label}_{stat}" for stat in stat_names]
        dfs.append(df)

# merge these into one data frame
output = gpd.pd.concat(dfs, axis=1, ignore_index=True)
columns = []
for df in dfs:
    columns += df.columns.to_list()
output.columns = columns

# save the results as a shape and a csv
logger.info(f"Saving to SHP: {output_shp}")
output.to_file(output_shp, driver="ESRI Shapefile")

logger.info(f"Saving to CSV: {output_csv}")
output.drop(["geometry"], axis=1).to_csv(output_csv, index=False)
