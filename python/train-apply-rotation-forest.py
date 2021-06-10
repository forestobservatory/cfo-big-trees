"""Trains a rotation forest model to predict carbon-relevant forest structure patterns from climate and CFO data"""

import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from osgeo import gdal
from scipy import stats
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

gdal.UseExceptions()


##########
# path / config setup

# raw paths
data = os.path.join("..", "data")
plots = os.path.join("..", "docs", "img")
training = os.path.join(data, "fia_calif_plot_level_climate_model.csv")

# veg data
v_metrics = ["CC", "CH", "LF", "LC"]
moments = ["MN", "VA", "SK", "KU"]
v_labels = []
for metric in v_metrics:
    for moment in moments:
        v_labels.append(f"{metric}_{moment}")

vegs = [
    "/vsigs/cfo-public/vegetation/California-Vegetation-CanopyCover-2020-Summer-00010m.tif",
    "/vsigs/cfo-public/vegetation/California-Vegetation-CanopyHeight-2020-Summer-00010m.tif",
    "/vsigs/cfo-public/vegetation/California-Vegetation-CanopyLayerCount-2020-Summer-00010m.tif",
    "/vsigs/cfo-public/vegetation/California-Vegetation-LadderFuelDensity-2020-Summer-00010m.tif",
]
veg_vrt = os.path.join(data, "veg.vrt")

# climate data
c_labels = ["aet", "aprpck", "cwd", "ppt", "tmn", "tmx"]
clims = [os.path.join(f"{c_label}_sierra_sierra.tif") for c_label in c_labels]
clim_tif = os.path.join(data, "clim-utm.tif")

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s %(levelname)s %(name)s [%(funcName)s] | %(message)s"),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info(f"Running {__file__}")


##########
# data setup

# create a VRT file to read the cloud tifs from
if not os.path.exists(veg_vrt):
    logger.info("Building Cloud VRT")
    vrt_options = gdal.BuildVRTOptions(
        separate=True,
    )

    vrt = gdal.BuildVRT(veg_vrt, vegs, options=vrt_options)
    vrt.FlushCache()

# read the training data
logger.info("Reading training data")
df = pd.read_csv(training)

# prep the data
xvars = v_labels + c_labels
yvar = "BAPA"
x = df[xvars]
y = df[yvar]

# PCA training
transformer = PCA(whiten=True)
xt = transformer.fit_transform(x)

# train test split
xtrain, xtest, ytrain, ytest = train_test_split(xt, y, train_size=0.7)


##########
# model training

# set weights
ymax = np.percentile(y, 95)
hist, edges = np.histogram(y, range=(0, ymax))
edges[-1] = y.max()

weights = np.zeros_like(ytrain)
bin_weights = 1 / (hist / hist.sum())
for i in range(len(hist)):
    in_range = (edges[i] < ytrain) & (ytrain <= edges[i + 1])
    weights[in_range] = bin_weights[i]

model = GradientBoostingRegressor(max_depth=10, validation_fraction=0.2)

model.fit(xtrain, ytrain, sample_weight=weights)
ypred = model.predict(xtest)

# save the model
with open(os.path.join(data, f"{yvar}.pck"), "wb") as out:
    pickle.dump(model, out)

# run the numbers
rsq = metrics.r2_score(ytest, ypred)
mae = metrics.mean_absolute_error(ytest, ypred)

logging.info("Model performance:")
logging.info(f"r-squared: {rsq:0.2f}")
logging.info(f"mae: {mae:0.2f}")


##########
# plotting performance

# set figure parameters
plt.figure(figsize=(5, 4), dpi=125)

# hex plot
plt.hexbin(
    ytest,
    ypred,
    gridsize=50,
    mincnt=1,
    cmap=plt.cm.Spectral_r,
)

# 1:1 line
plt.plot(
    (0, ymax),
    (0, ymax),
    color="black",
    linestyle="--",
    label=f"$r^2$: {rsq:0.2f}\n$MAE$: {mae:0.2f}",
)

# styling
plt.box(False)
cbar = plt.colorbar()
plt.xlim(0, ymax)
plt.ylim(0, ymax)

# labels
plt.title(f"{yvar} - Rotation Forest Model")
plt.xlabel("observed")
plt.ylabel("predicted")
cbar.set_label("Number of samples")
plt.legend(fancybox=True, loc="upper left")
plt.tight_layout()

# saving
plot_path = os.path.join(plots, f"{yvar}-rotation-forest-model.png")
plt.savefig(plot_path, dpi=200)
plt.close()


##########
# model inference

# get the output dimensions and profile
with rio.open(clim_tif, "r") as src:
    profile = src.profile
    xres, yres = src.res
    mask = src.read_masks(1)
    yvals, xvals = np.where(mask == 255)

with rio.open(veg_vrt, "r") as src:
    vtransform = src.transform

profile.update(count=1)

# set the output array
env = np.zeros((1, x.shape[1]), dtype=np.float32)

# set the output file
outpath = os.path.join(data, f"{yvar}.tif")

with rio.open(veg_vrt, "r") as vsrc, rio.open(clim_tif, "r") as csrc, rio.open(
    outpath, "w", **profile
) as out:
    for xval, yval in tqdm(zip(xvals, yvals), total=len(xvals), desc="Tile"):

        # get spatial references
        xmin, ymin = csrc.xy(xval, yval, "ll")
        xmax, ymax = (xmin + yres, ymin + yres)
        vwindow = rio.windows.from_bounds(xmin, ymin, xmax, ymax, vtransform)
        cwindow = rio.windows.Window(xval, yval, 1, 1)

        # read the data first
        veg = vsrc.read(masked=True, window=vwindow)
        clim = csrc.read(window=cwindow)

        # assign values to the array
        env[0, 0] = veg[0].mean()
        env[0, 1] = veg[0].std() ** 2
        env[0, 2] = stats.skew(veg[0], axis=None, nan_policy="omit")
        env[0, 3] = stats.kurtosis(veg[0], axis=None, nan_policy="omit")
        env[0, 4] = veg[1].mean()
        env[0, 5] = veg[1].std() ** 2
        env[0, 6] = stats.skew(veg[1], axis=None, nan_policy="omit")
        env[0, 7] = stats.kurtosis(veg[1], axis=None, nan_policy="omit")
        env[0, 8] = veg[2].mean()
        env[0, 9] = veg[2].std() ** 2
        env[0, 10] = stats.skew(veg[2], axis=None, nan_policy="omit")
        env[0, 11] = stats.kurtosis(veg[2], axis=None, nan_policy="omit")
        env[0, 12] = veg[3].mean()
        env[0, 13] = veg[3].std() ** 2
        env[0, 14] = stats.skew(veg[3], axis=None, nan_policy="omit")
        env[0, 15] = stats.kurtosis(veg[3], axis=None, nan_policy="omit")
        env[0, 16:] = np.squeeze(clim)

        nans = ~np.isfinite(env)
        if nans.any():
            env[nans] = 0

        # transform, apply, write
        xpc = transformer.transform(env)
        ypred = model.predict(xpc).astype(np.float32)
        out.write(ypred.reshape(-1, 1), 1, window=cwindow)
