"""Trains a rotation forest model to predict carbon-relevant forest structure patterns from climate and CFO data."""

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
this_dir, this_script = os.path.split(os.path.abspath(__file__))
data = os.path.join(this_dir, "..", "data")
plots = os.path.join(this_dir, "..", "docs", "img")
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
veg_tif = os.path.join(data, "veg.tif")

# climate data
c_labels = ["aet", "aprpck", "cwd", "ppt", "tmn", "tmx"]
clims = [os.path.join(data, f"{c_label}_sierra_sierra.tif") for c_label in c_labels]
clim_tif = os.path.join(data, "clim-utm.tif")
clim_vrt = os.path.join(data, "clim-utm.vrt")

# set some raster parameters
clim_epsg = 3310
utm_epsg = 32610
ndvalue = -9999

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s %(levelname)s %(name)s [%(funcName)s] | %(message)s"),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info(f"Running {__file__}")


##########
# data preprocessing

# replace climate data nan values with nodata, reproject to utm and stack the bands
if not os.path.exists(clim_tif):
    logger.info("Reprojecting and stacking climate data")
    output_data = []

    for clim_path in clims:
        with rio.open(clim_path, "r+") as src:
            src.crs = rio.crs.CRS.from_epsg(clim_epsg)
            with rio.vrt.WarpedVRT(src, crs=rio.crs.CRS.from_epsg(utm_epsg)) as vrt:
                output_profile = vrt.profile
                cdata = vrt.read(1)
                is_nan = ~np.isfinite(cdata)
                is_nd = cdata == vrt.nodata
                to_nd = is_nan | is_nd
                cdata[to_nd] = ndvalue
                output_data.append(cdata)

    # set the output raster data profile
    xsize, xoff, xmin, yoff, ysize, ymax, *other = output_profile["transform"]
    transform = rio.transform.Affine(int(xsize), xoff, xmin, yoff, int(ysize), ymax)
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        nodata=ndvalue,
        count=len(c_labels),
        blockxsize=512,
        blockysize=512,
        transform=transform,
    )

    # write each band to disk
    with rio.open(clim_tif, "w", **output_profile) as dst:
        for idx, cdata in enumerate(output_data):
            dst.write(cdata, idx + 1)

# create a VRT file to read the cloud tifs from
if not os.path.exists(veg_vrt):
    logger.info("Building Cloud VRT")

    # get the spatial extent from the climate data
    with rio.open(clim_tif, "r") as src:
        bounds = src.bounds

    vrt_options = gdal.BuildVRTOptions(separate=True, outputBounds=bounds)

    vrt = gdal.BuildVRT(veg_vrt, vegs, options=vrt_options)
    vrt.FlushCache()

# and write these data to a local tif file for faster reading
if not os.path.exists(veg_tif):
    logger.info("Building Veg. Raster")

    creation_options = [
        "COMPRESS=DEFLATE",
        "TILED=YES",
        "NUM_THREADS=ALL_CPUS",
        "BIGTIFF=YES",
    ]
    translate_options = gdal.TranslateOptions(
        creationOptions=creation_options,
    )

    translate = gdal.Translate(veg_tif, veg_vrt, options=translate_options)
    translate.FlushCache()

##########
# model data setup

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
ymax = np.percentile(y, 95)


##########
# model training

model_path = os.path.join(data, f"{yvar}.pck")
if os.path.exists(model_path):
    with open(model_path, "rb") as inf:
        model = pickle.loads(inf.read())

else:
    logger.info("Training rotation forest model")

    # set weights
    hist, edges = np.histogram(y, range=(0, ymax))
    edges[-1] = y.max()

    weights = np.zeros_like(ytrain)
    bin_weights = 1 / (hist / hist.sum())
    for i in range(len(hist)):
        in_range = (edges[i] < ytrain) & (ytrain <= edges[i + 1])
        weights[in_range] = bin_weights[i]

    model = GradientBoostingRegressor(max_depth=10, validation_fraction=0.2)
    model.fit(xtrain, ytrain, sample_weight=weights)

    # save it
    with open(model_path, "wb") as out:
        pickle.dump(model, out)

# run performance the numbers
ypred = model.predict(xtest)
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

# set the output file
outpath = os.path.join(data, f"{yvar}.tif")


with rio.open(veg_tif, "r") as vsrc, rio.open(clim_tif, "r") as csrc:
    with rio.open(outpath, "w", **profile) as out:
        for xval, yval in tqdm(zip(xvals, yvals), total=len(xvals), desc="Tile"):

            # set the output array
            env = np.zeros((1, x.shape[1]), dtype=np.float32)

            # get spatial references
            xmin, ymin = csrc.xy(yval, xval, "ll")
            xmax, ymax = (xmin + yres, ymin + yres)
            vwindow = rio.windows.from_bounds(xmin, ymin, xmax, ymax, vtransform)
            cwindow = rio.windows.Window(xval, yval, 1, 1)

            # read the climate data first and move on if nodata
            clim = csrc.read(window=cwindow, masked=True)
            veg = vsrc.read(masked=True, window=vwindow)

            if np.all(veg.mask):
                continue

            else:
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

                # transform and apply the model
                xpc = transformer.transform(env)
                ypred = model.predict(xpc)

                # write to disk
                out.write(ypred.reshape(-1, 1), 1, window=cwindow)
