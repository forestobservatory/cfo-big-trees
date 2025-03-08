"""Compute three-class maps for wildfire hazard and for bioclimatic suitability change."""

import os
import rasterio as rio
import numpy as np


# directories
base = "/Users/christopher/src/cfo-big-trees/data"
plots = os.path.join(base, "plots")
models = os.path.join(base, "models")
raster = os.path.join(base, "raster")

# input paths
hazard = os.path.join(raster, "SierraEcoregion-Wildfire-ScaledHazard-2020-Summer-00030m.tif")
current_mn = os.path.join(models, "ecoregion_model_current_mean.tif")
current_sd = os.path.join(models, "ecoregion_model_current_stdv.tif")
rcp45_mn = os.path.join(models, "ecoregion_model_rcp45_mean.tif")
rcp45_sd = os.path.join(models, "ecoregion_model_rcp45_stdv.tif")
rcp85_mn = os.path.join(models, "ecoregion_model_rcp85_mean.tif")
rcp85_sd = os.path.join(models, "ecoregion_model_rcp85_stdv.tif")
mask = os.path.join(models, "cropped-current.tif")

# output paths
hazard_classes = os.path.join(raster, "SierraEcoregion-Wildfire-HazardClasses-2020-Summer-00030m.tif")
rcp45_change = os.path.join(raster, "SierraEcoregion-Suitability-ChangeRCP45-2020-Summer-02000m.tif")
rcp85_change = os.path.join(raster, "SierraEcoregion-Suitability-ChangeRCP85-2020-Summer-02000m.tif")
rcp45_bivariate = os.path.join(raster, "SierraEcoregion-Suitability-BivariateRCP45-2020-Summer-00030m.tif")
rcp85_bivariate = os.path.join(raster, "SierraEcoregion-Suitability-BivariateRCP85-2020-Summer-00030m.tif")

# nodata
nodata_uint8 = np.iinfo(np.uint8).max
nodata_int16 = np.iinfo(np.int16).max

# standard deviation threshold for estimating significant change
sd_threshold = 3

# hazard classes
# 0.0 - 0.33: low (1)
# 0.33 - 0.66: moderate (2)
# 0.66 - 1.0: high (3)
print('Computing hazard classes')
with rio.open(hazard, "r") as src:
    profile = src.profile.copy()
    profile.update(dtype="uint8", nodata=nodata_uint8)
    with rio.open(hazard_classes, "w", **profile) as dst:
        for _, window in src.block_windows():
            data = src.read(1, window=window, masked=True)
            data[data < 33] = 1
            data[(data >= 33) & (data < 66)] = 2
            data[data >= 66] = 3
            dst.write(data, 1, window=window)

# bioclimatic suitability change - RCP45
# -1: decreased suitability
# 0: no change
# 1: increased suitability
print('Computing bioclimatic suitability change for RCP45')
with rio.open(current_mn, "r") as csrc, rio.open(current_sd, "r") as cstd, \
        rio.open(rcp45_mn, "r") as rsrc, rio.open(rcp45_sd, "r") as rstd, \
        rio.open(mask, "r") as msrc:
    profile = csrc.profile.copy()
    profile.update(dtype="int16", nodata=nodata_int16)
    with rio.open(rcp45_change, "w", **profile) as dst:
        cmean = csrc.read(1)
        cstdv = cstd.read(1)
        rmean = rsrc.read(1)
        rstdv = rstd.read(1)
        mask = msrc.read(1, masked=True)
        data = np.zeros(cmean.shape, dtype=np.int16)

        # difference
        diff = rmean - cmean
        margin = sd_threshold * cstdv + sd_threshold * rstdv
        neg_change = diff < -margin
        pos_change = diff > margin
        data[neg_change] = -1
        data[pos_change] = 1
        data[mask.mask] = nodata_int16
        dst.write(data, 1)

# bioclimatic suitability change - RCP85
print('Computing bioclimatic suitability change for RCP85')
with rio.open(current_mn, "r") as csrc, rio.open(current_sd, "r") as cstd, \
        rio.open(rcp85_mn, "r") as rsrc, rio.open(rcp85_sd, "r") as rstd, \
        rio.open(mask, "r") as msrc:
    profile = csrc.profile.copy()
    profile.update(dtype="int16", nodata=nodata_int16)
    with rio.open(rcp85_change, "w", **profile) as dst:
        cmean = csrc.read(1)
        cstdv = cstd.read(1)
        rmean = rsrc.read(1)
        rstdv = rstd.read(1)
        mask = msrc.read(1, masked=True)
        data = np.zeros(cmean.shape, dtype=np.int16)

        # difference
        diff = rmean - cmean
        margin = sd_threshold * cstdv + sd_threshold * rstdv
        neg_change = diff < -margin
        pos_change = diff > margin
        data[neg_change] = -1
        data[pos_change] = 1
        data[mask.mask] = nodata_int16
        dst.write(data, 1)

# bioclimatic suitability bivariate - RCP45
#             | -1|   | 1 | 2 | 3 |
# suitability | 0 |   | 4 | 5 | 6 |
#             | 1 |   | 7 | 8 | 9 |
#
#                     | 3 | 2 | 1 |
#                        hazard
print('Computing bioclimatic suitability bivariate for RCP45')
with rio.open(rcp45_change, "r") as csrc, rio.open(hazard_classes, "r") as hsrc:
    profile = csrc.profile.copy()

    # reproject the high res hazard data to match the 2km suitability data
    with rio.vrt.WarpedVRT(hsrc, resampling=rio.enums.Resampling.mode, **profile) as vrt:

        # set the output file
        profile.update(dtype="uint8", nodata=nodata_uint8)
        with rio.open(rcp45_bivariate, "w", **profile) as dst:

            for _, window in csrc.block_windows():
                cdata = csrc.read(1, window=window, masked=True)
                hdata = vrt.read(1, window=window, masked=True)
                data = np.zeros(cdata.shape, dtype=np.uint8) + nodata_uint8
                data[(cdata == -1) & (hdata == 3)] = 1
                data[(cdata == -1) & (hdata == 2)] = 2
                data[(cdata == -1) & (hdata == 1)] = 3
                data[(cdata == 0) & (hdata == 3)] = 4
                data[(cdata == 0) & (hdata == 2)] = 5
                data[(cdata == 0) & (hdata == 1)] = 6
                data[(cdata == 1) & (hdata == 3)] = 7
                data[(cdata == 1) & (hdata == 2)] = 8
                data[(cdata == 1) & (hdata == 1)] = 9
                dst.write(data, 1, window=window)

# bioclimatic suitability bivariate - RCP85
print('Computing bioclimatic suitability bivariate for RCP85')
with rio.open(rcp85_change, "r") as csrc, rio.open(hazard_classes, "r") as hsrc:
    profile = csrc.profile.copy()

    # reproject the high res hazard data to match the 2km suitability data
    with rio.vrt.WarpedVRT(hsrc, resampling=rio.enums.Resampling.mode, **profile) as vrt:

        # set the output file
        profile.update(dtype="uint8", nodata=nodata_uint8)
        with rio.open(rcp85_bivariate, "w", **profile) as dst:

            for _, window in csrc.block_windows():
                cdata = csrc.read(1, window=window, masked=True)
                hdata = vrt.read(1, window=window, masked=True)
                data = np.zeros(cdata.shape, dtype=np.uint8) + nodata_uint8
                data[(cdata == -1) & (hdata == 3)] = 1
                data[(cdata == -1) & (hdata == 2)] = 2
                data[(cdata == -1) & (hdata == 1)] = 3
                data[(cdata == 0) & (hdata == 3)] = 4
                data[(cdata == 0) & (hdata == 2)] = 5
                data[(cdata == 0) & (hdata == 1)] = 6
                data[(cdata == 1) & (hdata == 3)] = 7
                data[(cdata == 1) & (hdata == 2)] = 8
                data[(cdata == 1) & (hdata == 1)] = 9
                dst.write(data, 1, window=window)