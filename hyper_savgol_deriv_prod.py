#!/usr/bin/env python
# coding: utf-8

# ### hyper_savgol_deriv.py - (based on meeting 7 July 2023) -FINAL VERSION-
# ##### Daniel J. Donahoe - 7 July 2023
# ##### Cleaned up - MacKinnon - 2 Oct 2023

# * Open 1km^2 MLBS hyperspectral images,
# * Perform per-pixel Savitzky-Golay smoothing (with derivative),
# * Remove bad-bands (four regions total) from band-stack (cube),
# * And write-out new Savgol'd raster.

import os
import sys

import numpy as np
import rasterio as rio
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

# Create function for opening raster, creating Savgol ndarray,
#  and writing-out to drive. Window size of 7, polynomial order of 2,
#  1st derivative is what our SME suggeseted
def createSavgol(list_of_rasters,
                 savgol_window=7,
                 savgol_polyorder=2,
                 savgol_deriv=1,
                 write_tif=False):

    # if passed a single path convert to list
    # mackinnon TODO: maybe use pathlib instead or something
    if isinstance(list_of_rasters, str):
        list_of_rasters = [list_of_rasters]

    # list for holding savgoled arrays
    savgol_arrays = []

    # loop through list of hyperspectral images to open files, perform
    #  z-wise Savgol filtering, and delete bad-bands.
    for in_file in list_of_rasters:

        # Open and read raster
        with rio.open(in_file) as src:
            src_profile = src.profile
            src_crs = src.crs
            src_bounds = src.bounds
            src_dtype = src.dtypes[0]
            src_all_bands = src.read()

        # Perform Savgol smoothing
        savgol_filtered = savgol_filter(x=src_all_bands,
                                        axis=0,
                                        window_length=savgol_window,
                                        polyorder=savgol_polyorder,
                                        deriv=savgol_deriv,
                                        mode='nearest')

        # Convert both atmospheric absorption bands to
        #  np.nan AFTER Savgol filtering.
        #---
        # Visible: 400 - 670 nm
        # Red edge: 670 - 780 nm
        # Near infrared (NIR): 780 - 1320 nm
        # [-*-*atmospheric absorption region 1*-*-]
        # Shortwave infrared 1 (SWIR-1): 1460 - 1775 nm
        # [-*-*atmospheric absorption region 2*-*-]
        # Shortwave infrared 2 (SWIR-2):  1990 - 2455 nm
        #---
        savgol_filtered[:4] = np.nan      # trim beginning
        savgol_filtered[189:216] = np.nan # trim atmos. 1
        savgol_filtered[280:322] = np.nan # trim atmos. 2
        savgol_filtered[416:] = np.nan    # trim end

        savgol_arrays.append(savgol_filtered)

        # Write rasters out to drive if desired
        if write_tif:
            out_file = in_file.replace(".tif", "_savgolDeriv.tif")
            with rio.open(out_file,
                          mode = 'w',
                          drive = 'GTiff',
                          bounds = src_bounds,
                          height = src.shape[0],
                          width = src.shape[1],
                          count = savgol_filtered.shape[0],
                          dtype = src_dtype,
                          crs = src_crs,
                          profile = src_profile,
                          transform = src.transform,
                          compress = 'lzw') as new_dataset:
                new_dataset.write(savgol_filtered)

    return savgol_arrays


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not enough args")
    else:
        raster_files = [sys.argv[1]+x for x in os.listdir(sys.argv[1])
                        if x.endswith(".tif")]

    # Test the function on one tif
    print(f"running {raster_files[0]}")
    svout = createSavgol(raster_files[0], write_tif=True)
    plt.plot(svout[0][:,100,100])
    plt.show()
