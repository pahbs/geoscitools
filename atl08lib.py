import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
import pandas as pd

import glob
import os
import time

def atl08_io(ATL08_CSV_OUTPUT_DIR, YEAR_SEARCH, DO_PICKLE=True, LENGTH_SEG=100):
    
    '''Read all ATL08 from CSVs of a given year after extract_filter_atl08.py
        Write to a pickle file by year
        Return a geodataframe
    '''

    DIR_PICKLE = ATL08_CSV_OUTPUT_DIR
    print("Building list of ATL08 csvs...")
    all_atl08_csvs = glob.glob(ATL08_CSV_OUTPUT_DIR + "/"+YEAR_SEARCH+"/ATL08*"+str(LENGTH_SEG)+"m.csv", recursive=True)
    #all_atl08_csvs = random.sample(all_atl08_csvs, 500)
    print(len(all_atl08_csvs))
    
    if len(all_atl08_csvs) == 0:
        print(f"No csvs for a gdf for {YEAR_SEARCH} @ {LENGTH_SEG}m.")
        atl08_gdf = None
    else:  
        # Merge all files in the list
        print("Creating pandas data frame...")
        atl08_gdf = pd.concat((pd.read_csv(f) for f in all_atl08_csvs ), sort=False, ignore_index=True) # <--generator is (), list is []
        atl08_gdf = GeoDataFrame(atl08_gdf, geometry=gpd.points_from_xy(atl08_gdf.lon, atl08_gdf.lat), crs='epsg:4326')#.sample(frac=SAMP_FRAC)

        if DO_PICKLE:
            # Pickle the file
            if YEAR_SEARCH == "**":
                YEAR_SEARCH = 'allyears'
            cur_time = time.strftime("%Y%m%d%H%M")
            out_pickle_fn = os.path.join(DIR_PICKLE, "atl08_"+YEAR_SEARCH+"_filt_gdf_"+cur_time+".pkl")
            atl08_gdf.to_pickle(out_pickle_fn)

    return(atl08_gdf)