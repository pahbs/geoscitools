import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
import pandas as pd

import glob
import os
import time

def atl08_io(ATL08_CSV_OUTPUT_DIR, YEAR_SEARCH, DO_PICKLE=True, LENGTH_SEG=100, INTERSECT=None, BUF_DIST=0):
    
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
        if INTERSECT is None:
            atl08_gdf = pd.concat((pd.read_csv(f) for f in all_atl08_csvs ), sort=False, ignore_index=True) # <--generator is (), list is []
        else:
            if isinstance(INTERSECT, pd.DataFrame):
                pass
            else:
                INTERSECT = gpd.read_file(INTERSECT)
            
            print(f"Concatenating multiple gdfs for {YEAR_SEARCH} @ {LENGTH_SEG}m...")
            # Intersect each file with INTERSECT gdf and concat    
            atl08_gdf = pd.concat( ( gpd.overlay(get_seg_df_atl08(pd.read_csv(f), LENGTH_SEG), INTERSECT.to_crs(4326), how='intersection') for f in all_atl08_csvs ), sort=False, ignore_index=True)           
        
        
        if 'geometry' not in atl08_gdf.columns:
            print(f"Creating a gdf for {YEAR_SEARCH} @ {LENGTH_SEG}m...")
            atl08_gdf = GeoDataFrame(atl08_gdf, geometry=gpd.points_from_xy(atl08_gdf.lon, atl08_gdf.lat), crs='epsg:4326')#.sample(frac=SAMP_FRAC)

        if DO_PICKLE:
            # Pickle the file
            if YEAR_SEARCH == "**":
                YEAR_SEARCH = 'allyears'
            cur_time = time.strftime("%Y%m%d%H%M")
            out_pickle_fn = os.path.join(DIR_PICKLE, "atl08_"+YEAR_SEARCH+"_filt_gdf_"+cur_time+".pkl")
            atl08_gdf.to_pickle(out_pickle_fn)

    return(atl08_gdf)

def get_seg_df_atl08(atl08_df, LENGTH_SEG):
    
    if LENGTH_SEG == 20:
        
        atl08_df.rename(columns={'lon': 'lon_100m', 'lat': 'lat_100m'}, inplace=True)
        atl08_df.rename(columns={'lon_20m': 'lon', 'lat_20m': 'lat'}, inplace=True)
        atl08_df = atl08_df[atl08_df.h_can_20m < 3.402823 * 1e38]
        
    return(GeoDataFrame(atl08_df, geometry=gpd.points_from_xy(atl08_df.lon, atl08_df.lat), crs='epsg:4326'))