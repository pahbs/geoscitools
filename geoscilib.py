import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

from rasterio.coords import BoundingBox
from rasterio.coords import disjoint_bounds

import multiprocessing as mp

from scipy.spatial import cKDTree
from shapely.geometry import Point

gpd1 = gpd.GeoDataFrame([['John', 1, Point(1, 1)], ['Smith', 1, Point(2, 2)],
                         ['Soap', 1, Point(0, 2)]],
                        columns=['Name', 'ID', 'geometry'])
gpd2 = gpd.GeoDataFrame([['Work', Point(0, 1.1)], ['Shops', Point(2.5, 2)],
                         ['Home', Point(1, 1.1)]],
                        columns=['Place', 'geometry'])

def ckdnearest(gdA, gdB):
    '''Find nearest point
    https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    '''

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

def query_db_catid_NEW(catID, prod_code='M1BS', out_dir='/att/nobackup/pmontesa', db_table='nga_footprint_master_V2'):
    '''Query and select scenes from latest database
    '''
    with psycopg2.connect(database="arcgis", user="pmontesa", password="baaaad", host="arcdb04", port="5432") as dbConnect:

        cur = dbConnect.cursor() # setup the cursor
        selquery =  "SELECT S_FILEPATH, SENSOR, CATALOG_ID, ACQ_TIME FROM %s WHERE CATALOG_ID = '%s' AND PROD_CODE = '%s'" %(db_table, catID, prod_code)
        #selquery =  "SELECT * FROM %s WHERE CATALOG_ID = '%s' AND PROD_CODE = '%s'" %(db_table, catID, prod_code)
        cur.execute(selquery)
        selected=cur.fetchall()

    return selected

def reproject_raster(in_path, out_path, to_crs = CRS.from_string('EPSG:4326')):

    """https://stackoverflow.com/questions/60288953/how-to-change-the-crs-of-a-raster-with-rasterio
    """
    # reproject raster to project crs
    with rasterio.open(in_path) as src:
        src_crs = src.crs
        transform, width, height = calculate_default_transform(src_crs, to_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()

        kwargs.update({
            'crs': to_crs,
            'transform': transform,
            'width': width,
            'height': height})

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=to_crs,
                    resampling=Resampling.nearest)
    return(out_path)

def getSample(raster, pts, data):
    src = rasterio.open(raster)
    sample = src.sample(pts)
    arr = np.fromiter(sample, dtype=np.uint8)
    src.close()
    for ii in range(len(arr)):
        data[ii] = min(arr[ii], data[ii])

def extract_value_multi_thread(r_path: str, pt_gdf, sample_name: str, max_valid_value = 200, RESULT_DATA_TYPE = np.int16, NODATA_VAL = -9999):
    # Check if variable points to a dir of rasters or a single raster
    if len(os.path.splitext(r_path)[-1]) == 0:
        # get a list of tiles
        print("Get list of rasters")
        t_list = glob.glob(os.path.join(r_path, "*.tif"))
        print(f'Total number of tiles {len(t_list)}')
    else:
        print('Single raster')
        t_list = [r_path]
        
    DO_REPROJ = True
    pt_gdf_prj_str = str(pt_gdf.crs).split(':')[-1]
    with rasterio.open(t_list[0]) as src:
        
        if pt_gdf_prj_str in str(src.crs):
            print(pt_gdf.crs)
            print(src.crs)
            print('no re-projection needed.')
            DO_REPROJ = False
            
    if DO_REPROJ:
        # project gdf to rasters's CRS
        print("Re-project points to match raster")
        tmp = rasterio.open(t_list[0])
        pt_gdf = pt_gdf.to_crs(tmp.crs)
        tmp.close()
            
    print('extract points coords')
    pt_coord = [(pt.x, pt.y) for pt in pt_gdf.geometry]
    
    print('get boundingbox from points')
    bnd = pt_gdf.total_bounds
    pt_bnd = BoundingBox(bnd[0], bnd[1], bnd[2], bnd[3])
    
    print('loop through tiles & update sample values')
    print("Sampling rasters ...")
    r = mp.Array('i', np.full(len(pt_coord), 500)) # <-- this background value should not be negative and not be within the valid data range
    jobs = []
    
    ### TO RUN A SHORT TEST
    ### for fn in t_list[:100]:
    for fn in t_list:
        
        ds = rasterio.open(fn)
        r_bnd = ds.bounds
        ds.close()
        if not disjoint_bounds(r_bnd, pt_bnd):
            j = mp.Process(target=getSample, args=(fn, pt_coord, r))
            jobs.append(j)
            j.start()
        #else:
        #    print(f"PASS   {os.path.basename(fn)}")
            
    for j in jobs:
        j.join()
        
    ## merge sample back to gdf
    print("Add new column")
    rslt = np.array(r[:]).astype(RESULT_DATA_TYPE)
    rslt = np.where(rslt <= max_valid_value, rslt, NODATA_VAL)
    pt_gdf[sample_name] = pd.Categorical(rslt)
    
    print("Complete")
    
    return pt_gdf