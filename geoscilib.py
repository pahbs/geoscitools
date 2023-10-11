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

import datetime
import json
import warnings

import certifi
import urllib3
from urllib.parse import urlencode

from pprint import pprint

COLLECTIONCONCEPTID_DICT = {
                        'ATL08.003': "C2003772626-NSIDC_ECS",
                        'ATL08.005': "C2144424132-NSIDC_ECS",
                        'ATL08.006': "C2565090645-NSIDC_ECS",
                        'GLIHT': "C2013348111-LPDAAC_ECS",
                        # our ICESat-2 derived AGBD map c2020
                        'BOREALAGB2020': "C2756302505-ORNL_CLOUD" 
}

gpd1 = gpd.GeoDataFrame([['John', 1, Point(1, 1)], ['Smith', 1, Point(2, 2)],
                         ['Soap', 1, Point(0, 2)]],
                        columns=['Name', 'ID', 'geometry'])
gpd2 = gpd.GeoDataFrame([['Work', Point(0, 1.1)], ['Shops', Point(2.5, 2)],
                         ['Home', Point(1, 1.1)]],
                        columns=['Place', 'geometry'])

def make_points_grid(extent_df, spacing_m, grid_crs=3857):
    
    x_spacing = spacing_m #The point spacing you want
    y_spacing = spacing_m

    xmin, ymin, xmax, ymax = extent_df.to_crs(grid_crs).total_bounds #Find the bounds of all polygons in the df
    xcoords = [c for c in np.arange(xmin, xmax, x_spacing)] #Create x coordinates
    ycoords = [c for c in np.arange(ymin, ymax, y_spacing)] #And y

    coordinate_pairs = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1, 2) #Create all combinations of xy coordinates
    geometries = gpd.points_from_xy(coordinate_pairs[:,0], coordinate_pairs[:,1]) #Create a list of shapely points

    pointdf = gpd.GeoDataFrame(geometry=geometries, crs=grid_crs).to_crs(4326) #Create the point df
    
    return pointdf

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

def extract_zonal_gdf_point(r_fn, GDF, bandnames: list, reproject=True, TEST=False):
    
    from rasterstats import zonal_stats
    
    '''
    Extract pixel values of a raster to point in a Point geodataframe
   
    '''
    
    with rasterio.open(r_fn) as r_src:
        print("\tExtracting raster values from: ", r_fn)

        if reproject:
            
            print("\tRe-project geodataframe to match raster...")
            GDF = GDF.to_crs(r_src.crs)
            
        for i, bandname in enumerate(bandnames):
            bnum = i + 1
            
            r = r_src.read(bnum, masked=True)

            pt_coord = [(pt.x, pt.y) for pt in GDF.geometry]

            # Use 'sample' from rasterio
            
            pt_sample = r_src.sample(pt_coord, bnum)

            pt_sample_eval = np.fromiter(pt_sample, dtype=r.dtype)

            # Deal with no data...
            pt_sample_eval_ma = np.ma.masked_equal(pt_sample_eval, r_src.nodata)
            GDF[bandname] = pt_sample_eval_ma.astype(float).filled(np.nan)

            # Rename cols
            #GDF = rename_columns(GDF, bandname, stats_list = None)
            GDF = GDF.rename(columns = {bandname: 'val_'+bandname})
            
    return GDF

def extract_zonal_gdf_poly(r_fn, GDF, bandnames: list, reproject=True, stats_list = ['max','min','median','mean','percentile_98','count']):
    
    from rasterstats import zonal_stats
    
    with rasterio.open(r_fn) as r_src:
        print("\tExtracting raster values from: ", r_fn)

        if reproject:
            
            print("\tRe-project geodataframe to match raster...")
            GDF = GDF.to_crs(r_src.crs)

        for i, bandname in enumerate(bandnames):
            bnum = i + 1
            GDF = GDF.join(
                pd.DataFrame(
                    zonal_stats(
                        vectors=GDF.to_crs(r_src.crs), 
                        raster= r_src.read(bnum, masked=True),
                        affine= r_src.transform,
                        stats=stats_list
                    )
                ),
                how='left'
            )
        
            # Rename cols
            GDF = rename_columns(GDF, bandname, stats_list)
            
    return GDF

def rename_columns(GDF, bandname, stats_list):
    if stats_list is not None:
        names_list = ['val_'+ bandname + '_' + s for s in stats_list]
        rename_dict = dict(zip(stats_list, names_list))      
        GDF = GDF.rename(columns = rename_dict)
        
    return GDF

def extract_zonal_gdf(r_fn, GDF, bandnames: list, reproject=True, return_src_path=False, OUTDIR=None, OUTNAME=None):
    
    """Extract raster band values to the obs of a geodataframe 
    - modified version from ExtractUtils - captures raster name in output pdf (return_src_path)
    """

    if 'Polygon' in GDF.geometry.iloc[0].geom_type:
        feature_type = 'polygon'
        print(feature_type)
        GDF = extract_zonal_gdf_poly(r_fn, GDF, bandnames, reproject=True)
    else:
        feature_type = 'point'
        print(feature_type)
        GDF = extract_zonal_gdf_point(r_fn, GDF, bandnames, reproject=True)


    if return_src_path:
        GDF['src_file'] = os.path.basename(r_fn) 
        GDF['src_path'] = os.path.dirname(r_fn)
    
    print(f'\Finished zonal_stats for {len(GDF)} features ({feature_type}) with raster info from {len(bandnames)} bands: {bandnames}')
    if OUTDIR is None:
        return(GDF)
    else:
        GDF.to_file(os.path.join(OUTDIR, f'zonal_{OUTNAME}_{os.path.splitext(os.path.basename(r_fn))[0]}.gpkg'), driver='GPKG')
        
# -----------------------------------------------------------------------------
# class CmrProcess
#
# @author: Caleb Spradlin, caleb.s.spradlin@nasa.gov
# @version: 12.30.2021
#
# https://cmr.earthdata.nasa.gov/search/
# https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html
# -----------------------------------------------------------------------------
class CmrProcess(object):

    CMR_BASE_URL = 'https://cmr.earthdata.nasa.gov' +\
        '/search/granules.umm_json_v1_4?'

    # Range for valid lon/lat
    LATITUDE_RANGE = (-90, 90)
    LONGITUDE_RANGE = (-180, 180)

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self,
                 mission,
                 dateTime,
                 lonLat=None,
                 error=False,
                 dayNightFlag='',
                 pageSize=150,
                 maxPages=50):

        self._error = error
        self._dateTime = dateTime
        self._mission = mission
        self._pageSize = pageSize
        self._maxPages = maxPages
        
        self._lonLat = lonLat
        self._dayNightFlag = dayNightFlag

    # -------------------------------------------------------------------------
    # run()
    #
    # Given a set of parameters on init (time, location, mission), search for
    # the most relevant file. This uses CMR to search metadata for
    # relevant matches.
    # -------------------------------------------------------------------------
    def run(self):
        print('Starting query')
        outout = set()
        for i in range(self._maxPages):
            
            d, e = self._cmrQuery(pageNum=i+1)
            
            if e and i > 1:
                return sorted(list(outout))
            
            if not e:
                print('Results found on page: {}'.format(i+1))
                out = [r['file_url'] for r in d.values()]
                outout.update(out)
                
        outout = sorted(list(outout))
        return outout
        

    # -------------------------------------------------------------------------
    # cmrQuery()
    #
    # Search the Common Metadata Repository(CMR) for a file that
    # is a temporal and spatial match.
    # -------------------------------------------------------------------------
    def _cmrQuery(self, pageNum=1):

        requestDictionary = self._buildRequest(pageNum=pageNum)
        totalHits, resultDictionary = self._sendRequest(requestDictionary)

        if self._error:
            return None, self._error

        if totalHits <= 0:
            print('No hits on page number: {}, ending search.'.format(pageNum))
            #warnings.warn(msg)
            return None, True

        resultDictionaryProcessed = self._processRequest(resultDictionary)
        return resultDictionaryProcessed, self._error

    # -------------------------------------------------------------------------
    # buildRequest()
    #
    # Build a dictionary based off of parameters given on init.
    # This dictionary will be used to encode the http request to search
    # CMR.
    # -------------------------------------------------------------------------
    def _buildRequest(self, pageNum=1):
        requestDict = dict()
        requestDict['page_num'] = pageNum
        requestDict['page_size'] = self._pageSize
        requestDict['concept_id'] = self._mission
        requestDict['bounding_box'] = self._lonLat
        requestDict['day_night_flag'] = self._dayNightFlag
        requestDict['temporal'] = self._dateTime
        return requestDict

    # -------------------------------------------------------------------------
    # _sendRequest
    #
    # Send an http request to the CMR server.
    # Decode data and count number of hits from request.
    # -------------------------------------------------------------------------
    def _sendRequest(self, requestDictionary):
        with urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                                 ca_certs=certifi.where()) as httpPoolManager:
            encodedParameters = urlencode(requestDictionary, doseq=True)
            requestUrl = self.CMR_BASE_URL + encodedParameters
            try:
                requestResultPackage = httpPoolManager.request('GET',
                                                               requestUrl)
            except urllib3.exceptions.MaxRetryError:
                self._error = True
                return 0, None

            requestResultData = json.loads(
                requestResultPackage.data.decode('utf-8'))
            status = int(requestResultPackage.status)

            if not status == 400:
                totalHits = len(requestResultData['items'])
                return totalHits, requestResultData

            else:
                msg = 'CMR Query: Client or server error: ' + \
                    'Status: {}, Request URL: {}, Params: {}'.format(
                        str(status), requestUrl, encodedParameters)
                warnings.warn(msg)
                return 0, None

    # -------------------------------------------------------------------------
    # _processRequest
    #
    # For each result in the CMR query, unpackage relevant information to
    # a dictionary. While doing so set flags if data is not desirable (too
    # close to edge of dataset).
    #
    #  REVIEW: Make the hard-coded names class constants? There are a lot...
    # -------------------------------------------------------------------------
    def _processRequest(self, resultDict):

        resultDictProcessed = dict()

        for hit in resultDict['items']:

            fileName = hit['umm']['RelatedUrls'][0]['URL'].split(
                '/')[-1]

            # ---
            # These are hardcoded here because the only time these names will
            # ever change is if we changed which format of metadata we wanted
            # the CMR results back in.
            #
            # These could be placed as class constants in the future.
            # ---
            fileUrl = hit['umm']['RelatedUrls'][0]['URL']
            temporalRange = hit['umm']['TemporalExtent']['RangeDateTime']
            dayNight = hit['umm']['DataGranule']['DayNightFlag']

 
            spatialExtent = hit['umm']['SpatialExten' +
                                          't']['HorizontalSpatialDom' +
                                               'ain']

            key = fileName

            resultDictProcessed[key] = {
                'file_name': fileName,
                'file_url': fileUrl,
                'temporal_range': temporalRange,
                'spatial_extent': spatialExtent,
                'day_night_flag': dayNight}

        return resultDictProcessed