import geopandas as gpd
import numpy as np
import pandas as pd

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
    with psycopg2.connect(database="arcgis", user="pmontesa", password="UzkEnLNgNXWnDZGeCQOH", host="arcdb04", port="5432") as dbConnect:

        cur = dbConnect.cursor() # setup the cursor
        selquery =  "SELECT S_FILEPATH, SENSOR, CATALOG_ID, ACQ_TIME FROM %s WHERE CATALOG_ID = '%s' AND PROD_CODE = '%s'" %(db_table, catID, prod_code)
        #selquery =  "SELECT * FROM %s WHERE CATALOG_ID = '%s' AND PROD_CODE = '%s'" %(db_table, catID, prod_code)
        cur.execute(selquery)
        selected=cur.fetchall()

    return selected