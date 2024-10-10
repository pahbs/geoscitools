import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import show
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling

from scipy import ndimage

import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx

import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
import fiona
from fiona.crs import from_epsg
import pprint

import glob
import os
#import s3fs

import xml.etree.ElementTree as et

from osgeo import gdal, ogr, osr
from shapely import wkt
wgs_srs = osr.SpatialReference()
wgs_srs.SetWellKnownGeogCS('WGS84')
def xml2wkt(xml_fn):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_fn)
    #There's probably a cleaner way to do this with a single array instead of zipping
    taglon = ['ULLON', 'URLON', 'LRLON', 'LLLON']
    taglat = ['ULLAT', 'URLAT', 'LRLAT', 'LLLAT']
    #dg_mosaic.py doesn't preserve the BAND_P xml tags 
    #However, these are preserved in the STEREO_PAIR xml tags
    #taglon = ['ULLON', 'LRLON', 'LRLON', 'ULLON']
    #taglat = ['ULLAT', 'ULLAT', 'LRLAT', 'LRLAT']
    x = []
    y = []
    for tag in taglon:
        elem = tree.find('.//%s' % tag)
        #NOTE: need to check to make sure that xml has these tags (dg_mosaic doesn't preserve)
        x.append(elem.text)
    for tag in taglat:
        elem = tree.find('.//%s' % tag)
        y.append(elem.text)
    #Want to complete the polygon by returning to first point
    x.append(x[0])
    y.append(y[0])
    geom_wkt = 'POLYGON(({0}))'.format(', '.join(['{0} {1}'.format(*a) for a in zip(x,y)]))
    return geom_wkt

def geom_union(geom_list, **kwargs):
    convex=False
    union = geom_list[0]
    for geom in geom_list[1:]:
        union = union.Union(geom)
    if convex:
        union = union.ConvexHull()
    return union

def xml2geom(xml_fn):
    """
    Get OGR Geometry object
    """
    geom_wkt = xml2wkt(xml_fn)
    geom = ogr.CreateGeometryFromWkt(geom_wkt)
    #Hack for GDAL3, should reorder with (lat,lon) as specified
    if int(gdal.__version__.split('.')[0]) >= 3:
        wgs_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geom.AssignSpatialReference(wgs_srs)
    return geom

def ogr2shapely(geom):
    return wkt.loads(geom.ExportToWkt())

def get_vhr_xml_image_attribute(xml_fn, ATTRIB_NAME):
    xtree = et.parse(xml_fn)
    xroot = xtree.getroot()
    return(xroot.find('IMD').find('IMAGE').find(ATTRIB_NAME).text)

def get_vhr_xml_band_attribute(xml_fn, BAND_NAME, ATTRIB_NAME):
    xtree = et.parse(xml_fn)
    xroot = xtree.getroot()
    return(xroot.find('IMD').find(BAND_NAME).find(ATTRIB_NAME).text)

def make_vhr_xml_dataframe(xml_fn: str, 
                  BAND_FOR_BOUNDS = 'BAND_N',
                  CORNER_COLS_LIST = ['ULLON','ULLAT','ULHAE','URLON','URLAT','URHAE','LLLON','LLLAT','LLHAE','LRLON','LRLAT','LRHAE'],
                  DF_COLS_LIST: list = ['SATID','CATID','TLCTIME','MEANPRODUCTGSD', 'MEANSUNAZ','MEANSATEL','MEANSATAZ','MEANSATEL','MEANINTRACKVIEWANGLE','MEANCROSSTRACKVIEWANGLE','MEANOFFNADIRVIEWANGLE','CLOUDCOVER','SCANDIRECTION']):
    
    '''Read the XML of a VHR image and return a dataframe of its metadata
    '''
    try:
        #print(xml_fn)
        xtree = et.parse(xml_fn)
        xroot = xtree.getroot()
        for child in xroot:
            #print(child.tag, child.attrib)
            if 'IMD' in child.tag:
                #print(child.tag, child.attrib)
                for child_1 in child:
                    if BAND_FOR_BOUNDS == child_1.tag:
                        for child_2 in child_1:
                            if CORNER_COLS_LIST[0] in child_2.tag:
                                df1 = pd.DataFrame([get_vhr_xml_band_attribute(xml_fn, BAND_FOR_BOUNDS, COL) for COL in CORNER_COLS_LIST] ).transpose()
                                df1.columns = CORNER_COLS_LIST
                    if 'IMAGE' == child_1.tag:
                        #print(child_1.tag, child_1.attrib)
                        for child_2 in child_1:
                            if DF_COLS_LIST[0] in child_2.tag:
                                #print(child_2.tag, child_2.attrib)
                                df2 = pd.DataFrame([get_vhr_xml_image_attribute(xml_fn, COL) for COL in DF_COLS_LIST] ).transpose()
                                df2.columns = DF_COLS_LIST
        df1['geom_poly'] = ogr2shapely(xml2geom(xml_fn))
        return(pd.concat([df2, df1], axis=1))
    except Exception as e: 
        print(e)
        print(xml_fn)

def parse_aws_creds(credentials_fn):
    
    import configparser
    
    config = configparser.ConfigParser()
    config.read(credentials_fn)
    profile_name = config.sections()[0]
    #[print(key) for key in config['boreal_pub']]
    aws_access_key_id = config['boreal_pub']['aws_access_key_id']
    aws_secret_access_key = config['boreal_pub']['aws_secret_access_key']
    
    if False:
        credentials_df = pd.read_csv(credentials_fn, sep=" ", header=None)
        profile_name=credentials_df.iloc[0].to_list()[0].replace(']','').replace('[','')
        aws_access_key_id = credentials_df.iloc[1].to_list()[0].split("=")[1]
        aws_secret_access_key = credentials_df.iloc[2].to_list()[0].split("=")[1]
    
    return profile_name, aws_access_key_id, aws_secret_access_key

def get_rio_aws_session_from_creds(credentials_fn):
    
    import s3fs
    import rasterio as rio
    from rasterio.session import AWSSession
    import boto3
    import pandas as pd
    
    profile_name, aws_access_key_id, aws_secret_access_key = parse_aws_creds(credentials_fn)

    boto3_session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            #aws_session_token=credentials['SessionToken'],
            profile_name=profile_name
        )

    rio_aws_session = AWSSession(boto3_session)
    
    return rio_aws_session

def get_s3_fs_from_creds(credentials_fn=None, anon=False):
    
    if anon:
        s3_fs = s3fs.S3FileSystem(anon=anon)
    else:
        if credentials_fn is not None:
            profile_name, aws_access_key_id, aws_secret_access_key = parse_aws_creds(credentials_fn)
            s3_fs = s3fs.S3FileSystem(profile=profile_name, key=aws_access_key_id, secret=aws_secret_access_key)
        else:
            print("Must provide a credentials filename if anon=False")
            os._exit(1)
    return s3_fs

def resample_raster(raster, out_path=None, scale=2):
    """ Resample a raster
        multiply the pixel size by the scale factor
        divide the dimensions by the scale factor
        i.e
        given a pixel size of 250m, dimensions of (1024, 1024) and a scale of 2,
        the resampled raster would have an output pixel size of 500m and dimensions of (512, 512)
        given a pixel size of 250m, dimensions of (1024, 1024) and a scale of 0.5,
        the resampled raster would have an output pixel size of 125m and dimensions of (2048, 2048)
        returns a DatasetReader instance from either a filesystem raster or MemoryFile (if out_path is None)
    """
    data, profile = get_data_from_resample(raster, scale)

    if out_path is None:
        with write_mem_raster(data, **profile) as dataset:
            del data
            yield dataset

    else:
        with write_raster(out_path, data, **profile) as dataset:
            del data
            yield dataset


def write_mem_raster(data, **profile):
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(data)

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return


def write_raster(path, data, **profile):

    with rasterio.open(path, 'w', **profile) as dataset:  # Open as DatasetWriter
        dataset.write(data)

    with rasterio.open(path) as dataset:  # Reopen as DatasetReader
        yield dataset

'''
def do_resample(dataset, scale_factor):
    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * scale_factor),
            int(dataset.width * scale_factor)
        ),
        resampling=Resampling.bilinear
    )

    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    )
    return data, transform
'''

def get_data_from_resample(dataset, scale_factor):
    t = dataset.transform

    # rescale the metadata
    transform = Affine(t.a * scale_factor, t.b, t.c, t.d, t.e * scale_factor, t.f)
    height = int(raster.height / scale_factor)
    width = int(raster.width / scale_factor)

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    data = dataset.read(
            out_shape=(dataset.count, height, width),
            resampling=Resampling.bilinear,
        )
    
    return data, profile

def r_getgeom(r_fn, TO_GCS = True, scale_factor=100, out_res=250):
    
    '''Function to get a raster mask and other info to make a raster footprint'''
    
    # Alt attempt: use pygeotools to write out a coarsened raster that gets footprinted
    # This is slower and creates intermediate coarsened files..not ideal
    #warplib.diskwarp_multi_fn( [r_fn], res=out_res, extent='first', t_srs='first', r='cubic', verbose=True, outdir=os.path.dirname(r_fn), dst_ndv=None):
    
    with rasterio.open(r_fn) as dataset:
        name = os.path.basename(r_fn)
        # Attempt to use rasterio to downsample with a scale factor, the create a dataset that gets footprinted..
        # TODO: this was a fail - try to do it correctly
        #out_fn = os.path.splitext(r_fn)[0] + f'_sf{scale_factor}.tif'
        #dataset = resample_raster(dataset, out_path=out_fn, scale=scale_factor)
        
        
        # Read the dataset's valid data mask as a ndarray.
        data = dataset.read(1)
        mask = dataset.dataset_mask()
        data_crs = dataset.crs
        
        if False:
            results = (
                {'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(rasterio.features.shapes(mask, transform=dataset.transform))
            )
            geom = list(results)
            print(geom[-1])
        else:
            # Extract feature shapes and values from the array.
            # TODO: this might only get 1 of the geoms?
            
            #for geom, val in rasterio.features.shapes(data, mask=mask, transform=dataset.transform):
            for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
            #for geom, val in rasterio.features.shapes(data=dataset.read(1), transform=dataset.transform):

                if TO_GCS:
                    # Transform shapes from the dataset's own coordinate
                    # reference system to CRS84 (EPSG:4326).
                    geom = rasterio.warp.transform_geom(
                        dataset.crs, 'EPSG:4326', geom, precision=6)
                    #geom_list.append(geom)

                #print(val)
            #print(geom)
            
        return geom, name, os.path.basename(r_fn), data_crs
    
def raster_footprint_gdf(r_fn_list, OUT_F_NAME='footprints.gpkg', OUT_LYR_NAME='footprints', TO_GCS=False, WRITE_GPKG=True):
    
    # Build the components of the GeoDataFrame dict
    polys = [r_getgeom(r_fn, TO_GCS=False)[0] for r_fn in r_fn_list ]    
    names = [r_getgeom(r_fn, TO_GCS=False)[1] for r_fn in r_fn_list ]  
    files = [r_getgeom(r_fn, TO_GCS=False)[2] for r_fn in r_fn_list ]
    
    # Use shapely to convert list of polys to actual shape geoms
    geoms = [shapely.geometry.shape(i) for i in polys]
    
    # Return the CRS for output
    out_crs = r_getgeom(r_fn_list[0], TO_GCS=TO_GCS)[3]
    
    # Build the GeoDataFrame
    #footprints_gdf  = gpd.GeoDataFrame.from_features(polys)
    footprints_gdf = gpd.GeoDataFrame({'geometry':geoms, 'name':names, 'file':files}, crs=out_crs)
    
    if not TO_GCS:
        # Get area
        footprints_gdf["area_km2"] = footprints_gdf['geometry'].area/1000000
        footprints_gdf["area_ha"] = footprints_gdf['geometry'].area/10000
           
    if WRITE_GPKG:
        footprints_gdf.to_file(OUT_F_NAME, driver="GPKG", layer=OUT_LYR_NAME)
        print(f"Wrote out spatial footprints to {OUT_F_NAME}")
    
    return footprints_gdf

def get_geom_from_datasetmask(rio_dataset, GET_ONLY_DATASETMASK=True, MASK_OUT_VALUE=None):
    
    if GET_ONLY_DATASETMASK:
        results = (
                    {'properties': {'raster_value': v}, 'geometry': s} for i, (s, v) in enumerate(rasterio.features.shapes(rio_dataset.dataset_mask(), transform=rio_dataset.transform))
                )
    else:
        ma = np.ma.masked_invalid(rio_dataset.read())
        if rio_dataset.profile['nodata'] is not None:
            ma = np.ma.masked_where(ma == rio_dataset.profile['nodata'], ma)
        if MASK_OUT_VALUE is not None:
            ma = np.ma.masked_where(ma == MASK_OUT_VALUE, ma)
        ma[ma!=np.nan]=1
        results = (
                    #{'properties': {'raster_value': v}, 'geometry': s} for i, (s, v) in enumerate(rasterio.features.shapes(rio_dataset.read_masks(), transform=rio_dataset.transform))
                    {'properties': {'raster_value': v}, 'geometry': s} for i, (s, v) in enumerate(rasterio.features.shapes(ma, transform=rio_dataset.transform))
                )
        
    geom = list(results)

    #return(geom, rio_dataset.crs)
    return(geom)

def get_geom_from_bounds(rio_dataset, footprint_name=None):
    #result=[{"properties":{"id":1},"geometry": [mapping(box(*rio_dataset.bounds))] } ]
    ##geom = {'properties': {'raster_val': None }, 'geometry': mapping(box(*rio_dataset.bounds)) } 
    #geom = {'properties': {'raster_val': None }, 'geometry': {'type': 'Polygon', 'coordinates': [box(*rio_dataset.bounds)] } } 
    results = [{"properties":{'footprint_name': footprint_name}, "geometry": {'type': 'Polygon', 'coordinates': [list(box(*rio_dataset.bounds).exterior.coords)] } } ]
    geom = list(results)
    #print(geom)
    return(geom)

def raster_footprint(r_fn, DO_DATAMASK=True, GET_ONLY_DATASETMASK=True, R_READ_MODE='r+', MANY_CRS=False, DISSOLVE_FIELD='file', MASK_OUT_VALUE=None, OUTDIR=None):
    try:
        with rasterio.open(r_fn, mode=R_READ_MODE) as dataset:

            if DO_DATAMASK:
                # TODO: Fix this. This flag does nothing right now
                if GET_ONLY_DATASETMASK:
                    job_string = 'valid data mask (high memory)'
                else:
                    job_string = 'valid data mask + the nodata (most memory)'
                print(job_string)
                #geom, dataset.crs = get_geom_from_datasetmask(dataset)
                geom = get_geom_from_datasetmask(dataset, GET_ONLY_DATASETMASK=GET_ONLY_DATASETMASK, MASK_OUT_VALUE=MASK_OUT_VALUE)

            else:
                job_string = 'raster image bounds (low memory)'
                geom = get_geom_from_bounds(dataset)
            
            # Dissolve so you dont return 1 polygon for each unique raster value...
            footprints_gdf  = gpd.GeoDataFrame.from_features(geom, crs=dataset.crs)
            
            #print(footprints_gdf.crs.axis_info[0].unit_name)
            #print(dataset.crs)
            
            if not isinstance(r_fn, str):
                r_fn = str(r_fn).split(',')[1].replace('>','').replace(' ','')

            footprints_gdf['path'], footprints_gdf['file'] = os.path.split(r_fn)
            
            if DISSOLVE_FIELD in footprints_gdf.columns:
                footprints_gdf = footprints_gdf.dissolve(by=DISSOLVE_FIELD, as_index=False)
                if 'raster_value' in footprints_gdf.columns:
                    footprints_gdf = footprints_gdf.drop(columns=['raster_value'])
            else:
                print(f"Can't dissolve by {DISSOLVE_FIELD}. Field not found.")

            if False:
                print(f'Getting {job_string} for: {os.path.basename(r_fn)} ...')

            if 'm' in footprints_gdf.crs.axis_info[0].unit_name:
                # Get area
                footprints_gdf["area_km2"] = footprints_gdf['geometry'].area/1e6
                footprints_gdf["area_ha"] = footprints_gdf['geometry'].area/1e4

            if MANY_CRS:
                #print('There are multiple CRSs in this set, so reprojecting everything to 4326...')
                footprints_gdf = footprints_gdf.to_crs(4326)
                
            if OUTDIR is not None:
                footprints_gdf.to_file(os.path.join(OUTDIR, os.path.basename(r_fn).replace('.tif','.gpkg') ), driver='GPKG')
                footprints_gdf = None
            else:
                return footprints_gdf
    except Exception as e: 
        print(e)
        print(r_fn)
    
def build_footprint_db(gdf_list, TO_GCS=True, WRITE_GPKG=True, OUT_F_NAME='footprints.gpkg', OUT_LYR_NAME='footprints', DROP_DUPLICATES=True):
    
    print("Building GDF from list...")
    if TO_GCS:
        print("Converting to each to GCS...")
        #footprints_gdf = gpd.GeoDataFrame( pd.concat( gdf_list, ignore_index=True) )
        footprints_gdf = pd.concat( [gdf.to_crs(4326) for gdf in gdf_list], ignore_index=True)
    else:
        footprints_gdf = pd.concat( gdf_list, ignore_index=True)
    if DROP_DUPLICATES:
        footprints_gdf = footprints_gdf.drop_duplicates()
    
    #if TO_GCS:
    #    print("Converting to GCS...")
    #    footprints_gdf = footprints_gdf.to_crs({'init': 'epsg:4326'})
        
    if WRITE_GPKG:
        footprints_gdf.to_file(OUT_F_NAME, driver="GPKG", layer=OUT_LYR_NAME)
        print(f"Wrote out spatial footprints to {OUT_F_NAME}")
        
    return footprints_gdf

def MAP_FOOTPRINTS(gdf, COL_NAME, CAT=True, CMAP='coolwarm', ax=None, fig=None, VMIN=None, VMAX=None):
    
    if VMIN is None or VMAX is None:
        VMIN = min(gdf[COL_NAME])
        VMAX = max(gdf[COL_NAME])
    
    if ax is None:
        fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        
    #colors = {'TOA':'tab:red', 'SR':'tab:green'}
    #colors = {'QB02':'tab:pink', 'GE01':'tab:green', 'WV01':'tab:blue', 'WV02':'tab:red', 'WV03':'tab:purple'}



    # Plot type
    #ax = footprints_gdf_ahri_20220818_adapt.plot(ax=ax, alpha=0.5, ec='k', column='sensor', label='file', categorical=True, legend=True, cmap='viridis')

    # Plot sensor of SR
    ax = gdf.to_crs(4326).plot(ax=ax, alpha=0.75, 
                                                   #ec='k', 
                                                   column=COL_NAME, categorical=CAT, legend=True, cmap=CMAP, vmin=VMIN, vmax=VMAX)


    ax = ctx.add_basemap(ax, crs=4326, 
        #source = ctx.providers.Gaode.Satellite
        #source = ctx.providers.Esri.WorldShadedRelief
        source = ctx.providers.Esri.WorldGrayCanvas
        #source = ctx.providers.Esri.NatGeoWorldMap
        #source = ctx.providers.Esri.WorldImagery
        #source = ctx.providers.Esri.DeLorme
    )
    # For continuous colorbars
    if not CAT and fig is not None:
        
        cax = fig.add_axes(ax)
        fig.colorbar(ax, cax=cax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #plt.colorbar(ax, cax=cax)
    
    return(ax)
    
def MAKE_DIR_POLAR_PLOT(d_list, indir):
    '''Make a polar plot of the acquisition characteristics of the sun and the sensor for a list of .xml files
    '''
    
    #plot_list = []
    
    f = plt.figure(figsize=(7,5))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    # And a corresponding grid
    #ax.grid(which='both')
    #ax.grid(which='minor', alpha=0.02)
    ax.grid(which='major', alpha=0.5, linestyle='--')


    colors = {'QB02':'tab:pink', 'GE01':'tab:green', 'WV01':'tab:blue', 'WV02':'tab:red', 'WV03':'tab:purple'}

    # title = p['pairname']
    # title += '\nCenter datetime: %s' % p['cdate']
    # title += '\nTime offset: %0.2f s' % abs(p['dt'].total_seconds())
    # title += '\nConv. angle: %0.2f, Int. area: %0.2f km2' % (p['conv_ang'], p['intersection_area'])
    # title += '\n%s gsd:%0.2f az:%0.1f el:%0.1f off:%0.1f %s %i' % (p['id1_dict']['id'], p['id1_dict']['gsd'], p['id1_dict']['az'], (90-p['id1_dict']['el']), p['id1_dict']['offnadir'], p['id1_dict']['scandir'], p['id1_dict']['tdi'])
    # title += '\n%s gsd:%0.2f az:%0.1f el:%0.1f off:%0.1f %s %i' % (p['id2_dict']['id'], p['id2_dict']['gsd'], p['id2_dict']['az'], (90-p['id2_dict']['el']), p['id2_dict']['offnadir'], p['id2_dict']['scandir'], p['id2_dict']['tdi'])

    # title = d['id']
    # title += '\nCenter datetime: %s' % d['date']
    # title += '\n%s gsd:%0.2f az:%0.1f el:%0.1f off:%0.1f %s %i' % (d['id'], d['gsd'], d['az'], (90-d['el']), d['offnadir'], d['scandir'], d['tdi'])

    title = f'Acquisition geometry of SRLite input'
    title += f"\n{indir.split('nobackup/')[-1]}"

    # Mark the target at the center of the polar plot
    ax.plot(0,0,marker='+',color='k')
    
    marker_kwargs = {'marker':'o', 'alpha': 0.75}

    for d in d_list:
        
        # Make a df b/c its easier to handle during plotting of cmaps
        #df = pd.DataFrame([d])

        # Map colors to sensor like this?
        ax.plot(np.radians(d['az']), (90-d['el']), markersize=3, \
                #label='ID1', \
                c='k',
                # To map color to each sensor
                #c=pd.DataFrame([d])['sensor'].map(colors)[0], \
                **marker_kwargs   )
        # or
        #ax.plot(np.radians(df['az']), (90-df['el']), markersize=3, column=df['sensor'], categorical=True, **marker_kwargs ) #categorical=True, legend=True, cmap='viridis', 
       
        # Add the Sun positions
        ax.plot(np.radians(d['sunaz']), (90-d['sunel']), markersize=10, label='Sun', c='orange', mfc='none', **marker_kwargs )
        #ax.plot(np.radians(p['id2_dict']['az']), (90-p['id2_dict']['el']), marker='o', label='ID2')
        #ax.plot([np.radians(p['id1_dict']['az']), np.radians(p['id2_dict']['az'])], [90-p['id1_dict']['el'], 90-p['id2_dict']['el']], \
        #       color='k', ls=':')

        #ax.legend()

        #This sets elevation range
        ax.set_rmin(0)
        ax.set_rmax(90)
        #ax.patch.set_facecolor('lightgray')

        ax.set_title(title, fontsize=10)
        
        #plot_list.append(ax)
        
    plt.tight_layout()
    
    # TODO: how do i return the ax so that I can place the figure I just made with other figures..
    return ax

def fix_no_data_value(input_file, output_file, no_data_value=0):
    # https://gis.stackexchange.com/questions/369064/how-to-convert-0-values-to-nodata-values-with-rasterio
    with rasterio.open(input_file, "r+") as src:
        src.nodata = no_data_value
        with rasterio.open(output_file, 'w',  **src.profile) as dst:
            for i in range(1, src.count + 1):
                band = src.read(i)
                band = np.where(band==no_data_value,no_data_value,band)
                dst.write(band,i)
                
def array_to_polygons(array, transform=None):
    from rasterio.features import shapes
    from shapely.geometry import shape
    import geopandas
    """
    https://github.com/brycefrank/pyfor/blob/2d8e5b461b81578ea698f06df6b3736ae9959c41/pyfor/gisexport.py#L50
    Returns a geopandas dataframe of polygons as deduced from an array.

    :param array: The 2D numpy array to polygonize.
    :param affine: The affine transformation.
    :return:
    """
    if transform == None:
        results = [
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v)
                in enumerate(shapes(array))
        ]
    else:
        results = [
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v)
            in enumerate(shapes(array, transform=transform))
        ]


    tops_df = geopandas.GeoDataFrame({'geometry': [shape(results[geom]['geometry']) for geom in range(len(results))],
                                      'raster_val': [results[geom]['properties']['raster_val'] for geom in range(len(results))]})

    return(tops_df)


def footprint_cloudmask(r_fn, NEW_NDV, TO_DTYPE = 'uint8', INVERT=True, N_ITER=1, OUTDIR=None):
    
    with rasterio.open(r_fn) as dataset:

        ma = dataset.read(1)
        #print(ma)

        #for i, dtype, nodataval, crs in zip(dataset.indexes, dataset.dtypes, dataset.nodatavals, dataset.crs):
        #    print(i, dtype, nodataval, crs)

        # Set nodata value to the NEW_NDV    
        ma = np.where(ma==dataset.nodatavals, NEW_NDV, ma).astype(TO_DTYPE)
        
        if INVERT:
            ma = 1 - ma
        
        #ma = np.where(ma==NEW_NDV, dataset.nodatavals, ma).astype(TO_DTYPE)
        #ma[ma!=dataset.nodatavals] = 1
        
        #print(type(ma.dtype))

        # Binary dilation by n pixels
        ma = ndimage.binary_dilation(ma, iterations=N_ITER).astype(ma.dtype)
        #ma = ndimage.binary_erosion(ma, iterations=N_ITER).astype(ma.dtype)

        gdf = array_to_polygons(ma, transform = dataset.transform)
        gdf['file'] = os.path.basename(r_fn)
        gdf = gdf[gdf.raster_val != NEW_NDV]
        #print(ma)
        
        print(f'Footprinted cloudmask: {os.path.basename(r_fn)}')
        gdf.set_crs(crs=dataset.crs, inplace=True)
        
        if OUTDIR is not None:
            gdf.to_file(os.path.join(OUTDIR, os.path.basename(r_fn).replace('.tif','.gpkg') ), driver='GPKG')
            
        return gdf
    
def get_attributes_from_filename(footprint_gdf, image_type: str, file_split_str: str, filename = None, file_col = 'file', DROP_FILE_DUPLICATES=True):
    
    if file_col not in footprint_gdf.columns:
        footprint_gdf[file_col] = filename
    
    # Customize attributes
    footprint_gdf['type'] = image_type
    footprint_gdf['footprint_name'] = footprint_gdf[file_col].str.split(file_split_str, expand=True)[0]
    footprint_gdf['catid'] = footprint_gdf['footprint_name'].str.split('_', expand=True)[3]
    footprint_gdf['sensor'] = footprint_gdf[file_col].str.split('_', expand=True)[0]
    footprint_gdf['year'] = footprint_gdf[file_col].str.split('_', expand=True)[1].str[0:4].astype(int)
    footprint_gdf['month'] = footprint_gdf[file_col].str.split('_', expand=True)[1].str[4:6].astype(int)
    footprint_gdf['date'] = pd.to_datetime(footprint_gdf[file_col].str.split('_', expand=True)[1] , format="%Y%m%d")

    if DROP_FILE_DUPLICATES:
        # Drop dups from 'repair'
        footprint_gdf.drop_duplicates(subset=file_col, keep='last', inplace=True)

    if False:
        print( f"# {image_type} obs. : {footprint_gdf.shape[0]}")
    
    return footprint_gdf