import os
import numpy as np
import h5py
import rasterio as rio 
from rasterio.crs import CRS

from pathlib import Path

from rasterio.transform import from_origin
from rasterio.profiles import DefaultGTiffProfile
#
# Functions for NEON data stacking
#
# pulled from notebooks:
#                      NEON_hyperspectral_ancillary.ipynb

list_imagery = ['Aerosol_Optical_Depth','Aspect','Cast_Shadow',
                'Dark_Dense_Vegetation_Classification','Data_Selection_Index',
                'Haze_Cloud_Water_Map','Illumination_Factor','Path_Length',
                'Sky_View_Factor','Slope','Smooth_Surface_Elevation','Visibility_Index_Map',
                'Water_Vapor_Column','Weather_Quality_Indicator']

def stack_hyper_imagery(fn, SITE, subdataset='Metadata/Ancillary_Imagery', 
                            outdir=None, 
                            list_imagery = list_imagery,
                            RETURN_DATA=True
                           ):
    if outdir is None:
        outdir = os.path.dirname(fn)
    
    replace_string = '.tif'
    if 'ancillary' in subdataset.lower():
        replace_string = '_ancillary.tif'
    out_fn = os.path.join(outdir, os.path.basename(fn).replace('.h5', replace_string))
    
    f = h5py.File(fn,'r')
    
    SITE_refl = f[SITE]['Reflectance']
    #print(SITE_refl)
    
    prj4 = SITE_refl['Metadata']['Coordinate_System']['Proj4']
    epsg = SITE_refl['Metadata']['Coordinate_System']["EPSG Code"]
    prj4_str = prj4[()].decode("utf-8")
    epsg_str = epsg[()].decode("utf-8")
    
    mapInfo = SITE_refl['Metadata']['Coordinate_System']['Map_Info']
    mapInfo_split  = str(mapInfo[()]).split(',')
    
    #Extract the resolution & convert to floating decimal number
    res = float(mapInfo_split[5]),float(mapInfo_split[6])
    #print('Resolution:',res)

    #Extract the upper left-hand corner coordinates from mapInfo
    xMin = float(mapInfo_split[3]) 
    yMax = float(mapInfo_split[4])
     
    arr_list = []
    bandnames_list = []
    
    for n, img in enumerate(list_imagery):
        
        if 'Ancillary_Imagery' in subdataset:
            image_id = img # this is a name
            arr = SITE_refl[subdataset][image_id]
        else:
            image_id = n  # this is a number
            arr = SITE_refl[subdataset][:,:,list_imagery[image_id]].astype(np.float)
            
        #if RETURN_DATA: print(f'{SITE_refl[subdataset][image_id]}')
        
        #arr = np.moveaxis(SITE_refl['Metadata/Ancillary_Imagery'][img], -1, 0)

        #if RETURN_DATA: print(f'{arr.shape}:\t\t{image_id}')
        
        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, 2)
            
        if RETURN_DATA: print(f'{arr.shape}:\t\t{image_id}')
        
        # Append each band of array to array list: most arrays have only 1 band
        for i in range(0, arr.shape[2]):
            
            #print(f'arr i = {i}')
            arr_list.append(arr[:,:,i])
            
            bname = img
            if i > 0:
                bname = img + '_' +str(i + 1)
            bandnames_list.append(bname)
    
    # After arrays are re-configured...    
    # Calculate the xMax and yMin values from the dimensions
    # xMax = left corner + (# of columns * resolution)
    xMax = xMin + (arr.shape[2]*res[0])
    yMin = yMax - (arr.shape[1]*res[1]) 
    tile_ext = (xMin, xMax, yMin, yMax)
    
    #Can also create a dictionary of extent:
    tile_extDict = {}
    tile_extDict['xMin'] = xMin
    tile_extDict['xMax'] = xMax
    tile_extDict['yMin'] = yMin
    tile_extDict['yMax'] = yMax
    
    if RETURN_DATA: print(tile_extDict)
    
    # Create the stacked array
    stack = np.stack( arr_list, axis=0 )
    if RETURN_DATA: print(f'Stack shape: {stack.shape}')
    
    #origin is upper left
    transform = from_origin(tile_extDict['xMin'], tile_extDict['yMax'], res[0], res[1])

    # Set the profile for the output raster based on the ndarray stack
    out_profile = DefaultGTiffProfile(
        descriptions=bandnames_list,
        #interleave='pixel',
        driver="GTiff",
        height=stack.shape[1],
        width=stack.shape[2],
        count=stack.shape[0],
        dtype='float32',#str(stack.dtype),
        crs=CRS.from_epsg(epsg_str),
        resolution=res,
        transform=transform,
        #nodata=np.nan
    )
    if RETURN_DATA: print(out_profile)
    print(out_fn)
    with rio.open(out_fn, 'w+', **out_profile) as new_dataset:
        new_dataset.write(stack)
        #new_dataset.close()
    
    if RETURN_DATA:
        return stack, bandnames_list
    else:
        return out_fn

def plot_ancillary_stack(stack, bandnames_list):

    cmap_list = ['magma', 'bone', 'tab20',  'Greens', 'afmhot', 'winter',
                 'copper', 'summer', 'autumn', 'bone', 'bone', 'Wistia',
                 'hot', 'magma', 'inferno', 'plasma'
                ]

    f, axa = plt.subplots(nrows=4, ncols=5, sharex=True, sharey=True, dpi=150, figsize=(24,16)) 
    for n, ax in enumerate(axa.ravel()):
        #arr = SITE_refl['Metadata/Ancillary_Imagery'][list_imagery[n]]
        #print(f'{list_imagery[n]}:\t\t\t\t\t{arr.shape}')

        # This makes leftover axes empty and doesnt return an error
        if n >= len(bandnames_list):
            ax.axis('off')
        if n >= len(bandnames_list):
            pass
        else:
            ma = stack[n,:,:]
            #print(f'{list_imagery[n]}:\t\t\t\t\t{ma.shape}')

            # Hillshade
            if 'Aspect' in str(bandnames_list[n]) or 'Slope' in str(bandnames_list[n]) or 'Elevation' in str(bandnames_list[n]):
                from localtileserver import examples, helpers
                ma = helpers.hillshade(ma)
            if n < 11:
                im = ax.imshow(ma, interpolation='none', cmap=cmap_list[n])
            else:
                im = ax.imshow(ma, interpolation='none', cmap=cmap_list[n], clim=(np.nanpercentile(ma.data,10), np.nanpercentile(ma.data,90)) )
            #ax.set_title(layer_names[n] + '\n' + os.path.splitext(os.path.basename(fn_list[n]))[0], fontsize=5)
            ax.set_title(bandnames_list[n], fontsize=15)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = f.colorbar(im, cax=cax, orientation='vertical', extend='both')
            #cb.set_label(label_list[n])

    #fig.suptitle(f'Site {SITE} for year {YEAR}\n{DIR_SITE_EXTENT}', fontsize=16)
    plt.tight_layout()