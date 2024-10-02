import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import contextily as ctx

import os, sys
sys.path.append('/home/pmontesa/code/geoscitools')
import footprintlib


# Functions originally in notebook: vhr_xml_dataframe.ipynb
# pulled out into this lib file for easier testing

import pandas as pd
import xml.etree.ElementTree as et

def get_vhr_xml_image_attribute(xml_fn, ATTRIB_NAME):
    xtree = et.parse(xml_fn)
    xroot = xtree.getroot()
    return(xroot.find('IMD').find('IMAGE').find(ATTRIB_NAME).text)

def get_vhr_xml_band_attribute(xml_fn, BAND_NAME, ATTRIB_NAME):
    xtree = et.parse(xml_fn)
    xroot = xtree.getroot()
    return(xroot.find('IMD').find(BAND_NAME).find(ATTRIB_NAME).text)

def compute_unit_vector(elevation, azimuth):
    # Convert to radians
    elevation_rad = np.deg2rad(elevation)
    azimuth_rad = np.deg2rad(azimuth)
    
    # Compute x, y, z using spherical to Cartesian conversion, given the definitions of elevation and azimuth
    x = np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = np.cos(elevation_rad) * np.cos(azimuth_rad)
    z = np.sin(elevation_rad)
    
    return np.array([x, y, z])

def compute_angular_divergence(sun_vector, view_vector):
    dot_product = np.dot(sun_vector, view_vector)
    angle = np.arccos(dot_product)
    
    return np.rad2deg(angle)  # convert to degrees

def get_angular_divergence_col(df, cols_rename_dict: dict = {'sunel': 'sun_elevation', 'sunaz': 'sun_azimuth', 'el': 'view_elevation', 'az': 'view_azimuth'}):
    
    df.rename(columns=cols_rename_dict, inplace=True)
    
    # Input angles for sun and view directions
    sun_elevation = 30  # elevation angle for sun
    sun_azimuth = 60  # azimuth angle for sun

    view_elevation = 45  # elevation angle for view
    view_azimuth = 90  # azimuth angle for view

    sun_vector = compute_unit_vector(sun_elevation, sun_azimuth)
    view_vector = compute_unit_vector(view_elevation, view_azimuth)

    divergence = compute_angular_divergence(sun_vector, view_vector)

    print(f'The angular divergence between the sun and the view direction is {divergence} degrees.')

def make_vhr_xml_dataframe(xml_fn: str, 
                  BAND_FOR_BOUNDS = 'BAND_N',
                  CORNER_COLS_LIST = ['ULLON','ULLAT','ULHAE','URLON','URLAT','URHAE','LLLON','LLLAT','LLHAE','LRLON','LRLAT','LRHAE'],
                  DF_COLS_LIST: list = ['SATID','CATID','TLCTIME','MEANPRODUCTGSD', 'MEANSUNAZ','MEANSUNEL','MEANSATAZ','MEANSATEL','MEANINTRACKVIEWANGLE','MEANCROSSTRACKVIEWANGLE','MEANOFFNADIRVIEWANGLE','CLOUDCOVER','SCANDIRECTION'],
                DF_COLS_LIST_RENAME: list = ['satid','catid','tlctime','gsd', 'sunaz','sunel','az','el','intrack','crosstrack','offnadir','cloudcover','scandir'] 
                          ):
    
    '''Read the XML of a VHR image and return a dataframe of its metadata
    '''
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

    
    df1[CORNER_COLS_LIST] = df1[CORNER_COLS_LIST].apply(pd.to_numeric)
    df2[DF_COLS_LIST[2]] = df2[DF_COLS_LIST[2]].apply(pd.to_datetime)
    df2[DF_COLS_LIST[3:len(DF_COLS_LIST)-1]] = df2[DF_COLS_LIST[3:len(DF_COLS_LIST)-1]].apply(pd.to_numeric)
    df2.columns = DF_COLS_LIST_RENAME
    
    return(pd.concat([df2, df1], axis=1))

def MAKE_DIR_POLAR_PLOT(d_list, indir, title = None, FIGSIZE=(7,5)):
    '''Make a polar plot of the acquisition characteristics of the sun and the sensor for a list of .xml files
    '''
    from matplotlib  import cm
    #plot_list = []
    
    f = plt.figure(figsize=FIGSIZE)
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

    if title is None:
        title = f'Acquisition geometry of SRLite input'
        if indir is not None:
            #title += f"\n{indir.split('nobackup/')[-1]}"
            title += f"\n{indir}"

    # Mark the target at the center of the polar plot
    ax.plot(0,0,marker='+', markersize=10, color='red')
    
    marker_kwargs = {'marker':'o', 'alpha': 0.75}

    for d in d_list:
        
        # Make a df b/c its easier to handle during plotting of cmaps
        #df = pd.DataFrame([d])

        # Map colors to sensor like this?
        ax.plot(np.radians(d['az']), (90-d['el']), markersize=3, \
                #label='ID1', \
                c='k',
                #c=d['ang_div'], cmap = cm.jet,
                # To map color to each sensor
                #c=pd.DataFrame([d])['year'].map(colors)[0], \
                # To map color to values??
                #c=pd.DataFrame([d])['ang_div'].map(colors)[0], \
                **marker_kwargs 
               )
        if False:
            # or
            ax.plot(np.radians(d['az']), (90-d['el']), markersize=3, 
                    column=d['ang_div'], 
                    **marker_kwargs , 
                    categorical=False, 
                    legend=True, cmap='viridis') 
       
        # Add the Sun positions
        ax.plot(np.radians(d['sunaz']), (90-d['sunel']), markersize=10, label='Sun', c='orange', 
                mfc='none', 
                **marker_kwargs )
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
    
    # adding legend
    sun_marks = plt.scatter([],[],  facecolors='none', edgecolors='orange', label='Sun', s=75)
    sensor_marks = plt.scatter([],[],  color='k', label='Sensor')
    target_mark = plt.scatter([],[], marker='+', color='red', label='target')

    plt.legend(handles=[target_mark, sensor_marks, sun_marks])
    
    plt.tight_layout()
    
    # TODO: how do i return the ax so that I can place the figure I just made with other figures..
    return f # this lets you save it later with f.savefig("somefile.png")

def MAP_FOOTPRINTS(gdf, COL_NAME, CAT=True, CMAP='coolwarm'):
    
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    #colors = {'TOA':'tab:red', 'SR':'tab:green'}
    #colors = {'QB02':'tab:pink', 'GE01':'tab:green', 'WV01':'tab:blue', 'WV02':'tab:red', 'WV03':'tab:purple'}

    # Plot type
    #ax = footprints_gdf_ahri_20220818_adapt.plot(ax=ax, alpha=0.5, ec='k', column='sensor', label='file', categorical=True, legend=True, cmap='viridis')

    # Plot sensor of SR
    ax1 = gdf.to_crs(4326).cx[-180:-125,50:75].plot(ax=ax1, alpha=0.75, ec='k', column=COL_NAME, categorical=CAT, legend=True, cmap=CMAP)


    ax1 = ctx.add_basemap(ax1, crs=4326, 
        #source = ctx.providers.Gaode.Satellite
        #source = ctx.providers.Esri.WorldShadedRelief
        source = ctx.providers.Esri.WorldGrayCanvas
        #source = ctx.providers.Esri.NatGeoWorldMap
        #source = ctx.providers.Esri.WorldImagery
        #source = ctx.providers.Esri.DeLorme
    )
    # if not CAT:
    #     cax = fig.add_axes(ax1)
    #     fig.colorbar(ax1, cax=cax)
    #     #divider = make_axes_locatable(ax1)
    #     #cax = divider.append_axes("right", size="5%", pad=0.05)
    #     plt.colorbar(ax1, cax=cax)
    
# https://gis.stackexchange.com/questions/265520/groupby-ploting-give-each-plot-title-name
def map_image_band(cog_fn, band_num=13, vmin=0.20, vmax=0.45):
    
    with rio.open(cog_fn) as dataset:

        fig, ax = plt.subplots(figsize=(5, 5))

        # use imshow so that we have something to map the colorbar to
        image_hidden = ax.imshow(dataset.read(band_num), 
                                 cmap='nipy_spectral', 
                                 vmin=vmin, 
                                 vmax=vmax)

        # plot on the same axis with rio.plot.show
        image = show(dataset.read(band_num), 
                              transform=dataset.transform, 
                              ax=ax, 
                              cmap='nipy_spectral', 
                              vmin=vmin, 
                              vmax=vmax)

        # add colorbar using the now hidden image
        fig.colorbar(image_hidden, ax=ax)
        
def MAP_SUBPLOTS(gdf, out_gdf_fn, col_list = ['sensor', 'year', 'sunel'], cmap_list = ['RdBu', 'plasma', 'cividis'], categorical_list = [True, False, False], VMIN=None, VMAX=None):

    fig, axa = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, dpi=150, figsize=(40,4))

    for n, ax in enumerate(axa.ravel()):
        # This makes leftover axes empty and doesnt return an error
        if n >= len(col_list):
            ax.axis('off')
            pass
        else:
            ax.set_title(col_list[n], fontsize=15)
            ax = footprintlib.MAP_FOOTPRINTS(gdf, col_list[n], ax=ax, CMAP=cmap_list[n], CAT=categorical_list[n], VMIN=VMIN, VMAX=VMAX)

            #
            #if not categorical_list[n]:
            #cax = fig.add_axes(ax)
    #         fig.colorbar(ax, cax=cax)
    #         divider = make_axes_locatable(ax)
    #         cax = divider.append_axes("right", size="5%", pad=0.05)
    #         plt.colorbar(ax, cax=cax)

    fig.suptitle(f'n={len(gdf)} {out_gdf_fn}', horizontalalignment='left')
    plt.tight_layout()