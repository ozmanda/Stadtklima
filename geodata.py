import os
import numpy as np
import rasterio
import pickle
from warnings import warn
from pandas import read_csv
from spatialconvolutions import convolutions
import pvlib.irradiance as rad


def get_loc(stationid):
    """
    Determines the latitude and longitude of the measurement station using
    the input parameter stationid: the ID of the station, corresponding to
    stationid_new in the stations.csv list.

    Returns the latitude and longitude of the station corrected to LV09
    coordinates
    """
    stationscsv = read_csv(os.path.join(os.getcwd(), 'Messdaten\\stations.csv'), delimiter=';')
    targetlat = None
    targetlon = None
    for _, row in stationscsv.iterrows():
        if row['stationid_new'] == stationid:
            targetlat = row['CH_N'] if row['CH_N'] < 1000000 else row['CH_N'] - 1000000
            targetlon = row['CH_E'] if row['CH_E'] < 2000000 else row['CH_E'] - 2000000
            break
    if not targetlat and not targetlon:
        warn(f'The location of the sensor for station with ID {stationid} could not be determined', Warning)
        raise ValueError

    return targetlat, targetlon


def extract_feature(targetlat, targetlon, featurename, geopath, convs, num):
    if f'{featurename}.tif' in os.listdir(geopath):
        featuremap = rasterio.open(os.path.join(geopath, f'{featurename}.tif'))
        originlat = featuremap.meta['transform'][5]  # gives northern boundary
        originlon = featuremap.meta['transform'][2]  # gives western boundary
        featuremap = featuremap.read()
        featuremap[featuremap == -9999] = 0

        if targetlat > originlat or targetlon < originlon:
            warn(f'station outside of feature map area for feature {featurename}. \nLatitude: {targetlat} \n'
                 f'Longitude: {targetlon}', Warning)
            raise ValueError

        idxlat = originlat - targetlat
        idxlon = targetlon - originlon

        if idxlat > featuremap.shape[1] or idxlon > featuremap.shape[2]:
            warn(f'station outside of feature map area for feature {featurename}. \nLatitude: {targetlat} \n'
                 f'Longitude: {targetlon}', Warning)
            raise ValueError

    else:
        warn(f'The .tif file for the feature {featurename} is not available in the folder {geopath}, '
             f'check path', Warning)
        raise FileNotFoundError

    if not convs:
        feature = np.empty(shape=(num, 1))
        feature[:, 0] = featuremap[0, int(idxlat), int(idxlon)]
        return feature
    else:
        featureconvs = convolutions(convs, featuremap, idxlat, idxlon)
        feature = np.empty(shape=(num, len(convs)+1))
        feature[:, :] = featureconvs
        return feature


def get_geofeatures(stationid, geopath, num, convs, geofeaturelist):
    # list of static / geographic features to be considered
    geofeatures = np.empty(shape=(num, 0))

    targetlat, targetlon = get_loc(stationid)

    for idx, feature in enumerate(geofeaturelist):
        if feature == 'altitude':
            geofeatures = np.append(geofeatures, extract_feature(targetlat, targetlon, feature, geopath, None, num), axis=1)
        else:
            geofeatures = np.append(geofeatures, extract_feature(targetlat, targetlon, feature, geopath, convs, num), axis=1)

    return geofeatures, targetlat, targetlon


def load_geofeatures(geodir, shape):
    """
    This function either loads the geofeatures from an existing .PICKLE file, or it generate the
    feature data using the shape of the new NetCDF file and its layermasks.

    PARAMETERS:
    -----------
    topodir: topodir is the directory in which the topographical / geographical data can be found
      By default, it should be located in the (wd)/Data/geofeatures

    shape: this variable is the shape of the original NetCDF file, used to reformat the rasters
      to the dimensions of the training data: (time, layer, height, width)


    RETURNS:
    --------
    geofeatures: this variable contains the features of the 6 geographical features buildings,
      paved surfaces, forests, surface water, urban green and altitude. Geofeatures already have
      invalid observations removed according to the parameter layermasks.
      Attention: these values are hard-coded, as are as their convolutions.

    geolist: contians the names of the geographical variables for which features were loaded or
      generated (appended to the featurelist in NetCDF.py)

    """
    # altitude must always be the last geofeature to be processed
    geolist = ["buildings", "pavedsurfaces", "forests", "surfacewater", "urbangreen", "altitude"]
    rasterdim = (shape[2], shape[3])  # dimensions for the raster/map: layers x height x width

    # standardisted filename
    filename = f'geofeatures_{len(geolist)}_{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}.PICKLE'

    # check if the geofeatures file already exists and load data if it does, otherwise generate the array
    if os.path.isfile(os.path.join(geodir, filename)):
        geofeatures = pickle.load(open(os.path.join(geodir, filename), "rb"))  # without mask removed
    else:
        # create empty data array with 5 dimensions: times x layers x height x width x geofeatures
        geofeatures = np.zeros(shape=(shape[0], shape[1], shape[2], shape[3], len(geolist)))

        for idx, dataname in enumerate(geolist[0:-1]):
            # load either from .TIF or from .PICKLE
            if os.path.isfile(os.path.join(geodir, f'{dataname}.PICKLE')):
                data = pickle.load(open(os.path.join(geodir, f'{dataname}.PICKLE'), "rb"))
            elif os.path.isfile(os.path.join(geodir, f'{dataname}.tif')):
                data = rasterio.open(os.path.join(geodir, f'{dataname}.tif'))
                data = data.read(1)
                pickle.dump(data, open(os.path.join(geodir, f'{dataname}.PICKLE'), "wb"))  # save geofeatures as PICKLE
            else:
                print(f'Data for geofeature {dataname} is not available in .PICKLE or .tif format')
                break

            if dataname != "altitude":
                # convert the raster into the same dimensions as the training data (rasterdim)
                data = convert_raster(data, rasterdim)
                for time in range(shape[0]):
                    for layer in range(shape[1]):
                        geofeatures[time, layer, :, :, idx] = data

            else:
                # correct the altitude for each layer
                for layer in range(shape[1]):
                    layerdat = data + np.full(shape=data.shape, fill_value=layer*4)
                    for time in range(shape[0]):
                        geofeatures[time, layer, :, :, idx] = layerdat

    return geofeatures, geolist


def convert_raster(data, newdim):
    newraster = np.zeros(shape=newdim)

    istep = np.floor(data.shape[0] / newdim[0])
    jstep = np.floor(data.shape[1] / newdim[1])

    for i in range(0, newdim[0]):
        for j in range(0, newdim[1]):
            newraster[i, j] = np.mean(data[int(i*istep):int(i*istep+istep+1), int(j*jstep):int(j*jstep+jstep+1)])

    return newraster

