import os
import numpy as np
from utils import lv03_to_lv95, roundTime, manhatten_distance, reduce_resolution, remove_emptytimes
from warnings import catch_warnings, warn, simplefilter
import rasterio
from irradiation import irradiationmap
from scipy import signal
from pandas import read_csv, to_datetime, DataFrame


def in_window(timeframe: list, times: list):
    afterStart = (timeframe[0] <= times).to_list()
    beforeEnd = (times <= timeframe[1]).to_list()
    return [a and b for a,b in zip(afterStart, beforeEnd)]


def res_adjustment(featuremap: np.ndarray, res: int):
    featuremap[featuremap == 0] = np.NaN
    # Case for geofeatures
    if len(featuremap.shape) == 4:
        newmap = np.empty(shape=(featuremap.shape[0], featuremap.shape[1],
                                 int(np.round(featuremap.shape[2] / res)),
                                 int(np.round(featuremap.shape[3] / res))))
        for time in range(featuremap.shape[0]):
            for feature in range(featuremap.shape[1]):
                for idxs, _ in np.ndenumerate(newmap[time, feature, :, :]):
                    with catch_warnings():
                        simplefilter("ignore", category=RuntimeWarning)
                        newmap[time, feature, idxs[0], idxs[1]] = np.nanmean(featuremap[time, feature,
                                                                             idxs[0] * res: idxs[0] * res + res - 1,
                                                                             idxs[1] * res: idxs[1] * res + res - 1])
    # case for temp / humi / irrad
    elif len(featuremap.shape) == 3:
        newmap = np.empty(shape=(featuremap.shape[0],
                                 int(np.round(featuremap.shape[1] / res)),
                                 int(np.round(featuremap.shape[2] / res))))
        for time in range(featuremap.shape[0]):
            for idxs, _ in np.ndenumerate(newmap):
                newmap[time, idxs[1], idxs[2]] = np.nanmean(
                    featuremap[time, idxs[0] * res:idxs[0] * res + res - 1, idxs[1] * res:idxs[1] * res + res - 1])
    # simple case
    elif len(featuremap.shape) == 2:
        newmap = np.empty(shape=(int(np.round(featuremap.shape[0] / res)), int(np.round(featuremap.shape[1] / res))))
        for idxs, _ in np.ndenumerate(newmap):
            newmap[idxs[0], idxs[1]] = np.nanmean(
                featuremap[idxs[0] * res:idxs[0] * res + res - 1, idxs[1] * res:idxs[1] * res + res - 1])
    else:
        warn(f'Map cannot be adjusted to resolution {res}, check shape and size.', Warning)
        raise ValueError

    # convert NaNs back to 0 for use in further functions
    newmap[newmap == np.NaN] = 0

    return newmap


def generate_geomap(geopath: str, boundary: dict, shape: tuple, geofeaturelist: list, 
                    convs: list, sigma:int = 3, resolution=16):
    '''
    Generates a geomap from the given geofeatures and convolutions.
    :param geopath: path to geofeatures
    :param boundary: boundary of the map
    :param shape: shape of the map, already reduced to the resolution of the PALM simulation files
    :param geofeaturelist: list of geofeatures
    :param convs: list of convolutions
    :param sigma: sigma for gaussian convolution
    :return: geomap with shape (n_geofeatures * n_convolutions+1, height of humimap, width of humimap)
    '''
    # geomaps is in the shape which considers the resolution (i.e., it is already reduced)
    geomaps = np.zeros(shape=(len(geofeaturelist) * (len(convs) + 1), shape[0], shape[1], shape[2]))
    print('\nGenerating Geofeatures')
    for idx, geofeature in enumerate(geofeaturelist):
        print(f'    {geofeature}...')
        if geofeature == 'altitude':
            geofeature_idx = 0
        else:
            geofeature_idx = idx*(len(convs)+1)-5

        #* issue with geofeatures is not from loading
        # load feature map, get border and check that it is complete. Any negative values are assign NaN
        featuremap, geo_border = load_geomap(os.path.join(geopath, f'{geofeature}.tif'))
        featuremap[featuremap < 0] = np.nan
        if boundary['CH_E'] > geo_border['E'] or boundary['CH_W'] < geo_border['W'] or \
                boundary['CH_S'] < geo_border['S'] or boundary['CH_N'] > geo_border['N']:
            warn(f'geofeature map {geofeature} incomplete', Warning)
            print(f'Border geomap:     N {int(geo_border["N"])}, S {int(geo_border["S"])}, W {int(geo_border["W"])}, E {int(geo_border["E"])}')
            print(f'Border featuremap: N {int(boundary["CH_N"])}, S {int(boundary["CH_S"])}, W {int(boundary["CH_W"])}, E {int(boundary["CH_E"])}')
            raise ValueError

        # indices locating the PALM simulation map in the geofeature map
        palm_geoidxs = {'N': int(np.round(geo_border['N'] - boundary['CH_N'])),
                        'S': int(np.round(geo_border['N'] - boundary['CH_S'])),
                        'W': int(np.round(boundary['CH_W'] - geo_border['W'])),
                        'E': int(np.round(boundary['CH_E'] - geo_border['W']))}
        
        #* there is data within the geomap at this point
        # add uncovoluted feature map to geomaps after adjusting to the proper resolution
        geomaps[geofeature_idx, :, :, :] = reduce_resolution(featuremap[palm_geoidxs['N']:palm_geoidxs['S'], 
                                                                        palm_geoidxs['W']:palm_geoidxs['E']],
                                                                resolution)
        # print(f'geomap min/max = {np.min(geomaps[geofeature_idx, 0, :, :])}/{np.max(geomaps[idx*(len(convs)+1), 0, :, :])}')
        
        
        if geofeature != 'altitude':
            # create padded feature map for convolutions - full resolution
            padded_featuremap = np.empty(shape=((shape[1]*resolution + np.max(convs)), 
                                                (shape[2]*resolution + np.max(convs))))
            padded_featuremap[:] = np.NaN

            # calculate the amount that padding exceeds geofeature map per edge
            #* negative values indicate that padding exceeds the map in that direction
            padding_over = {'N': int(palm_geoidxs['N'] - (np.max(convs)/2)),
                    'S': featuremap.shape[0] - int(palm_geoidxs['S'] + (np.max(convs)/2)),
                    'W': int(palm_geoidxs['W'] - (np.max(convs)/2)),
                    'E': featuremap.shape[1] - int(palm_geoidxs['E'] + (np.max(convs)/2))}

            for key in padding_over.keys():
                if padding_over[key] < 0:
                    padding_over[key] = abs(padding_over[key])
                else: 
                    padding_over[key] = 0

            # two index sets: 1. start and end of padded featuremap covered by the geomap (assumption is the whole padded map)
            #                 2. indices for the start and end of the padded featuremap within the geomap (used to extract data)
            filled_paddedmap = {'N': 0,
                                'S': padded_featuremap.shape[0]-1,
                                'W': 0,
                                'E': padded_featuremap.shape[1]-1}
            geomapidxs_padded = {'N': int(palm_geoidxs['N'] - (np.max(convs)/2)),
                                'S': int(palm_geoidxs['S'] + (np.max(convs)/2) -1),
                                'W': int(palm_geoidxs['W'] - (np.max(convs)/2)),
                                'E': int(palm_geoidxs['E'] + (np.max(convs)/2) -1)}
            if padding_over['N']:
                geomapidxs_padded['N'] = 0
                filled_paddedmap['N'] = padding_over['N']
            if padding_over['S']:
                geomapidxs_padded['S'] = featuremap.shape[0]
                filled_paddedmap['S'] = padded_featuremap.shape[0] - padding_over['S']
            if padding_over['W']:
                geomapidxs_padded['W'] = 0
                filled_paddedmap['W'] = padding_over['W']
            if padding_over['E']:
                geomapidxs_padded['E'] = 0
                filled_paddedmap['E'] = padded_featuremap.shape[1] - padding_over['E']

            # fill geodata into padded map using the indices
            padded_featuremap[filled_paddedmap['N']:filled_paddedmap['S']+1, 
                            filled_paddedmap['W']:filled_paddedmap['E']+1] = featuremap[geomapidxs_padded['N']:geomapidxs_padded['S']+1,
                                                                                        geomapidxs_padded['W']:geomapidxs_padded['E']+1]

            # reduce padded_featuremap resolution
            padded_featuremap = reduce_resolution(padded_featuremap, resolution=resolution)
            # padded_featuremap[padded_featuremap == 0] = np.nan
            # add convoluted feature maps to geomaps
            print('    convolutions...')
            max_conv_pad = (np.max(convs)/resolution)/2
            array_idxs = {'N': int(0+max_conv_pad), 
                        'S': int(padded_featuremap.shape[0]-max_conv_pad),
                        'W': int(0+max_conv_pad), 
                        'E': int(padded_featuremap.shape[1]-max_conv_pad)}
            for conv_idx, conv in enumerate(convs):
                conv_pad = (conv/2)/resolution
                print(f'      conv {conv}')
                # empty array for convolutions in reduced size
                conv_array = np.zeros(shape=(shape[1], shape[2]))
                kernel = signal.gaussian(conv/resolution + 1, std=sigma) # type: ignore
                kernel = np.outer(kernel, kernel)

                # fill by column in row
                
                for lat in range(0, conv_array.shape[0]):
                    for lon in range(0, conv_array.shape[1]): 
                        geo_array = padded_featuremap[int(lat+array_idxs['N']-conv_pad): int(lat+array_idxs['N']+conv_pad)+1,
                                                    int(lon+array_idxs['W']-conv_pad): int(lon+array_idxs['W']+conv_pad)+1]
                        if geo_array.shape != kernel.shape:
                            print(geo_array.shape)
                            print('padded feature map excerpt:')
                            print(f'    {int(lat+array_idxs["N"]-conv_pad)}:{int(lat+array_idxs["N"]+conv_pad)+1}')
                            print(f'    {int(lon+array_idxs["W"]-conv_pad)}:{int(lon+array_idxs["W"]+conv_pad)+1}')
                            print('Cell:')
                            print(f'    lat: {lat}/{conv_array.shape[0]}\n    lon: {lon}/{conv_array.shape[1]}')
                        gaussian_array = geo_array * kernel
                        conv_array[lat, lon] = np.nanmean(gaussian_array)
                if np.sum(np.isnan(conv_array)) != 0:
                    print(np.sum(np.isnan(conv_array)))
                geomaps[geofeature_idx+conv_idx+1, :, :, :] = conv_array
    return geomaps


def load_geomap(path):
    featuremap = rasterio.open(path)
    geo_N = featuremap.meta['transform'][5]  # gives northern boundary
    geo_W = featuremap.meta['transform'][2]  # gives western boundary
    # transform to LV95 coordinates
    geo_N, geo_W = lv03_to_lv95(geo_N, geo_W)
    geo_S = geo_N - featuremap.shape[1]
    geo_E = geo_W + featuremap.shape[0]
    featuremap = featuremap.read()[0, :, :]
    featuremap[featuremap < 0] = 0
    borders = {'N': geo_N, 'S': geo_S, 'W': geo_W, 'E': geo_E}
    return featuremap, borders


def extract_measurement(datapath: str, times: list):
    measurementfile: DataFrame = read_csv(datapath, delimiter=";")

    # catch both tz aware and tz naive
    try:
        if to_datetime(str(measurementfile.iloc[0, 0])).tz_localize('UTC') > times[-1] or \
                to_datetime(str(measurementfile.iloc[-1, 0])).tz_localize('UTC') < times[0]:
            return None
    except TypeError or AttributeError:
        if to_datetime(str(measurementfile.iloc[0, 0])) > times[-1] or \
            to_datetime(str(measurementfile.iloc[-1, 0])) < times[0]:
            return None

    # round times in csv file for matching
    for idx, row in measurementfile.iterrows():
        measurementfile['datetime'].iloc[idx] = roundTime(to_datetime(row['datetime'])).tz_localize('UTC') # type: ignore

    measurements = []
    for time in times:
        incsv = measurementfile['datetime'] == time
        if np.sum(incsv) == 0:
            measurements.append(0)
        else:
            measurements.append(measurementfile.loc[incsv].iloc[0, 1])

    # check that measurements exist
    if np.sum(measurements) == 0:
        return None

    return measurements


def generate_measurementmaps(datapath: str, stations: dict, times: list, boundary: dict, 
                             resolution: int, purpose: str):
    # TODO: create a logging function to log station within the dataset and those without data at these times.
    # create empty array for humidities
    measurementmaps = np.zeros(shape=(len(times),
                               int((boundary['CH_E'] - boundary['CH_W']) / resolution),
                               int((boundary['CH_N'] - boundary['CH_S']) / resolution)))
    s = 0
    for file in os.listdir(datapath):
        if not file.startswith(purpose):
            continue
        stationname = file.split(".")[0].split("_")[1]
        if stationname not in stations.keys():
            continue

        s += 1 # counter for progress visualisation
        print(f'Extracting {"humidities" if purpose == "humi" else "temperatures"} '
              f'({s}/{len(stations.keys())}) for {stationname} ')

        # print(f'generate_measurementmaps: {times.shape}')
        measurements = extract_measurement(os.path.join(datapath, file), times)
        if measurements is None:
            warn(f'Station {stationname} does not have any {"humidity" if purpose == "humi" else "temperature"} '
                 f'measurements for this time period, skipping station', Warning)
            continue
        try:
            # idx = statnames.index(stationname)
            idxlat = int((int(stations[stationname]['lat']) - boundary['CH_S'])/resolution)
            idxlon = int((int(stations[stationname]['lon']) - boundary['CH_W'])/resolution)
            measurementmaps[:, idxlon, idxlat] = measurements
        except Exception:
            warn(f'Adding {"humidity" if purpose == "humi" else "temperature"} for station {stationname} failed,' 
                 f'skipping this station.')
            continue

    if np.sum(measurementmaps) == 0:
        return None, None

    measurementmaps, times = remove_emptytimes(measurementmaps, times) # type: ignore

    return measurementmaps, times


def tempgen(datapath: str, stations: dict, times: list, boundary: dict, resolution: int):
    temps, times = generate_measurementmaps(datapath, stations, times, boundary, resolution, 'temp') # type: ignore
    if temps is None:
        print(f'N Stations: {len(stations)}')
        warn('No temperature measurements found for the given boundary and times')
        raise ValueError
    # print('Adjusting temperature map resolution')
    # temps = res_adjustment(temps, res)
    return temps, times


def humigen(datapath: str, stations: dict, times: list, boundary: dict, resolution: int):
    humimaps, times = generate_measurementmaps(datapath, stations, times, boundary, resolution, 'humi') # type: ignore
    print('Adjusting humidity map resolution')
    # humimaps = res_adjustment(humimaps, res)
    print('\nCalculating Manhattan distance for humidity maps')
    humimaps = manhatten_distance(humimaps) #type: ignore
    try:
        humimaps.dump(os.path.join(os.getcwd(), 'humimaps.pickle'))
    except OverflowError:
        warn('Data larger than 4GiB and cannot be serialised for saving.')

    return humimaps, times


def geogen(geopath: str, boundary: dict, humimaps: np.ndarray):
    geofeaturelist = ["altitude", "buildings", "forests", "pavedsurfaces", "surfacewater", "urbangreen"]
    #! fixed to be meaningful for a resolution of 16m, original convs are [10, 30, 100, 200, 500]
    #* convs must always be even when divided by the resolution (considers either side)
    convs = [32, 48, 112, 208, 512]
    geomaps = generate_geomap(geopath, boundary, humimaps.shape, geofeaturelist, convs)
    try:
        geomaps.dump(os.path.join(os.getcwd(), 'geomaps_nores.pickle'))
    except OverflowError:
        warn('Data larger than 4GiB and cannot be serialised for saving.')

    return geomaps


def radgen(boundary: dict, geomaps: np.ndarray, times: list):
    print('    irradiation.....................')
    irrad = irradiationmap(boundary, times, geomaps[0, 0, :, :])
    return irrad