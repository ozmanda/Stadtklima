import os
import numpy as np
from utils import lv03_to_lv95, roundTime, manhatten_distance, moving_average, remove_emptytimes
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
                    convs: list, sigma:int = 3):
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
    geomaps = np.zeros(shape=(len(geofeaturelist) * (len(convs) + 1), shape[0], shape[1], shape[2]))
    print('\nGenerating Geofeatures')
    for idx, geofeature in enumerate(geofeaturelist):
        print(f'    {geofeature}...')

        # load feature map, get border and check that it is complete
        featuremap, geo_border = load_geomap(os.path.join(geopath, f'{geofeature}.tif'))
        if boundary['CH_E'] > geo_border['E'] or boundary['CH_W'] < geo_border['W'] or \
                boundary['CH_S'] < geo_border['S'] or boundary['CH_N'] > geo_border['N']:
            warn(f'geofeature map {geofeature} incomplete', Warning)
            print(f'Border geomap:     N {int(geo_border['N'])}, S {int(geo_border['S'])}, W {int(geo_border['W'])}, E {int(geo_border['E'])}')
            print(f'Border featuremap: N {int(boundary["CH_N"])}, S {int(boundary["CH_S"])}, W {int(boundary["CH_W"])}, E {int(boundary["CH_E"])}')
            raise ValueError
        
        # create padded feature map for convolutions
        padded_featuremap = np.empty(shape=(featuremap.shape[0] + np.max(convs), 
                                            featuremap.shape[1] + np.max(convs)))
        padded_featuremap[:] = np.NaN

        # indices locating the PALM simulation map in the geofeature map
        palm_geoidxs = {'N': int(np.round(geo_border['N'] - boundary['CH_N'])),
                        'S': int(np.round(geo_border['N'] - boundary['CH_S'])),
                        'W': int(np.round(boundary['CH_W'] - geo_border['W'])),
                        'E': int(np.round(boundary['CH_E'] - geo_border['W']))}
        
        # add uncovoluted feature map to geomaps
        geomaps[idx*(len(convs)+1), :, :, :] = featuremap[palm_geoidxs['N']:palm_geoidxs['S'], 
                                                          palm_geoidxs['W']:palm_geoidxs['E']]
        
        # indices locating padded feature map in the geofeature map
        #* can be negative or larger than the geofeature map, indicating that padding is required in that dimension
        padded_geoidxs = {'N': palm_geoidxs['N'] - (np.max(convs)/2),
                          'S': palm_geoidxs['S'] + (np.max(convs)/2),
                          'W': palm_geoidxs['W'] - (np.max(convs)/2),
                          'E': palm_geoidxs['E'] + (np.max(convs)/2)}
        
        # indices indicating where the geofeature map is located in the padded map (starting with the assumption
        # that the geofeature map is sufficient and no padding is needed)
        featuremap_paddedidxs = {'N': 0,
                                  'S': padded_featuremap.shape[0]-1,
                                  'W': 0,
                                  'E': padded_featuremap.shape[1]-1}
        
        #* if padding exceeds the geofeature map, the featuremap_paddedidxs must be adjusted
        for i in ['N', 'W']:
            if padded_geoidxs[i] < 0:
                featuremap_paddedidxs[i] = abs(padded_geoidxs[i])
                padded_geoidxs[i] = 0
        if padded_geoidxs['S'] > featuremap.shape[0]:
            padding = padded_geoidxs['S'] - featuremap.shape[0]
            featuremap_paddedidxs['S'] = padded_featuremap.shape[0] - padding
            padded_geoidxs['S'] = featuremap.shape[0]
        if padded_geoidxs['E'] > featuremap.shape[1]:
            padding = padded_geoidxs['E'] - featuremap.shape[1]
            featuremap_paddedidxs['E'] = padded_featuremap.shape[1] - padding
            padded_geoidxs['E'] = featuremap.shape[1]

        # fill padded feature map with values from feature map
        padded_featuremap[featuremap_paddedidxs['N']:featuremap_paddedidxs['S'],
                          featuremap_paddedidxs['W']:featuremap_paddedidxs['E']] = featuremap[padded_geoidxs['N']:padded_geoidxs['S'],
                                                                                               padded_geoidxs['W']:padded_geoidxs['E']]

        # add convoluted feature maps to geomaps
        print('    convolutions...')
        for conv_idx, conv in enumerate(convs):
            kernel = signal.gaussian(conv + 1, std=sigma) # type: ignore
            kernel = np.outer(kernel, kernel)

            # fill by column in row
            for lon in range(palm_geoidxs['W'], palm_geoidxs['E']+1):
                for lat in range(palm_geoidxs['N'], palm_geoidxs['S']+1):
                    geo_array = padded_featuremap[int(lat - conv / 2):int(lat + conv / 2 + 1),
                                                    int(lon - conv / 2):int(lon + conv / 2 + 1)]
                    gaussian_array = geo_array * kernel
                    geomaps[idx * len(geofeaturelist)+conv_idx+1, :, lon, lat] = np.nanmean(gaussian_array)
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
    #* convs must always be even, 
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
    # irradiation = res_adjustment(irradiation, res)
    try:
        irrad.dump(os.path.join(os.getcwd(), 'irradiation.pickle'))
    except OverflowError:
        warn('Data larger than 4GiB and cannot be serialised for saving.')
    return irrad