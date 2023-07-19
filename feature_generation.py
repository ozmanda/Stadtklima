import os
import numpy as np
from utils import lv03_to_lv95, roundTime, manhatten_distance, moving_average, remove_emptytimes
from warnings import catch_warnings, warn, simplefilter
import rasterio
from irradiation import irradiationmap
from scipy import signal
from pandas import read_csv, to_datetime


def in_window(timeframe, times):
    afterStart = (timeframe[0] <= times).to_list()
    beforeEnd = (times <= timeframe[1]).to_list()
    return [a and b for a,b in zip(afterStart, beforeEnd)]


def res_adjustment(featuremap, res):
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


def generate_geomap(geopath, boundary, shape, geofeaturelist, convs, sigma=3):
    geomaps = np.zeros(shape=(shape[0], len(geofeaturelist) * (len(convs) + 1), shape[1], shape[2]))
    print('\nGenerating Geofeatures')
    for idx, geofeature in enumerate(geofeaturelist):
        print(f'    {geofeature}...')
        featuremap = rasterio.open(os.path.join(geopath, f'{geofeature}.tif'))
        originlat = featuremap.meta['transform'][5]  # gives northern boundary
        originlon = featuremap.meta['transform'][2]  # gives western boundary
        # transform to LV95 coordinates
        originlat, originlon = lv03_to_lv95(originlat, originlon)
        featuremap = featuremap.read()
        featuremap[featuremap == -9999] = 0

        if boundary['CH_E'] < originlon or originlon + featuremap.shape[1] < boundary['CH_W'] or \
                originlat < boundary['CH_S'] or originlat-featuremap.shape[2] > boundary['CH_N']:
            warn(f'geofeature map {geofeature} incomplete', Warning)
            print(f'Border geomap: N {originlat}, S {originlat-featuremap.shape[2]}, W {originlon}, E {originlon + featuremap.shape[1]}')
            print(f'Border featuremap: N {boundary["CH_N"]}, S {boundary["CH_S"]}, W {boundary["CH_W"]}, E {boundary["CH_E"]}')
            raise ValueError

        lons = [int(np.round(boundary['CH_E'] - originlon)), int(np.round(boundary['CH_W'] - originlon))]
        lats = [int(np.round(originlat - boundary['CH_N'])), int(np.round(originlat - boundary['CH_S']))]
        print('    convolutions...')
        for idx, conv in enumerate(convs):
            if conv % 2 != 0:
                if conv % 2 == 1:
                    conv -= 1
                else:
                    warn('Convolution sizes must be integer numbers', Warning)

            kernel = signal.gaussian(conv + 1, std=sigma)
            kernel = np.outer(kernel, kernel)

            for lon in range(lons[0], lons[1]):
                for lat in range(lats[0] - lats[1]):
                    idxs = np.array([[int(lat - conv / 2), int(lat + conv / 2 + 1)],
                                     [int(lon - conv / 2), int(lon + conv / 2 + 1)]])
                    geomaps[:, idx * len(geofeaturelist)+conv+1, lon, lat] = np.nanmean(featuremap[0,
                                                                                        idxs[0, 0]:idxs[0, 1],
                                                                                        idxs[1, 0]:idxs[1, 1]] * kernel)
    return geomaps


def extract_measurement(datapath, times):
    measurementfile = read_csv(datapath, delimiter=";")

    # catch both tz aware and tz naive
    try:
        # print(f'{to_datetime(measurementfile.iloc[0, 0]).tz_localize("UTC")} > {times[-1]}')
        # print(f'{to_datetime(measurementfile.iloc[-1, 0]).tz_localize("UTC")} < {times[0]}')
        if to_datetime(measurementfile.iloc[0, 0]).tz_localize('UTC') > times[-1] or \
                to_datetime(measurementfile.iloc[-1, 0]).tz_localize('UTC') < times[0]:
            return None
    except TypeError or AttributeError:
        if to_datetime(measurementfile.iloc[0, 0]) > times[-1] or to_datetime(measurementfile.iloc[-1, 0]) < times[0]:
            return None

    # round times in csv file for matching
    # print(measurementfile.iloc[0:10, 0])
    for idx, row in measurementfile.iterrows():
        measurementfile.iloc[idx, 0] = roundTime(to_datetime(row['datetime'])).tz_localize('UTC')
    # print(measurementfile.iloc[0:10, 0])

    measurements = []
    for time in times:
        # print(measurementfile['datetime'])
        incsv = measurementfile['datetime'] == time
        if np.sum(incsv) == 0:
            measurements.append(0)
        else:
            measurements.append(measurementfile.loc[incsv].iloc[0, 1])

    # check that measurements exist
    if np.sum(measurements) == 0:
        return None

    # fill in empty spots using neighbouring values, catches all possible cases
    # for idx, mes in enumerate(measurements):
    #     if mes != 0:
    #         continue
    #     elif idx == 0 and mes == 0:
    #         measurements[idx] = measurements[idx + 1]
    #     elif idx == len(measurements) - 1 and mes == 0:
    #         measurements[idx] = measurements[idx - 1]
    #     elif mes == 0 and measurements[idx - 1] != 0 and measurements[idx + 1] == 0:
    #         measurements[idx] = measurements[idx - 1]
    #     elif mes == 0 and measurements[idx - 1] == 0 and measurements[idx + 1] != 0:
    #         measurements[idx] = measurements[idx + 1]
    #     elif mes == 0 and measurements[idx - 1] != 0 and measurements[idx + 1] != 0:
    #         measurements[idx] = (measurements[idx - 1] + measurements[idx + 1]) / 2

    return measurements


def generate_measurementmaps(datapath, stations, times, boundary, resolution, purpose):
    # create empty array for humidities
    measurementmaps = np.zeros(shape=(len(times),
                               int((boundary['CH_W'] - boundary['CH_E']) / resolution),
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
            idxlon = int((int(stations[stationname]['lon']) - boundary['CH_E'])/resolution)
            measurementmaps[:, idxlon, idxlat] = measurements
        except Exception:
            warn(f'Adding humidities for station {stationname} failed, skipping this station.')
            continue

    if np.sum(measurementmaps) == 0:
        return None, None

    measurementmaps, times = remove_emptytimes(measurementmaps, times)

    return measurementmaps, times


def tempgen(datapath, stations, times, boundary, resolution):
    temps, times = generate_measurementmaps(datapath, stations, times, boundary, resolution, 'temp')
    if temps is None:
        warn('No temperature measurements found for the given boundary and times')
        raise ValueError
    # print('Adjusting temperature map resolution')
    # temps = res_adjustment(temps, res)
    return temps, times


def humigen(datapath, stations, times, boundary, resolution):
    humimaps, times = generate_measurementmaps(datapath, stations, times, boundary, resolution, 'humi')
    print('Adjusting humidity map resolution')
    # humimaps = res_adjustment(humimaps, res)
    print('\nCalculating Manhattan distance for humidity maps')
    # TIMES REDUCED FOR TESTING!!!!
    humimaps = manhatten_distance(humimaps)
    try:
        humimaps.dump(os.path.join(os.getcwd(), 'humimaps.pickle'))
    except OverflowError:
        warn('Data larger than 4GiB and cannot be serialised for saving.')

    return humimaps, times


def geogen(geopath, boundary, humimaps):
    geofeaturelist = ["altitude", "buildings", "forests", "pavedsurfaces", "surfacewater", "urbangreen"]
    convs = [10, 30, 100, 200, 500]
    geomaps = generate_geomap(geopath, boundary, humimaps.shape, geofeaturelist, convs)
    try:
        geomaps.dump(os.path.join(os.getcwd(), 'geomaps_nores.pickle'))
    except OverflowError:
        warn('Data larger than 4GiB and cannot be serialised for saving.')

    return geomaps


def radgen(boundary, geomaps, times):
    print('    irradiation.....................')
    irrad = irradiationmap(boundary, times, geomaps[0, 0, :, :])
    # irradiation = res_adjustment(irradiation, res)
    try:
        irrad.dump(os.path.join(os.getcwd(), 'irradiation.pickle'))
    except OverflowError:
        warn('Data larger than 4GiB and cannot be serialised for saving.')
    return irrad