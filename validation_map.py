import os
import argparse
import rasterio
import numpy as np
from utils import roundTime, extract_times, manhatten_distance, moving_average, extract_surfacetemps
from irradiation import irradiationmap
from scipy import signal
from warnings import warn, catch_warnings, simplefilter
from netCDF4 import Dataset
from pandas import read_csv, to_datetime, DataFrame


def load_data(palmpath, stationdata, res):
    """
    Extract time, boundary coordinates and stations within this boundary from PALM file
    palmpath: relative path from WD to PALM file
    """
    if os.path.isfile(f'{palmpath.split(".nc")[0]}_temps.pickle'):
        print('Loading stored data')
        temps = np.load(f'{palmpath.split(".nc")[0]}_temps.pickle', allow_pickle=True)
        times = np.load(f'{palmpath.split(".nc")[0]}_times.pickle', allow_pickle=True)
        boundary = np.load(f'{palmpath.split(".nc")[0]}_boundary.pickle', allow_pickle=True)
        stations = np.load(f'{palmpath.split(".nc")[0]}_stations.pickle', allow_pickle=True)

    else:
        print(f'Loading PALM file {palmpath.split("/")[-1]}')
        try:
            palmfile = Dataset(os.path.join(os.getcwd(), palmpath), "r", format="NETCDF4")
        except FileNotFoundError as e:
            warn(f'File {palmpath.split("/")[-1]} not found, check file path and try again')
            raise e
        print(f'   extracting times')
        times = np.array(extract_times(to_datetime(palmfile.origin_time), palmfile['time']))

        print('   extracting surface temperatures')
        temps = extract_surfacetemps(palmfile['theta'])

        print('   determining boundary')
        boundary = np.zeros(shape=(2, 2))
        boundary[0, 0] = palmfile.origin_x
        boundary[0, 1] = boundary[0, 0] + palmfile.dimensions['x'].size * res
        boundary[1, 0] = palmfile.origin_y
        boundary[1, 1] = boundary[1, 0] + palmfile.dimensions['y'].size * res

        print('   identifying stations within the boundary')
        stations = stations_loc(boundary, stationdata)

        try:
            temps.dump(f'{palmpath.split(".nc")[0]}_temps.pickle')
            times.dump(f'{palmpath.split(".nc")[0]}_times.pickle')
            boundary.dump(f'{palmpath.split(".nc")[0]}_boundary.pickle')
            stations.dump(f'{palmpath.split(".nc")[0]}_stations.pickle')
        except OverflowError:
            warn('Data larger than 4GiB and cannot be serialised for saving.')

    return temps, times, boundary, stations


def stations_loc(boundary, stationdata):
    stationscsv = read_csv(stationdata, delimiter=";")
    stations = np.empty(shape=(0, 3))
    for _, row in stationscsv.iterrows():
        if boundary[0, 0] <= int(row["CH_E"])-2000000 <= boundary[0, 1] and boundary[1, 0] <= int(row["CH_N"])-1000000 <= boundary[1, 1]:
            stations = np.append(stations, [[row["stationid_new"],
                                             int(row["CH_E"])-2000000,
                                             int(row["CH_N"])-1000000]], axis=0)
    return stations


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


def generate_geomap(geopath, boundary, shape, geofeaturelist, convs, res, sigma=3):
    geomaps = np.zeros(shape=(shape[0], len(geofeaturelist) * (len(convs) + 1), shape[1], shape[2]))
    print('\nGenerating Geofeatures')
    for idx, geofeature in enumerate(geofeaturelist):
        print(f'{geofeature}...')
        featuremap = rasterio.open(os.path.join(geopath, f'{geofeature}.tif'))
        originlat = featuremap.meta['transform'][5]  # gives northern boundary
        originlon = featuremap.meta['transform'][2]  # gives western boundary
        featuremap = featuremap.read()
        featuremap[featuremap == -9999] = 0

        if boundary[0, 0] < originlon or originlon + featuremap.shape[1] < boundary[0, 1] or \
                originlat < boundary[1, 0] or originlat-featuremap.shape[2] > boundary[1, 1]:
            warn(f'geofeature map {geofeature} incomplete', Warning)
            raise ValueError

        lons = [int(np.round(boundary[0, 0] - originlon)), int(np.round(boundary[0, 1] - originlon))]
        lats = [int(np.round(originlat - boundary[1, 1])), int(np.round(originlat - boundary[1, 0]))]

        # geomaps[:, idx * len(geofeaturelist) - 1, :, :] = res_adjustment(featuremap[0, lons[0]:lons[1],
        #                                                                  lats[0]:lats[1]], res)
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

    if to_datetime(measurementfile.iloc[0, 0]).tz_localize('UTC') > times[-1] or \
            to_datetime(measurementfile.iloc[-1, 0]).tz_localize('UTC') < times[0]:
        return None

    # round times in csv file for matching
    for idx, row in measurementfile.iterrows():
        measurementfile.iloc[idx, 0] = roundTime(to_datetime(row['datetime'])).tz_localize('UTC')

    measurements = []
    for time in times:
        incsv = measurementfile['datetime'] == time
        if np.sum(incsv) == 0:
            measurements.append(0)
        else:
            measurements.append(measurementfile.loc[incsv].iloc[0, 1])

    # fill in empty spots using neighbouring values, catches all possible cases
    for idx, mes in enumerate(measurements):
        if mes != 0:
            continue
        elif idx == 0 and mes == 0:
            measurements[idx] = measurements[idx + 1]
        elif idx == len(measurements) - 1 and mes == 0:
            measurements[idx] = measurements[idx - 1]
        elif mes == 0 and measurements[idx - 1] != 0 and measurements[idx + 1] == 0:
            measurements[idx] = measurements[idx - 1]
        elif mes == 0 and measurements[idx - 1] == 0 and measurements[idx + 1] != 0:
            measurements[idx] = measurements[idx + 1]
        elif mes == 0 and measurements[idx - 1] != 0 and measurements[idx + 1] != 0:
            measurements[idx] = (measurements[idx - 1] + measurements[idx + 1]) / 2

    return measurements


def generate_measurementmaps(datapath, stations, times, boundary, purpose):
    statnames = list(stations[:, 0])
    # create empty array for humidities
    measurementmaps = np.zeros(shape=(len(times),
                               int(boundary[0, 1] - boundary[0, 0]),
                               int(boundary[1, 1] - boundary[1, 0])))
    s = 0
    for file in os.listdir(datapath):
        if not file.startswith(purpose):
            continue
        stationname = file.split(".")[0].split("_")[1]
        if stationname not in stations[:, 0]:
            continue

        s += 1 # counter for progress visualisation
        print(f'Extracting {"humidities" if purpose == "humi" else "temperatures"} '
              f'({s}/{len(stations[:, 0])}) for {stationname} ')

        measurements = extract_measurement(os.path.join(datapath, file), times)
        if measurements is None:
            warn(f'Station {stationname} does not have any {"humidity" if purpose == "humi" else "temperature"} '
                 f'measurements for this time period, skipping station', Warning)
            continue
        try:
            idx = statnames.index(stationname)
            idxlon = int(int(stations[idx, 1]) - boundary[0, 0])
            idxlat = int(int(stations[idx, 2]) - boundary[1, 0])
            measurementmaps[:, idxlon, idxlat] = measurements
        except Exception:
            warn(f'Adding humidities for station {stationname} failed, skipping this station.')
            continue

    return measurementmaps


def generate_features(datapath, geopath, stations, boundary, times, resolution):
    print('Temperatures....................')
    if 'movingaverage.pickle' not in os.listdir(os.getcwd()):
        # TEMPERATURE MEASUREMENTS
        if 'temps_manhattan.pickle' not in os.listdir(os.getcwd()):
            temps = generate_measurementmaps(datapath, stations, times, boundary, 'temp')
            print('Adjusting temperature map resolution')
            # temps = res_adjustment(temps, res)
            print('\nCalculating Manhattan distance for temperature maps')
            temps = manhatten_distance(temps)
            try:
                temps.dump(os.path.join(os.getcwd(), 'temps_manhattan.pickle'))
            except OverflowError:
                warn('Data larger than 4GiB and cannot be serialised for saving.')
        else:
            temps = np.load(os.path.join(os.getcwd(), 'temps_manhattan.pickle'), allow_pickle=True)

        print('Moving average..................')
        # calculate moving average with measured temps
        ma = np.empty(shape=temps.shape)
        for idxs, _ in np.ndenumerate(temps[0, :, :]):
            ma[:, idxs[0], idxs[1]] = moving_average(temps[:, idxs[0], idxs[1]], times)
        del temps

        try:
            ma.dump(os.path.join(os.getcwd(), 'movingaverage.pickle'))
        except OverflowError:
            warn('Data larger than 4GiB and cannot be serialised for saving.')

    else:
        ma = np.load(os.path.join(os.getcwd(), 'movingaverage.pickle'), allow_pickle=True)


    # HUMIDITIY MEASUREMENTS
    print('Humidities..........................')
    if 'humimaps.pickle' not in os.listdir(os.getcwd()):
        humimaps = generate_measurementmaps(datapath, stations, times, boundary, 'humi')
        print('Adjusting humidity map resolution')
        # humimaps = res_adjustment(humimaps, res)
        print('\nCalculating Manhattan distance for humidity maps')
        humimaps = manhatten_distance(humimaps)
        try:
            humimaps.dump(os.path.join(os.getcwd(), 'humimaps.pickle'))
        except OverflowError:
            warn('Data larger than 4GiB and cannot be serialised for saving.')
    else:
        humimaps = np.load(os.path.join(os.getcwd(), 'humimaps.pickle'), allow_pickle=True)

    # GEOFEATURES AND IRRADIATION
    print('Geofeatures.........................')
    if 'irradiation.pickle' not in os.listdir(os.getcwd()):
        if 'geomaps_nores.pickle' not in os.listdir(os.getcwd()):
            geofeaturelist = ["altitude", "buildings", "forests", "pavedsurfaces", "surfacewater", "urbangreen"]
            convs = [10, 30, 100, 200, 500]
            geomaps = generate_geomap(geopath, boundary, humimaps.shape, geofeaturelist, convs, resolution)
        try:
            geomaps.dump(os.path.join(os.getcwd(), 'geomaps_nores.pickle'))
        except OverflowError:
            warn('Data larger than 4GiB and cannot be serialised for saving.')
        else:
            geomaps = np.load(os.path.join(os.getcwd(), 'geomaps_nores.pickle'), allow_pickle=True)

        irradiation = irradiationmap(boundary, times, geomaps[0, 0, :, :])
        # irradiation = res_adjustment(irradiation, res)
        try:
            irradiation.dump(os.path.join(os.getcwd(), 'irradiation.pickle'))
        except OverflowError:
            warn('Data larger than 4GiB and cannot be serialised for saving.')
    else:
        irradiation = np.load(os.path.join(os.getcwd(), 'irradiation.pickle'), allow_pickle=True)
        geomaps = np.load(os.path.join(os.getcwd(), "geomaps_nores.pickle"), allow_pickle=True)

    # if 'geomaps.pickle' not in os.listdir(os.getcwd()):
    #     if 'geomaps_nores.pickle' not in os.listdir(os.getcwd()):
    #         geofeaturelist = ["altitude", "buildings", "forests", "pavedsurfaces", "surfacewater", "urbangreen"]
    #         convs = [10, 30, 100, 200, 500]
    #         geomaps = generate_geomap(geopath, boundary, humimaps.shape, geofeaturelist, convs, resolution)
    #         geomaps.dump(os.path.join(os.getcwd(), 'geomaps_nores.pickle'))
    #     else:
    #         geomaps = np.load(os.path.join(os.getcwd(), 'geomaps_nores.pickle'), allow_pickle=True)
    #
    #     geomaps = res_adjustment(geomaps, resolution)
    #     geomaps.dump(os.path.join(os.getcwd(), 'geomaps.pickle'))
    # else:
    #     geomaps = np.load(os.path.join(os.getcwd(), 'geomaps.pickle'), allow_pickle=True)

    # separation of time and datetime
    datetime_full = times.copy()
    for idx, time in enumerate(times):
        t = times.tz_localize(None)
        times[idx] = t.time()
        datetime_full[idx] = t
    return datetime_full, times, humimaps, geomaps, irradiation, ma


def create_dataset(palmpath, savepath, stationdata, datapath, geopath, resolution):
    palmtemps, times, boundary, stations = load_data(palmpath, stationdata, resolution)
    datetime, times, humis, geo, rad, ma = generate_features(datapath, geopath, stations, boundary,
                                                             times, resolution)
    datetime = np.repeat(datetime, geo.shape[2]*geo.shape[3])
    times = np.repeat(times, geo.shape[2]*geo.shape[3])

    print('Creating Dataset....................')
    df = DataFrame({
        'datetime': datetime, 'time': times, 'altitude': np.ravel(geo[:, 0, :, :]),
        'buildings': np.ravel(geo[:, 1, :, :]), 'buildings_10': np.ravel(geo[:, 2, :, :]),
        'buildings_30': np.ravel(geo[:, 3, :, :]), 'buildings_100': np.ravel(geo[:, 4, :, :]),
        'buildings_200': np.ravel(geo[:, 5, :, :]), 'buildings_500': np.ravel(geo[:, 6, :, :]),
        'forests': np.ravel(geo[:, 7, :, :]), 'forests_10': np.ravel(geo[:, 8, :, :]),
        'forests_30': np.ravel(geo[:, 9, :, :]), 'forests_100': np.ravel(geo[:, 10, :, :]),
        'forests_200': np.ravel(geo[:, 11, :, :]), 'forests_500': np.ravel(geo[:, 12, :, :]),
        'pavedsurfaces': np.ravel(geo[:, 13, :, :]), 'pavedsurfaces_10': np.ravel(geo[:, 14, :, :]),
        'pavedsurfaces_30': np.ravel(geo[:, 15, :, :]), 'pavedsurfaces_100': np.ravel(geo[:, 16, :, :]),
        'pavedsurfaces_200': np.ravel(geo[:, 17, :, :]), 'pavedsurfaces_500': np.ravel(geo[:, 18, :, :]),
        'surfacewater': np.ravel(geo[:, 19, :, :]), 'surfacewater_10': np.ravel(geo[:, 20, :, :]),
        'surfacewater_30': np.ravel(geo[:, 21, :, :]), 'surfacewater_100': np.ravel(geo[:, 22, :, :]),
        'surfacewater_200': np.ravel(geo[:, 23, :, :]), 'surfacewater_500': np.ravel(geo[:, 24, :, :]),
        'urbangreen': np.ravel(geo[:, 25, :, :]), 'urbangreen_10': np.ravel(geo[:, 26, :, :]),
        'urbangreen_30': np.ravel(geo[:, 27, :, :]), 'urbangreen_100': np.ravel(geo[:, 28, :, :]),
        'urbangreen_200': np.ravel(geo[:, 29, :, :]), 'urbangreen_500': np.ravel(geo[:, 30, :, :]),
        'humidity': np.ravel(humis), 'irradiation': np.ravel(rad), 'moving_average': np.ravel(ma),
        'temperature': np.ravel(palmtemps)
    })

    domain = palmpath.split('_')[-2]
    filename = f'validation_dataset_{datetime[0].year}{datetime[0].month}{datetime[0].day}.{datetime[0].hour}_' \
               f'{datetime[-1].year}{datetime[-1].month}{datetime[-1].day}.{datetime[-1].hour}_Domain-{domain}_' \
               f'{geo.shape[2]}x{geo.shape[3]}.csv'
    savepath = os.path.join(savepath, filename)

    print('Saving Dataset......................')
    df.to_csv(savepath, sep=';', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--palmpath', type=str, default='Data/mb_8_multi_stations_LCZ_masked_N11_M01.00m.nc',
                        help='relative path to the PALM file to be used for validation')
    parser.add_argument('--resolution', type=int, default=32, help='resolution of the PALM file in metres')
    parser.add_argument('--datapath', type=str, default='Messdaten/Daten_Meteoblue',
                        help='relative path to measurement data')
    parser.add_argument('--geopath', type=str, default='Data/geodata',
                        help='relative path to spatial / geographic data')
    parser.add_argument('--stationdata', type=str, default='Messdaten/stations.csv',
                        help='relative path to .csv file containing station information')
    parser.add_argument('--savepath', type=str, default='Data/VALIDATION', help='relative path to save folder')

    args = parser.parse_args()

    create_dataset(os.path.join(os.getcwd(), args.palmpath),
                   os.path.join(os.getcwd(), args.savepath),
                   os.path.join(os.getcwd(), args.stationdata),
                   os.path.join(os.getcwd(), args.datapath),
                   os.path.join(os.getcwd(), args.geopath),
                   args.resolution)

