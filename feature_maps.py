import argparse
import os
from warnings import warn
import pandas as pd
from pandas import read_csv, to_datetime, DataFrame, Timedelta
import numpy as np
from netCDF4 import Dataset
from utils import lv_to_wgs84, wgs84_to_lv, extract_times, extract_surfacetemps, roundTime, manhatten_distance, moving_average

# GLOBAL VARIABLES
geofeatures = ['altitude', 'buildings', 'buildings_10', 'buildings_30', 'buildings_100', 'buildings_200',
               'buildings_500', 'forests', 'forests_10', 'forests_30', 'forests_100', 'forests_200', 'forests_500',
               'pavedsurfaces', 'pavedsurfaces_10', 'pavedsurfaces_30', 'pavedsurfaces_100', 'pavedsurfaces_200',
               'pavedsurfaces_500', 'surfacewater', 'surfacewater_10', 'surfacewater_30', 'surfacewater_100',
               'surfacewater_200', 'surfacewater_500', 'urbangreen', 'urbangreen_10', 'urbangreen_30', 'urbangreen_100',
               'urbangreen_200', 'urbangreen_500']


def format_boundaries(boundary):
    """
    Takes boundary definition in ° and transforms it into a dictionary containing LV95 projection coordinates
    boundary: [lat_S, lat_N, lon_E, lon_W]
    """
    assert boundary[0] < boundary[1], 'Northern latitude must be larger than the southern latitude'
    assert boundary[2] < boundary[3], 'Eastern longitude must be larger than the western longitude'
    CH_S, CH_E = wgs84_to_lv(boundary[0], boundary[2], 'lv95')
    CH_N, CH_W = wgs84_to_lv(boundary[1], boundary[3], 'lv95')
    return {'CH_S': CH_S, 'CH_N': CH_N, 'CH_E': CH_E, 'CH_W': CH_W}


def extract_palm_data(palmpath, res):
    """
    Extracts times, temperature and boundary coordinates from PALM file. PALM coordinates are extracted as latitude
    and longitude (WGS84) and converted to LV95 projection coordinates.
    """
    print('Extracting PALM File data....................')
    print('    loading PALM file........................')
    palmfile = Dataset(palmpath, 'r', format='NETCDF4')

    print('    determining boundary.....................')
    CH_S, CH_E = wgs84_to_lv(palmfile.origin_lat, palmfile.origin_lon, 'lv95')
    CH_N = CH_S + palmfile.dimensions['x'].size * res
    CH_W = CH_E + palmfile.dimensions['y'].size * res
    boundary = {'CH_S': CH_S, 'CH_N': CH_N, 'CH_E': CH_E, 'CH_W': CH_W}

    print('   extracting times..........................')
    times = np.array(extract_times(to_datetime(palmfile.origin_time), palmfile['time']))

    print('   extracting surface temperatures...........')
    temps = extract_surfacetemps(palmfile['theta_xy'])

    return boundary, times, temps


def stations_loc(boundary, stationdata):
    print('   identifying stations within the boundary')
    stationscsv = read_csv(stationdata, delimiter=";")
    stations = np.empty(shape=(0, 3))
    for _, row in stationscsv.iterrows():
        if boundary['CH_E'] <= int(row["CH_E"]) <= boundary['CH_W'] and boundary['CH_S'] <= int(row["CH_N"]) <= boundary['CH_N']:
            stations = np.append(stations, [[row["stationid_new"], int(row["CH_E"]), int(row["CH_N"])]], axis=0)
    return stations


def in_window(timeframe, times):
    afterStart = (timeframe[0] <= times).to_list()
    beforeEnd = (times <= timeframe[1]).to_list()
    return [a and b for a,b in zip(afterStart, beforeEnd)]


def time_generation(timeframe):
    times = []
    time = timeframe[0]
    while time < timeframe[1]:
        times.append(time)
        time += Timedelta(minutes=5)
    return times

    
# MEASUREMENT AND FEATURE GENERATION ----------------------------------------------------------------------------------
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
        print(f'{geofeature}...')
        featuremap = rasterio.open(os.path.join(geopath, f'{geofeature}.tif'))
        originlat = featuremap.meta['transform'][5]  # gives northern boundary
        originlon = featuremap.meta['transform'][2]  # gives western boundary
        featuremap = featuremap.read()
        featuremap[featuremap == -9999] = 0

        if boundary['CH_E'] < originlon or originlon + featuremap.shape[1] < boundary['CH_W'] or \
                originlat < boundary['CH_S'] or originlat-featuremap.shape[2] > boundary['CH_N']:
            warn(f'geofeature map {geofeature} incomplete', Warning)
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
                               int(boundary['CH_W'] - boundary['CH_E']),
                               int(boundary['CH_N'] - boundary['CH_S'])))
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
            idxlon = int(int(stations[idx, 1]) - boundary['CH_E'])
            idxlat = int(int(stations[idx, 2]) - boundary['CH_S'])
            measurementmaps[:, idxlon, idxlat] = measurements
        except Exception:
            warn(f'Adding humidities for station {stationname} failed, skipping this station.')
            continue

    return measurementmaps
    
def generate_features(datapath, geopath, stations, boundary, times, resolution=None):
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
            geomaps = generate_geomap(geopath, boundary, humimaps.shape, geofeaturelist, convs)
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

    # separation of time and datetime
    datetime_full = times.copy()
    for idx, time in enumerate(times):
        t = times.tz_localize(None)
        times[idx] = t.time()
        datetime_full[idx] = t
    return datetime_full, times, humimaps, geomaps, irradiation, ma


def add_geos(dict, geos):
    for idx, feature in enumerate(geofeatures):
        dict[feature] = np.ravel(geos[:, idx, :, :])
    return dict


# WRAPPER FUNCTIONS ---------------------------------------------------------------------------------------------------

def generate_featuremaps(type, datapath, geopath, stationinfo, savepath, palmpath=None, res=None, boundary=None, times=None):
    if type == 'validation':
        boundary, temps, times = extract_palm_data(palmpath, res)
    elif type == 'inference':
        boundary = format_boundaries(args.boundary)
        times = time_generation(times)
    else:
        warn('Invalid type passed to feature map generation, only "validation" and "inference" accepted')
        raise ValueError

    # identify stations and generate features within the boundaries based on these stations
    stations = stations_loc(boundary, stationinfo)
    datetimes, times, humis, geo, rad, ma = generate_features(datapath, geopath, stations, boundary, times, res)

    # create time and datetime maps
    datetime_map = np.empty(shape=rad.shape())
    time_map = datetime
    for idx, dt in enumerate(datetimes):
        datetime_map[idx, :, :] = dt
        time_map[idx, :, :] = times[idx]

    # create feature dictionary and then DataFrame
    maps = {'datetime': np.ravel(datetime_map), 'time': np.ravel(time_map)}
    maps = add_geos(maps, geo)
    maps = {**maps, 'humidity': np.ravel(humis), 'irradiation': np.ravel(rad), 'moving average': np.ravel(ma)}
    feature_dataframe = DataFrame(maps)

    # generate filename and save dataset
    filename = f'{type}_dataset_{datetime[0].year}{datetime[0].month}{datetime[0].day}.{datetime[0].hour}_' \
               f'{datetime[-1].year}{datetime[-1].month}{datetime[-1].day}.{datetime[-1].hour}_{boundary["CH_S"]}-' \
               f'{boundary["CH_N"]}_{boundary["CH_E"]}-{boundary["CH_W"]}.csv'
    savepath = os.path.join(savepath, filename)
    feature_dataframe.to_csv(savepath, sep=';', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='Type of feature map to be generated validation (requires PALM file for '
                                               'target temperature) or inference')
    parser.add_argument('--measurementpath', type=str, default='S:/Meteoblue/Data/Messdaten/Daten_Meteoblue',
                        help='Path to station-based measurement data')
    parser.add_argument('--stationinfo', type=str, default='S:/Meteoblue/Data/Messdaten/stations.csv',
                        help='Path to station information')
    parser.add_argument('--geopath', type=str, default='S:/Meteoblue/QRF/Data/geodata',
                        help='Path to spatial / geographic data')
    parser.add_argument('--savepath', type=str, default='VALIDATION', help='Path to save folder')

    # Inference arguments
    parser.add_argument('--boundary', type=float, nargs=4, help='Boundary coordinates for the feature map in the order'
                                                                '[S latitue, N latitude, E longitude, W longitude]')
    parser.add_argument('--time', type=str, nargs=2, help='Beginning and end time of the inference period in the '
                                                          'format %Y/%m/%d_%H:%M')
    # Validation arguments
    parser.add_argument('--palmfile', type=str, help='Path to PALM file')
    parser.add_argument('--res', type=int, help='PALM file resolution [m]')

    args = parser.parse_args()
    assert args.mode == 'inference' or args.mode == 'validation'
    assert os.path.isdir(args.measurementpath), 'Measurement path provided must be a directory'
    assert os.path.isfile(args.stationinfo), 'Path to station information must be a file'
    assert os.path.isdir(args.geopath), 'Path to geodata must be a directory'
    if not os.path.isdir(args.savepath):
        os.mkdir(args.savepath)

    if args.mode == 'inference':
        assert args.boundary, 'Boundary information must be provided'
        try:
            times = pd.to_datetime(args.time, format='%Y/%m/%d_%H:%M')
        except Error as e:
            warn('Inference times were entered in an unreadable format. Try again with "YYYY/MM/DD_HH:MM"')
            raise e

        generate_featuremaps(args.mode, args.measurementpath, args.geopath, args.stationinfo, args.savepath,
                             boundary=args.boundary, times=times)

        
    elif args.mode == 'validation':
        assert os.path.isfile(args.palmfile), 'Valid PALM simulation file must be given'
        generate_featuremaps(args.mode, args.measurementpath, args.geopath, args.stationinfo, args.savepath,
                             palmpath=args.palmfile, res=args.res)

