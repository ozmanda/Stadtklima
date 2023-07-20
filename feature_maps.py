import os
import _pickle as cPickle
import argparse
import pickle
from utils import manhatten_distance
import numpy as np
import pandas as pd
from warnings import warn
from netCDF4 import Dataset
from pandas import DataFrame, to_datetime, Timedelta, read_csv
from utils import wgs84_to_lv, extract_times, extract_surfacetemps
from feature_generation import tempgen, humigen, geogen, radgen


# GLOBAL VARIABLES
geofeatures = ['altitude', 'buildings', 'buildings_10', 'buildings_30', 'buildings_100', 'buildings_200',
               'buildings_500', 'forests', 'forests_10', 'forests_30', 'forests_100', 'forests_200', 'forests_500',
               'pavedsurfaces', 'pavedsurfaces_10', 'pavedsurfaces_30', 'pavedsurfaces_100', 'pavedsurfaces_200',
               'pavedsurfaces_500', 'surfacewater', 'surfacewater_10', 'surfacewater_30', 'surfacewater_100',
               'surfacewater_200', 'surfacewater_500', 'urbangreen', 'urbangreen_10', 'urbangreen_30', 'urbangreen_100',
               'urbangreen_200', 'urbangreen_500']


def extract_palm_data(palmpath, res):
    """
    Extracts times, temperature and boundary coordinates from PALM file. PALM coordinates are extracted as latitude
    and longitude (WGS84) and converted to LV95 projection coordinates.
    """
    print('Extracting PALM File data....................')
    print('    loading PALM file........................')
    palmfile = Dataset(palmpath, 'r', format='NETCDF4')

    print('    determining boundary.....................')
    CH_S, CH_W = wgs84_to_lv(palmfile.origin_lat, palmfile.origin_lon, 'lv95')
    CH_N = CH_S + palmfile.dimensions['x'].size * res
    CH_E = CH_W + palmfile.dimensions['y'].size * res
    boundary = {'CH_S': CH_S, 'CH_N': CH_N, 'CH_E': CH_E, 'CH_W': CH_W}

    print('    extracting times.........................')
    times = np.array(extract_times(to_datetime(palmfile.origin_time), palmfile['time']))

    return boundary, times


def stations_loc(boundary, stationdata):
    print('Identifying stations within the boundary')
    stationscsv = read_csv(stationdata, delimiter=";")
    stations = {}
    for _, row in stationscsv.iterrows():
        if boundary['CH_W'] <= int(row["CH_E"]) <= boundary['CH_E'] and boundary['CH_S'] <= int(row["CH_N"]) <= boundary['CH_N']:
            stations[row["stationid_new"]] = {'lat': int(row["CH_N"]), 'lon': int(row["CH_E"])}
    return stations

def time_generation(timeframe):
    times = []
    time = timeframe[0]
    while time < timeframe[1]:
        times.append(time)
        time += Timedelta(minutes=5)
    return times


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


def generate_features(datapath, geopath, stations, boundary, times, folder, resolution=None):
    # TEMEPRATURE GENERATION
    print('Temperatures........................')
    if os.path.isfile(os.path.join(folder, 'temps.z')):
        # load temperatures and times
        with open(os.path.join(folder, 'temps.z'), 'rb') as file:
            temps = cPickle.load(file)
            file.close()
        with open(os.path.join(folder, 'times.z'), 'rb') as file:
            times = cPickle.load(file)
            file.close()

        # if manhattan distance calculations done, load otherwise call function
        if os.path.isfile(os.path.join(folder, 'ma.z')):
            with open(os.path.join(folder, 'ma.z'), 'rb') as file:
                ma = cPickle.load(file)
                file.close()
        else:
            print('Moving average..................')
            ma = manhatten_distance(temps)
    else:
        # temperature and time generation
        temps, times = tempgen(datapath, stations, times, boundary, resolution)
        # save intermediate step
        with open(os.path.join(folder, 'temps.z'), 'wb') as file:
            cPickle.dump(temps, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()
        with open(os.path.join(folder, 'times.z'), 'wb') as file:
            cPickle.dump(times, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

        # generate manhattan distance maps and save
        ma = manhatten_distance(temps)
        with open(os.path.join(folder, 'ma.z'), 'wb') as file:
            cPickle.dump(ma, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

    print('Humidities..........................')
    if os.path.isfile(os.path.join(folder, 'humimaps.z')):
        with open(os.path.join(folder, 'humimaps.z'), 'rb') as file:
            humimaps = cPickle.load(file)
            file.close()
    else:
        humimaps, times = humigen(datapath, stations, times, boundary, resolution)
        with open(os.path.join(folder, 'humimaps.z'), 'wb') as file:
            cPickle.dump(humimaps, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

    print('Geofeatures.........................')
    if os.path.isfile(os.path.join(folder, 'geomaps.z')):
        with open(os.path.join(folder, 'geomaps.z'), 'rb') as file:
            geomaps = cPickle.load(file)
            file.close()
    else:
        geomaps = geogen(geopath, boundary, humimaps)
        with open(os.path.join(folder, 'geomaps.z'), 'wb') as file:
            cPickle.dump(geomaps, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

    print('Irradiation.........................')
    if os.path.isfile(os.path.join(folder, 'irrad.z')):
        with open(os.path.join(folder, 'irrad.z'), 'rb') as file:
            irrad = cPickle.load(file)
            file.close()
    else:
        irrad = radgen(boundary, geomaps, times)
        with open(os.path.join(folder, 'irrad.z'), 'wb') as file:
            cPickle.dump(irrad, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

    # separation of time and datetime
    datetime_full = times.copy()
    for idx, time in enumerate(times):
        t = time.tz_localize(None)
        times[idx] = t.time()
        datetime_full[idx] = t
    return datetime_full, times, humimaps, geomaps, irrad, ma


def add_geos_flattened(dict, geos):
    for idx, feature in enumerate(geofeatures):
        dict[feature] = np.ravel(geos[:, idx, :, :])
    return dict


def add_geos(dict, geos):
    for idx, feature in enumerate(geofeatures):
        dict[feature] = geos[:, idx, :, :]
    return dict


# WRAPPER FUNCTIONS ---------------------------------------------------------------------------------------------------

def generate_featuremaps(type, datapath, geopath, stationinfo, savepath, palmpath=None, res=None,
                         boundary_wgs84=None, times=None):
    # Set folder name for intermediate steps, boundaries and times for both validation and inference cases
    if type == 'validation':
        folder = f'VALIDATION/{os.path.basename(palmpath).split(".nc")[0]}'
        boundary, times = extract_palm_data(palmpath, res)
    elif type == 'inference':
        folder = f'INFERENCE/{args.boundary[0]}-{args.boundary[1]}_{args.boundary[2]}_{args.boundary[3]}-' \
                 f'{args.time[0].replace("/", "-")}_{args.time[1].replace("/", "-")}'
        boundary = format_boundaries(boundary_wgs84)
        times = time_generation(times)
    else:
        warn('Invalid type passed to feature map generation, only "validation" and "inference" accepted')
        raise ValueError

    # Ensure folder exists
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # identify stations and generate featuremaps within the boundaries based on these stations
    stations = stations_loc(boundary, stationinfo)
    datetimes, times, humis, geo, rad, ma = generate_features(datapath, geopath, stations, boundary, times, folder, res)

    # Extract surface temperature from PALM simulation for validation type
    if type == 'validation':
        print('    extracting PALM surface temperatures..........')
        try:
            temps = extract_surfacetemps(palmpath)
        except IndexError:
            temps = extract_surfacetemps(palmpath)
        with open(os.path.join(folder, 'PALM_surfacetemps.z'), 'wb') as file:
            cPickle.dump(temps, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.close()

    # create time and datetime maps
    datetime_map = np.empty(shape=rad.shape, dtype=np.dtype('U20'))
    time_map = np.empty(shape=rad.shape, dtype=np.dtype('U20'))
    for idx, dt in enumerate(datetimes):
        datetime_map[idx, :, :] = str(dt)
        time_map[idx, :, :] = str(times[idx])

    # create feature dictionary and then DataFrame - flattened version
    # maps = {'datetime': np.ravel(datetime_map), 'time': np.ravel(time_map)}
    # maps = add_geos(maps, geo)
    # maps = {**maps, 'humidity': np.ravel(humis), 'irradiation': np.ravel(rad), 'moving_average': np.ravel(ma)}
    # feature_dataframe = DataFrame(maps)

    # create feature dictionary - unflattened version
    maps = {'datetime': datetime_map, 'time': time_map}
    maps = add_geos(maps, geo)
    maps = {**maps, 'humidity': humis, 'irradiation': rad, 'moving_average': ma, 'temperature': temps}

    # generate filename and save dataset
    starttime = f'{datetime_map[0, 0, 0][0:10]}-{datetime_map[0, 0, 0][11:16]}'.replace(':', '.')
    endtime = f'{datetime_map[-1, 0, 0][0:10]}-{datetime_map[-1, 0, 0][11:16]}'.replace(':', '.')
    filename = f'{type}_{starttime}_{endtime}_{boundary_wgs84[0]}-{boundary_wgs84[1]}_' \
               f'{boundary_wgs84[2]}-{boundary_wgs84[3]}.json'
    savepath = os.path.join(savepath, filename)
    with open(savepath, 'wb') as file:
        cPickle.dump(maps, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()
    # feature_dataframe.to_csv(savepath, sep=';', index=False)


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
        except Exception as e:
            warn('Inference times were entered in an unreadable format. Try again with "YYYY/MM/DD_HH:MM"')
            raise e

        generate_featuremaps(args.mode, args.measurementpath, args.geopath, args.stationinfo, args.savepath,
                             boundary_wgs84=args.boundary, times=times)

        
    elif args.mode == 'validation':
        assert os.path.isfile(args.palmfile), 'Valid PALM simulation file must be given'
        generate_featuremaps(args.mode, args.measurementpath, args.geopath, args.stationinfo, args.savepath,
                             palmpath=args.palmfile, res=args.res)

