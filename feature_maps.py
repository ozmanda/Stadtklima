import os
import argparse
from utils import manhatten_distance
import numpy as np
import pandas as pd
from warnings import warn
from netCDF4 import Dataset #type: ignore
from metpy.calc import relative_humidity_from_mixing_ratio
from metpy.units import units
from pandas import to_datetime, Timedelta, read_csv
from utils import wgs84_to_lv, extract_times, extract_surfacetemps, dump_file, load_file, moving_average
from feature_generation import tempgen, humigen, geogen, radgen


# GLOBAL VARIABLES
geofeatures = ['altitude', 'buildings', 'buildings_10', 'buildings_30', 'buildings_100', 'buildings_200',
               'buildings_500', 'forests', 'forests_10', 'forests_30', 'forests_100', 'forests_200', 'forests_500',
               'pavedsurfaces', 'pavedsurfaces_10', 'pavedsurfaces_30', 'pavedsurfaces_100', 'pavedsurfaces_200',
               'pavedsurfaces_500', 'surfacewater', 'surfacewater_10', 'surfacewater_30', 'surfacewater_100',
               'surfacewater_200', 'surfacewater_500', 'urbangreen', 'urbangreen_10', 'urbangreen_30', 'urbangreen_100',
               'urbangreen_200', 'urbangreen_500']
pressure = 1013.25

def extract_palm_data(palmpath: str, res: int):
    """
    Extracts times, temperature and boundary coordinates from PALM file. PALM coordinates are extracted as latitude
    and longitude (WGS84) and converted to LV95 projection coordinates.
    """
    print('Extracting PALM File data....................')
    print('    loading PALM file........................')
    palmfile: Dataset = Dataset(palmpath, 'r', format='NETCDF4')

    print('    determining boundary.....................')
    CH_S, CH_W, _ = wgs84_to_lv(palmfile.origin_lat, palmfile.origin_lon, 'lv95') #type: ignore
    CH_N = CH_S + palmfile.dimensions['x'].size * res
    CH_E = CH_W + palmfile.dimensions['y'].size * res
    boundary = {'CH_S': CH_S, 'CH_N': CH_N, 'CH_E': CH_E, 'CH_W': CH_W}


    print('    extracting times.........................')
    times = np.array(extract_times(to_datetime(palmfile.origin_time), palmfile['time']))

    return boundary, times


def palm_humi(palmpath):
    """
    Extracts water vapor mixing ratio from PALM file and transforms it into rela00tive humidity maps
    :param palmpath: path to PALM simulation file
    :return: 3-dimensional humidity map
    """
    print('Extracting PALM humidity data................')
    print('    loading PALM file........................')
    palmfile = Dataset(palmpath, 'r', format='NETCDF4')
    all_mr = palmfile['q_xy']
    surf_humis = np.zeros(shape=(all_mr.shape[0], all_mr.shape[2], all_mr.shape[3]))
    for time in range(all_mr.shape[0]):
        for idxs, _ in np.ndenumerate(all_mr[time, 0, :, :]):
            for layer in range(all_mr.shape[1]):
                if all_mr[time, layer, idxs[0], idxs[1]] != -9999:
                    surface_mixing_ratio = all_mr[time, layer, idxs[0], idxs[1]]
                    temp = palmfile['theta_xy'][time, layer, idxs[0], idxs[1]]
                    relative_humidity = relative_humidity_from_mixing_ratio(pressure*units.hPa,
                                                                            (temp - 273.15) * units.degC,
                                                                            surface_mixing_ratio).to('percent')
                    relative_humidity = round(float(relative_humidity), 2)
                    assert 0 < relative_humidity < 100, 'Relative humidity must be between 0 and 100 (percent)'
                    if np.isnan(relative_humidity):
                        warn(f'Relative humidity at {idxs} for time {time} is NaN')
                        raise ValueError
                    if np.isinf(relative_humidity):
                        warn(f'Relative humidity at {idxs} for time {time} is inf')
                        raise ValueError
                    surf_humis[time, idxs[0], idxs[1]] = relative_humidity
                    break
                else:
                    continue
    # flip maps to account for PALM having origin at the bottom left, not top left
    surf_humis = np.flip(surf_humis, axis=1)

    return surf_humis


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
    CH_S, CH_E = wgs84_to_lv(boundary[0], boundary[2], 'lv95') #type: ignore
    CH_N, CH_W = wgs84_to_lv(boundary[1], boundary[3], 'lv95') #type: ignore
    return {'CH_S': CH_S, 'CH_N': CH_N, 'CH_E': CH_E, 'CH_W': CH_W}


def generate_features(datapath: str, geopath: str, stations: dict, boundary: dict, times: list[pd.Timestamp], 
                      folder: str, resolution: int = 0, palmhumis: bool = False, palmpath: str = ''):
    # TEMEPRATURE GENERATION
    print('Temperatures........................')
    if os.path.isfile(os.path.join(folder, 'temps.z')):
        # load temperatures and times
        temps = load_file(os.path.join(folder, 'temps.z'))
        times = load_file(os.path.join(folder, 'times.z'))

        # if manhattan distance calculations done, load otherwise call function
        print('Moving average..................')
        if os.path.isfile(os.path.join(folder, 'ma.z')):
            ma = load_file(os.path.join(folder, 'ma.z'))

        elif os.path.isfile(os.path.join(folder, 'manhattan.z')):
            md = load_file(os.path.join(folder, 'manhattan.z'))

            ma = moving_average(md, times)
            dump_file(os.path.join(folder, 'ma.z'), ma)

        else:
            print('Manhattan Distance..............')
            md = manhatten_distance(temps)
            dump_file(os.path.join(folder, 'manhattan.z'), md)

            ma = moving_average(md, times)
            dump_file(os.path.join(folder, 'ma.z'), ma)
    else:
        # temperature and time generation
        temps, times = tempgen(datapath, stations, times, boundary, resolution)

        # save intermediate step
        dump_file(os.path.join(folder, 'temps.z'), temps)
        dump_file(os.path.join(folder, 'times.z'), times)

        # generate manhattan distance maps and save
        md = manhatten_distance(temps)
        dump_file(os.path.join(folder, 'manhattan.z'), md)

        # calculate moving average of manhattan distance maps
        ma = moving_average(md, times)
        dump_file(os.path.join(folder, 'ma.z'), ma)

    print('Humidities..........................')
    if palmhumis:
        print('    palm humidity')
        if os.path.isfile(os.path.join(folder, 'humimaps_palm.z')):
            humimaps = load_file(os.path.join(folder, 'humimaps_palm.z'))
        else:
            humimaps = palm_humi(palmpath)
            dump_file(os.path.join(folder, 'humimaps_palm.z'), humimaps)

    else:
        if os.path.isfile(os.path.join(folder, 'humimaps.z')):
            humimaps = load_file(os.path.join(folder, 'humimaps.z'))
        else:
            humimaps, times = humigen(datapath, stations, times, boundary, resolution)
            dump_file(os.path.join(folder, 'humimaps.z'), humimaps)
            dump_file(os.path.join(folder, 'times.z'), times)

    print('Geofeatures.........................')
    if os.path.isfile(os.path.join(folder, 'geomaps.z')):
        geomaps = load_file(os.path.join(folder, 'geomaps.z'))
    else:
        geomaps = geogen(geopath, boundary, humimaps)
        dump_file(os.path.join(folder, 'geomaps.z'), geomaps)

    print('Irradiation.........................')
    if os.path.isfile(os.path.join(folder, 'irrad.z')):
        irrad = load_file(os.path.join(folder, 'irrad.z'))
    else:
        irrad = radgen(boundary, geomaps, times)
        dump_file(os.path.join(folder, 'irrad.z'), irrad)

    # separation of time and datetime
    datetime_full = []
    times_only = []
    for idx, time in enumerate(times):
        t = time.tz_localize(None)
        datetime_full.append(t)
        times_only.append(t.time())
    return datetime_full, times_only, humimaps, geomaps, irrad, ma


def add_geos_flattened(d: dict, geos: np.ndarray):
    for idx, feature in enumerate(geofeatures):
        d[feature] = np.ravel(geos[:, idx, :, :])
    return d


def add_geos(d: dict, geos: np.ndarray):
    for idx, feature in enumerate(geofeatures):
        d[feature] = geos[:, idx, :, :]
    return d


# WRAPPER FUNCTIONS ---------------------------------------------------------------------------------------------------
# TODO: turn this into two separate functions for validation and inference cases
# TODO: identify repeated functions and generalise them for validation and inference cases
def generate_featuremaps(type: str, datapath: str, geopath: str, stationinfo: str, savepath: str, palmpath: str = '', 
                         res: int = 16, boundary_wgs84: list = [], times: list = [], palmhumi: bool = False):
    """
    Function to generate feature maps for validation and inference cases. For validation cases, PALM simulation data is 
    used to generate feature and target temperature maps. For inference cases, the target temperature maps are not generated.

    Args:
        type (str): Type of feature map to be generated, either "validation" or "inference"
        datapath (str): Path to station-based measurement data
        geopath (str): Path to geospatial data
        stationinfo (str): Path to station information
        savepath (str): Path to save folder
        palmpath (str, optional): Path to PALM simulation file. Defaults to None.
        res (int, optional): Resolution of the PALM simulation file. Defaults to 16.
        boundary_wgs84 (list, optional): Boundary of the maps (inference case). Defaults to None.
        times (list, optional): Start and end times for the generated maps. Defaults to None.
        palmhumi (bool, optional): Boolean indicating whether humidity values should be taken from PALM. Defaults to False.

    Raises:
        ValueError: Upon receiving a type other than "validation" or "inference"
    """
    # Set folder name for intermediate steps, boundaries and times for both validation and inference cases
    boundary: dict = {}
    times: np.ndarray = np.array([])
    if type == 'validation':
        folder = f'DATA/QRF_Inference_Feature_Maps/{os.path.basename(palmpath).split(".nc")[0]}_palmhumi'
        boundary, times = extract_palm_data(palmpath, res)
        dump_file(f'{os.path.splitext(palmpath)[0]}_boundary.z', boundary)
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
    datetimes, times, humis, geo, rad, ma = generate_features(datapath, geopath, stations, boundary, times, folder, res,
                                                              palmhumis=palmhumi, palmpath=palmpath)

    # Extract surface temperature from PALM simulation for validation type
    if type == 'validation':
        print('PALM surface temperatures..........')
        if os.path.isfile(os.path.join(folder, 'PALM_surfacetemps.z')):
            temps = load_file(os.path.join(folder, 'PALM_surfacetemps.z'))
        else:
            temps = extract_surfacetemps(palmpath)
            dump_file(os.path.join(folder, 'PALM_surfacetemps.z'), temps)

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
    print(f'moving_average shape: {ma.shape}')

    # generate filename and save dataset
    starttime = f'{datetime_map[0, 0, 0][0:10]}-{datetime_map[0, 0, 0][11:16]}'.replace(':', '.')
    endtime = f'{datetime_map[-1, 0, 0][0:10]}-{datetime_map[-1, 0, 0][11:16]}'.replace(':', '.')
    if boundary_wgs84:
        filename = f'{type}_{starttime}_{endtime}_{boundary_wgs84[0]}-{boundary_wgs84[1]}_' \
                   f'{boundary_wgs84[2]}-{boundary_wgs84[3]}.json'
    elif palmpath:
        filename = f'{os.path.basename(palmpath).split(".nc")[0]}_palmhumi.json'

    savepath = os.path.join(savepath, filename)
    dump_file(savepath, maps)


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
    parser.add_argument('--palmhumi', type=bool, default=False, help='Should humidity be taken from PALM simulation')
    parser.add_argument('--res', type=int, help='PALM file resolution [m]', default=16)

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
                             boundary_wgs84=args.boundary, times=times, palmhumi=args.palmhumi)

        
    elif args.mode == 'validation':
        assert os.path.isfile(args.palmfile), 'Valid PALM simulation file must be given'
        generate_featuremaps(args.mode, args.measurementpath, args.geopath, args.stationinfo, args.savepath,
                             palmpath=args.palmfile, res=args.res, palmhumi=args.palmhumi)

