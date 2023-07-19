import os
import argparse
from warnings import warn
from utils import roundTime
from pandas import read_csv, to_datetime, Timedelta, DataFrame
from netCDF4 import Dataset


def times(datapath, filename):
    if filename:
        ncfile = Dataset(os.path.join(datapath, filename), 'r', format='NETCDF4')

        # check if the unit is seconds or days (both occurs)
        if ncfile['time'][0,] < 1:
            seconds_from_origin = ncfile['time'][:, ] * 60 * 60 * 24
        else:
            seconds_from_origin = ncfile['time'][:, ]

        # boolean array indicating which times are not zero
        seconds_from_origin = seconds_from_origin[seconds_from_origin > 0]

        # determine the start and end time of the PALM simulation
        origintime = to_datetime(ncfile.origin_time, format='%Y-%m-%d %H:%M:%S')
        origintime = roundTime(origintime.tz_localize(None))
        endtime = roundTime(origintime + Timedelta(seconds=seconds_from_origin[-1]))
    else:
        origintime = to_datetime('2019-06-01 12:00:00', format='%Y-%m-%d %H:%M:%S')
        endtime = to_datetime('2019-08-31 12:00:00', format='%Y-%m-%d %H:%M:%S')

    return origintime, endtime


def empty_df(datapath, stationids):
    for stationid in stationids:
        try:
            stationcsv = read_csv(os.path.join(datapath, f'{stationid}.csv'), delimiter=';')
            cols = stationcsv.columns
            break
        except Exception as e:
            continue

    return DataFrame(columns=cols)


def stationdata(datapath, stationid, start, end):
    stationcsv = read_csv(os.path.join(datapath, f'{stationid}.csv'), delimiter=';')
    if start and end:
        if start > to_datetime(stationcsv.iloc[-1, 0]) or end < to_datetime(stationcsv.iloc[0, 0]):
            warn('PALM simulation outside of measurement window', Warning)
            raise ValueError

        startidx = 0
        if to_datetime(stationcsv.iloc[startidx, 0]) > start:
            warn(f'PALM simulation begins before measurement period, beginning at '
                 f'{to_datetime(stationcsv.iloc[startidx, 0])}', Warning)
        else:
            while to_datetime(stationcsv.iloc[startidx, 0]) < start:
                startidx += 1
                continue

        endidx = startidx
        try:
            while to_datetime(stationcsv.iloc[endidx, 0]) <= end:
                if endidx >= len(stationcsv):
                    warn(f'PALM simulation is longer than the measured period, cutting off at '
                         f'{to_datetime(stationcsv.iloc[endidx, 0])}', Warning)
                    break
                endidx += 1
                continue
        except IndexError:
            warn('Index Error', Warning)
            pass

    else:
        startidx = 0
        endidx = len(stationcsv)

    return stationcsv.iloc[startidx:endidx, :]


def create_dataset(datapath, palmpath, filename, stationids, savepath):
    start, end = times(palmpath, filename)

    if not stationids:
        stationids = []
        for station in os.listdir(datapath):
            stationids.append(station.split('.')[0])

    dataset = empty_df(datapath, stationids)
    for stationid in stationids:
        try:
            dataset = dataset.append(stationdata(datapath, stationid, start, end), ignore_index=True)
        except ValueError:
            continue

    if start and end:
        filename = f'dataset_{start.year}{start.month}{start.day}.{start.hour}_{end.year}{end.month}{end.day}.' \
                   f'{end.hour}_{len(stationids)}stations.csv'
    else:
        filename = f'dataset_all_{len(stationids)}stations.csv'

    savepath = os.path.join(os.getcwd(), f'{savepath}/{filename}') if savepath else os.path.join(os.getcwd(), f'Data/{filename}')
    dataset.to_csv(savepath, sep=";", index_label='index')

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--palmpath', type=str, default='Data', help='relative path to folder containing PALM files')
    parser.add_argument('--palmfile', type=str, default='mb_8_multi_stations_LCZ_masked_N03_M01.00m.nc',
                        help='Name of the PALM file to be loaded')
    parser.add_argument('--measurementpath', type=str, default='Data/MeasurementFeatures_v3',
                        help='relative path to folder containing measurement data')
    parser.add_argument('--savepath', type=str, default='Data/v4', help='relative path for the save file')
    args = parser.parse_args()

    stations = None
    args.palmfile = None
    dataset = create_dataset(datapath=os.path.join(os.getcwd(), args.measurementpath),
                             palmpath=args.palmpath,
                             filename=args.palmfile,
                             stationids=stations,
                             savepath=args.savepath)

