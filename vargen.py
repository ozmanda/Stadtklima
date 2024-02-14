import os
import argparse
import numpy as np
from warnings import warn
from geodata import get_geofeatures
from utils import roundTime, DST_TZ, moving_average
from irradiation import irradiationcalc
from pandas import read_csv, to_datetime, DataFrame

#! faulty stations are hardcoded and used in generate_dfs
faultystations = ['C059A2225266','D07769DF208C','D63DFE9B164B','DF15D23E4B15','E2A0DF1A4941','E437CB2AF225','F033A8C6BB79','F4683D808CFB','F5C16A4B6340']

def preprocessing(filepath):
    """
    Loads measurements files and executes two preprocessing steps: rounding
    times to 5 minutes and removes duplicate lines.

    Parameters:
        filepath: absolute filepath to the measurement file

    Returns:
        csvfile: loaded and preprocessed data as a pandas DataFrame
    """
    # load csv
    csvfile = read_csv(filepath, delimiter=";")

    # round times
    csvfile['datetime'] = to_datetime(csvfile['datetime'])
    for idx, row in csvfile.iterrows():
        csvfile.iloc[idx, 0] = roundTime(row['datetime'])

    # get unique rows (some files contain duplicates)
    csvfile = csvfile.drop_duplicates()

    return csvfile


def file_matching(tempfile, humifile):
    # new arrays
    newhumi = np.empty(shape=(0, 2))
    newtemp = np.empty(shape=(0, 2))

    if len(tempfile) < len(humifile):
        humitimes = humifile['datetime'].to_list()
        for _, row in tempfile.iterrows():
            try:
                idx = humitimes.index(row['datetime'])
                humidat = [[humitimes[idx], humifile.iloc[idx, 1]]]
                tempdat = [[row['datetime'], row['temp']]]
            except ValueError:
                continue
            newtemp = np.append(newtemp, tempdat, axis=0)
            newhumi = np.append(newhumi, humidat, axis=0)

    elif len(humifile) < len(tempfile):
        temptimes = tempfile['datetime'].to_list()
        for _, row in humifile.iterrows():
            try:
                idx = temptimes.index(row['datetime'])
                tempdat = [[temptimes[idx], tempfile.iloc[idx, 1]]]
                humidat = [[row['datetime'], row['humi']]]
            except ValueError:
                continue
            newtemp = np.append(newtemp, tempdat, axis=0)
            newhumi = np.append(newhumi, humidat, axis=0)

    else:
        warn("file_matching function called unnecessarily", Warning)
        return tempfile, humifile

    newtemp = DataFrame(newtemp, columns=['datetime', 'temp'])
    newhumi = DataFrame(newhumi, columns=['datetime', 'humi'])
    return newtemp, newhumi


def create_lists(tempfile, stationid, geopath, geoconvs, infofile):
    try:
        geofeatures, targetlat, targetlon = get_geofeatures(stationid, geopath, len(tempfile), geoconvs,
                                                            ["altitude", "buildings", "forests",
                                                             "pavedsurfaces", "surfacewater", "urbangreen"],
                                                            infofile)
    except ValueError or FileNotFoundError as e:
        raise e

    times = DST_TZ(tempfile['datetime'].to_list())
    temps = tempfile.reset_index()['temp']

    return times, geofeatures, temps, targetlat, targetlon


def generate_dfs(datapath, geopath, savepath, geoconvs, infofile):
    filenames = os.listdir(datapath)

    for filename in filenames:
        if filename.startswith('temp'):
            stationid = filename.split('.csv')[0].split('_')[1]

            if f'{stationid}.csv' in os.listdir(savepath) or stationid in faultystations:
                continue

            print(f'Processing station {stationid}')
            print('........')

            try:
                humifile = preprocessing(os.path.join(datapath, f'humi_{stationid}.csv'))
            except FileNotFoundError as e:
                warn(f'No humidity file found for {stationid}, skipping this station')
                continue

            try:
                tempfile = preprocessing(os.path.join(datapath, filename))
            except FileNotFoundError as e:
                warn(f'No temperature file found for {stationid}, skipping this station')
                continue

            if len(humifile) != len(tempfile):
                warn(f'the number of measured temperatures and measured humidities do not coincide, unmatched', Warning)
                tempfile, humifile = file_matching(tempfile, humifile)

            try:
                times, geofeatures, temps, targetlat, targetlon = create_lists(tempfile, stationid, geopath,
                                                                               geoconvs, infofile)
                humi = humifile['humi'].to_list()
                irad = irradiationcalc(times, targetlat, targetlon)
            except Exception as e:
                print(f'Station {stationid} failed, continuing with next station.')
                continue


            datetime = times.copy()
            for idx, time in enumerate(times):
                times[idx] = time.tz_localize(None).time()
                datetime[idx] = datetime[idx].tz_localize(None)

            ma_temps = moving_average(temps, datetime)


            df = DataFrame({'datetime': datetime, 'time': times, 'altitude': geofeatures[:, 0],
                            'buildings': geofeatures[:, 1], 'buildings_10': geofeatures[:, 2],
                            'buildings_30': geofeatures[:, 3], 'buildings_100': geofeatures[:, 4],
                            'buildings_200': geofeatures[:, 5], 'buildings_500': geofeatures[:, 6],
                            'forests': geofeatures[:, 7], 'forests_10': geofeatures[:, 8],
                            'forests_30': geofeatures[:, 9], 'forests_100': geofeatures[:, 10],
                            'forests_200': geofeatures[:, 11], 'forests_500': geofeatures[:, 12],
                            'pavedsurfaces': geofeatures[:, 13], 'pavedsurfaces_10': geofeatures[:, 14],
                            'pavedsurfaces_30': geofeatures[:, 15], 'pavedsurfaces_100': geofeatures[:, 16],
                            'pavedsurfaces_200': geofeatures[:, 17], 'pavedsurfaces_500': geofeatures[:, 18],
                            'surfacewater': geofeatures[:, 19], 'surfacewater_10': geofeatures[:, 20],
                            'surfacewater_30': geofeatures[:, 21], 'surfacewater_100': geofeatures[:, 22],
                            'surfacewater_200': geofeatures[:, 23], 'surfacewater_500': geofeatures[:, 24],
                            'urbangreen': geofeatures[:, 25], 'urbangreen_10': geofeatures[:, 26],
                            'urbangreen_30': geofeatures[:, 27], 'urbangreen_100': geofeatures[:, 28],
                            'urbangreen_200': geofeatures[:, 29], 'urbangreen_500': geofeatures[:, 30],
                            'humidity': humi, 'irradiation': irad, 'moving_average': list(ma_temps),
                            'temperature': list(temps)})

            df.to_csv(f'{savepath}/{stationid}.csv', sep=';', index=False)
            print('Done\n')

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='Messdaten/Daten_Meteoblue',
                        help='relative path to measurement data')
    parser.add_argument('--geopath', type=str, default='Data/geodata',
                        help='relative path to the static / geographic data')
    parser.add_argument('--infofile', type=str, default=None)
    parser.add_argument('--savepath', type=str, default='Data/MeasurementFeatures_final_v1',
                        help='relative path to save folder')
    args = parser.parse_args()

    assert os.path.isdir(args.datapath), 'Invalid path to measurement data'
    assert os.path.isdir(args.geopath), 'Invalid path to static / geographic data'
    assert os.path.isfile(args.infofile), 'Invalid station file'
    if not os.path.isdir(args.savepath):
        os.mkdir(args.savepath)

    generate_dfs(os.path.join(os.getcwd(), args.datapath),
                 os.path.join(os.getcwd(), args.geopath),
                 os.path.join(os.getcwd(), args.savepath),
                 geoconvs=[10, 30, 100, 200, 500],
                 infofile=args.infofile)
