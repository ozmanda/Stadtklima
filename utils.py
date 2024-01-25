import numpy as np
import time
import _pickle as cPickle # type: ignore
import pickle
import pandas as pd
# from typing import Literal
from pandas import to_datetime, Timedelta, Timestamp
from warnings import warn
from netCDF4 import Dataset # type: ignore
from scipy.spatial.distance import cdist

# _types = Literal["lv03", "lv95"]

def roundTime(dt, roundTo=5*60):
    """
    Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 5 minutes.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    dt = Timestamp.to_pydatetime(dt)
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return to_datetime(dt + Timedelta(seconds=rounding-seconds, microseconds=-dt.microsecond))


def lv95_to_lv03(lv95_lat: float, lv95_lon: float):
    return lv95_lat - 1000000, lv95_lon - 2000000


def lv03_to_lv95(lv03_lat: float, lv03_lon: float):
    return lv03_lat + 1000000, lv03_lon + 2000000


# float specification!
def lv_to_wgs84(lv_lat: float, lv_lon: float, type: float, h_lv: float = 0):
    if type == 'lv03':
        y_prime: float = (lv_lon - 600000) / 1000000
        x_prime: float = (lv_lat - 200000) / 1000000
    elif type == 'lv95':
        y_prime: float = (lv_lon - 2600000) / 1000000
        x_prime: float = (lv_lat - 1200000) / 1000000
    else:
        warn(f'Invalid type ({type}) passed for conversion (only "lv95" or "lv03" accepted).')
        raise ValueError

    lambda_prime: float = 2.6779094 + \
                   4.728982 * y_prime + \
                   0.791484 * y_prime * x_prime + \
                   0.130600 * y_prime * x_prime**2 - \
                   0.043600 * y_prime**3

    phi_prime: float = 16.9023892 + \
                3.238272 * x_prime - \
                0.270978 * y_prime**2 - \
                0.002528 * x_prime**2 - \
                0.044700 * x_prime + y_prime**2 - \
                0.014000 * x_prime**3

    wgs84_lat: float = (phi_prime * 100) / 36
    wgs84_lon: float = (lambda_prime * 100) / 36

    if h_lv:
        h_wgs: float = h_lv + 49.55 \
                - 12.9 * y_prime \
                - 22.64 * x_prime
        return wgs84_lat, wgs84_lon, h_wgs
    else:
        return wgs84_lat, wgs84_lon


def wgs84_to_lv(wgs84_lat: float, wgs84_lon: float, type, 
                h_wgs: float = 0, unit: str = 'deg'):
    if unit == 'deg':
        wgs84_lat *= 3600
        wgs84_lon *= 3600
    # Breite = latitude = phi, Länge = longitude = lambda
    phi_prime = (wgs84_lat - 169028.66) / 10000
    lambda_prime = (wgs84_lon - 26782.5) / 10000


    # E = longitude, N = latitude
    lv95_lon: float =  2600072.37 \
                + 211455.93 * lambda_prime \
                - 10938.51 * lambda_prime * phi_prime \
                - 0.36 * lambda_prime * phi_prime**2 \
                - 44.54 * lambda_prime**3

    lv95_lat: float = 1200147.07 \
               + 308807.95 * phi_prime \
               + 3745.25 * lambda_prime**2 \
               + 76.63 * phi_prime**2 \
               - 194.56 * lambda_prime**2 * phi_prime \
               + 119.79 * phi_prime**3

    if h_wgs:
        h_lv = h_wgs - 49.55 \
               + 2.73 * lambda_prime \
               + 6.94 * phi_prime

    if type == 'lv95' and h_wgs:
        return lv95_lat, lv95_lon, h_lv #type: ignore
    elif type == 'lv95' and not h_wgs:
        return lv95_lat, lv95_lon, 0
    elif type == 'lv03':
        lv03_lat, lv03_lon = lv95_to_lv03(lv95_lat, lv95_lon)
        if h_wgs:
            return lv03_lat, lv03_lon, h_lv #type: ignore
        else:
            return lv03_lat, lv03_lon, 0


def DST_TZ(times: list):
    for idx, time in enumerate(times):
        if Timestamp('2019-03-31 02:00') <= time <= Timestamp('2019-10-27 02:00'):
            time -= Timedelta(hours=2)
            time = time.tz_localize('utc')
            time = time.tz_convert('Europe/Zurich')
            times[idx] = time
        else:
            time -= Timedelta(hours=1)
            time = time.tz_localize('utc')
            time = time.tz_convert('Europe/Zurich')
            times[idx] = time
    return times


def extract_times(origintime: np.datetime64, times_list: list):
    """
    Extracts the time vector, formatting it as a datetime. The time contained within the PALM file is given as
    seconds since origin. Additionally, a boolean vector is generated, indicating the start of the useable time
    series (certain observations are required to create the moving average).
    """
    times = []
    print(f'times list length: {len(times_list)}')
    for _, time in enumerate(times_list):
        times.append(origintime + Timedelta(minutes=np.round(time * 24 * 60)))

    #* lost observations fixed, for most PALM files it is 2/56 which are lost
    # lost observations for the one hour moving average: 60 min / timedelta
    td_minutes = (times[2]-times[1]).total_seconds() / 60
    lost_obs = int(60/td_minutes)
    t_bool = [True] * len(times)
    t_bool[0:lost_obs] = [False] * lost_obs
    if not times:
        raise ValueError
    return times, t_bool


def remove_emptytimes(maps: np.ndarray, times: np.ndarray):
    print(f'remove_emptylines before: {times.shape}')
    emptytimes = []
    for time in range(maps.shape[0]):
        if not np.sum(maps[time, :, :]):
            emptytimes.append(time)
            continue
    if emptytimes:
        maps = np.delete(maps, emptytimes, axis=0)
        times = np.delete(times, emptytimes, axis=0)
    print(f'remove_emptylines after: {times.shape}')
    return maps, times


def dump_file(path: str, object):
    with open(path, 'wb') as file:
        cPickle.dump(object, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()


def load_file(path: str):
    with open(path, 'rb') as file:
        object = cPickle.load(file)
        file.close()
    return object


def moving_average(temps: np.ndarray, datetimes: list, timedelta=Timedelta(minutes=60)):
    movingaverage = []
    for i, time in enumerate(datetimes):
        print(f'time {i}: ', end='')
        ma = []
        for idx, t in enumerate(datetimes):
            if time-timedelta <= t <= time:
                ma.append(temps[idx])
            elif t > time:
                break
        if not ma:
            print(f'moving average shape: {np.array(movingaverage).shape}')
            movingaverage.append(temps[i])
        else:
            print(f'moving average shape: {np.array(movingaverage).shape}')
            movingaverage.append(np.mean(ma, axis=0))

    movingaverage = np.array(movingaverage)
    if movingaverage.shape != temps.shape:
        warn(f'Shape of moving average vector ({movingaverage.shape}) is not equivalent to the length of the '
             f'temperature vector ({temps.shape})')
        raise ValueError

    print(f'movingaverage final shape: {movingaverage.shape}')
    return movingaverage


def manhatten_distance_(featuremaps: np.ndarray):
    print(f'featuremaps shape: {featuremaps.shape}')
    times = featuremaps.shape[0]
    for time in range(times):
        print(f'time {time}/{times}')
        start_timer()
        full = np.transpose(np.array(np.where(featuremaps[time, :, :] != 0)))
        for idxs, val in np.ndenumerate(featuremaps[time]):
            if not val:
                dists = cdist(np.array([idxs]), full, 'cityblock').astype(int)
                nearest = np.where(dists == dists.min())[1]
                nearest_vals = featuremaps[time, full[nearest][:, 0], full[nearest][:, 1]]
                featuremaps[time, idxs[0], idxs[1]] = np.mean(nearest_vals)
        print('Manhattan Distance Function: ', end=' ')
        end_timer()
    return featuremaps


def manhatten_distance(featuremaps: np.ndarray):
    times: int = featuremaps.shape[0]
    for t in range(times):
        print(f'time {t+1}/{times}', end='\r')
        full = np.where(featuremaps[t, :, :] != 0)
        for idxs, val in np.ndenumerate(featuremaps[t]):
            if val:
                featuremaps[t, idxs[0], idxs[1]] = val
                continue
            else:
                filled = False
                d = 1
                rows = []
                cols = []
                while not filled:
                    lims = {'row_start': idxs[0]-d,
                            'row_end': idxs[0]+d,
                            'col_start': idxs[1]-d,
                            'col_end': idxs[1]+d}
                    # if one of the row numbers correspond to the row number of a cell with value
                    if lims['row_start'] in full[0] or lims['row_end'] in full[0]:
                        # find idx of matching row
                        pos = np.where([x == lims['row_start'] or x == lims['row_end'] for x in full[0]])[0]
                        for p in pos:
                            if full[1][p] in range(lims['col_start'], lims['col_end'] + 1):
                                rows.append(full[0][p])
                                cols.append(full[1][p])
                    if lims['col_start'] in full[1] or lims['col_end'] in full[1]:
                        # find idx of matching column
                        pos = np.where([x == lims['col_start'] or x == lims['col_end'] for x in full[1]])[0]
                        for p in pos:
                            if full[0][p] in range(lims['row_start'] + 1, lims['row_end']):
                                rows.append(full[0][p])
                                cols.append(full[1][p])
                    if rows and cols:
                        filled = True
                    else:
                        d += 1
                featuremaps[t, idxs[0], idxs[1]] = np.mean(featuremaps[t, rows, cols])
    return featuremaps


def manhatten_distance_old(featuremaps: np.ndarray):
    times = featuremaps.shape[0]
    for time in range(times):
        print(f'time {time}/{times}')

        # two lists of indexes where humimaps has a value
        full = np.where(featuremaps[time, :, :] != 0)
        for idxs, val in np.ndenumerate(featuremaps[time, :, :]):
            maxdist = featuremaps.shape[1] + featuremaps.shape[2]  # maximum possible distance between two cells
            if val == 0:
                nearest = []
                for measurement in range(len(full[0])):
                    dist = np.abs(idxs[0] - full[0][measurement]) + np.abs(idxs[1] - full[1][measurement])
                    if dist < maxdist:
                        maxdist = dist
                        nearest = [measurement]
                    elif dist == maxdist:
                        nearest.append(measurement)

                if len(nearest) != 1:
                    mes = 0
                    for _, val in enumerate(nearest):
                        mes += featuremaps[time, full[0][val], full[1][val]]
                    try:
                        featuremaps[time, idxs[0], idxs[1]] = mes/len(nearest)
                    except ZeroDivisionError:
                        featuremaps[time, idxs[0], idxs[1]] = mes / len(nearest)
                else:
                    featuremaps[time, idxs[0], idxs[1]] = featuremaps[time, full[0][nearest[0]], full[1][nearest[0]]]
            else:
                continue

    return featuremaps


def extract_surfacetemps(palmpath):
    palmfile = Dataset(palmpath, 'r', format='NETCDF4')
    try:
        temps = palmfile['theta_xy']
    except IndexError:
        temps = palmfile['theta']
    surf_temps = np.zeros(shape=(temps.shape[0], temps.shape[2], temps.shape[3]))
    for time in range(temps.shape[0]):
        for idxs, _ in np.ndenumerate(temps[time, 0, :, :]):
            for layer in range(temps.shape[1]):
                if temps[time, layer, idxs[0], idxs[1]] != -9999:
                    surf_temps[time, :, :][idxs] = temps[time, layer, idxs[0], idxs[1]] - 273.15
                    break
                else:
                    continue
    # flip maps to account for PALM having origin at the bottom left, not top left
    surf_temps = np.flip(surf_temps, axis=1)

    return surf_temps


#! this used to obscure the earlier manhattan_distance function, if something breaks, check here first --> reinstated to see if its faster
#! TODO: I think this is the faster function, needs to be properly evaluated tho (ushe)
def manhatten_distance(featuremaps: np.ndarray):
    times = featuremaps.shape[0]
    full_array = np.zeros(featuremaps.shape)
    for time in range(times):
        print(f'time {time}/{times}')
        full = np.transpose(np.array(np.where(featuremaps[time, :, :] != 0)))
        for idxs, val in np.ndenumerate(featuremaps[time]):
            if val:
                full_array[time, idxs[0], idxs[1]] = val
                continue
            else:
                dists = cdist(np.array([idxs]), full).astype(int)
                nearest = np.where(dists == dists.min())[1]
                nearest_vals = featuremaps[time, full[nearest][:, 0], full[nearest][:, 1]]
                full_array[time, idxs[0], idxs[1]] = np.mean(nearest_vals)
    return full_array


def start_timer():
    global _start_time
    _start_time = time.time()


def end_timer():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(f'Time: {t_hour}:{t_min}:{t_sec}')


def reduce_resolution(array, resolution):
    """
    Reduces the dimension of a given array by the resolution. A 10x10 array with a resolution of 2 would return a 5x5 
    array. The method averages all 
    """
    new_array = np.zeros(shape=(int(array.shape[0]/resolution), int(array.shape[1]/resolution)))
    for row in range(new_array.shape[0]):
        for col in range(new_array.shape[1]):
            new_array[row, col] = np.mean(array[row*resolution:row*resolution+resolution, 
                                                col*resolution:col*resolution+resolution])
        return new_array
    