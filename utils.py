import numpy as np
from pandas import to_datetime, Timedelta, Timestamp
from warnings import warn


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


def DST_TZ(times):
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


def extract_times(origintime, times):
    t = []
    for idx, time in enumerate(times):
        t.append(origintime + Timedelta(minutes=np.round(time * 24 * 60)))
    return t


def moving_average(temps, datetime, timedelta=Timedelta(minutes=30)):
    moving_average = []
    for time in datetime:
        ma = []
        for idx, t in enumerate(datetime):
            if t-timedelta <= time <= t+timedelta:
                ma.append(temps[idx])
            elif t > t+timedelta:
                break
        moving_average.append(np.sum(ma)/len(ma))

    if len(moving_average) != len(temps):
        warn(f'Length of moving average vector ({len(moving_average)}) is not equivalent to the length of the '
             f'temperature vector ({len(temps)})')
        raise ValueError

    return moving_average


def manhatten_distance(featuremaps):
    times = featuremaps.shape[0]
    for time in range(times):
        print(f'time {time}/{times}')
        full = np.where(featuremaps[time, :, :] != 0)  # two lists of indexes where humimaps has a value

        for idxs, val in np.ndenumerate(featuremaps[time, :, :]):
            maxdist = featuremaps.shape[1] + featuremaps.shape[2] # maximum possible distance between two cells
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
                    featuremaps[time, idxs[0], idxs[1]] = mes/len(nearest)
                else:
                    featuremaps[time, idxs[0], idxs[1]] = featuremaps[time, full[0][nearest[0]], full[1][nearest[0]]]
            else:
                continue

    return featuremaps


def extract_surfacetemps(temps):
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
