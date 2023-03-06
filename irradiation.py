import datetime
import numpy as np
import pandas as pd
from pytz import timezone
from utils import lv_to_wgs84
from pysolar import solar, radiation


def irradiationcalc(times, targetlat, targetlon):
    # if irradiationcalc throws a TypeError somewhere altitude calculation, the problem is likely caused by pd.Timestamp
    #   in the lines 97 & 723 of solartime.py from the pysolar package, where .utctimetuple() doesn't work for a
    #   tz-aware Timestamp (https://github.com/pandas-dev/pandas/issues/32174)
    #   add "if when.tzinfo is None else when.timetuple()" to the end of when.utctimetuple() to get the expected result
    targetlat, targetlon = lv_to_wgs84(targetlat, targetlon, type='lv95')
    irradiation = list()
    for time in times:
        time = time.tz_localize(timezone('Europe/Zurich'))
        alt = solar.get_altitude(targetlat, targetlon, time)
        if alt <= 0:
            irradiation.append(0)
        else:
            irradiation.append(radiation.get_radiation_direct(time, alt))

    return irradiation


def irradiationmap(boundary, times, altitudes):
    irradmap = np.empty(shape=(len(times), altitudes.shape[0], altitudes.shape[1]))
    for idxs, _ in np.ndenumerate(irradmap):
        rads = irradiationcalc(times, boundary['CH_S'] + idxs[0], boundary['CH_W'] + idxs[1])
        irradmap[:, idxs[0]-1, idxs[1]] = rads

    return irradmap



