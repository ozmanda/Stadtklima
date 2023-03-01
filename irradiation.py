import datetime
import numpy as np
import pandas as pd
from pytz import timezone
from utils import lv_to_wgs84
from pysolar import solar, radiation


def irradiationcalc(times, targetlat, targetlon):
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



