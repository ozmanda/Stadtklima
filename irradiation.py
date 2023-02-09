import numpy as np
from utils import lv_to_wgs84
from pysolar import solar, radiation


def irradiationcalc(times, targetlat, targetlon):
    targetlat, targetlon = lv_to_wgs84(targetlat, targetlon, type='lv95')
    irradiation = list()
    for time in times:
        time = time.to_pydatetime()
        alt = solar.get_altitude(targetlat, targetlon, time)
        if alt <= 0:
            irradiation.append(0)
        else:
            irradiation.append(radiation.get_radiation_direct(time, alt))

    return irradiation


def irradiationmap(boundary, times, altitudes):
    irradmap = np.empty(shape=(len(times), altitudes.shape[0], altitudes.shape[1]))
    for idxs, _ in np.ndenumerate(irradmap):
        rads = irradiationcalc(times, boundary[0, 0] + idxs[0], boundary[1, 0] + idxs[1])
        irradmap[:, idxs[0]-1, idxs[1]] = rads

    return irradmap



