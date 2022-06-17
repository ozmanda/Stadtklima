import numpy as np
from pysolar import solar, radiation


def convert_coordinates(targetlat, targetlon):
    x_prime = (targetlat - 200000) / 1000000
    y_prime = (targetlon - 600000) / 1000000

    lambda_prime = 2.6779094 + \
                   4.728982 * y_prime + \
                   0.791484 * y_prime * x_prime + \
                   0.130600 * y_prime * x_prime**2 - \
                   0.043600 * y_prime**3

    phi_prime = 16.9023892 + \
                3.238272 * x_prime - \
                0.270978 * y_prime**2 - \
                0.002528 * x_prime**2 - \
                0.044700 * x_prime + y_prime**2 - \
                0.014000 * x_prime**3

    lat = (phi_prime * 100) / 36
    lon = (lambda_prime * 100) / 36

    return lat, lon


def irradiationcalc(times, targetlat, targetlon):
    targetlat, targetlon = convert_coordinates(targetlat, targetlon)
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



