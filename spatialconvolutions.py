import numpy as np
from scipy import signal


# CAREFUL - THIS IS PROGRAMMED ASSUMING CELL SIZE 16x16 METERS
res = 16
# convolution kernels should have a size of approx. 10, 30, 100, 200 and 500 meters and must be uneven
# kernel size and loss must be adjusted when adjusting the resolution (i.e. cell size)


def create_kernels(sigma=3):
    discretekernels = {"size": [1, 3, 7, 13, 31],
                       "loss": [0, 1, 3, 6, 15],
                       "kernel": [1, np.ones(shape=(3, 3)),
                                  np.ones(shape=(7, 7)),
                                  np.ones(shape=(13, 13)),
                                  np.ones(shape=(31, 31)), ]
                       }
    continuouskernels = {"size": [1, 3, 7, 13, 31],
                         "loss": [0, 1, 3, 6, 15]
                         }

    kernels = []
    for size in continuouskernels["size"]:
        kernel = signal.gaussian(size, std=sigma).reshape(size, 1)
        kernel = np.outer(kernel, kernel)
        kernels.append(kernel)

    continuouskernels["kernel"] = kernels

    return discretekernels, continuouskernels, discretekernels["loss"][-1]


def spatial_convolutions(feature, kernels, shape):
    # create empty convolution map with the same shape as the original maps
    convs = np.zeros(shape=(5, shape[0], shape[1], shape[2], shape[3]))
    maxloss = kernels["loss"][-1]

    for idx in range(0, 5):
        loss = kernels["loss"][idx]
        kernel = kernels["kernel"][idx]

        for time in range(shape[0]):
            for layer in range(shape[1]):
                for i in range(maxloss, shape[1] - maxloss):
                    for j in range(maxloss, shape[2] - maxloss):
                        convsum = np.sum(feature[time, i - loss:i + loss + 1, j - loss:j + loss + 1] * kernel)
                        convs[idx, layer, time, i, j] = convsum

    return convs


def convolutions(convs, fmap, lat, lon, sigma=3):
    featureconvs = np.empty(shape=0)
    featureconvs = np.append(featureconvs, fmap[0, int(lat), int(lon)])

    for conv in convs:
        if conv % 2 == 1:
            conv -= 1

        kernel = signal.gaussian(conv+1, std=sigma)
        kernel = np.outer(kernel, kernel)
        feature = np.nanmean(fmap[0, int(lat - conv/2):int(lat + conv/2 + 1),
                             int(lon - conv/2):int(lon + conv/2 + 1)] * kernel)

        featureconvs = np.append(featureconvs, [feature], axis=0)
    return featureconvs


