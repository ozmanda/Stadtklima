import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


def generate_maps(maps, date, imgdir):
    layerdir = os.path.join(imgdir, "Layermaps")
    os.mkdir(layerdir)

    if len(maps.shape) == 3:
        vmin = np.nanmin(maps)-5
        vmax = np.nanmax(maps)+5
        for mapidx in range(maps.shape[0]):
            map = maps[mapidx, :, :]
            tempmap = sns.heatmap(map, vmin=vmin, vmax=vmax, linewidth=0)
            plt.close()
            fig = tempmap.get_figure()
            fig.savefig(os.path.join(layerdir, f'{date}_tempmap{mapidx}.png'))
        animate(layerdir)

    elif len(maps.shape) == 4:
        vmin = np.nanmin(maps)-5
        vmax = np.nanmax(maps)+5
        for layer in range(maps.shape[1]):
            for mapidx in range(maps.shape[0]):
                map = maps[mapidx, layer, :, :]
                tempmap = sns.heatmap(map, vmin=vmin, vmax=vmax, linewidth=0, cbar=False,
                                      yticklabels=False, xticklabels=False)
                plt.close()
                fig = tempmap.get_figure()
                fig.savefig(os.path.join(layerdir, f'{date}_layer{layer}_tempmap{mapidx}.png'),
                            bbox_inches='tight', pad_inches=0)

        animate(layerdir)
        generate_surfacemaps(maps, date, imgdir)

    return


def generate_surfacemaps(maps, date, imgdir):
    surfacedir = os.path.join(imgdir, "Surfacemaps")
    os.mkdir(surfacedir)

    surfacemap = np.empty(shape=(maps.shape[0], maps.shape[2], maps.shape[3]))

    for idx, _ in np.ndenumerate(surfacemap[1, :, :]):
        for layer in range(maps.shape[1]):
            if not np.isnan(maps[1, layer, idx[0], idx[1]]):
                surfacemap[:, idx[0], idx[1]] = maps[:, layer, idx[0], idx[1]]
                break

    vmin = np.nanmin(maps) - 5
    vmax = np.nanmax(maps) + 5
    for mapidx in range(maps.shape[0]):
        map = surfacemap[mapidx, :, :]
        surfacetempmap = sns.heatmap(map, vmin=vmin, vmax=vmax, linewidth=0, cbar=False,
                                     yticklabels=False, xticklabels=False)
        plt.close()
        fig = surfacetempmap.get_figure()
        fig.savefig(os.path.join(imgdir, f'{date}_surfacetemp{mapidx}.png'),
                            bbox_inches='tight', pad_inches=0)

    animate(imgdir)

    return


def animate(imgdir=os.path.join(os.getcwd(), "Images")):
    folder = input("Which folder should pictures be imported from? (only enter folder name) ")
    picdir = os.path.join(imgdir, folder)

    frames = [Image.open(os.path.join(picdir, imagename)) for imagename in os.listdir(picdir) if imagename.endswith(".png")]
    frame = frames[0]
    frame.save(os.path.join(picdir, f'{folder}_animation.GIF'), format="GIF",
               append_images=frames, save_all=True, duration=100, loop=0)

