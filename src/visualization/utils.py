import numpy as np


def plot_occupancy_map(ax, image, config, save_array=False):
    """Save the occupancy map for the given image

    Parameters
    ----------
    image : 3D array
        The occupancy 3D array

    Returns
    -------
    None
    """

    temp = (image - np.mean(image)) / np.std(image)
    temp[temp > 0.2] = 1
    temp[(temp < 0.2) & (temp > 0)] = 0
    temp[temp < 0] = -1
    ax.imshow(np.flip(temp, axis=0), origin='lower')

    # Save the map
    if save_array:
        file_name = config['map_save_path'] + 'occupancy_map.npy'
        np.save(file_name, np.flip(temp, axis=0))

    return None
