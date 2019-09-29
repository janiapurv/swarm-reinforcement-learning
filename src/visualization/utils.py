import numpy as np
import matplotlib.pyplot as plt


def plot_occupancy_map(image, config, save_array=False):
    """Save the occupancy map for the given image

    Parameters
    ----------
    image : 3D array
        The occupancy 3D array

    Returns
    -------
    None
    """
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.flip(image, axis=0))
    plt.show()

    # Save the map
    if save_array:
        file_name = config['map_save_path'] + 'occupancy_map.npy'
        np.save(file_name, np.flip(image, axis=0))

    return None
