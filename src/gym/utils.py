import math

import numpy as np
import pandas as pd


def haversine_dist(init_point, final_point):
    """Gives the Haversine distance between the initial and final point

    Parameters
    ----------
    init_point : array
        The initial point
    final_point : array
        The final point

    Returns
    -------
    float
        The Haversine distance
    """

    # Assuming the input is in degrees
    init_rad = init_point * math.pi / 180
    final_rad = final_point * math.pi / 180

    d_latitude = final_rad[0] - init_rad[0]
    d_longitude = final_rad[1] - init_rad[1]

    a = math.sin(d_latitude / 2)**2 + math.cos(init_rad[0]) * math.cos(
        final_rad[0]) * math.sin(d_longitude / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    if d_longitude < 0:
        c = c * (-1)

    return 6356.752e3 * c


def get_xy_position():
    """Get the x and y position of all the buildings
    """
    data = np.genfromtxt('src/gym/latitude_longitude.csv',
                         delimiter=',',
                         skip_header=True)

    init_point = data[0]
    xy_pos = np.zeros(data.shape)
    for i, item in enumerate(data):
        temp_x = item.copy()
        temp_x[0] = init_point[0]
        xy_pos[i, 0] = haversine_dist(init_point, temp_x)

        temp_y = item.copy()
        temp_y[1] = init_point[1]
        xy_pos[i, 1] = haversine_dist(init_point, temp_y)

    df = pd.DataFrame(xy_pos)
    filepath = 'src/gym/co_ordinates.xlsx'
    df.to_excel(filepath, index=False)

    return None
