"""
    Helper methods and variables
    ============================
"""

import plotly.graph_objs as go


def obstacle_generator(obstacle_map):
    """
    Generates a grid map with obstacles for testing

    Args:
        obstacle_map: Numpy array of zeros of shape atleast 100x100x100

    Returns:
        Numpy array with obstacles marked as 1
    """

    for i in range(20, 41):
        for j in range(20, 41):
            for k in range(20, 41):
                obstacle_map[i][j][k] = 0
    for i in range(60, 81):
        for j in range(20, 41):
            for k in range(20, 41):
                obstacle_map[i][j][k] = 0
    for i in range(20, 41):
        for j in range(20, 41):
            for k in range(60, 81):
                obstacle_map[i][j][k] = 0
    for i in range(60, 81):
        for j in range(20, 41):
            for k in range(60, 81):
                obstacle_map[i][j][k] = 0
    for i in range(60, 81):
        for j in range(60, 81):
            for k in range(20, 41):
                obstacle_map[i][j][k] = 0
    for i in range(20, 41):
        for j in range(60, 81):
            for k in range(20, 41):
                obstacle_map[i][j][k] = 0
    for i in range(60, 81):
        for j in range(60, 81):
            for k in range(60, 81):
                obstacle_map[i][j][k] = 0
    for i in range(20, 41):
        for j in range(60, 81):
            for k in range(60, 81):
                obstacle_map[i][j][k] = 0

    return obstacle_map


obs = go.Mesh3d(x=[20, 20, 40, 40, 20, 20, 40, 40],
                y=[20, 40, 40, 20, 20, 40, 40, 20],
                z=[20, 20, 20, 20, 40, 40, 40, 40],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                color='purple')
obs1 = go.Mesh3d(x=[60, 60, 80, 80, 60, 60, 80, 80],
                 y=[20, 40, 40, 20, 20, 40, 40, 20],
                 z=[20, 20, 20, 20, 40, 40, 40, 40],
                 i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                 j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                 k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                 color='purple')
obs2 = go.Mesh3d(x=[20, 20, 40, 40, 20, 20, 40, 40],
                 y=[20, 40, 40, 20, 20, 40, 40, 20],
                 z=[60, 60, 60, 60, 80, 80, 80, 80],
                 i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                 j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                 k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                 color='purple')
obs3 = go.Mesh3d(x=[60, 60, 80, 80, 60, 60, 80, 80],
                 y=[20, 40, 40, 20, 20, 40, 40, 20],
                 z=[60, 60, 60, 60, 80, 80, 80, 80],
                 i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                 j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                 k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                 color='purple')
obs4 = go.Mesh3d(x=[60, 60, 80, 80, 60, 60, 80, 80],
                 y=[60, 80, 80, 60, 60, 80, 80, 60],
                 z=[20, 20, 20, 20, 40, 40, 40, 40],
                 i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                 j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                 k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                 color='purple')
obs5 = go.Mesh3d(x=[20, 20, 40, 40, 20, 20, 40, 40],
                 y=[60, 80, 80, 60, 60, 80, 80, 60],
                 z=[20, 20, 20, 20, 40, 40, 40, 40],
                 i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                 j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                 k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                 color='purple')
obs6 = go.Mesh3d(x=[60, 60, 80, 80, 60, 60, 80, 80],
                 y=[60, 80, 80, 60, 60, 80, 80, 60],
                 z=[60, 60, 60, 60, 80, 80, 80, 80],
                 i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                 j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                 k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                 color='purple')
obs7 = go.Mesh3d(x=[20, 20, 40, 40, 20, 20, 40, 40],
                 y=[60, 80, 80, 60, 60, 80, 80, 60],
                 z=[60, 60, 60, 60, 80, 80, 80, 80],
                 i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                 j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                 k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                 color='purple')

meshes = [obs, obs1, obs2, obs3, obs4, obs5, obs6, obs7]
