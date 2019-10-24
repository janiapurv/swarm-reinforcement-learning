import time
import yaml
from pathlib import Path

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from envs.environments import Environment
from envs.utils import get_xy_position

# from models.torch_network import Actor, Critic
# from models.torch_train import AdvantageCritic

from visualization.utils import plot_occupancy_map

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'convert lat-long to cartesian') as check, check():
    get_xy_position(config)

with skip_run('skip', 'plot occupancy grid') as check, check():
    env = Environment(config)
    program_starts = time.time()
    fig, ax = plt.subplots()
    for i in range(10000):
        if i < 1:
            rgbImg, depthImg, segImg = env.get_camera_image()
            plot_occupancy_map(ax,
                               np.rot90(depthImg),
                               config,
                               save_array=False)
            plt.show()
        p.stepSimulation()
        time.sleep(1 / 250)
    program_ends = time.time()

with skip_run('run', 'learning tactic') as check, check():

    env = Environment(config)
    # ['n_robots', 'primitive', 'target_node_id', 0, 0, 0]
    net_output_1 = [[9, 1, 38, 0, 0, 0], [6, 1, 39, 0, 0, 0],
                    [10, 1, 40, 0, 0, 0], [12, 1, 15, 0, 0, 0],
                    [9, 1, 12, 0, 0, 0], [4, 1, 11, 0, 0, 0]]
    start = time.time()
    for j in range(4):
        print(j)
        _, _, done = env.step(net_output_1)
        if done:
            break

    print(time.time() - start)

    net_output_2 = [[10, 2, 38, 0, 0, 0], [0, 2, 39, 0, 0, 0],
                    [15, 2, 40, 0, 0, 0], [13, 2, 15, 0, 0, 0],
                    [12, 2, 12, 0, 0, 0], [0, 2, 11, 0, 0, 0]]
    for j in range(2):
        print(j)
        _, _, done = env.step(net_output_2)
        if done:
            break

    print(time.time() - start)

    net_output_3 = [[25, 2, 37, 0, 0, 0], [0, 2, 39, 0, 0, 0],
                    [0, 2, 40, 0, 0, 0], [25, 2, 38, 0, 0, 0],
                    [0, 2, 12, 0, 0, 0], [0, 2, 11, 0, 0, 0]]
    for j in range(2):
        print(j)
        _, _, done = env.step(net_output_3)
        if done:
            break

    print(time.time() - start)

    net_output_4 = [[25, 1, 37, 0, 0, 0], [0, 2, 39, 0, 0, 0],
                    [0, 2, 40, 0, 0, 0], [25, 2, 37, 0, 0, 0],
                    [0, 2, 12, 0, 0, 0], [0, 2, 11, 0, 0, 0]]
    for j in range(2):
        print(j)
        _, _, done = env.step(net_output_4)
        if done:
            break

    print(time.time() - start)
