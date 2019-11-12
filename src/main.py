import time
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from envs.environments import Benning
from envs.utils import get_xy_position

from models.torch_network import Actor, Critic
from models.torch_train import AdvantageCritic

from visualization.utils import plot_occupancy_map

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'convert lat-long to cartesian') as check, check():
    get_xy_position(config)

with skip_run('skip', 'plot occupancy grid') as check, check():
    env = Benning(config)
    program_starts = time.time()
    fig, ax = plt.subplots()
    for i in range(10000):
        if i == 10:
            rgbImg, depthImg, segImg = env.get_camera_image()
            plot_occupancy_map(ax,
                               np.rot90(depthImg, k=2),
                               config,
                               save_array=True)
            plt.show()
        time.sleep(1 / 250)
    program_ends = time.time()

with skip_run('skip', 'hand crafted tactics') as check, check():

    env = Benning(config)
    # ['n_robots', 'primitive', 'target_node_id', 0, 0, 0]
    net_output_1 = [[20, 1, 38, 0, 0, 0], [10, 1, 39, 0, 0, 0],
                    [20, 1, 40, 0, 0, 0], [12, 1, 15, 0, 0, 0],
                    [9, 1, 12, 0, 0, 0], [4, 1, 11, 0, 0, 0]]
    start = time.time()
    for j in range(10):
        print(j)
        _, _, done = env.step(net_output_1)
        env.reset()
        if done:
            break

    net_output_2 = [[25, 1, 38, 0, 0, 0], [0, 1, 39, 0, 0, 0],
                    [25, 1, 40, 0, 0, 0], [13, 1, 15, 0, 0, 0],
                    [12, 1, 12, 0, 0, 0], [0, 1, 11, 0, 0, 0]]
    for j in range(1):
        print(j)
        _, _, done = env.step(net_output_2)
        if done:
            break

    net_output_3 = [[50, 1, 37, 0, 0, 0], [0, 1, 39, 0, 0, 0],
                    [0, 1, 40, 0, 0, 0], [25, 1, 38, 0, 0, 0],
                    [0, 1, 12, 0, 0, 0], [0, 1, 11, 0, 0, 0]]
    for j in range(1):
        print(j)
        _, _, done = env.step(net_output_3)
        if done:
            break

    net_output_4 = [[50, 1, 37, 0, 0, 0], [0, 1, 39, 0, 0, 0],
                    [0, 1, 40, 0, 0, 0], [25, 1, 37, 0, 0, 0],
                    [0, 1, 12, 0, 0, 0], [0, 1, 11, 0, 0, 0]]
    for j in range(1):
        print(j)
        _, _, done = env.step(net_output_4)
        if done:
            break

with skip_run('skip', 'check learning tactics') as check, check():
    env = Benning(config)
    net_output_1 = [[20, 1, 38, 0, 0, 0], [10, 1, 39, 0, 0, 0],
                    [20, 1, 40, 0, 0, 0], [12, 1, 15, 0, 0, 0],
                    [9, 1, 12, 0, 0, 0], [4, 1, 11, 0, 0, 0]]
    for j in range(10):
        rand_input = np.random.rand(18)
        print(j)
        _, _, done = env.step(rand_input)
        # env.reset()
        if done:
            break

with skip_run('run', 'check learning tactics') as check, check():
    env = Benning(config)
    actor_critic = AdvantageCritic(config)
    actor_critic.train(env, Actor, Critic)
