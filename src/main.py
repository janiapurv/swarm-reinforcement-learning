import time
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import pybullet as p

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
    for i in range(2):
        if i < 1:
            rgbImg, depthImg, segImg = env.get_camera_image()
            plot_occupancy_map(ax, segImg, config, save_array=True)
        plt.show()
        p.stepSimulation()
        time.sleep(1 / 250)
    program_ends = time.time()

with skip_run('run', 'learning tactic') as check, check():
    env = Environment(config)
    # ['n_robots', 'primitive', 'target_node_id', 0, 0, 0]
    net_output = [[10, 2, 38, 0, 0, 0], [5, 2, 39, 0, 0, 0],
                  [10, 2, 40, 0, 0, 0], [10, 2, 38, 0, 0, 0],
                  [5, 2, 39, 0, 0, 0], [10, 2, 40, 0, 0, 0]]
    for j in range(1000):
        start = time.time()
        # print(j)
        env.step(net_output)
        # print(time.time() - start)
