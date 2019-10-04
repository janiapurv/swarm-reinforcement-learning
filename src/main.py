import time
import yaml
from pathlib import Path

import pybullet as p
import matplotlib.pyplot as plt
from gym.environments import Environment

from visualization.utils import plot_occupancy_map
from gym.utils import get_xy_position

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'convert lat-long to cartesian') as check, check():
    get_xy_position(config)

with skip_run('skip', 'plot occupancy grid') as check, check():
    env = Environment(25, 25, headless=False)
    program_starts = time.time()
    fig, ax = plt.subplots()
    for i in range(10000):
        if i < 1:
            rgbImg, depthImg, segImg = env.get_camera_image()
            plot_occupancy_map(ax, segImg, config, save_array=True)
        plt.show()
        p.stepSimulation()
        time.sleep(1 / 250)
    program_ends = time.time()

with skip_run('skip', 'planning tactic') as check, check():
    env = Environment(25, 25, headless=False)
    program_starts = time.time()
    fig, ax = plt.subplots()
    for i in range(10000):
        env.take_action()
        p.stepSimulation()
        time.sleep(1 / 250)
    program_ends = time.time()
