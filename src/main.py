import time
import yaml
from pathlib import Path

import pybullet as p
from gym.environments import Environment

from visualization.utils import plot_occupancy_map

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('run', 'Planning_tactics') as check, check():
    env = Environment(25, 25, headless=False)
    program_starts = time.time()
    for i in range(10000):
        if i < 1:
            rgbImg, depthImg, segImg = env.get_camera_image()
            plot_occupancy_map(segImg, config, save_array=True)
        # env.take_action()
        env.get_camera_image()
        p.stepSimulation()
        time.sleep(1 / 250)
    program_ends = time.time()
    print('Total time taken = ', program_ends - program_starts)
