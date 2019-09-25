import time
import pybullet as p

from gym.environments import Environment

env = Environment(1, 1, headless=False)

for i in range(10000):
    env.get_camera_image()
    p.stepSimulation()
    time.sleep(1 / 250)
