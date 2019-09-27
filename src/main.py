import time
import pybullet as p

from gym.environments import Environment

env = Environment(25, 25, headless=False)

program_starts = time.time()
for i in range(10000):
    env.get_camera_image()
    env.take_action()
    p.stepSimulation()
    time.sleep(1 / 250)

program_ends = time.time()

print('Total time taken = ', program_ends - program_starts)
