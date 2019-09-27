import math
import numpy as np

import pybullet as p
import pybullet_data
from .agents import Ground, Arial


def get_position(agent):
    grid = np.arange(25).reshape(5, 5)
    pos_xy = np.where(grid == agent)
    return pos_xy


class Environment():
    def __init__(self, n_ground, n_arial, headless=True):
        self.n_ground = n_ground
        self.n_arial = n_arial
        if headless:
            p.connect(p.DIRECT)  # Non-graphical version
        else:
            p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)

        self.ground_vehicles = []
        self.arial_vehicles = []

        # Create the ground and arial vehicles
        plane_orientation = p.getQuaternionFromEuler([0, 0, math.pi / 2])
        init_orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])

        # Setup ground
        p.loadURDF("plane.urdf", [0, 0, 0],
                   plane_orientation,
                   useFixedBase=True)
        p.loadURDF("src/gym/urdf/env.urdf", [10, 0, 0],
                   plane_orientation,
                   useFixedBase=True,
                   flags=p.URDF_INITIALIZE_SAT_FEATURES)
        for i, item in enumerate(range(n_ground)):
            position = get_position(item)
            init_pos = [position[0] * 0.25 + 2.5, position[1] * 0.25, 2]
            self.ground_vehicles.append(
                Ground(init_pos, init_orientation, i, 1 / 10))

        for i, item in enumerate(range(n_arial)):
            position = get_position(item)
            init_pos = [position[0] * 0.25 + 2.5, position[1] * 0.25 - 1.5, 2]
            self.arial_vehicles.append(
                Arial(init_pos, init_orientation, i, 1 / 10))

    def get_camera_image(self):
        roll = 0
        upAxisIndex = 2
        camDistance = 4
        pixelWidth = 256
        pixelHeight = 256
        camTargetPos = [0, 0, 0]

        fov = 120
        aspect = pixelWidth / pixelHeight
        near = 0.01
        far = -10
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, 0, -90, roll, upAxisIndex)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)
        # Get depth values using the OpenGL renderer
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            pixelWidth,
            pixelHeight,
            view_matrix,
            projection_matrix,
            shadow=True,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return rgbImg, depthImg, segImg

    def reset(self):
        """
        Resets the position of all the robots
        """
        for vehicle in self.ground_vehicles:
            vehicle.reset()

        for vehicle in self.arial_vehicles:
            vehicle.reset()
        p.stepSimulation()

    def take_action(self):
        # Take random action as of now
        rand_step_x = np.random.rand(1)
        rand_step_y = np.random.rand(1)
        for vehicle in self.ground_vehicles:
            current_pos, _ = vehicle.get_pos_and_orientation()
            current_pos[0] = vehicle.init_pos[0] + rand_step_x * 2 - 4.5
            current_pos[1] = vehicle.init_pos[1] + rand_step_y * 2 - 2.5
            current_pos[2] = 1
            vehicle.set_position(current_pos)

        for vehicle in self.arial_vehicles:
            current_pos, _ = vehicle.get_pos_and_orientation()
            current_pos
            vehicle.set_position(current_pos)
        p.stepSimulation()

    def calculate_reward(self, vehicle):
        pos = vehicle.get_pos_and_orientation()
        # Calculate reward
        reward = 0 * pos
        return reward

    def reward(self):
        for vehicle in self.ground_vehicles:
            self.reward.append(self.calculate_reward(vehicle))

        for vehicle in self.ground_vehicles:
            self.reward.append(self.calculate_reward(vehicle))
