import math

import pybullet as p
import pybullet_data
from .agents import Ground, Arial


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
        p.loadURDF("src/gym/urdf/environment.urdf", [-5, 0, 0],
                   init_orientation,
                   useFixedBase=True)
        for i, item in enumerate(range(n_ground)):
            position = [i + 2, item, 0.1]
            self.ground_vehicles.append(
                Ground(position, init_orientation, i, 1 / 10))

        for i, item in enumerate(range(n_arial)):
            position = [i + 2.5, item, 0.1]
            self.arial_vehicles.append(
                Arial(position, init_orientation, i, 1 / 10))

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
        for vehicle in self.ground_vehicles:
            position = 0  # from external file
            vehicle.set_position(position)

        for vehicle in self.arial_vehicles:
            position = 0  # from external file
            vehicle.set_position(position)
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
