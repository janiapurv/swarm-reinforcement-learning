import time
import math
from pathlib import Path

import pybullet as p
import pybullet_data

from .state_manager import StateManager
from .states import State
from .actions import Action
from .action_manager import ActionManager
from .rewards import BenningReward


class Benning(object):
    def __init__(self, config):
        if config['simulation']['headless']:
            p.connect(p.DIRECT)  # Non-graphical version
        else:
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=150,
                                         cameraYaw=0,
                                         cameraPitch=-89.999,
                                         cameraTargetPosition=[0, 80, 0])
        # Environment parameters
        self.n_ugv = config['simulation']['n_ugv']
        self.n_uav = config['simulation']['n_uav']
        self.current_time = config['simulation']['current_time']
        self.done = False
        self.config = config

        # Parameters for simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optional
        p.setGravity(0, 0, -9.81)
        # p.setTimeStep(config['simulation']['time_step'])
        # p.setRealTimeSimulation(1)

        # Initial setup of the environment
        self._initial_setup()

        # Initialize the state and action components
        self.state_manager = StateManager(self.current_time, self.config)
        self.state_manager.initial_vehicles_setup()
        self.state_manager.initial_mission_setup()
        # Composition of state manager in subsquent operations
        self.state = State(self.state_manager)
        self.reward = BenningReward(self.state_manager)
        self.action = Action(self.state_manager)
        self.action_manager = ActionManager(self.state_manager)

    def _initial_setup(self):
        # Setup ground
        plane = p.loadURDF("plane.urdf", [0, 0, 0],
                           p.getQuaternionFromEuler([0, 0, math.pi / 2]),
                           useFixedBase=True,
                           globalScaling=20)
        texture = p.loadTexture('src/envs/images/mud.png')
        p.changeVisualShape(plane, -1, textureUniqueId=texture)

        if self.config['simulation']['collision_free']:
            path = Path(
                __file__).parents[0] / 'urdf/environment_collision_free.urdf'
        else:
            path = Path(__file__).parents[0] / 'urdf/environment.urdf'
        p.loadURDF(str(path), [58.487, 23.655, 0.1],
                   p.getQuaternionFromEuler([0, 0, math.pi / 2]),
                   useFixedBase=True)

        return None

    def get_camera_image(self):
        """Get the camera image of the scene

        Returns
        -------
        tuple
            Three arrays corresponding to rgb, depth, and segmentation image.
        """
        upAxisIndex = 2
        camDistance = 500
        pixelWidth = 350
        pixelHeight = 700
        camTargetPos = [0, 80, 0]

        far = camDistance
        near = -far
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, 0, 90, 0, upAxisIndex)
        projection_matrix = p.computeProjectionMatrix(-90, 60, 150, -150, near,
                                                      far)
        # Get depth values using the OpenGL renderer
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            pixelWidth,
            pixelHeight,
            view_matrix,
            projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return rgbImg, depthImg, segImg

    def reset(self):
        """
        Resets the position of all the robots
        """
        for vehicle in self.state_manager.uav:
            vehicle.reset()

        for vehicle in self.state_manager.ugv:
            vehicle.reset()

        for i in range(20):
            time.sleep(1 / 240)
            p.stepSimulation()

        # reset time
        self.state_manager.current_time = 0
        # call the state manager
        state = self.state.get_state()
        done = False
        return state, done

    def step(self, action, hand_crafted=False):
        """Take a step in the environement
        """
        # Action splitting
        if hand_crafted:
            decoded_act_uav = action[0:3]
            decoded_act_ugv = action[3:]
        else:
            decoded_act_uav, decoded_act_ugv = self.action.get_action(action)
        # Execute the actions
        self.action_manager.primitive_execution(decoded_act_uav,
                                                decoded_act_ugv, p)
        print(self.state_manager.current_time)
        # Update the progress
        mission_done = self.state_manager.update_progress()
        # Get the new encoded state
        new_state = self.state.get_state()
        # Get reward
        reward = self.get_reward()
        return new_state, reward, mission_done

    def check_episode_done(self):
        done = False
        if self.state_manager.current_time >= self.config['simulation'][
                'total_time']:
            done = True
        return done

    def get_reward(self):
        """Update reward of all the agents
        """
        # Calculate the reward
        total_reward = self.reward.mission_reward(self.state_manager.ugv,
                                                  self.state_manager.uav,
                                                  self.config)
        for vehicle in self.state_manager.ugv:
            vehicle.reward = total_reward

        for vehicle in self.state_manager.uav:
            vehicle.reward = total_reward

        return total_reward
