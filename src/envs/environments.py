import math
from pathlib import Path
import numpy as np

import pybullet as p
import pybullet_data

from .state_manager import StateManager
from .states import State
from .actions import Action
from .action_manager import ActionManager
from .rewards import BenningReward
from .agents import UAV, UGV


def get_initial_position(agent, n_agents):
    grid = np.arange(n_agents).reshape(n_agents // 5, 5)
    pos_xy = np.where(grid == agent)
    return [pos_xy[0][0] * 20 + 10, pos_xy[1][0] * 20]


class Benning():
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
        p.setRealTimeSimulation(1)

        # Create the ground and arial vehicles list
        self.ugv = []
        self.uav = []

        # Initial setup of the environment
        self._initial_setup(uav=self.uav, ugv=self.ugv)

        # Initialize the state and action components
        self.state_manager = StateManager(self.uav, self.ugv,
                                          self.current_time, self.config)
        self.state_manager._initial_mission_setup()
        self.state = State(self.state_manager)
        self.reward = BenningReward(self.state_manager)
        self.action = Action(self.state_manager)
        self.action_manager = ActionManager(self.state_manager)

    def _initial_setup(self, uav, ugv):
        # Setup ground

        ground = p.loadURDF("plane.urdf", [0, 0, 0],
                            p.getQuaternionFromEuler([0, 0, math.pi / 2]),
                            useFixedBase=True,
                            globalScaling=20)
        texture = p.loadTexture('src/envs/images/white_background.jpg')
        p.changeVisualShape(ground, -1, textureUniqueId=texture)
        p.configureDebugVisualizer(shadowMapWorldSize=20,
                                   lightPosition=[10, 70, 100])
        p.configureDebugVisualizer(shadowMapResolution=8192)

        if self.config['simulation']['collision_free']:
            path = Path(
                __file__).parents[0] / 'urdf/environment_collision_free.urdf'
        else:
            path = Path(__file__).parents[0] / 'urdf/environment.urdf'
        # p.loadURDF(str(path), [58.487, 23.655, 0.1],
        #            p.getQuaternionFromEuler([0, 0, math.pi / 2]),
        #            useFixedBase=True)
        p.loadURDF(str(path), [25, 140, 44],
                   p.getQuaternionFromEuler([
                       -0.45 * math.pi / 180, -24.5 * math.pi / 180,
                       -20.0 * math.pi / 180
                   ]),
                   flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                   useFixedBase=True)

        # Initialise the UGV and UAV
        init_orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        for i, item in enumerate(range(self.n_ugv)):
            position = get_initial_position(item, self.n_ugv)
            init_pos = [position[0] * 0.25 + 2.5, position[1] * 0.25, 5]
            ugv.append(UGV(init_pos, init_orientation, i, 1 / 10, self.config))

        for i, item in enumerate(range(self.n_uav)):
            position = get_initial_position(item, self.n_uav)
            init_pos = [position[0] * 0.25 + 2.5, position[1] * 0.25 - 1.5, 5]
            uav.append(UAV(init_pos, init_orientation, i, 1 / 10, self.config))

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
        for vehicle in self.ugv:
            vehicle.reset()

        for vehicle in self.uav:
            vehicle.reset()

        p.stepSimulation()

        # call the state manager
        state = self.state.get_state()
        done = False
        return state, done

    def step(self, action):
        """Take a step in the environement
        """
        # Action splitting
        decoded_actions_uav = action[0:3]
        decoded_actions_ugv = action[3:]

        # Execute the actions
        done = self.action_manager.primitive_execution(decoded_actions_uav,
                                                       decoded_actions_ugv, p)
        print(self.action_manager.current_time)
        self.state_manager.update_progress()

        # Get the new encoded state
        new_state = self.state.get_state()

        # Get reward
        reward = self.get_reward()
        # Is episode done
        # done = self.check_episode_done()

        return new_state, reward, done

    def simulate_motion(self, path_uav, path_ugv):
        # Update all the vehicles
        for vehicle in self.uav:
            pos, _ = vehicle.get_pos_and_orientation()
            pos[0] = pos[0] - 0.001 * path_uav[0]
            pos[1] = pos[1] - 0.001 * path_uav[1]
            pos[2] = 5
            vehicle.set_position(pos)

        for vehicle in self.ugv:
            pos, _ = vehicle.get_pos_and_orientation()
            pos[0] = pos[0] - 0.001 * path_ugv[0]
            vehicle.set_position(pos)
        p.stepSimulation()
        return

    def check_episode_done(self):
        done = False
        if self.current_time >= self.config['simulation']['total_time']:
            done = True
        return done

    def get_reward(self):
        """Update reward of all the agents
        """
        # Calculate the reward
        total_reward = self.reward.mission_reward(self.ugv, self.uav,
                                                  self.config)
        for vehicle in self.ugv:
            vehicle.reward = total_reward

        for vehicle in self.uav:
            vehicle.reward = total_reward

        return total_reward
