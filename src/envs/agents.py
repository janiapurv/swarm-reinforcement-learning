from pathlib import Path

import numpy as np
import pybullet as p


class UGV():
    """The class is the interface to a single robot
    """
    def __init__(self, init_pos, init_orientation, robot_id, dt, config):
        # Properties UGV
        self.vehicle_id = robot_id
        self.init_pos = init_pos
        self.current_pos = init_pos
        self.updated_pos = init_pos
        self.init_orientation = init_orientation
        self.cluster_id = 0
        self.idle = True
        self.ammo = 100
        self.functional = True
        self.speed = config['ugv']['speed']
        self.search_speed = config['ugv']['search_speed']
        self.type = 'ugv'

        # Config
        self.config = config

        # Simulation parameters
        self.reward = 0

        self._initial_setup()
        self.reset()

    def _initial_setup(self):
        if self.config['simulation']['collision_free']:
            path = Path(__file__).parents[
                0] / 'urdf/ground_vehicle_collision_free.urdf'
        else:
            path = Path(__file__).parents[0] / 'urdf/ground_vehicle.urdf'
        self.object_id = p.loadURDF(str(path), self.init_pos,
                                    self.init_orientation)
        self.constraint = p.createConstraint(self.object_id, -1, -1, -1,
                                             p.JOINT_FIXED, [0, 0, 0],
                                             [0, 0, 0], self.init_pos)
        return None

    def reset(self):
        """Moves the robot back to its initial position
        """
        p.resetBasePositionAndOrientation(self.object_id, self.init_pos,
                                          self.init_orientation)
        return None

    def get_pos_and_orientation(self):
        """
        Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.object_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]

    def get_info(self):
        """Returns the information about the UGV

        Returns
        -------
        dict
            A dictionary containing all the information
        """
        info = {}
        info['vehicle_id'] = self.vehicle_id
        info['current_pos'] = self.current_pos
        info['updated_pos'] = self.updated_pos
        info['idle'] = self.idle
        info['ammo'] = self.ammo
        info['functional'] = self.functional
        info['type'] = self.type

        return info

    def set_position(self, position):
        """This function moves the vehicles to given position

        Parameters
        ----------
        position : array
            The position to which the vehicle should be moved.
        """
        pos, _ = self.get_pos_and_orientation()
        self.current_pos = pos
        p.changeConstraint(self.constraint, position)

        return None


class UAV():
    """The class is the interface to a single robot
    """
    def __init__(self, init_pos, init_orientation, robot_id, dt, config):
        # Properties UGV
        self.vehicle_id = robot_id
        self.init_pos = init_pos
        self.current_pos = init_pos
        self.updated_pos = init_pos
        self.init_orientation = init_orientation
        self.cluster_id = 0
        self.idle = True
        self.battery = 100
        self.functional = True
        self.speed = config['uav']['speed']
        self.search_speed = config['uav']['search_speed']
        self.type = 'uav'

        # Config
        self.config = config

        # Simulation parameters
        self.reward = 0

        self._initial_setup()
        self.reset()

    def _initial_setup(self):
        if self.config['simulation']['collision_free']:
            path = Path(
                __file__).parents[0] / 'urdf/arial_vehicle_collision_free.urdf'
        else:
            path = Path(__file__).parents[0] / 'urdf/arial_vehicle.urdf'
        self.object_id = p.loadURDF(str(path), self.init_pos,
                                    self.init_orientation)
        self.constraint = p.createConstraint(self.object_id, -1, -1, -1,
                                             p.JOINT_FIXED, [0, 0, 0],
                                             [0, 0, 0], self.init_pos)
        return None

    def reset(self):
        """Moves the robot back to its initial position
        """
        p.resetBasePositionAndOrientation(self.object_id, self.init_pos,
                                          (0., 0., 0., 1.))
        return None

    def get_pos_and_orientation(self):
        """Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.object_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]

    def get_info(self):
        """Returns the information about the UGV

        Returns
        -------
        dict
            A dictionary containing all the information
        """
        info = {}
        info['vehicle_id'] = self.vehicle_id
        info['current_pos'] = self.current_pos
        info['updated_pos'] = self.updated_pos
        info['idle'] = self.idle
        info['battery'] = self.battery
        info['functional'] = self.functional
        info['type'] = self.type

        return info

    def set_position(self, position):
        """This function moves the vehicles to given position

        Parameters
        ----------
        position : array
            The position to which the vehicle should be moved.
        """
        pos, _ = self.get_pos_and_orientation()
        self.current_pos = pos
        p.changeConstraint(self.constraint, position)

        return None
