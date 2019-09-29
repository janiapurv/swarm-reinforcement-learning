import numpy as np
import pybullet as p


class Ground():
    """The class is the interface to a single robot
    """
    def __init__(self, init_pos, init_orientation, robot_id, dt):
        self.id = robot_id
        self.init_pos = init_pos
        self.init_orientation = init_orientation
        self.dt = dt
        self.velocity = np.array([0, 0, 2])
        self.pybullet_id = p.loadURDF("src/gym/urdf/ground_vehicle.urdf",
                                      self.init_pos, self.init_orientation)
        self.reset()

    def reset(self):
        """Moves the robot back to its initial position
        """
        p.resetBasePositionAndOrientation(self.pybullet_id, self.init_pos,
                                          self.init_orientation)

    def get_pos_and_orientation(self):
        """
        Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]

    def set_position(self, position):
        """This function moves the vehicles to given position

        Parameters
        ----------
        position : array
            The position to which the vehicle should be moved.
        """
        p.resetBasePositionAndOrientation(self.pybullet_id, position,
                                          self.init_orientation)


class Arial():
    """The class is the interface to a single robot
    """
    def __init__(self, init_pos, init_orientation, robot_id, dt):
        self.id = robot_id
        self.init_pos = init_pos
        self.init_orientation = init_orientation
        self.dt = dt
        self.velocity = np.array([0, 0, 2])
        self.pybullet_id = p.loadURDF("src/gym/urdf/arial_vehicle.urdf",
                                      self.init_pos, self.init_orientation)
        self.joint_ids = list(range(p.getNumJoints(self.pybullet_id)))
        self.reset()

    def reset(self):
        """Moves the robot back to its initial position
        """
        p.resetBasePositionAndOrientation(self.pybullet_id, self.init_pos,
                                          (0., 0., 0., 1.))

    def get_pos_and_orientation(self):
        """Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]

    def set_position(self, position):
        """This function moves the vehicles to given position

        Parameters
        ----------
        position : array
            The position to which the vehicle should be moved.
        """
        p.resetBasePositionAndOrientation(self.pybullet_id, position,
                                          self.init_orientation)
