import numpy as np
from scipy import spatial


class FormationControl():
    """ Formation control primitive using region based shape control.
    Coded by: Apurvakumar Jani, Date: 18/9/2019
    """
    def __init__(self):
        # Initialise the parameters
        return None

    def get_vel(self, j, curr_pos, min_dis, centroid_pos, alpha, gamma,
                path_vel, vel_max, a, b, knn, formation_type):

        # Calculate pairwise distance
        curr_loc = curr_pos[j, :]
        if len(curr_pos) < 6:
            knn = len(curr_pos)
        peers_pos = curr_pos[spatial.KDTree(curr_pos).query(curr_loc, k=knn
                                                            )[1], :]

        # Calculate the velocity of each neighboor particle
        k = 1 / len(peers_pos)  # constant
        g_lij = (min_dis**2) - np.linalg.norm(
            curr_loc - peers_pos, axis=1, ord=2)
        del_g_ij = 2 * (peers_pos - curr_loc)
        temp = np.maximum(0, g_lij / (min_dis**2))**2
        P_ij = k * np.dot(temp, del_g_ij)

        temp = (curr_loc - centroid_pos) / np.array([a, b])
        f_g_ij = np.linalg.norm(temp, ord=2) - 1

        # Calculate path velocity
        kl = 1  # constant
        del_f_g_ij = 1 * (curr_loc - centroid_pos)
        del_zeta_ij = (kl * max(0, f_g_ij)) * del_f_g_ij
        vel = path_vel - (alpha * del_zeta_ij) - (gamma * P_ij)

        if vel_max is not None:
            vel[0] = self.getFeasibleSpeed(vel[0], vel_max)
            vel[1] = self.getFeasibleSpeed(vel[1], vel_max)

        return vel

    def getFeasibleSpeed(self, vel, vel_max):
        """This function limit the velocity returned
        by get_vel function for the stability

        Parameters
        ----------
        vel : float
            Calculated velocity
        vel_max : float
            Maximum allowed velocity
        """
        if vel > 0:
            vel = min(vel_max, vel)
        else:
            vel = max(-vel_max, vel)

        return vel

    def execute(self, vehicles, next_pos, centroid_pos, dt, formation_type):
        """Get the position of the formation control

        Parameters
        ----------
        vehicles : list
            A list containing UAV or UGV class
        centroid_pos : array
            An array containing the x, y, and z position
        dt : float
            Time step to be used for distance calculation
        """

        # Parameters
        vel_max = 100
        a = 3
        b = 3
        knn = 6
        vmax = vehicles[0].speed
        alpha = 1
        gamma = 1
        min_dis = 2

        all_drones_pose = np.zeros((len(vehicles), 2))
        for i, vehicle in enumerate(vehicles):
            all_drones_pose[i] = vehicle.current_pos[0:2]

        vel_combined = []
        for j, vehicle in enumerate(vehicles):
            path = np.array([next_pos[0], next_pos[1]]) - centroid_pos
            path_vel = (1 / dt) * path
            vel = self.get_vel(j, all_drones_pose, min_dis, centroid_pos,
                               alpha, gamma, path_vel, vel_max, a, b, knn,
                               formation_type)
            # Normalize the velocity
            if np.linalg.norm(vel) > vmax:
                vel = (vmax / np.linalg.norm(vel)) * vel
            vel_combined.append(vel)

            # New position
            new_pos = np.zeros(3)
            new_pos[0:2] = vehicle.current_pos[0:2] + vel * dt

            # Update position
            if vehicle.type == 'uav':
                new_pos[2] = 12.0
                vehicle.updated_pos = new_pos
            else:
                new_pos[2] = 1.5
                vehicle.updated_pos = new_pos

        vel_combined = np.linalg.norm(np.array(vel_combined), axis=1)

        if np.max(vel_combined) < 0.015 * len(all_drones_pose):
            formation_done = True
        else:
            formation_done = False

        return vehicles, formation_done
