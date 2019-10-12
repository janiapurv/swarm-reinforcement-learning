import numpy as np
from scipy.spatial import distance


class FormationControl():
    def __init__(self):
        # Initialise the parameters
        return None

    def get_vel(self, j, curr_pos, min_dis, centroid_pos, alpha, gamma,
                path_vel, vel_max, a, b, knn, formation_type):
        # Implementing pij which care of peers
        peers = curr_pos.copy()
        curr_loc = curr_pos[j, :]
        peers = np.delete(peers, (j), axis=0)
        dist_mat = np.zeros((len(peers), 2))
        # Finding distance of each uav f
        for j in range(len(peers)):
            dst = distance.euclidean(curr_loc, peers[j, :])
            dist_mat[j, :] = [int(j), dst]
        dist_t = dist_mat[np.argsort(
            dist_mat[:, 1])]  # sorting by lowest distance
        dist_t = (dist_t[:, 0])
        idx = []

        for i in range(len(dist_t)):
            id = int(dist_t[i])  # converting into integer
            idx.append(id)

        peers = peers[np.array(idx)]  # Arranging peers according to distance

        # knn =6
        peers = np.delete(peers, np.s_[knn:],
                          axis=0)  # deleting farther neighbours

        k = 1  # constant
        P_ij = np.array([0, 0])

        for j in range(len(peers)):
            g_lij = (min_dis**2) - (distance.euclidean(curr_loc,
                                                       peers[j, :]))**2
            del_xij = (-curr_loc + peers[j, :])
            del_g_ij = 2 * del_xij
            P_ij = P_ij + k * max(0, g_lij) * (del_g_ij)

        kl = 1  # constant
        f_g_ij = (((curr_loc[0] - centroid_pos[0])**2) / a**2) + ((
            (curr_loc[1] - centroid_pos[1])**2) / b**2) - 1

        if formation_type == 'ring':
            a_inner = a - 0.1 * a
            a_outer = a + 0.1 * a
            b_inner = b - 0.1 * b
            b_outer = b + 0.1 * b

            f_g_ij_RING_outer = ((
                (curr_loc[0] - centroid_pos[0])**2) / a_outer**2) + ((
                    (curr_loc[1] - centroid_pos[1])**2) / b_outer**2) - 1
            f_g_ij_RING_inner = 1 - ((
                (curr_loc[0] - centroid_pos[0])**2) / a_inner**2) + ((
                    (curr_loc[1] - centroid_pos[1])**2) / b_inner**2)

            f_g_ij = 0.5 * (f_g_ij_RING_outer + f_g_ij_RING_inner)

            del_f_g_ij = -2 * (-curr_loc + centroid_pos)
            del_zeta_ij = kl * max(0, f_g_ij)
        else:
            f_g_ij = (((curr_loc[0] - centroid_pos[0])**2) / a**2) + ((
                (curr_loc[1] - centroid_pos[1])**2) / b**2) - 1

            del_f_g_ij = -2 * (-curr_loc + centroid_pos)
            del_zeta_ij = kl * max(0, f_g_ij) * del_f_g_ij

        print(np.linalg.norm((alpha * del_zeta_ij) - (gamma * P_ij)))
        if np.linalg.norm((alpha * del_zeta_ij) -
                          (gamma * P_ij)) < 0.05 * vel_max:

            vel = path_vel - 0 * (alpha * del_zeta_ij) - (gamma * P_ij)
        else:
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

    def execute(self, vehicles, centroid_pos, dt, formation_type):
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
        a = 10
        b = 10
        knn = 6
        vmax = 2
        alpha = 2
        gamma = 0.5
        min_dis = 1

        all_drones_pose = np.zeros((len(vehicles), 3))
        pose_new = np.zeros((len(vehicles), 3))
        for i, vehicle in enumerate(vehicles):
            all_drones_pose[i, :] = vehicle.current_pos

        for j in range(len(vehicles)):  # Simulated-parallel run
            path_vel = np.array([
                centroid_pos[0] + dt, centroid_pos[1] + dt
            ]) - np.array([centroid_pos[0], centroid_pos[1]])
            path_vel = (1 / dt) * path_vel
            curr_pos = all_drones_pose[:, 0:2]

            vel = self.get_vel(j, curr_pos, min_dis, centroid_pos, alpha,
                               gamma, path_vel, vel_max, a, b, knn,
                               formation_type)
            dst = np.linalg.norm(vel)
            if dst > vmax:
                vel = (vmax / dst) * vel
            pos_j = all_drones_pose[j]
            pose_new[j, :] = [
                pos_j[0] + dt * vel[0], pos_j[1] + dt * vel[1],
                pos_j[2] * 0 + 3
            ]

        for j, vehicle in enumerate(vehicles):
            vehicle.updated_pos = pose_new[j, :]

        return vehicles
