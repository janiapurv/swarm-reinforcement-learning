import math as mt
import pickle
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

from .state_manager import StateManager

from primitives.planning.planners import RRT
from primitives.planning.maps import GridObstacleMap
from primitives.planning.plots import Plot2D

from primitives.formation.control import FormationControl
from primitives.mrta.task_allocation import MRTA


class ActionManager(StateManager):
    def __init__(self, state_manager):
        super(ActionManager,
              self).__init__(state_manager.uav, state_manager.ugv,
                             state_manager.current_time, state_manager.config)
        self.state_manager = state_manager
        self.mrta = MRTA()

        # Setup the platoons
        self._init_platoons_setup()
        return None

    def _init_platoons_setup(self):
        """Initial setup of platoons with primitive execution class
        """

        self.uav_platoon = []
        for i in range(self.config['simulation']['n_ugv_platoons']):
            self.uav_platoon.append(PrimitiveManager(self.state_manager))

        self.ugv_platoon = []
        for i in range(self.config['simulation']['n_ugv_platoons']):
            self.ugv_platoon.append(PrimitiveManager(self.state_manager))
        return None

    def get_robot_group_info(self, vehicles, decoded_actions):
        """Calculates the robot and group info needed for task allocation

        Parameters
        ----------
        vehicles : list
            A list of UAV or UGV vehicles class
        decoded_actions : array
            The decoded actions array

        Returns
        -------
        robotInfo, groupInfo, targetInfo
        """
        # Get the non idle vehicle info and update it
        robotInfo = np.zeros((len(vehicles), 3))
        for i, vehicle in enumerate(vehicles):
            robotInfo[i, 0:2] = vehicle.current_pos[0:2]
            if vehicle.type == 'uav':
                robotInfo[i, 2] = vehicle.battery
            else:
                robotInfo[i, 2] = vehicle.ammo

        # Get the group/target info
        groupInfo = np.zeros((len(decoded_actions), 6))
        for i, actions in enumerate(decoded_actions):
            info = self.node_info(actions[2])
            groupInfo[i, 0:2] = info['position'][0:2]
            groupInfo[i, 2] = mt.floor(actions[0])
            groupInfo[i, 3] = groupInfo[i, 3] * 0 + 1
            groupInfo[i, 4] = groupInfo[i, 4] * 0
            groupInfo[i, 5] = groupInfo[i, 5] * 0 + 600
        return robotInfo, groupInfo

    def primitive_parameters(self, decode_actions, vehicles_id, group_center,
                             type):
        info = {}
        info['vehicles_id'] = vehicles_id
        info['primitive_id'] = -1
        info['end_pos'] = [0, 0]
        info['formation_type'] = None
        info['vehicle_type'] = type

        # Decoded actions is of the form
        # ['n_vehicles', 'primitive_id', 'target_id']
        # should implement as a dict
        if decode_actions[1] == 1:
            target_info = self.node_info(decode_actions[2])
            info['end_pos'] = target_info['position']
            info['primitive_id'] = decode_actions[1]

        elif decode_actions[1] == 2:
            target_info = self.node_info(decode_actions[2])
            info['end_pos'] = target_info['position']
            if decode_actions[3] == 0:
                info['formation_type'] = 'solid'
            else:
                info['formation_type'] = 'ring'
            info['primitive_id'] = decode_actions[1]
        return info

    def perform_marta_task_allocation(self, decoded_actions_uav,
                                      decoded_actions_ugv):
        """Perfroms task allocation using MRTA

        Parameters
        ----------
        decoded_actions_uav : array
            UAV decoded actions
        decoded_actions_ugv : array
            UGV decoded actions
        """
        # UAV allocation
        robotInfo, groupInfo = self.get_robot_group_info(
            self.uav, decoded_actions_uav)

        # MRTA
        robotInfo, groupCenter = self.mrta.allocateRobots(robotInfo, groupInfo)
        for i in range(self.config['simulation']['n_uav_platoons']):
            vehicles_id = [
                j for j, item in enumerate(robotInfo) if item - 1 == i
            ]
            parameters = self.primitive_parameters(decoded_actions_uav[i],
                                                   vehicles_id, [0, 0], 'uav')
            self.uav_platoon[i]._init_setup(parameters)

        # UGV allocation
        robotInfo, groupInfo = self.get_robot_group_info(
            self.ugv, decoded_actions_uav)
        # MRTA
        robotInfo, groupCenter = self.mrta.allocateRobots(robotInfo, groupInfo)
        for i in range(self.config['simulation']['n_ugv_platoons']):
            vehicles_id = [
                j for j, item in enumerate(robotInfo) if item - 1 == i
            ]
            parameters = self.primitive_parameters(decoded_actions_ugv[i],
                                                   vehicles_id, [0, 0], 'ugv')
            self.ugv_platoon[i]._init_setup(parameters)
        return None

    def perform_task_allocation(self, decoded_actions_uav,
                                decoded_actions_ugv):
        """Perfroms task allocation using MRTA

        Parameters
        ----------
        decoded_actions_uav : array
            UAV decoded actions
        decoded_actions_ugv : array
            UGV decoded actions
        """
        ids = 0
        for i in range(self.config['simulation']['n_uav_platoons']):
            vehicles_id = list(range(ids, ids + decoded_actions_uav[i][0]))
            ids = ids + decoded_actions_uav[i][0]
            parameters = self.primitive_parameters(decoded_actions_uav[i],
                                                   vehicles_id, [0, 0], 'uav')
            self.uav_platoon[i]._init_setup(parameters)

        ids = 0
        for i in range(self.config['simulation']['n_ugv_platoons']):
            vehicles_id = list(range(ids, ids + decoded_actions_ugv[i][0]))
            ids = ids + decoded_actions_ugv[i][0]
            parameters = self.primitive_parameters(decoded_actions_ugv[i],
                                                   vehicles_id, [0, 0], 'ugv')
            self.ugv_platoon[i]._init_setup(parameters)
        return None

    def primitive_execution(self,
                            decoded_actions_uav,
                            decoded_actions_ugv,
                            p_simulation,
                            hand_coded=True):
        """Performs task execution

        Parameters
        ----------
        decoded_actions_uav : array
            UAV decoded actions
        decoded_actions_ugv : [type]
            UAV decoded actions
        p_simulation : bullet engine
            Bullet engine to execute the simulation
        hand_coded : bool
            Whether hand coded tactics are being used or not
        """

        if hand_coded:
            self.perform_task_allocation(decoded_actions_uav,
                                         decoded_actions_ugv)
        else:
            self.perform_marta_task_allocation(decoded_actions_uav,
                                               decoded_actions_ugv)

        done_rolling_primitive = False
        # Execute them
        for i in range(500):
            # Update the time
            self.current_time = self.current_time + self.config['simulation'][
                'time_step']
            done = []
            # Update all the uav vehicles
            for i in range(self.config['simulation']['n_uav_platoons']):
                if self.uav_platoon[i].n_vehicles > 0:
                    done.append(
                        self.uav_platoon[i].execute_primitive(p_simulation))

            # Update all the ugv vehicles
            for i in range(self.config['simulation']['n_ugv_platoons']):
                if self.ugv_platoon[i].n_vehicles > 0:
                    done.append(
                        self.ugv_platoon[i].execute_primitive(p_simulation))

            # if all(item for item in done):
            #     done_rolling_primitive = True
            #     break
            p_simulation.stepSimulation()

            # Video recording and logging
            if self.config['record_video']:
                p_simulation.startStateLogging(
                    p_simulation.STATE_LOGGING_VIDEO_MP4,
                    self.config['log_path'] + "tactic.mp4")
            if self.config['log_states']:
                p_simulation.startStateLogging(
                    p_simulation.STATE_LOGGING_GENERIC_ROBOT,
                    self.config['log_path'] + "LOG00048.TXT")
        return done_rolling_primitive


class PrimitiveManager(StateManager):
    def __init__(self, state_manager):
        super(PrimitiveManager,
              self).__init__(state_manager.uav, state_manager.ugv,
                             state_manager.current_time, state_manager.config)
        self.state_manager = state_manager
        obstacele_map = GridObstacleMap(grid=state_manager.grid_map)
        self.planning = RRT(obstacele_map,
                            k=5000,
                            dt=5,
                            init=(185, 65),
                            low=(0, 0),
                            high=(350, 700),
                            dim=2)

        # Save the RRT obstacele_map
        path = self.config['rrt_data_path'] + '/rrt_object.pkl'
        with open(path, 'wb') as output:
            pickle.dump(self.planning, output, pickle.HIGHEST_PROTOCOL)

        start_p = self.convert_pixel_ordinate([0, 0], ispixel=False)
        end_p = self.convert_pixel_ordinate([40, 200], ispixel=False)
        path = self.planning.find_path(start_p, end_p)
        for item in path:
            plt.scatter(item[0], item[1], s=50)
        Plot2D().draw_rrt(self.planning.rrt,
                          draw_nodes=False,
                          omap=state_manager.grid_map.transpose())
        self.formation = FormationControl()
        return None

    def _init_setup(self, primitive_info):
        """Peform initial setup of the primitive
        class with vehicles and primitive information

        Parameters
        ----------
        primitive_info: dict
            A dictionary containing information about vehicles
            and primitive realted parameters.
        """
        # Update vehicles
        self.vehicles_id = primitive_info['vehicles_id']

        if primitive_info['vehicle_type'] == 'uav':
            self.vehicles = [self.uav[j] for j in self.vehicles_id]
        else:
            self.vehicles = [self.ugv[j] for j in self.vehicles_id]
        self.n_vehicles = len(self.vehicles)

        # Primitive parameters
        self.primitive_id = primitive_info['primitive_id']
        self.formation_type = primitive_info['formation_type']
        self.end_pos = primitive_info['end_pos']
        self.count = 0

        return None

    def make_vehicles_idle(self):
        for vehicle in self.vehicles:
            vehicle.idle = True
        return None

    def make_vehicles_nonidle(self):
        for vehicle in self.vehicles:
            vehicle.idle = False
        return None

    def get_centroid(self):
        centroid = []
        for vehicle in self.vehicles:
            centroid.append(vehicle.current_pos)
        centroid = np.mean(np.asarray(centroid), axis=0)
        return centroid[0:2]  # only x and y

    def execute_primitive(self, p):
        """Perform primitive execution
        """
        primitives = [self.planning_primitive, self.formation_primitive]
        done = primitives[self.primitive_id - 1](p)
        return done

    def convert_pixel_ordinate(self, point, ispixel):
        if not ispixel:
            converted = [point[0] / 0.42871 + 145, point[1] / 0.42871 + 115]
        else:
            converted = [(point[0] - 145) * 0.42871,
                         (point[1] - 115) * 0.42871]

        return converted

    def get_spline_points(self):
        # Perform planning and fit a spline
        self.start_pos = self.get_centroid()
        pixel_start = self.convert_pixel_ordinate(self.start_pos,
                                                  ispixel=False)
        pixel_end = self.convert_pixel_ordinate(self.end_pos, ispixel=False)
        path = self.planning.find_path(pixel_start, pixel_end)

        # Convert to cartesian co-ordinates
        points = np.zeros((len(path), 2))
        for i, point in enumerate(path):
            points[i, :] = self.convert_pixel_ordinate(point, ispixel=True)

        if points.shape[0] > 3:
            tck, u = interpolate.splprep(points.T)
            unew = np.linspace(u[0], u[2], 5)
            x_new, y_new = interpolate.splev(unew, tck)
        else:
            f = interpolate.interp1d(points[:, 0], points[:, 1])
            x_new = np.linspace(points[0, 0], points[-1, 0], 10)
            y_new = f(x_new)

        new_points = np.array([x_new, y_new]).T
        return new_points

    def planning_primitive(self, p):
        """Performs path planning primitive
        """
        if self.count == 0:
            # First point of formation
            self.centroid_pos = self.get_centroid()
            self.next_pos = self.centroid_pos
            formation_done = self.formation_primitive(p)
            if formation_done:
                self.count = 1
                self.new_points = self.get_spline_points()
                print('yes')
                for item in self.new_points:
                    temp = item
                    pos = [temp[0], temp[1], 2]
                    a = p.createVisualShape(p.GEOM_SPHERE,
                                            radius=1,
                                            rgbaColor=[1, 0, 0, 1],
                                            visualFramePosition=pos)
                    p.createMultiBody(0, baseVisualShapeIndex=a)
        else:
            self.centroid_pos = self.get_centroid()
            self.new_points = self.get_spline_points()
            current_dist = np.linalg.norm(self.end_pos - self.centroid_pos)
            end_pos_dist = np.linalg.norm(self.end_pos - self.new_points,
                                          axis=1)
            index = np.where(end_pos_dist < 0.95 * current_dist)[0]
            if len(index) > 0:
                self.next_pos = self.new_points[1]
            else:
                self.next_pos = self.end_pos
            formation_done = self.formation_primitive(p)

        return formation_done

    def formation_primitive(self, p):
        """Performs formation primitive
        """
        if self.primitive_id == 2:
            self.centroid_pos = self.end_pos
            self.next_pos = self.end_pos

        # self.make_vehicles_nonidle()

        dt = self.config['simulation']['time_step']
        self.vehicles, formation_done = self.formation.execute(
            self.vehicles, self.next_pos, self.centroid_pos, dt,
            self.formation_type)
        for vehicle in self.vehicles:
            vehicle.set_position(vehicle.updated_pos)

        if formation_done:
            self.make_vehicles_idle()

        return formation_done


# for item in path:
#     temp = self.convert_pixel_ordinate(item, ispixel=True)
#     pos = [temp[0], temp[1], 2]
#     a = p.createVisualShape(p.GEOM_SPHERE,
#                             radius=1,
#                             rgbaColor=[1, 0, 0, 1],
#                             visualFramePosition=pos)
#     p.createMultiBody(0, baseVisualShapeIndex=a)

# start_p = self.convert_pixel_ordinate([0, 0], ispixel=False)
# end_p = self.convert_pixel_ordinate([40, 200], ispixel=False)
# path = self.planning.find_path(start_p, end_p)
# for item in path:
#     plt.scatter(item[0], item[1], s=50)
# Plot2D().draw_rrt(self.planning.rrt,
#                   draw_nodes=False,
#                   omap=state_manager.grid_map.transpose())
