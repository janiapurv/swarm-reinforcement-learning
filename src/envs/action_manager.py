import math as mt
import numpy as np
from scipy import interpolate

from primitives.planning.planners import SkeletonPlanning

from primitives.formation.control import FormationControl
from primitives.mrta.task_allocation import MRTA


class ActionManager(object):
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.config = state_manager.config
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
            info = self.state_manager.node_info(actions[2])
            groupInfo[i, 0:2] = info['position'][0:2]
            groupInfo[i, 2] = mt.floor(actions[0])
            groupInfo[i, 3] = groupInfo[i, 3] * 0 + 1
            groupInfo[i, 4] = groupInfo[i, 4] * 0
            groupInfo[i, 5] = groupInfo[i, 5] * 0 + 600
        return robotInfo, groupInfo

    def primitive_parameters(self, decode_actions, vehicles_id, type):
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
            target_info = self.state_manager.node_info(decode_actions[2])
            info['end_pos'] = target_info['position']
            info['primitive_id'] = decode_actions[1]

        elif decode_actions[1] == 2:
            target_info = self.state_manager.node_info(decode_actions[2])
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
            self.state_manager.uav, decoded_actions_uav)

        # MRTA
        robotInfo, groupCenter = self.mrta.allocateRobots(robotInfo, groupInfo)
        for i in range(self.config['simulation']['n_uav_platoons']):
            vehicles_id = [
                j for j, item in enumerate(robotInfo) if item - 1 == i
            ]
            parameters = self.primitive_parameters(decoded_actions_uav[i],
                                                   vehicles_id, 'uav')
            self.uav_platoon[i]._init_setup(parameters)

        # UGV allocation
        robotInfo, groupInfo = self.get_robot_group_info(
            self.state_manager.ugv, decoded_actions_uav)
        # MRTA
        robotInfo, groupCenter = self.mrta.allocateRobots(robotInfo, groupInfo)
        for i in range(self.config['simulation']['n_ugv_platoons']):
            vehicles_id = [
                j for j, item in enumerate(robotInfo) if item - 1 == i
            ]
            parameters = self.primitive_parameters(decoded_actions_ugv[i],
                                                   vehicles_id, 'ugv')
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
                                                   vehicles_id, 'uav')
            self.uav_platoon[i].set_parameters(parameters)

        ids = 0
        for i in range(self.config['simulation']['n_ugv_platoons']):
            vehicles_id = list(range(ids, ids + decoded_actions_ugv[i][0]))
            ids = ids + decoded_actions_ugv[i][0]
            parameters = self.primitive_parameters(decoded_actions_ugv[i],
                                                   vehicles_id, 'ugv')
            self.ugv_platoon[i].set_parameters(parameters)
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
        simulation_count = 0
        # Execute them
        for i in range(500):
            simulation_count += 1
            # Update the time
            done = []
            # Update all the uav vehicles
            for i in range(self.config['simulation']['n_uav_platoons']):
                if self.uav_platoon[i].n_vehicles > 0:
                    done.append(self.uav_platoon[i].execute_primitive())

            # Update all the ugv vehicles
            for i in range(self.config['simulation']['n_ugv_platoons']):
                if self.ugv_platoon[i].n_vehicles > 0:
                    done.append(self.ugv_platoon[i].execute_primitive())

            if all(item for item in done):
                done_rolling_primitive = True
                break
            p_simulation.stepSimulation()

            # Video recording and logging
            if self.config['record_video']:
                p_simulation.startStateLogging(
                    p_simulation.STATE_LOGGING_VIDEO_MP4,
                    self.config['log_path'] + "tactic.mp4")
            if self.config['log_states']:
                print('Need to implement')

        simulation_time = simulation_count * self.config['simulation'][
            'time_step']
        self.state_manager.current_time += simulation_time
        return done_rolling_primitive


class PrimitiveManager(object):
    def __init__(self, state_manager):
        self.config = state_manager.config
        self.state_manager = state_manager
        self.planning = SkeletonPlanning(self.state_manager.config,
                                         self.state_manager.grid_map)
        self.formation = FormationControl()
        return None

    def set_parameters(self, primitive_info):
        """Set up the parameters of the premitive execution

        Parameters
        ----------
        primitive_info: dict
            A dictionary containing information about vehicles
            and primitive realted parameters.
        """
        # Update vehicles
        self.vehicles_id = primitive_info['vehicles_id']

        if primitive_info['vehicle_type'] == 'uav':
            self.vehicles = [
                self.state_manager.uav[j] for j in self.vehicles_id
            ]
        else:
            self.vehicles = [
                self.state_manager.ugv[j] for j in self.vehicles_id
            ]
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

    def convert_pixel_ordinate(self, point, ispixel):
        if not ispixel:
            converted = [point[0] / 0.42871 + 145, point[1] / 0.42871 + 115]
        else:
            converted = [(point[0] - 145) * 0.42871,
                         (point[1] - 115) * 0.42871]

        return converted

    def get_spline_points(self):
        # Perform planning and fit a spline
        self.start_pos = self.centroid_pos
        pixel_start = self.convert_pixel_ordinate(self.start_pos,
                                                  ispixel=False)
        pixel_end = self.convert_pixel_ordinate(self.end_pos, ispixel=False)
        path = self.planning.find_path(pixel_start, pixel_end, spline=False)

        # Convert to cartesian co-ordinates
        points = np.zeros((len(path), 2))
        for i, point in enumerate(path):
            points[i, :] = self.convert_pixel_ordinate(point, ispixel=True)

        # Depending on the distance select number of points of the path
        dist = np.linalg.norm(self.start_pos - self.end_pos)
        n_steps = np.floor(dist / 200 * 250)

        if points.shape[0] > 3:
            tck, u = interpolate.splprep(points.T)
            unew = np.linspace(u.min(), u.max(), n_steps)
            x_new, y_new = interpolate.splev(unew, tck)
        else:
            f = interpolate.interp1d(points[:, 0], points[:, 1])
            x_new = np.linspace(points[0, 0], points[-1, 0], 10)
            y_new = f(x_new)

        new_points = np.array([x_new, y_new]).T
        return new_points, points

    def execute_primitive(self):
        """Perform primitive execution
            """
        primitives = [self.planning_primitive, self.formation_primitive]
        done = primitives[self.primitive_id - 1]()
        return done

    def planning_primitive(self):
        """Performs path planning primitive
        """
        # Make vehicles non idle
        self.make_vehicles_nonidle()
        done_rolling = False

        if self.count == 0:
            # First point of formation
            self.centroid_pos = self.get_centroid()
            self.next_pos = self.centroid_pos
            done = self.formation_primitive()
            if done:
                self.count = 1
                self.new_points, points = self.get_spline_points()
        else:
            self.centroid_pos = self.get_centroid()
            distance = np.linalg.norm(self.centroid_pos - self.end_pos)

            if len(self.new_points) > 2 and distance > 5:
                self.next_pos = self.new_points[1]
                self.new_points = np.delete(self.new_points, 0, 0)
            else:
                self.next_pos = self.end_pos
            done_rolling = self.formation_primitive()

        # Make vehicles idle
        if done_rolling:
            self.make_vehicles_idle()

        return done_rolling

    def formation_primitive(self):
        """Performs formation primitive
        """
        if self.primitive_id == 2:
            self.centroid_pos = self.end_pos
            self.next_pos = self.end_pos

        self.make_vehicles_nonidle()

        dt = self.config['simulation']['time_step']
        self.vehicles, done = self.formation.execute(self.vehicles,
                                                     self.next_pos,
                                                     self.centroid_pos, dt,
                                                     self.formation_type)
        for vehicle in self.vehicles:
            vehicle.set_position(vehicle.updated_pos)

        return done
