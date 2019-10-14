import math as mt
import numpy as np

from .state_manager import StateManager
from .primitives.planning import RRT
from .primitives.formation import FormationControl
from .primitives.task_allocation import MRTA


class ActionManager(StateManager):
    def __init__(self, state_manager):
        super(ActionManager,
              self).__init__(state_manager.uav, state_manager.ugv,
                             state_manager.current_time, state_manager.config)
        self.state_manager = state_manager
        self.mrta = MRTA()

        # Setup the platoons
        self._init_platoons_setup()

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

    def primitive_parameters(self, decode_actions, vehicles_id, type):
        info = {}
        info['vehicles_id'] = vehicles_id
        info['primitive_id'] = -1
        info['start_pos'] = [0, 0]
        info['end_pos'] = [0, 0]
        info['centroid_pos'] = [0, 0]
        info['formation_type'] = None
        info['vehicle_type'] = type

        # Decoded actions is of the form
        # ['n_vehicles', 'primitive_id', 'target_id']
        # should implement as a dict
        if decode_actions[1] < 2:
            target_info = self.node_info(decode_actions[2])
            info['end_pos'] = target_info['position']
            info['start_pos'] = info['centroid_pos']
            info['centroid_pos'] = info['start_pos']
            if decode_actions[3] == 0:
                info['formation_type'] = 'solid'
            else:
                info['formation_type'] = 'ring'
            info['primitive_id'] = decode_actions[1]

        elif decode_actions[1] > 1:
            target_info = self.node_info(decode_actions[2])
            info['centroid_pos'] = target_info['position']
            if decode_actions[3] == 0:
                info['formation_type'] = 'solid'
            else:
                info['formation_type'] = 'ring'
            info['primitive_id'] = decode_actions[1]

        return info

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
                                                   vehicles_id, 'uav')
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
                                                   vehicles_id, 'ugv')
            self.ugv_platoon[i]._init_setup(parameters)

        return None

    def primitive_execution(self, decoded_actions_uav, decoded_actions_ugv,
                            p_simulation):
        """Performs task execution

        Parameters
        ----------
        decoded_actions_uav : array
            UAV decoded actions
        decoded_actions_ugv : [type]
            UAV decoded actions
        p_simulation : bullet engine
            Bullet engine to execute the simulation
        """

        self.perform_task_allocation(decoded_actions_uav, decoded_actions_ugv)

        # Execute them
        for i in range(1000):
            # Update the time
            self.current_time = self.current_time + self.config['simulation'][
                'time_step']
            # Update all the vehicles
            for i in range(self.config['simulation']['n_uav_platoons']):
                self.uav_platoon[i].execute_primitive()

            # Update all the vehicles
            for i in range(self.config['simulation']['n_ugv_platoons']):
                self.ugv_platoon[i].execute_primitive()
            p_simulation.stepSimulation()

        return None


class PrimitiveManager(StateManager):
    def __init__(self, state_manager):
        super(PrimitiveManager,
              self).__init__(state_manager.uav, state_manager.ugv,
                             state_manager.current_time, state_manager.config)
        # Primitives
        self.planning = RRT(self.grid_map)
        self.formation = FormationControl()

        return None

    def _init_setup(self, primitive_info):
        """Peform initial setup of the primitive
        class with vehicles id and primitive id

        Parameters
        ----------
        vehicles_id : list
            A list of vehicle id belonging to a premitive
        primitive_id : int
            The primitive id to execute
        """
        # Update vehicles
        self.vehicles_id = primitive_info['vehicles_id']
        if primitive_info['vehicle_type'] == 'uav':
            self.vehicles = [self.uav[j] for j in self.vehicles_id]
        else:
            self.vehicles = [self.ugv[j] for j in self.vehicles_id]
        self.n_vehicles = len(self.vehicles)

        # Primitive parameters
        self.primitive_id = primitive_info['primitive_id'] - 1
        self.formation_type = primitive_info['formation_type']
        self.centroid_pos = primitive_info['centroid_pos']
        self.start_pos = primitive_info['start_pos']
        self.end_pos = primitive_info['end_pos']

        return None

    def execute_primitive(self):
        """Perform primitive execution
        """
        primitives = [
            self.planning_primitive, self.formation_primitive,
            self.mapping_primitive
        ]
        primitives[self.primitive_id]()
        return None

    def planning_primitive(self):
        """Performs path planning primitive
        """
        # First run the formation
        self.path = self.planning.find_path(self.start_pos, self.end_pos)

        return None

    def formation_primitive(self):
        """Performs formation primitive
        """
        if self.n_vehicles > 1:  # Cannot do formation with one vehicle
            dt = 0.1
            self.vehicles = self.formation.execute(self.vehicles,
                                                   self.centroid_pos, dt,
                                                   self.formation_type)
            for vehicle in self.vehicles:
                vehicle.set_position(vehicle.updated_pos)

        return None

    def mapping_primitive(self):
        """Performs mapping primitive
        """
        if self.n_vehicles > 1:  # Cannot do formation with one vehicle
            dt = 0.1
            self.vehicles = self.formation.execute(self.vehicles,
                                                   self.centroid_pos, dt,
                                                   self.formation_type)
            for vehicle in self.vehicles:
                vehicle.set_position(vehicle.updated_pos)
        return None
