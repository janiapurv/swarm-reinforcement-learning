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
        targetInfo = []
        for i, actions in enumerate(decoded_actions):
            info = self.node_info(actions[1])
            groupInfo[i, 0:2] = info['position']
            groupInfo[i, 2] = mt.floor(actions[0])
            groupInfo[i, 3] = groupInfo[i, 3] * 0 + 1
            groupInfo[i, 4] = groupInfo[i, 4] * 0
            groupInfo[i, 5] = groupInfo[i, 5] * 0 + 600

            # Target info (primitive id and target location)
            targetInfo.append([actions[0], actions[1]])
        return robotInfo, groupInfo, targetInfo

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
        robotInfo, groupInfo, targetInfo = self.get_robot_group_info(
            self.uav, decoded_actions_uav)
        # MRTA
        robotInfo = self.mrta.allocateRobots(robotInfo, groupInfo)
        for i in range(self.config['simulation']['n_uav_platoons']):
            vehicles_id = [
                j for j, item in enumerate(robotInfo) if item - 1 == i
            ]
            self.uav_platoon[i]._init_setup(vehicles_id, 1)

        # UGV allocation
        robotInfo, groupInfo, targetInfo = self.get_robot_group_info(
            self.ugv, decoded_actions_uav)
        # MRTA
        robotInfo = self.mrta.allocateRobots(robotInfo, groupInfo)
        for i in range(self.config['simulation']['n_ugv_platoons']):
            vehicles_id = [
                j for j, item in enumerate(robotInfo) if item - 1 == i
            ]
            self.ugv_platoon[i]._init_setup(vehicles_id, 1)

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

    def _init_setup(self, vehicles_id, primitive_id):
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
        self.vehicle_id = vehicles_id
        self.primitive_executing = primitive_id
        self.formation_type = 'solid'
        self.vehicles = [self.uav[j] for j in vehicles_id]
        self.n_vehicles = len(self.vehicles)

        if self.n_vehicles > 1:
            centroid_pos = []
            for vehicle in self.vehicles:
                vehicle.idle = False
                centroid_pos.append(vehicle.current_pos)
            self.centroid_pos = np.mean(np.asarray(centroid_pos)[:, 0:2],
                                        axis=0).tolist()
        return None

    def execute_primitive(self):
        """Perform primitive execution
        """
        primitives = [
            self.planning_primitive, self.formation_primitive,
            self.mapping_primitive
        ]
        primitives[self.primitive_executing]()

        return None

    def planning_primitive(self):
        """Performs path planning primitive
        """
        if self.n_vehicles > 1:  # Cannot do formation with one vehicle
            dt = 0.1
            self.vehicles = self.formation.execute(self.vehicles,
                                                   self.centroid_pos, dt,
                                                   self.formation_type)
            for vehicle in self.vehicles:
                vehicle.set_position(vehicle.updated_pos)
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
