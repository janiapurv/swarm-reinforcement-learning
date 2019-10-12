import numpy as np
from numpy import genfromtxt


class StateManager():
    def __init__(self, uav, ugv, current_time, config):
        super(StateManager, self).__init__()
        # Need to specify some parameters
        self.uav = uav
        self.ugv = ugv
        self.current_time = current_time
        self.config = config

        self.obstacle_map = np.load(self.config['map_save_path'] +
                                    'occupancy_map.npy')
        # Need to zeros to make it 3D
        temp = np.zeros(self.obstacle_map.shape)
        self.grid_map = np.dstack((self.obstacle_map, temp))

        self._initial_buildings_setup()
        self._initial_nodes_setup()
        self._initial_target_setup()

    def _initial_mission_setup(self):
        self.goal = self.config['simulation']['goal_node']
        self.progress_reward = self.config['reward']['progress_reward']
        self.indoor_reward = 2 * self.progress_reward
        self.n_keep_in_pareto = self.config['state']['n_keep_in_pareto']

    def _initial_buildings_setup(self):
        # Buildings setup (probably we might need to read it from a file)
        self.buildings = []
        for i in range(self.config['simulation']['n_buildings']):
            info = {}
            info['position'] = [0, 0]
            info['n_floors'] = 1
            info['primeter'] = 4 * (4 * 3)
            info['area'] = 4 * 4
            self.buildings.append(info)
        return None

    def _initial_nodes_setup(self):
        """Performs initial nodes setup
        """
        # Nodes setup
        self.nodes = []
        path = self.config['map_data_path'] + 'nodes.csv'
        position_data = genfromtxt(path, delimiter=',')
        for i in range(self.config['simulation']['n_nodes']):
            info = {}
            info['position'] = position_data[i]
            info['importance'] = 0
            self.nodes.append(info)
        return None

    def _initial_target_setup(self):
        """Performs target setup with properties such as goal probability,
        goal progress etc.
        """
        # Targets
        self.target = []
        n_targets = self.config['simulation']['n_targets']
        for target in self.config['simulation']['target_building_id']:
            info = {}
            info['target_id'] = target
            info['probability_goals'] = 1 / n_targets
            info['progress_goals'] = 0
            info['probability_goals_indoor'] = 1 / n_targets
            info['defence_perimeter'] = 0
            info['n_defence_perimeter'] = 0
            info['progress_goals_indoor'] = 0
            node_info = self.node_info(target)  # Get the position
            info['position'] = node_info['position']
            self.target.append(info)

    def target_info(self, id):
        """Get the information about the target.

        Parameters
        ----------
        id : int
            Target ID

        Returns
        -------
        dict
            A dictionary containing all the information about the target.
        """
        for target in self.target:
            if target['target_id'] == id:
                return target

    def node_info(self, id):
        """Get the information about a node.

        Parameters
        ----------
        id : int
            Node ID

        Returns
        -------
        dict
            A dictionary containing all the information about the node.
        """
        return self.nodes[id]

    def building_info(self, id):
        """Get the information about a building such as perimeter,
        position, number of floors.

        Parameters
        ----------
        id : int
            Building ID

        Returns
        -------
        dict
            A dictionary containing all the information about the building.
        """
        return self.buildings[id]

    def update(self, ugv, uav, current_time):
        self.ugv = ugv
        self.uav = uav
        self.current_time = current_time
        return None

    def check_vehicle(self, vehicle):
        if (not vehicle.idle) and vehicle.type == 'uav':
            if vehicle.primitive_executing == 1:
                return 'uav'
        elif (not vehicle.idle) and vehicle.type == 'ugv':
            if vehicle.primitive_executing == 1:
                return 'ugv'
        else:
            return None

    def check_closeness(self, vehicle, target):
        target_pos = target['position']
        vehicle_pos = vehicle['position']
        dist = np.linalg.norm(vehicle_pos - target_pos)
        if vehicle.type == 'uav':
            return dist <= self.config['uav']['search_dist']
        elif vehicle.type == 'ugv':
            return dist <= self.config['ugv']['defense_radius']
        else:
            return None

    def outdoor_progress(self, vehicle, target):
        req_progress = target['n_floor'] * target['perimeter']
        progesse_rate = vehicle.search_speed / req_progress
        progress_goals = (self.current_time -
                          vehicle.time_reached_target) * progesse_rate
        return progress_goals

    def indoor_progress(self, vehicle, target):
        # Need to implement
        return None

    def outdoor_target_progress(self, vehicles):
        for target in self.target:
            progress_goals = 0
            for vehicle in vehicles:
                if self.check_vehicle(vehicle) == 'uav':
                    if self.check_closeness(vehicle, target):
                        progress_goals += self.outdoor_progress(
                            vehicle, target)
                else:
                    if self.check_closeness(vehicle, target):
                        progress_goals += self.indoor_progress(vehicle, target)
            if progress_goals > 1:
                progress_goals = 1
            target['progress_goals'] = progress_goals

    def update_progress(self):
        # Parameters
        # need to implemenet

        return None
