import math as mt


class Action(object):
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.config = state_manager.config
        return None

    def action_decode(self, net_output, type):
        n_nodes = self.config['simulation']['n_nodes']
        if type == 'uav':
            # n_primitive = self.config['primitive']['uav']['n_uav_primitive']
            n_formations = self.config['primitive']['uav']['n_formations']
            max_size = self.config['primitive']['uav']['max_size']
            n_caution_status = self.config['primitive']['uav'][
                'n_caution_status']
        else:
            # n_primitive = self.config['primitive']['ugv']['n_ugv_primitive']
            n_formations = self.config['primitive']['ugv']['n_formations']
            max_size = self.config['primitive']['ugv']['max_size']
            n_caution_status = self.config['primitive']['ugv'][
                'n_caution_status']

        action = []  # We might have to change this code a lot
        action.append(net_output[0])
        # action.append(mt.floor(n_primitive * net_output[1] - 1) + 1)
        action.append(1)  # By default one
        action.append(mt.floor(n_nodes * net_output[1] - 1) + 1)
        action.append(mt.floor(n_formations * net_output[1] - 1) + 1)
        action.append(max_size * net_output[1])
        action.append(max_size * net_output[1])
        action.append(n_caution_status * net_output[1])

        return action[0:3]

    def get_idle_vehicles(self, vehicles):
        """Returns non idle vehicles

        Parameters
        ----------
        vehicles : list
            A list of UAV or UGV vehilces class
        """
        vehicles = list(filter(lambda vehicle: vehicle.idle, vehicles))
        return vehicles

    def format_net_output(self, net_output):
        temp = net_output.tolist()
        formated_output = {}
        uav_action, ugv_action = [], []
        for i in range(self.config['simulation']['n_uav_platoons']):
            uav_action.append(temp[3 * i:3 * i + 3])
        formated_output['uav'] = uav_action

        for i in range(self.config['simulation']['n_ugv_platoons']):
            ugv_action.append(temp[3 * i + 9:3 * i + 18])
        formated_output['ugv'] = ugv_action
        return formated_output

    def get_action(self, net_output):
        n_uav_platoons = self.config['simulation']['n_uav_platoons']
        n_ugv_platoons = self.config['simulation']['n_ugv_platoons']

        # Need to normalise
        decoded_actions_uav, decoded_actions_ugv = [], []

        # Format the net output
        formated_output = self.format_net_output(net_output)

        # NEED TO REIMPLEMENT THE PART OF NORMALISING % OF DRONES NEEDED
        # Decode for uav's actions
        total_probability = 0
        for platoon_id in range(n_uav_platoons):
            output = formated_output['uav'][
                platoon_id]  # net_output is dictionary
            decoded_actions = self.action_decode(output, 'uav')
            decoded_actions_uav.append(decoded_actions)
            total_probability += decoded_actions[0]

        idle_vehicles = self.get_idle_vehicles(self.state_manager.uav)
        for i, actions in enumerate(decoded_actions_uav):
            if i == 2:
                decoded_actions_uav[i][0] = len(idle_vehicles) - (
                    decoded_actions_uav[0][0] + decoded_actions_uav[1][0])
            else:
                n_vehicle = len(idle_vehicles)
                if i % 2 == 0:
                    decoded_actions_uav[i][0] = mt.floor(
                        actions[0] / total_probability * n_vehicle)
                else:
                    decoded_actions_uav[i][0] = mt.ceil(
                        actions[0] / total_probability * n_vehicle)

        # Decode for ugv's actions
        total_probability = 0
        for platoon_id in range(n_ugv_platoons):
            output = formated_output['ugv'][
                platoon_id]  # net_output is dictionary
            decoded_actions = self.action_decode(output, 'ugv')
            decoded_actions_ugv.append(decoded_actions)
            total_probability += decoded_actions[0]

        idle_vehicles = self.get_idle_vehicles(self.state_manager.ugv)
        for i, actions in enumerate(decoded_actions_ugv):
            if i == self.config['simulation']['n_ugv_platoons'] - 1:
                decoded_actions_ugv[i][0] = len(idle_vehicles) - (
                    decoded_actions_ugv[0][0] + decoded_actions_ugv[1][0])
            else:
                n_vehicle = len(idle_vehicles)
                if i % 2 == 0:
                    decoded_actions_ugv[i][0] = mt.floor(
                        actions[0] / total_probability * n_vehicle)
                else:
                    decoded_actions_ugv[i][0] = mt.ceil(
                        actions[0] / total_probability * n_vehicle)

        return decoded_actions_uav, decoded_actions_ugv
