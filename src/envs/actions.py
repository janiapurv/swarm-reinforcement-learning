import math as mt


class Action(object):
    def __init__(self, config):
        super(Action, self).__init__()
        self.config = config
        return None

    def action_decode(self, net_output, type):
        n_nodes = self.config['simulation']['n_nodes']
        if type == 'uav':
            n_primitive = self.config['primitive']['uav']['n_uav_primitive']
            n_formations = self.config['primitive']['uav']['n_formations']
            max_size = self.config['primitive']['uav']['max_size']
            n_caution_status = self.config['primitive']['uav'][
                'n_caution_status']
        else:
            n_primitive = self.config['primitive']['ugv']['n_ugv_primitive']
            n_formations = self.config['primitive']['ugv']['n_formations']
            max_size = self.config['primitive']['ugv']['max_size']
            n_caution_status = self.config['primitive']['ugv'][
                'n_caution_status']

        action = []  # We might have to change this code a lot
        action.append(net_output[0])
        action.append(mt.floor(n_primitive * net_output[1] - 1) + 1)
        action.append(mt.floor(n_nodes * net_output[2] - 1) + 1)
        action.append(mt.floor(n_formations * net_output[3] - 1) + 1)
        action.append(max_size * net_output[4])
        action.append(max_size * net_output[5])
        action.append(n_caution_status * net_output[6])

        return action

    def get_action(self, net_output):
        n_uav_platoons = self.config['simulation']['n_uav_platoons']
        n_ugv_platoons = self.config['simulation']['n_ugv_platoons']

        # Need to normalise
        decoded_actions_uav, decoded_actions_ugv = [], []

        # NEED TO REIMPLEMENT THE PART OF NORMALISING % OF DRONES NEEDED
        # Decode for uav's actions
        total_vehicles = 0
        for platoon_id in range(n_uav_platoons):
            output = net_output[platoon_id]  # net_output is dictionary
            decoded_actions = self.action_decode(output, 'uav')
            decoded_actions_uav.append(decoded_actions)
            total_vehicles += decoded_actions[0]

        for i, actions in enumerate(decoded_actions_uav):
            decoded_actions_uav[i][0] = actions[0] / total_vehicles

        # Decode for ugv's actions
        total_vehicles = 0
        for platoon_id in range(n_ugv_platoons):
            output = net_output[platoon_id + 3]  # net_output is dictionary
            decoded_actions = self.action_decode(output, 'ugv')
            decoded_actions_ugv.append(decoded_actions)
            total_vehicles += decoded_actions[0]

        for i, actions in enumerate(decoded_actions_ugv):
            decoded_actions_ugv[i][0] = actions[0] / total_vehicles

        return decoded_actions_uav, decoded_actions_ugv
