import numpy as np


def get_time_dist(path_planning, vehicle, target_pos):
    """Calculated the information of probable goal building

    Parameters
    ---------
    vehicle : vehicle class
        We use the speed and current position
    target_pos : array
        final desired position of robot

    Returns
    -------
    float
        time to reach the target [minimum time]
    """
    # Read from co-ordinate file
    if vehicle.type == 'uav':
        time_to_reach = np.linalg.norm(vehicle.current_pos -
                                       target_pos) / vehicle.speed
    elif vehicle.type == 'ugv':
        path_RRT = path_planning(vehicle.current_pos, target_pos)
        total_distance = np.sum(np.linalg.norm(np.diff(path_RRT)))
        time_to_reach = total_distance / vehicle.speed

    return time_to_reach


def goal_information(goal_id, config):
    """Calculated the information of probable goal building

    Parameters
    ----------
    goal_id : int
        Goal ID
    config : yaml
        The configuration file

    Returns
    -------
    dict
        A dictionary containing goal position, perimeter, floors,
        progress, and probability
    """
    # Read from co-ordinate file
    info = {}
    info['goal_position'] = 0
    info['perimeter'] = config['goal']['goal_building_perimeter'][goal_id]
    info['floors'] = config['goal']['goal_building_floors'][goal_id]
    info['goal_progress'] = 0
    info['goal_probability'] = 0
    return info


def mission_reward(ugv, uav, config):
    """Caculated the total mission reward depending on the progress

    Parameters
    ----------
    ugv : list
        List of all UGVs
    uav : list
        List of all UAVs
    config : yaml
        The configuration file

    Returns
    -------
    float
        The reward for the mission at an instant of time.
    """

    # Simulation parameters
    total_time = config['simulation']['total_time']
    n_goals = config['goal']['n_goals']

    # UAV reward weight parameters
    w_time_uav = config['reward']['w_time_uav']
    w_battery_uav = config['reward']['w_battery_uav']
    w_b_UAV_0 = 1  # Need to implement

    # Reward for UAV
    r_uav_time = 0
    r_uav_battery = 0

    # Calculate the reward for UAV
    for vehicle in uav:
        r_uav_battery += w_battery_uav * vehicle.battery / w_b_UAV_0
        for goal in range(n_goals):
            position, _ = vehicle.get_pos_and_orientation
            info = goal_information(goal)
            time_to_goal = get_time_dist(vehicle, info['goal_position'])
            r_uav_time += w_time_uav * (1 - info['goal_progress']) * (
                total_time - time_to_goal) / total_time

    # Reward for UGV
    r_ugv_time = 0
    r_ugv_ammo = 0

    # UGV reward weight parameters
    w_time_ugv = config['reward']['w_time_ugv']
    w_battery_ugv = config['reward']['w_ammo_ugv']
    w_b_ugv_0 = 1  # Need to implement

    # Calculate the reward for UGV
    for vehicle in ugv:
        r_ugv_ammo += w_battery_ugv * vehicle.ammo / w_b_ugv_0
        for goal in range(n_goals):
            position, _ = vehicle.get_pos_and_orientation
            info = goal_information(goal)
            time_to_goal = get_time_dist(vehicle, info['goal_position'])
            r_ugv_time += w_time_ugv * (1 - info['goal_progress']) * (
                total_time - time_to_goal) / total_time

    # Search reward parameters
    # w_search = config['reward']['w_search']
    r_search = 0

    # for vehicle in ugv:
    #     position = vehicle.get_pos_and_orientation()
    #     for goal in range(n_goals):
    #         info = goal_information(goal)
    #         time_to_goal = get_time_dist(vehicle, info['goal_position'])
    #         # Need to implement inside buidling search
    #         # inside_search_time = get_t_search_inside(info['perimeter'] *
    #         #                                          info['floors'])

    #         # r_search += w_search * info['goal_probability'] * info[
    #         #     'goal_progress'] * (total_time -
    #         #                         inside_search_time) / total_time

    reward = r_ugv_time + r_ugv_ammo + r_uav_time + r_uav_battery + r_search

    return reward
