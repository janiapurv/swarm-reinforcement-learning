"""
Path Planning Algorithms: RRT
=============================
"""

import math
import numpy as np
import networkx as nx

from .planning_utils import GridObstacleMap


class RRT:
    """
    RRT Path Planner
    """
    def __init__(self,
                 obstacle_map,
                 k=50,
                 dt=5,
                 init=(0, 0),
                 low=0,
                 high=100,
                 dim=2):
        """
        Generates RRT graph with obstacle avoidance

        Args:
            k: Number of iterations of branching
            dt: Time interval for applying control velocity
            init: Starting point for RRT
            low: Minimum range in the coordinate system
            high: Maximum range in the coordinate system
            dim: 2 for 2D, 3 for 3D
        """

        self.k = k
        self.dt = dt
        self.init = init
        self.low = low
        self.high = high
        self.dim = dim
        self.map = GridObstacleMap(grid=obstacle_map)
        self.rrt = self.generate_rrt(k, dt, init)

    def generate_rrt(self, k, dt, x_init):
        """
        Generates RRT graph avoiding obstacles

        Args:
            k: Number of iterations of branching
            dt: Time interval for applying control velocity
            x_init: Starting point for RRT

        Returns:
            RRT graph
        """

        T = nx.Graph()
        T.add_node(x_init)

        for kk in range(self.k):
            x_new, x_near, u, distance = self.make_new_edge(T)
            T.add_node(self.np2tup(x_new))
            T.add_edge(self.np2tup(x_near),
                       self.np2tup(x_new),
                       u=u,
                       weight=distance)

        return T

    def find_path(self, start, goal):
        """
        Finds a path from start to goal avoiding obstacles

        Args:
            start: Start pose
            goal: Goal pose

        Returns:
            path: List of poses to visit to reach the goal
        """

        x_start = self.nearest_neighbor(self.tup2np(start), self.rrt)
        x_goal = self.nearest_neighbor(self.tup2np(goal), self.rrt)

        path = nx.shortest_path(self.rrt, self.np2tup(x_start),
                                self.np2tup(x_goal))
        path.insert(0, start)
        path.append(goal)

        return path

    def make_new_edge(self, T):
        x_rand, x_near, u = None, None, None

        while True:
            x_rand = self.random_state()
            x_near = self.nearest_neighbor(x_rand, T)
            u = self.select_input(x_rand, x_near)
            x_new = self.new_state(x_near, u, self.dt)

            if self.map.has_collision(x_new, x_near):
                break

        return x_new, x_near, u, self.distance(x_near, x_new)

    def np2tup(self, x):
        return tuple(x)

    def tup2np(self, x):
        return np.asarray(x, dtype=np.float64)

    def distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def random_state(self):
        return np.random.uniform(self.low, self.high, self.dim)

    def nearest_neighbor(self, x_rand, T):
        min_distance = math.inf
        result = None

        for x in T.nodes:
            x = self.tup2np(x)
            d = self.distance(x, x_rand)
            if d < min_distance:
                min_distance = d
                result = x

        return result.astype(dtype=np.float64)

    def select_input(self, x_rand, x_near):
        return x_rand - x_near

    def new_state(self, x_near, u, dt):
        return x_near + u * dt
