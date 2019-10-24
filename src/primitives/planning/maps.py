"""
Maps for collision detection: Grid, Octomap
===========================================
"""

import numpy as np

# import octomap as om
from .lines import DDA2D, DDA3D


class ObstacleMap:
    """
    Base class defining the interface for maps
    """
    def __init__(self, line):
        self.line = line if line is not None else DDA3D()

    def round_points(self, points):
        return [np.asarray(p, dtype=np.int32) for p in points]

    def has_collision(self, start, end):
        """
        Detects collision between a line segment and a map

        Args:
            start: One endpoint of a line segment
            end: Other endpoint of a line segment

        Returns:
            Boolean value indicating if the line segments collides on the map

        """
        return True


class GridObstacleMap(ObstacleMap):
    """
    Fixed resolution grid based obstacle map
    """
    def __init__(self, grid, line=None):
        """
        GridObstacleMap constructor

        Args:
            grid: Numpy array with obstacles marked
            with 1s and 0s everywhere else
        """
        line = DDA2D() if grid.ndim == 2 else DDA3D()
        super(GridObstacleMap, self).__init__(line)
        self.grid = grid

    def has_collision(self, start, end):
        """
        Detects collision between a line segment and a map

        Args:
            start: One endpoint of a line segment
            end: Other endpoint of a line segment

        Returns:
            Boolean value indicating if the line segments collides on the map

        """
        points = self.round_points(self.line.get_points(start, end))
        try:
            for point in points:
                if self.grid[tuple(point)] > 0:
                    return True
        except Exception:
            pass

        return False


class OctoObstacleMap(ObstacleMap):
    """
    Efficient Octree based obstacle map
    """
    def __init__(self, filename=None, octomap=None, line=None):
        """
        OctoObstacleMap constructor

        Args:
            octomap: Octomap object from octomap-python package
        """
        super(OctoObstacleMap, self).__init__(line)

        self.octomap = None

        if filename is not None:
            octree = om.OcTree(0.1)  # noqa
            octree.readBinary(filename)
            self.octomap = octree
        elif octomap is not None:
            self.octomap = octomap

    def has_collision(self, start, end):
        """
        Detects collision between a line segment and a map

        Args:
            start: One endpoint of a line segment
            end: Other endpoint of a line segment

        Returns:
            Boolean value indicating if the line segments collides on the map

        """
        hit = np.ndarray(3, dtype=np.float64)
        distance = np.linalg.norm(end - start)
        return self.octomap.castRay(start, (end - start) / distance,
                                    hit,
                                    maxRange=distance)
