"""
Maps for collision detection: Grid, Octomap
===========================================
"""

import numpy as np
# import octomap as om


class ObstacleMap:
    """
    Base class defining the interface for maps
    """
    def __init__(self, line):
        self.line = line if line is not None else DDA()

    def round_points(self, points):
        return [(int(round(p[0])), int(round(p[1])), int(round(p[2])))
                for p in points]

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
                m, n, o = point
                if self.grid[m, n, o] < 255:
                    return False
        except Exception:
            pass

        return True


"""
Line generation algorithms
==========================
"""


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


class DDA:
    """
    Digital Differential Analyzer line generation algorithm
    """
    def __init__(self):
        pass

    def get_points(self, start, end):
        """
        Generates points on a line segment using DDA algorithm

        Args:
            start: One endpoint of a line segment
            end: The other endpoint of a line segment

        Returns:
            List of points on the line segment
        """

        points = []
        sx, sy, sz = start
        ex, ey, ez = end

        dx, dy, dz = (ex - sx), (ey - sy), (ez - sz)

        if abs(dx) >= abs(dy) and abs(dx) >= abs(dz):
            step = abs(dx)
        elif abs(dy) >= abs(dz):
            step = abs(dy)
        else:
            step = abs(dz)

        dx, dy, dz = (dx / step), (dy / step), (dz / step)

        x, y, z, i = sx, sy, sz, 1
        while i <= step:
            x, y, z, i = x + dx, y + dy, z + dz, i + 1
            points.append((x, y, z))

        return points
