"""
Line generation algorithms
==========================
"""


class DDA2D:
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
        sx, sy = start
        ex, ey = end

        dx, dy = (ex - sx), (ey - sy)
        step = abs(dx) if (abs(dx) >= abs(dy)) else abs(dy)

        dx, dy = (dx / step), (dy / step)

        x, y, i = sx, sy, 1
        while i <= step:
            x, y, i = x + dx, y + dy, i + 1
            points.append((x, y))

        return points


class DDA3D:
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
