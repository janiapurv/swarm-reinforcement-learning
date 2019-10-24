"""
Plotters for visualization
==========================
"""

# import plotly.graph_objs as go


class Plot2D:
    """
    3D plotter
    """
    def path2points(self, path):
        xs, ys = [], []

        for point in path:
            for points in point:
                xs.append(point[0])
                ys.append(point[1])

        return xs, ys

    def get_rrt_points(self, nodes):
        xs, ys = [], []
        for point in nodes:
            xs.append(point[0])
            ys.append(point[1])
        return xs, ys

    def get_rrt_lines(self, edges):
        lines = []
        for edge in edges:
            lines.append(list(edge))
        return lines

    def draw_rrt(self, rrt, draw_nodes=True, draw_edges=True, omap=None):
        import matplotlib.pyplot as plt
        from matplotlib import collections as mc
        ax = plt.gca()

        if omap is not None:
            plt.imshow(omap, alpha=0.5, cmap='Greys', interpolation='nearest')

        if draw_nodes:
            xs, ys = self.get_rrt_points(rrt.nodes)
            plt.scatter(xs, ys, s=5, color='black')

        if draw_edges:
            lines = mc.LineCollection(self.get_rrt_lines(rrt.edges),
                                      color='green')
            ax.add_collection(lines)

        plt.axis('equal')
        plt.show()

    def draw(self, path, obstacle_map=None, meshes=[]):
        """
        Draw a 3D plot of the path

        Args:
            path: List of points
            obstacle_map: Map of the environment as a numpy array
            meshes: Blobs/meshes in the environment that form obstacles
        """

        xs, ys = self.path2points(path)
        import matplotlib.pyplot as plt

        plt.plot(xs, ys, color='gray')
        plt.scatter(path[0][0], path[0][1], color='green')
        plt.scatter(path[-1][0], path[-1][1], color='red')
        if obstacle_map is not None:
            plt.imshow(obstacle_map,
                       cmap='gray',
                       extent=(0, 100, 0, 100),
                       origin='lower')
        plt.axis('equal')
        plt.show()


class Plot3D:
    """
    3D plotter
    """
    def path2points(self, path):
        xs, ys, zs = [], [], []

        for point in path:
            for points in point:
                xs.append(point[0])
                ys.append(point[1])
                zs.append(point[2])

        return xs, ys, zs

    def draw(self, path, obstacle_map=None, meshes=[]):
        """
        Draw a 3D plot of the path

        Args:
            path: List of points
            obstacle_map: Map of the environment as a numpy array
            meshes: Blobs/meshes in the environment that form obstacles
        """

        xs, ys, zs = self.path2points(path)

        fig1 = go.Scatter3d(x=xs,
                            y=ys,
                            z=zs,
                            mode='lines',
                            line=dict(color="red", width=4))

        fig_start = go.Scatter3d(x=[path[0][0]],
                                 y=[path[0][1]],
                                 z=[path[0][2]],
                                 mode='markers',
                                 line=dict(color="orange", width=10))
        fig_end = go.Scatter3d(x=[path[-1][0]],
                               y=[path[-1][1]],
                               z=[path[-1][2]],
                               mode='markers',
                               line=dict(color="yellow", width=10))

        data = [fig1, fig_start, fig_end] + meshes

        fig = go.Figure(data=data)

        fig.show()
