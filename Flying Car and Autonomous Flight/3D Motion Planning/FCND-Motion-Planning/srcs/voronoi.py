import numpy as np
import networkx as nx
import numpy.linalg as LA
from bresenham import bresenham
from scipy.spatial import Voronoi
from srcs.grid import GridPlanner
from srcs.a_star import AStarGraph

class VoronoiPlanner(GridPlanner):
    def __init__(self, obstacles_data, target_altitude, safety_distance, name, save_plt=False, vis_dir='images', **kwargs):
        super(VoronoiPlanner, self).__init__(obstacles_data, target_altitude, safety_distance, name, obs_cen=True, h_type='norm', save_plt=save_plt, vis_dir=vis_dir)
        self.astar = AStarGraph()

        self.edges = self.create_edges(self.grid, target_altitude)
        # Create the graph with the weight of the edges        
        # set to the Euclidean distance between the points
        self.graph = nx.Graph()
        for e in self.edges:
            p1 = e[0]
            p2 = e[1]
            dist = LA.norm(np.array(p2) - np.array(p1))
            self.graph.add_edge(p1, p2, weight=dist)

        self.graph_pts = np.array(self.graph.nodes)

    def create_edges(self, grid, target_altitude):
        """
        Returns a grid representation of a 2D configuration space
        along with Voronoi graph edges given obstacle data and the
        drone's altitude.
        """

        # TODO: create a voronoi graph based on
        # location of obstacle centres    
        graph = Voronoi(self.obstacle_centre)

        # TODO: check each edge from graph.ridge_vertices for collision
        edges = []
        for v in graph.ridge_vertices:
            p1 = graph.vertices[v[0]]
            p2 = graph.vertices[v[1]]

            hit = self.check_obstacles(grid, p1, p2)

            # If the edge does not hit on obstacle
            # add it to the list
            if not hit:
                # array to tuple for future graph creation step)
                p1 = (p1[0], p1[1], target_altitude)
                p2 = (p2[0], p2[1], target_altitude)
                edges.append((p1, p2))

        return edges

    def check_obstacles(self, grid, p1, p2):
        # Test each pair p1 and p2 for collision using Bresenham
        # (need to convert to integer if using prebuilt Python package)
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        
        hit = False
        for c in cells:
            # First check if we're off the map
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                hit = True
                break

            # Next check if we're in collision
            if grid[c[0], c[1]] == 1:
                hit = True
                break
        return hit

    def a_star(self, grid_start, grid_goal):
        grid_start = self.closest_point(self.graph_pts, grid_start)
        grid_goal = self.closest_point(self.graph_pts, grid_goal)
        return self.astar.search(self.graph, grid_start, grid_goal)

    def plot_grid(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        super().plot_grid(drone_home, drone_start=drone_start, drone_goal=drone_goal, path=path, edges=self.edges, title='Voronoi')

    def plot_voxmap(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        pass
