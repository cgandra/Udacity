import numpy as np
import networkx as nx
import numpy.linalg as LA
from srcs.sampling import Sampler
from srcs.grid import GridPlanner
from srcs.a_star import AStarGraph
from sklearn.neighbors import KDTree
from shapely.geometry import LineString

class ProbRoadMapPlanner(GridPlanner):
    def __init__(self, obstacles_data, target_altitude, safety_distance, name, n_nodes=300, n_neigh=10, kd_query='rad', kd_neigh=3, init_b=True, save_plt=False, vis_dir='images'):
        super(ProbRoadMapPlanner, self).__init__(obstacles_data, target_altitude, safety_distance, name, obs_con_h=True, h_type='norm', save_plt=save_plt, vis_dir=vis_dir)

        self.n_nodes = n_nodes
        self.n_neigh = n_neigh
        self.kd_query = kd_query
        self.kd_neigh = kd_neigh

        grid_shape = [0, self.grid.shape[0], 0, self.grid.shape[1], target_altitude, target_altitude]
        self.sampler = Sampler(self.obstacle_corners, self.obstacle_height, self.obstacle_max_xy, grid_shape)
        self.polygons = self.sampler._polygons
        if init_b:
            self.init_graph(grid_shape)

        self.astar = AStarGraph()

    def init_graph(self, grid_shape):
        self.sampler.set_grid_shape(grid_shape)
        self.nodes = self.sampler.sample(self.n_nodes, self.kd_query, self.kd_neigh)
        print('Nodes: ', len(self.nodes))
        self.graph = self.create_graph(self.nodes, self.n_neigh)
        print('Graph Created')
        self.graph_pts = np.array(self.graph.nodes)

    def can_connect(self, n1, n2):
        # casts two points as a shapely LineString() object
        # tests for collision with a shapely Polygon() object
        # returns True if connection is possible, False otherwise
        l = LineString([n1, n2])
        for p in self.polygons:
            if p.crosses(l) and p.height >= min(n1[2], n2[2]):
                return False
        return True

    def create_graph(self, nodes, k):
        # defines a networkx graph as g = Graph()
        # defines a tree = KDTree(nodes)
        # test for connectivity between each node and 
        # k of it's nearest neighbors
        # if nodes are connectable, add an edge to graph
        # Iterate through all candidate nodes!
        graph = nx.Graph()
        tree = KDTree(nodes, metric="euclidean")
        for n1 in nodes:
            # for each node connect try to connect to k nearest nodes
            dists, idxs = tree.query([n1], k, return_distance=True)
        
            for idx, dist in zip(idxs[0], dists[0]):
                n2 = nodes[idx]
                if n2 == n1:
                    continue

                if graph.has_edge(n1, n2):
                    continue

                if self.can_connect(n1, n2):
                    graph.add_edge(n1, n2, weight=dist)
        return graph

    def a_star(self, grid_start, grid_goal):
        grid_start = self.closest_point(self.graph_pts, grid_start)
        grid_goal = self.closest_point(self.graph_pts, grid_goal)
        return self.astar.search(self.graph, grid_start, grid_goal)

    def plot_grid(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        edges = [([n1[0], n1[1]], [n2[0], n2[1]]) for (n1, n2) in self.graph.edges]
        c_nodes = [[n[0], n[1]] for n in self.graph.nodes]
        all_nodes = [[n[0], n[1]] for n in self.nodes]
        super().plot_grid(drone_home, drone_start=drone_start, drone_goal=drone_goal, path=path, edges=edges, nodes=c_nodes, all_nodes=all_nodes, title='Prob Roadmap')

    def plot_voxmap(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        pass