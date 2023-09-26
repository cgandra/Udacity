import numpy as np
import networkx as nx
import numpy.linalg as LA
from enum import Enum, auto
from queue import PriorityQueue
from bresenham import bresenham
from srcs.grid import GridPlanner
from srcs.a_star import AStarGraph
from timeit import default_timer as timer

# Rapidly-Exploring Random Tree
class RRT:
    def __init__(self, x_init, id=0):
        # A tree is a special case of a graph with
        # directed edges and only one path to any vertex.
        self.id = id
        self.tree = nx.DiGraph()
        self.tree.add_node(x_init)
                
    def add_vertex(self, x_new):
        self.tree.add_node(tuple(x_new))
    
    def add_edge(self, x_near, x_new, u, w, c=0):
        self.tree.add_edge(tuple(x_near), tuple(x_new), orientation=u, weight=w, pathcost=c)
        
    @property
    def vertices(self):
        return self.tree.nodes()
    
    @property
    def edges(self):
        return self.tree.edges()

class State(Enum):
    Trapped = auto()
    Advanced = auto()
    Reached = auto()


class RRTPlanner(GridPlanner):
    def __init__(self, obstacles_data, target_altitude, safety_distance, name, algo='rrt', dt=1, n_samples=300, min_samples=100, min_dist=0.5, n_neighbors=8, seed=0, save_plt=False, vis_dir='images'):
        super(RRTPlanner, self).__init__(obstacles_data, target_altitude, safety_distance, algo, obs_con_h=True, h_type='norm', save_plt=save_plt, vis_dir=vis_dir)

        print('RRT Algo: ', algo)
        self.dt = dt
        self.algo = algo
        self.min_dist = min_dist
        self.n_samples = n_samples
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors
        self.target_altitude = target_altitude
        self.astar = AStarGraph()
        self.swapped = False
        self.seed = seed

        self.rrt_funcs = {}
        self.rrt_funcs['rrt'] = self.generate_RRT
        self.rrt_funcs['rrt_connect'] = self.generate_RRT_Connect
        self.rrt_funcs['rrt_star'] = self.generate_RRT_Star
        self.rrt_funcs['rrt_star_bi'] = self.generate_RRT_Star_Bi

        self.titles = {}
        self.titles['rrt'] = 'RRT'
        self.titles['rrt_connect'] = 'RRT Connect'
        self.titles['rrt_star'] = 'RRT*'
        self.titles['rrt_star_bi'] = 'RRT* Bidirectional'
        
    # Based on algorithm: Rapidly-exploring random trees: A new tool for path planning
    # http://lavalle.pl/papers/Lav98c.pdf
    def generate_RRT(self, start, goal):
        grid = self.grid
        alt = self.target_altitude
        rrt = [RRT(start)]

        i = -1
        while i < self.n_samples:
            x_rand = self.sample(grid, alt)
            ret, x_new = self.extend(rrt[0], x_rand, grid.shape)
            if ret == State.Trapped:
                continue

            if i >= self.min_samples:
                ret = self.connect_goal(rrt[0], goal)
                if ret == State.Reached:
                    print('Found goal in {} samples'.format(i))
                    break
            i += 1

        if goal not in rrt[0].vertices:
            x_near, dist = self.nearest_neighbor(goal, rrt[0])
            print('Could not find goal in {} samples. Closest point {} @ dist {}'.format(i, x_near, dist))
            goal = x_near

        return rrt, [goal]

    def generate_RRT_Org(self, start, goal):
        grid = self.grid
        alt = self.target_altitude
        rrt = [RRT(start)]

        i = -1
        while i < self.n_samples:
            x_rand = self.sample(grid, alt)
            ret, x_new = self.extend(rrt[0], x_rand, grid.shape)

            if ret != State.Trapped and i >= self.min_samples:
                ret = self.connect_goal(rrt[0], goal)
                if ret == State.Reached:
                    print('Found goal in {} samples'.format(i))
                    break
            i += 1

        if goal not in rrt[0].vertices:
            x_near, dist = self.nearest_neighbor(goal, rrt[0])
            print('Could not find goal in {} samples. Closest point {} @ dist {}'.format(i, x_near, dist))
            goal = x_near

        return rrt, [goal]

    # Based on algorithm: RRT-Connect: An Efficient Approach to Single-Query Path Planning
    # https://www.cs.cmu.edu/afs/cs/academic/class/15494-s14/readings/kuffner_icra2000.pdf
    def generate_RRT_Connect(self, start, goal):
        grid = self.grid
        alt = self.target_altitude
        rrt = [RRT(start), RRT(goal, id=1)]
        x_near_s = x_near_g = None

        i = -1
        while i < self.n_samples:
            x_rand = self.sample(grid, alt)
            ret, x_new = self.extend(rrt[0], x_rand, grid.shape)
            if ret == State.Trapped:
                continue

            ret = self.connect(rrt[1], x_new, grid.shape)
            if ret == State.Reached:
                print('Found goal in {} samples'.format(i))
                break

            i += 1
            rrt = self.swap_trees(rrt)

        if ret == State.Reached:
            if self.swapped:
                rrt = self.swap_trees(rrt)
            
            x_near_s, _ = self.nearest_neighbor(x_new, rrt[0])
            x_near_g, _ = self.nearest_neighbor(x_new, rrt[1])
        else:
            print('Could not find goal in {} samples'.format(i))

        return rrt, [x_near_s, x_near_g]

    def generate_RRT_Connect_Org(self, start, goal):
        grid = self.grid
        alt = self.target_altitude
        rrt = [RRT(start), RRT(goal, id=1)]
        x_near_s = x_near_g = None

        i = -1
        while i < self.n_samples:
            x_rand = self.sample(grid, alt)
            ret, x_new = self.extend(rrt[0], x_rand, grid.shape)
            if ret != State.Trapped:
                ret = self.connect(rrt[1], x_new, grid.shape)
                if ret == State.Reached:
                    print('Found goal in {} samples'.format(i))
                    break

            i += 1
            rrt = self.swap_trees(rrt)

        if ret == State.Reached:
            if self.swapped:
                rrt = self.swap_trees(rrt)
            
            x_near_s, _ = self.nearest_neighbor(x_new, rrt[0])
            x_near_g, _ = self.nearest_neighbor(x_new, rrt[1])
        else:
            print('Could not find goal in {} samples'.format(i))

        return rrt, [x_near_s, x_near_g]

    # Based on algorithm: Sampling-based Algorithms for Optimal Motion Planning
    # https://arxiv.org/pdf/1105.1186.pdf
    def generate_RRT_Star(self, start, goal):
        grid = self.grid
        alt = self.target_altitude
        rrt = [RRT(start)]

        i = -1
        while i < self.n_samples:
            x_rand = self.sample(grid, alt)
            x_near, x_new, u = self.get_near_new(rrt[0], x_rand, grid.shape)
            if not self.check_valid_sample(rrt[0], x_new, x_near):
                continue

            i += 1
            X_near = self.connect_best_parent(rrt[0], x_new)

            if X_near is not None:
                self.rewire(rrt[0], x_new, X_near)

            if i >= self.min_samples:
                ret = self.connect_goal(rrt[0], goal)
                if ret == State.Reached:
                    print('Found goal in {} samples'.format(i))
                    break

        if goal not in rrt[0].vertices:
            x_near, dist = self.nearest_neighbor(goal, rrt[0])
            print('Could not find goal in {} samples. Closest point {} @ dist {}'.format(i, x_near, dist))
            goal = x_near

        return rrt, [goal]

    # Based on algorithm: Optimal Bidirectional Rapidly-Exploring Random Trees
    # http://dspace.mit.edu/bitstream/handle/1721.1/79884/MIT-CSAIL-TR-2013-021.pdf
    def generate_RRT_Star_Bi(self, start, goal):
        grid = self.grid
        alt = self.target_altitude

        rrt = [RRT(start), RRT(goal, id=1)]
        goals = [start, goal]
        x_near_s = x_near_g = None
        min_cost = float('inf')
        min_dist = 0

        i = -1
        while i < self.n_samples:
            x_rand = self.sample(grid, alt)
            x_near, x_new, u = self.get_near_new(rrt[0], x_rand, grid.shape)
            if not self.check_valid_sample(rrt[0], x_new, x_near):
                continue

            i += 1
            X_near = self.connect_best_parent(rrt[0], x_new)

            if X_near is not None:
                self.rewire(rrt[0], x_new, X_near)

            x_conn_new, cost, dist = self.connect_rs(rrt[1], x_new, grid.shape)
            cost +=  self.get_path_cost([rrt[0]], [x_new])
            if x_conn_new is not None and cost < min_cost:
                x_conn = (rrt[1].id, x_new, x_conn_new, i)
                min_cost = cost
                min_dist = dist

            if min_cost < float('inf') and i >= self.min_samples:
                print('Found goal in {} samples'.format(i))
                break

            rrt = self.swap_trees(rrt)
            goals = goals[::-1]

        if self.swapped:
            rrt = self.swap_trees(rrt)

        if min_cost == float('inf'):
            print('Could not find goal in {} samples'.format(i))
        else:
            u = self.select_input(x_conn[1], x_conn[2])
            self.add_edge(rrt[x_conn[0]], x_conn[1], x_conn[2], u, pathcost=0)
            x_near_s, x_near_g = x_conn[1], x_conn[1]
            print('Bi RRT* Final: ', min_cost, min_dist, x_conn)
            
        return rrt, [x_near_s, x_near_g]

    def sample(self, grid, alt):
        x_rand = self.sample_state(grid, alt)
        # sample states until a free state is found
        while grid[int(x_rand[0]), int(x_rand[1])] == 1:
            x_rand = self.sample_state(grid, alt)

        return x_rand

    def get_path_cost(self, rrt, x):
        cost_p = 0.
        for i in range(len(rrt)):
            pred = [pred for pred in rrt[i].tree.predecessors(x[i])]
            if len(pred):
                cost_p += rrt[i].tree.edges[pred[-1], x[i]]['pathcost']

        return cost_p

    def swap_trees(self, rrt):
        self.swapped = not self.swapped
        return rrt[::-1]

    def sample_state(self, grid, alt):
        xval = np.random.uniform(0, grid.shape[0])
        yval = np.random.uniform(0, grid.shape[1])
        return (xval, yval, alt)

    def nearest_neighbor(self, x_rand, rrt):
        pts = np.array(rrt.vertices)
        dist = LA.norm(pts - np.array(x_rand), axis=1)
        idx = np.argmin(dist)
        return tuple(pts[idx]), dist[idx]

    def select_input(self, x_rand, x_near):
        # Select input which moves x_near closer to x_rand. 
        # This should return the angle or orientation of the vehicle
        return np.arctan2(x_rand[1] - x_near[1], x_rand[0] - x_near[0])

    def steer(self, x_near, u, grid_shape):
        # The new vertex x_new is calculated by travelling from 
        # the current vertex x_near with a orientation u for time dt
        nx = x_near[0] + np.cos(u)*self.dt
        ny = x_near[1] + np.sin(u)*self.dt
        nx = np.clip(nx, 1, grid_shape[0] - 2)
        ny = np.clip(ny, 1, grid_shape[1] - 2)
        return (nx, ny, x_near[2])

    def obstacle_free(self, grid, p1, p2, min_altitude):
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
            if grid[c[0], c[1]] >= min_altitude:
                hit = True
                break
        return not hit

    def check_valid_sample(self, rrt, x_new, x_near):
        return rrt.tree.has_node(x_new) == False and self.obstacle_free(self.grid25, x_new, x_near, self.target_altitude)

    def get_move_state(self, x, x_new):
        dist = LA.norm(np.array(x_new) - np.array(x))
        if dist <=  self.min_dist:
            ret = State.Reached
        else:
            ret = State.Advanced

        return ret

    def add_edge(self, rrt, x_new, x_near, u, pathcost=0):
        dist = LA.norm(np.array(x_new) - np.array(x_near))
        rrt.add_edge(x_near, x_new, u, dist, pathcost)

    def add_valid_edge(self, rrt, x, x_new, x_near, u):
        if self.check_valid_sample(rrt, x_new, x_near):
            self.add_edge(rrt, x_new, x_near, u, pathcost=0)
            ret = self.get_move_state(x, x_new)
        else:
            ret = State.Trapped
        return ret

    def get_near_new(self, rrt, x, grid_shape, plt_pt=False):
        x_near, _ = self.nearest_neighbor(x, rrt)
        u = self.select_input(x, x_near)
        x_new = self.steer(x_near, u, grid_shape)

        if plt_pt:
            import matplotlib.pyplot as plt
            plt.imshow(self.grid, cmap='Greys', origin='lower')
            plt.plot(x[1], x[0], 'bx')
            plt.plot(x_near[1], x_near[0], 'gx')
            plt.plot(x_new[1], x_new[0], 'rx')
            plt.plot([x[1], x_near[1]], [x[0], x_near[0]], 'y-')

            plt.show()

        return x_near, x_new, u

    def extend(self, rrt, x, grid_shape):
        x_near, x_new, u = self.get_near_new(rrt, x, grid_shape)
        ret = self.add_valid_edge(rrt, x, x_new, x_near, u)
        return ret, x_new

    def connect(self, rrt, x, grid_shape):
        ret = State.Advanced
        while ret == State.Advanced:
            ret, x_new = self.extend(rrt, x, grid_shape)
        return ret

    def connect_goal(self, rrt, goal):
        x_near, _ = self.nearest_neighbor(goal, rrt)
        u = self.select_input(goal, x_near)
        ret = self.add_valid_edge(rrt, goal, goal, x_near, u)
        return ret

    def sample_cost(self, rrt, x_near, x_new):
        dist = LA.norm(np.array(x_near) - np.array(x_new))
        pred = [pred for pred in rrt.tree.predecessors(x_near)]
        if len(pred):
            cost_p = rrt.tree.edges[pred[-1], x_near]['pathcost']
        else:
            cost_p = 0.
        return  cost_p + dist, cost_p

    def n_nearest_neighbors(self, x, rrt, n_neighbors):
        pts = np.array(rrt.vertices)
        dist = LA.norm(pts - np.array(x), axis=1)
        if len(dist) > n_neighbors:
            idx = np.argpartition(dist, n_neighbors, axis=0)
            idx = idx[0:n_neighbors]
        else:
            idx = np.arange(0, len(dist))

        x_neighbors = tuple(map(tuple, pts[idx]))
        return x_neighbors

    def find_best_parent(self, rrt, x, x_neighbors):
        X_near = PriorityQueue()
        for x_near in x_neighbors:
            if self.obstacle_free(self.grid25, x, x_near, self.target_altitude):
                cost, cost_p = self.sample_cost(rrt, x_near, x)
                X_near.put((cost, cost_p, x_near))

        return X_near

    def add_best_parent(self, rrt, x, x_neighbors):
        X_near = self.find_best_parent(rrt, x, x_neighbors)

        if not X_near.empty():
            cost, cost_p, x_min = X_near.get()
            u = self.select_input(x, x_min)
            self.add_edge(rrt, x, x_min, u, pathcost=cost)
            ret = State.Reached
        else:
            ret = State.Trapped
        return X_near, ret

    def connect_best_parent(self, rrt, x_new):
        x_neighbors = self.n_nearest_neighbors(x_new, rrt, self.n_neighbors)
        X_near, _ = self.add_best_parent(rrt, x_new, x_neighbors)
        return X_near

    def rewire(self, rrt, x_new, X_near):
        while not X_near.empty():
            _, cost_p, x_near = X_near.get()
            cost_new, _ = self.sample_cost(rrt, x_new, x_near)
            if cost_new < cost_p:
                pred = [pred for pred in rrt.tree.predecessors(x_near)]
                pred = pred[-1]
                # if the path through xnew has lower cost than the path through the current parent
                # the edge linking the vertex to its current parent is deleted
                rrt.tree.remove_edge(pred, x_near)
                # New edges are created from xnew to x_near
                u = self.select_input(x_near, x_new)
                self.add_edge(rrt, x_near, x_new, u, pathcost=cost_new)

    def connect_rs(self, rrt, x, grid_shape):
        cost = float('inf'), 
        dist = 0
        x_min = None
        x_near, x_new, u = self.get_near_new(rrt, x, grid_shape)

        x_neighbors = self.n_nearest_neighbors(x_new, rrt, self.n_neighbors)
        X_near = self.find_best_parent(rrt, x, x_neighbors)

        if not X_near.empty():
            cost, cost_p, x_min = X_near.get()
            dist = cost-cost_p

        return x_min, cost, dist

    def a_star(self, grid_start, grid_goal):
        np.random.seed(self.seed)
        start_t = timer()
        path = None
        self.rrt, goals = self.rrt_funcs[self.algo](grid_start, grid_goal)
        print("Generate RRT time: {}s".format(timer() - start_t))

        if goals[0] is not None:
            path = [goals[0]]
            pred = goals[0]
            while pred != grid_start:
                pred = list(self.rrt[0].tree.predecessors(pred))
                path = path + pred
                pred = pred[-1]
            path = path[::-1]

            #path = self.astar.search(self.rrt[0].tree, grid_start, goals[0])

        if len(self.rrt) > 1 and goals[1] is not None:
            path_s = path
            #path_g = self.astar.search(self.rrt[1].tree, grid_goal, goals[1])
            #path = path_s + path_g[::-1]

            path_g = [goals[1]]
            pred = goals[1]
            while pred != grid_goal:
                pred = list(self.rrt[1].tree.predecessors(pred))
                path_g = path_g + pred
                pred = pred[-1]
            path = path_s + path_g

        if len(path):
            print('Found a path.')

        return path

    def plot_grid(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        edges = [([n1[0], n1[1]], [n2[0], n2[1]]) for (n1, n2) in self.rrt[0].edges]
        if len(self.rrt) > 1:
            edges1 = [([n1[0], n1[1]], [n2[0], n2[1]]) for (n1, n2) in self.rrt[1].edges]
        else:
            edges1 = None

        super().plot_grid(drone_home, drone_start=drone_start, drone_goal=drone_goal, path=path, edges=edges, edges_g=edges1, title=self.titles[self.algo])

    def plot_voxmap(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        pass