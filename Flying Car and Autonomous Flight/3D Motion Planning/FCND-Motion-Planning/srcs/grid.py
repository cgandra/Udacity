import os
import math
import numpy as np
import matplotlib.pyplot as plt
from srcs.a_star import AStarGrid 

from timeit import default_timer as timer

class GridPlanner():
    def __init__(self, obstacles_data, target_altitude, safety_distance, name, h_type='norm', obs_cen=False, obs_con_h=False, save_plt=False, vis_dir='images'):
        self.h_type = h_type
        self.callback_fn = None

        # Define a grid for a particular altitude and safety margin around obstacles
        self.grid, self.grid25, self.north_offset, self.east_offset, self.obstacle_centre, self.obstacle_corners, self.obstacle_height, self.obstacle_max_xy = self.create_grid(obstacles_data, target_altitude, safety_distance, obs_cen=obs_cen, obs_con_h=obs_con_h)

        print("North offset = {0}, east offset = {1}".format(self.north_offset, self.east_offset))
        self.grid_center = (-self.north_offset, -self.east_offset)

        self.grid_tl = self.get_grid_tl()
        self.grid_tl.append(0)
        self.grid_br = self.get_grid_br()
        self.grid_br.append(0)

        self.astar = AStarGrid()

        # For saving images
        self.fcnt = 0
        self.fcnt_v = 0
        self.name = name
        self.save_plt = save_plt
        self.vis_dir = os.path.join(vis_dir, self.name)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        self.layout_data = []

    def need_local_replan(self):
        return False

    def set_callback(self, func_cb=None):
        self.callback_fn = func_cb

    def set_heuristics_type(self, h_type):
        self.h_type = h_type

    def get_grid_tl(self):
        return [self.north_offset, self.east_offset]

    def get_grid_br(self):
        return [self.north_offset+self.grid.shape[0], self.east_offset+self.grid.shape[1]]

    def get_grid_coord(self, local_pos, target_altitude):
        pos_0 = np.clip(int(np.round(local_pos[0]+self.grid_center[0])), 0, self.grid.shape[0])
        pos_1 = np.clip(int(np.round(local_pos[1]+self.grid_center[1])), 0, self.grid.shape[1])
        local_pos[2] = max(target_altitude, local_pos[2], self.grid25[pos_0, pos_1])
        return (pos_0, pos_1, int(np.round(local_pos[2])))

    def check_obstacle(self, loc):
        return self.grid[loc[0], loc[1]]

    def closest_point(self, pts, pt):
        """
        Find the closest point in the graph/skeleton etc
        to the `current_point`.
        """
        dist = np.linalg.norm(pts - np.array(pt), axis=1)
        idx = np.argmin(dist)
        return tuple(pts[idx])

    def a_star(self, grid_start, grid_goal):
        return self.astar.search(self.grid, grid_start, grid_goal, self.h_type)

    def get_grid_size(self, data):

        # minimum and maximum north coordinates
        north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
        north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

        # minimum and maximum east coordinates
        east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
        east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

        # maximum altitude
        alt_max = int(np.ceil(np.max(data[:, 2] + data[:, 5])))

        # given the minimum and maximum coordinates we can
        # calculate the size of the grid.
        north_size = int(north_max - north_min)
        east_size = int(east_max - east_min)

        return north_size, east_size, north_min, east_min, alt_max

    def get_obstacle_corners(self, data, i, north_min, east_min, north_size, east_size, safety_distance, voxel_size=1):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        y_min = int(np.clip(np.floor(north - d_north - safety_distance - north_min)/voxel_size, 0, north_size-1))
        y_max = int(np.clip(np.ceil(north + d_north + safety_distance - north_min)/voxel_size, 0, north_size-1))
        x_min = int(np.clip(np.floor(east - d_east - safety_distance - east_min)/voxel_size, 0, east_size-1))
        x_max = int(np.clip(np.ceil(east + d_east + safety_distance - east_min)/voxel_size, 0, east_size-1))

        return [y_min, y_max, x_min, x_max]

    def create_grid(self, data, drone_altitude, safety_distance, obs_cen=False, obs_con_h=False):
        """
        Returns a grid representation of a 2D configuration space
        based on given obstacle data, drone altitude and safety distance
        arguments.
        """

        north_size, east_size, north_min, east_min, _ = self.get_grid_size(data)
        if obs_con_h:
            obstacle_max_xy = np.max((data[:, 3], data[:, 4])) + safety_distance
        else:
            obstacle_max_xy = 0

        # Initialize an empty grid
        grid = np.zeros((north_size, east_size))
        grid25 = np.zeros((north_size, east_size))

        # Define a list to hold Voronoi points
        obstacle_centre = []

        # Define lists to hold obstacle corners/heights
        obstacle_corners = []
        obstacle_height = []

        obstacles_t = []
        # Populate the grid with obstacles
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i, :]
            height = alt + d_alt + safety_distance
            obstacle = self.get_obstacle_corners(data, i, north_min, east_min, north_size, east_size, safety_distance)
            grid25[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = height
            if height > drone_altitude:
                grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
                obstacles_t.append([obstacle[0], obstacle[1], obstacle[2], obstacle[3], 0, height])
                
                # add center of obstacles to list
                if obs_cen:
                    obstacle_centre.append([north - north_min, east - east_min])

                # add obstacle corners/heights to list
                if obs_con_h:
                    obstacle_corners.append([(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])])                
                    obstacle_height.append(int(np.ceil(height)))

        return grid, grid25, int(north_min), int(east_min), obstacle_centre, obstacle_corners, obstacle_height, obstacle_max_xy

    def create_grid25(self, data, drone_altitude, safety_distance, obs_cen=False, obs_con_h=False):
        """
        Returns a grid representation of a 3D configuration space
        based on given obstacle data, drone altitude and safety distance
        arguments.
        """

        north_size, east_size, north_min, east_min, _ = self.get_grid_size(data)
        if obs_con_h:
            obstacle_max_xy = np.max((data[:, 3], data[:, 4])) + safety_distance
        else:
            obstacle_max_xy = 0

        # Initialize an empty grid
        grid = np.zeros((north_size, east_size))

        # Define a list to hold Voronoi points
        obstacle_centre = []

        # Define lists to hold obstacle corners/heights
        obstacle_corners = []
        obstacle_height = []

        # Populate the grid with obstacles
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i, :]
            height = alt + d_alt + safety_distance
            obstacle = self.get_obstacle_corners(data, i, north_min, east_min, north_size, east_size, safety_distance)
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = height
            
            # add center of obstacles to list
            if obs_cen:
                obstacle_centre.append([north - north_min, east - east_min])

            # add obstacle corners/heights to list
            if obs_con_h:
                obstacle_corners.append([(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])])                
                obstacle_height.append(int(np.ceil(height)))

        return grid, int(north_min), int(east_min), obstacle_centre, obstacle_corners, obstacle_height, obstacle_max_xy

    def get_lat_lon_range(self, callback_fn):
        min_lon, min_lat = callback_fn(self.grid_tl)
        max_lon, max_lat = callback_fn(self.grid_br)
        lon_rng = [min_lon, max_lon]
        lat_rng = [min_lat, max_lat]
        return lon_rng, lat_rng

    def convert_to_lat_lon(self, drone_home, drone_start, drone_goal, path, path_lr, edges, edges_g, nodes, all_nodes):
        gc = self.grid_center
        cbfn = self.callback_fn
        if drone_start is not None:
            drone_start = (cbfn(drone_start))[::-1]
        if drone_goal is not None:
            drone_goal = (cbfn(drone_goal))[::-1]
            
        drone_home = (cbfn(drone_home))[::-1]
        x_rng, y_rng = self.get_lat_lon_range(self.callback_fn)
        
        if edges is not None:
            edges = [[((cbfn([e[0][0]-gc[0], e[0][1]-gc[1], 0]))[0:2])[::-1], ((cbfn([e[1][0]-gc[0], e[1][1]-gc[1], 0]))[0:2])[::-1]] for e in edges]
            
        if edges_g is not None:
            edges_g = [[((cbfn([e[0][0]-gc[0], e[0][1]-gc[1], 0]))[0:2])[::-1], ((cbfn([e[1][0]-gc[0], e[1][1]-gc[1], 0]))[0:2])[::-1]] for e in edges_g]

        if nodes is not None:
            nodes = [((cbfn([n[0]-gc[0], n[1]-gc[1], 0]))[0:2])[::-1] for n in nodes]

        if all_nodes is not None:
            all_nodes = [((cbfn([n[0]-gc[0], n[1]-gc[1], 0]))[0:2])[::-1] for n in all_nodes]
            
        if path is not None:
            path = [((cbfn([p[0], p[1], 0]))[0:2])[::-1] for p in path]

        if path_lr is not None:
            path_lr = [((cbfn([p[0], p[1], 0]))[0:2])[::-1] for p in path_lr]

        return drone_home, drone_start, drone_goal, path, path_lr, edges, edges_g, nodes, all_nodes, x_rng, y_rng

    def plot_grid(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None, skeleton=None, 
                  edges=None, edges_g=None, nodes=None, all_nodes=None, title='Grid'):
        gc = self.grid_center
        if self.callback_fn is not None:
            x_label, y_label = ['Longitude', 'Latitude']
            drone_home, drone_start, drone_goal, path, path_lr, edges, edges_g, nodes, all_nodes, x_rng, y_rng =\
            self.convert_to_lat_lon(drone_home, drone_start, drone_goal, path, path_lr, edges, edges_g, nodes, all_nodes)
        else:
            x_label, y_label = ['X-East', 'Y-North']
            x_rng = [self.grid_tl[1],self.grid_br[1]]
            y_rng = [self.grid_tl[0],self.grid_br[0]]

            if edges is not None:
                edges = [[[e[0][0]-gc[0], e[0][1]-gc[1]], [e[1][0]-gc[0], e[1][1]-gc[1]]] for e in edges]

            if edges_g is not None:
                edges_g = [[[e[0][0]-gc[0], e[0][1]-gc[1]], [e[1][0]-gc[0], e[1][1]-gc[1]]] for e in edges_g]

            if nodes is not None:
                nodes = [[n[0]-gc[0], n[1]-gc[1]] for n in nodes]

            if all_nodes is not None:
                all_nodes = [[n[0]-gc[0], n[1]-gc[1]] for n in all_nodes]

        fig, ax = plt.subplots()
        if skeleton is not None:
            skeleton = (skeleton.copy()).astype(self.grid.dtype)
            ax.imshow(self.grid, origin='lower', cmap='Greys', extent=(x_rng[0],x_rng[1],y_rng[0],y_rng[1]))
            ax.imshow(skeleton, origin='lower', cmap='Greys', alpha=0.7, extent=(x_rng[0],x_rng[1],y_rng[0],y_rng[1]))
        else:
            ax.imshow(self.grid, origin='lower', cmap='Greys', extent=(x_rng[0],x_rng[1],y_rng[0],y_rng[1]))
        
        if edges is not None:
            for e in edges:
                p1 = e[0]
                p2 = e[1]
                ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'y-')
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'y-', label="Edges")

        if edges_g is not None:
            for e in edges_g:
                p1 = e[0]
                p2 = e[1]
                ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'c-')
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'c-', label="Edges-G")

        if all_nodes is not None:
            n = np.array(all_nodes)
            ax.scatter(n[:, 1], n[:, 0], c='m', label="All Nodes")

        if nodes is not None:
            n = np.array(nodes)
            ax.scatter(n[:, 1], n[:, 0], c='c', label="Connected Nodes")

        ax.plot(drone_home[1], drone_home[0], 'm*', label="Home")

        if drone_start is not None:
            ax.plot(drone_start[1], drone_start[0], 'bx', label="Start")

        if drone_goal is not None:
            ax.plot(drone_goal[1], drone_goal[0], 'gx', label="Goal")

        if path_lr is not None:
            wps = np.array(path_lr)
            ax.plot(wps[:, 1], wps[:, 0], 'y', label="Path_lr")

        if path is not None:
            wps = np.array(path)
            ax.plot(wps[:, 1], wps[:, 0], 'r', label="Path")

        ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', borderaxespad=0)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if self.save_plt:
            file_str = os.path.join(self.vis_dir, self.name + '_' + str(self.fcnt) + '.png')
            self.fcnt += 1
            fig.set_size_inches(19.2, 19.2)
            plt.tight_layout()
            plt.savefig(file_str)
            plt.close()
        else:
            self.maximize()
            plt.tight_layout()
            plt.show()

    def plot_voxmap(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None, voxmap=None, voxel_size=1, title='Grid'):
        if voxmap is None:
            return

        vs = voxel_size
        gc = self.grid_center
        x_label, y_label = ['North', 'East']

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        start_t = timer()
        xmin = gc[0]//vs
        ymin = gc[1]//vs
        x1 = np.arange(-xmin, (voxmap.shape[0]-xmin)+1)
        y1 = np.arange(-ymin, (voxmap.shape[1]-ymin)+1)
        z1 = np.arange(0, voxmap.shape[2]+1)
        x, y, z = np.meshgrid(x1, y1, z1)
        ax.voxels(x, y, z, voxmap, edgecolor='none', alpha=0.1)
        ax.set_xlim(x1[-1], x1[0])
        ax.set_ylim(y1[0], y1[-1])
        ax.set_zlim(z1[0], z1[-1])
        print("Voxels Plot Elapsed Time: {}s".format(timer() - start_t))

        drone_home = (drone_home[0]/vs, drone_home[1]/vs, drone_home[2]/vs)
        ax.plot3D(drone_home[0], drone_home[1], drone_home[2], 'y*', label="Home")

        if drone_start is not None:
            drone_start = (drone_start[0]/vs, drone_start[1]/vs, drone_start[2]/vs)
            ax.plot3D(drone_start[0], drone_start[1], drone_start[2], 'bx', label="Start")

        if drone_goal is not None:
            drone_goal = (drone_goal[0]/vs, drone_goal[1]/vs, drone_goal[2]/vs)
            ax.plot3D(drone_goal[0], drone_goal[1], drone_goal[2], 'gx', label="Goal")

        if path_lr is not None:
            path_lr = [[p[0]/vs, p[1]/vs, p[2]/vs] for p in path_lr]
            wps = np.array(path_lr)
            ax.plot3D(wps[:, 0], wps[:, 1], wps[:, 2], 'm', label="Path_lr")

        if path is not None:
            path = [[p[0]/vs, p[1]/vs, p[2]/vs] for p in path]
            wps = np.array(path)
            max_z = max(wps[:, 2]) + 1
            ax.plot3D(wps[:, 0], wps[:, 1], wps[:, 2], 'r', label="Path")

        ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', borderaxespad=0)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if self.save_plt:
            file_str = os.path.join(self.vis_dir, self.name + '_3D_' + str(self.fcnt_v) + '.png')
            self.fcnt_v += 1
            fig.set_size_inches(19.2, 19.2)
            plt.tight_layout()
            plt.savefig(file_str)
            plt.close()
        else:
            self.maximize()
            plt.show()

    def maximize(self):
        plot_backend = plt.get_backend()
        mng = plt.get_current_fig_manager()
        if plot_backend == 'TkAgg':
            mng.resize(*mng.window.maxsize())
        elif plot_backend == 'wxAgg':
            mng.frame.Maximize(True)
        elif plot_backend == 'Qt5Agg':
            mng.window.showMaximized()