import numpy as np
import numpy.linalg as LA
from srcs.a_star import AStar3D
from srcs.grid import GridPlanner

class VoxMapPlanner(GridPlanner):
    def __init__(self, obstacles_data, target_altitude, safety_distance, name, h_type='norm', voxel_size=5, save_plt=False, vis_dir='images'):
        super(VoxMapPlanner, self).__init__(obstacles_data, target_altitude, safety_distance, name, h_type='norm', save_plt=save_plt, vis_dir=vis_dir)

        self.voxel_size = voxel_size
        self.voxmap, self.north_offset, self.east_offset = self.create_voxmap(obstacles_data, safety_distance, voxel_size)

        self.astar = AStar3D()

    def create_voxmap(self, data, safety_distance, voxel_size=5):
        """
        Returns a grid representation of a 3D configuration space
        based on given obstacle data, drone altitude and safety distance
        arguments.
        """

        north_size, east_size, north_min, east_min, alt_max = self.get_grid_size(data)

        # Initialize an empty voxmap
        voxmap = np.zeros((north_size//voxel_size, east_size//voxel_size, alt_max//voxel_size), dtype=np.bool)

        # Populate the grid with obstacles
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i, :]
            height = int(np.ceil(alt + d_alt + safety_distance))//voxel_size
            obstacle = self.get_obstacle_corners(data, i, north_min, east_min, north_size, east_size, safety_distance, voxel_size)
            voxmap[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1, 0:height+1] = True

        return voxmap, int(north_min), int(east_min)

    def check_obstacle(self, loc):
        return False

    def a_star(self, grid_start, grid_goal, step=1):
        vs = self.voxel_size
        shp = self.voxmap.shape

        gs = (grid_start[0]//vs, grid_start[1]//vs, grid_start[2]//vs)
        while self.voxmap[gs[0], gs[1], gs[2]] and gs[0] < shp[0] and gs[1] < shp[1] and gs[2] < shp[2]:
            gs = (gs[0], gs[1], gs[2]+1)

        gg = (grid_goal[0]//vs, grid_goal[1]//vs, grid_goal[2]//vs)
        while self.voxmap[gg[0], gg[1], gg[2]] and gg[0] < shp[0] and gg[1] < shp[1] and gg[2] < shp[2]:
            gg = (gg[0], gg[1], gg[2]+1)
        print('Voxel Start and Goal: ', gs, gg)

        path = self.astar.search(self.voxmap, gs, gg, self.h_type, step=step)
        path = [[p[0]*vs, p[1]*vs, p[2]*vs] for p in path]
        return path

    def plot_grid(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None, save_plt=False):
        super().plot_grid(drone_home, drone_start=drone_start, drone_goal=drone_goal, path=path, title='Voxmap')

    def plot_voxmap(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None, save_plt=False):
        super().plot_voxmap(drone_home, drone_start=drone_start, drone_goal=drone_goal, path=path, path_lr=path_lr, voxmap=self.voxmap, voxel_size=self.voxel_size, title='Voxmap')
