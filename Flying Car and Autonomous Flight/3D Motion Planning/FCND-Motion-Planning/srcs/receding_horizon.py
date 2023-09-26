import numpy as np
import numpy.linalg as LA
from bresenham import bresenham
from srcs.grid import GridPlanner
from srcs.voxmap import VoxMapPlanner
from timeit import default_timer as timer

class RecedingHorizonPlanner(GridPlanner):
    def __init__(self, obstacles_data, target_altitude, safety_distance, name, lr_altitude, search_step, lr_search_step, h_type='norm', save_plt=False, vis_dir='images'):
        super(RecedingHorizonPlanner, self).__init__(obstacles_data, target_altitude, safety_distance, name, h_type='norm', save_plt=save_plt, vis_dir=vis_dir)

        self.voxel_size = 1
        self.search_step = search_step
        self.lr_search_step = lr_search_step
        self.target_altitude = target_altitude
        self.safety_distance = safety_distance
        self.obstacles_data = obstacles_data

        self.motion_p = VoxMapPlanner(obstacles_data, target_altitude, safety_distance, name, h_type=h_type, voxel_size=self.voxel_size, vis_dir=vis_dir)
        self.grid_lr, _, _, _, _, _, _, _ = self.create_grid(obstacles_data, lr_altitude, safety_distance)

    def need_local_replan(self):
        return True

    def check_obstacle(self, loc):
        return False

    def a_star(self, grid_start, grid_goal):
        grid_goal = (grid_goal[0], grid_goal[1], grid_start[2])
        print(self.grid_lr[grid_start[0], grid_start[1]], self.grid_lr[grid_goal[0], grid_goal[1]])
        path = self.astar.search(self.grid_lr, grid_start, grid_goal, self.h_type, step=self.lr_search_step)
        return path

    def get_obstacles(self, grid, p1, p2, min_altitude):
        obstacles = []
        max_z = 0
        # Test each pair p1 and p2 for collision using Bresenham
        # (need to convert to integer if using prebuilt Python package)
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        
        hit = False
        for c in cells:
            # Check if we're in collision
            if grid[c[0], c[1]] >= min_altitude:
                hit = True
                obstacles.append([c[0], c[1], grid[c[0], c[1]]])
            elif len(obstacles):
                break
        c = (c[0], c[1], max(min_altitude, int(np.ceil(grid[c[0], c[1]]))+1))

        if hit:
            obstacles = np.array(obstacles)
            max_z = max(obstacles[:, 2])
            
        return hit, c, max_z

    def local_path_plan(self, cur_loc, target_loc):
        gc = self.grid_center
        p1 = cur_loc = self.get_grid_coord(list(cur_loc), self.target_altitude)
        p2 = target_loc = (target_loc[0]+gc[0], target_loc[1]+gc[1], target_loc[2])
        hit, c, max_z = self.get_obstacles(self.grid25, p1, p2, self.target_altitude)
        print('Replan: ', hit, cur_loc, target_loc)
        if hit:
            max_z += self.safety_distance
            if c[0] == p2[0] and c[1] == p2[1]:
                p2 = None

            print('Local Start and Goal: ', p1, c)
            start_t = timer()
            path = self.motion_p.a_star(p1, c, step=self.search_step)
            print("Path Search Time: {}s".format(timer() - start_t))
        else:
            p2 = None
            path = [cur_loc, target_loc]
        return path, p2

    def plot_grid(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        super().plot_grid(drone_home, drone_start=drone_start, drone_goal=drone_goal, path=path, path_lr=path_lr, title='Receding Horizon')

    def plot_voxmap(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        vs = 4
        gc = self.grid_center
        voxmap, _, _ = self.motion_p.create_voxmap(self.obstacles_data, self.safety_distance, voxel_size=vs)
        super().plot_voxmap(drone_home, drone_start=drone_start, drone_goal=drone_goal, path=path, path_lr=path_lr, voxmap=voxmap, voxel_size=vs, title='Receding Horizon')