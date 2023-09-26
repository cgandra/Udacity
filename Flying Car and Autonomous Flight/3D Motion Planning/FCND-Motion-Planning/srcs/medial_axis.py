import numpy as np
from skimage.util import invert
from skimage.morphology import medial_axis
from srcs.grid import GridPlanner

class MedialAxisPlanner(GridPlanner):
    def __init__(self, obstacles_data, target_altitude, safety_distance, name, h_type='norm', save_plt=False, vis_dir='images'):
        super(MedialAxisPlanner, self).__init__(obstacles_data, target_altitude, safety_distance, name, h_type=h_type, save_plt=save_plt, vis_dir=vis_dir)
        self.skeleton = medial_axis(invert(self.grid))
        self.inv_skeleton = invert(self.skeleton).astype(np.int)
        self.skel_pts = np.transpose(np.nonzero(self.skeleton))

    def a_star(self, grid_start, grid_goal):
        cp = self.closest_point(self.skel_pts, grid_start[0:2])
        grid_start = (cp[0], cp[1], grid_start[2])
        cp = self.closest_point(self.skel_pts, grid_goal[0:2])
        grid_goal = (cp[0], cp[1], grid_goal[2])
        return self.astar.search(self.inv_skeleton, grid_start, grid_goal, self.h_type)

    def plot_grid(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        super().plot_grid(drone_home, drone_start=drone_start, drone_goal=drone_goal, path=path, skeleton=self.skeleton, title='Medial Axis')

    def plot_voxmap(self, drone_home, drone_start=None, drone_goal=None, path=None, path_lr=None):
        pass