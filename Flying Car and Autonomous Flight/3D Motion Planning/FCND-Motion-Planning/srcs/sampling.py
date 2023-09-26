import numpy as np
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point

class Poly:

    def __init__(self, coords, height):
        self._polygon = Polygon(coords)
        self._height = height

    @property
    def height(self):
        return self._height

    @property
    def coords(self):
        return list(self._polygon.exterior.coords)[:-1]
    
    @property
    def area(self):
        return self._polygon.area

    @property
    def center(self):
        return (self._polygon.centroid.x, self._polygon.centroid.y)

    def contains(self, point):
        point = Point(point)
        return self._polygon.contains(point)

    def crosses(self, other):
        return self._polygon.crosses(other)

def extract_polygons(obstacle_corners, obstacle_height):

    polygons = []
    for corner, height in zip(obstacle_corners, obstacle_height):
        p = Poly(corner, height)
        polygons.append(p)

    return polygons

class Sampler:

    def __init__(self, obstacle_corners, obstacle_height, obstacle_max_xy, grid_shape):
        self._polygons = extract_polygons(obstacle_corners, obstacle_height)
        self._ymin = grid_shape[0]
        self._ymax = grid_shape[1]
        self._xmin = grid_shape[2]
        self._xmax = grid_shape[3]
        self._zmin = grid_shape[4]
        self._zmax = grid_shape[5]

        # Record maximum polygon dimension in the xy plane
        # multiply by 2 since given sizes are half widths
        # This is still rather clunky but will allow us to 
        # cut down the number of polygons we compare with by a lot.
        self._max_poly_xy = 2 * obstacle_max_xy
        centers = np.array([p.center for p in self._polygons])
        self._tree = KDTree(centers, metric='euclidean')

    def set_grid_shape(self, grid_shape):
        self._ymin = grid_shape[0]
        self._ymax = grid_shape[1]
        self._xmin = grid_shape[2]
        self._xmax = grid_shape[3]
        self._zmin = grid_shape[4]
        self._zmax = grid_shape[5]

    def sample(self, num_samples, kd_query, kd_neigh=3):
        """Implemented with a k-d tree for efficiency."""
        xvals = np.random.uniform(self._xmin, self._xmax, num_samples)
        yvals = np.random.uniform(self._ymin, self._ymax, num_samples)
        zvals = np.random.uniform(self._zmin, self._zmax, num_samples)
        samples = list(zip(yvals, xvals, zvals))

        pts = []
        for s in samples:
            in_collision = False
            if kd_query == 'rad':
                idxs = list(self._tree.query_radius(np.array([s[0], s[1]]).reshape(1, -1), r=self._max_poly_xy)[0])
            if kd_query == 'nn':
                idxs = list(self._tree.query(np.array([s[0], s[1]]).reshape(1, -1), kd_neigh, return_distance=False)[0])

            if len(idxs) > 0:
                for ind in idxs: 
                    p = self._polygons[int(ind)]
                    if p.contains(s) and p.height >= s[2]:
                        in_collision = True
            if not in_collision:
                pts.append(s)
                
        return pts

    @property
    def polygons(self):
        return self._polygons
