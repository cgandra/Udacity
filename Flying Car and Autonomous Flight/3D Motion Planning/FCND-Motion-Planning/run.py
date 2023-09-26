import csv
import yaml
import argparse
import numpy as np

from srcs.rrt import RRTPlanner
from srcs.grid import GridPlanner
from srcs.voxmap import VoxMapPlanner
from srcs.voronoi import VoronoiPlanner
from srcs.medial_axis import MedialAxisPlanner
from srcs.probabilistic_roadmap import ProbRoadMapPlanner
from srcs.receding_horizon  import RecedingHorizonPlanner
from srcs.motion_planning import drone_run

from timeit import default_timer as timer
from udacidrone.frame_utils import global_to_local, local_to_global

def get_lon_lat(file):
    f=open(file, newline='')
    reader = csv.reader(f)
    data = next(reader)
    f.close()
    lat0 = float(data[0].split()[1])
    lon0 = float(data[1].split()[1])

    return [lon0, lat0]

def get_global_pos(loc_pos):
    lon, lat, _ = local_to_global(loc_pos, drone_home_g)
    return [lon, lat]

def check_valid_goal(motion_p, cfg, drone_home_g):
    drone_goal = global_to_local(cfg['common_p']['goal'], drone_home_g)
    drone_goal = np.round(drone_goal).tolist()
    drone_goal = motion_p.get_grid_coord(np.array(drone_goal), cfg['common_p']['target_altitude'])
    if motion_p.check_obstacle(drone_goal):
        print("Error: Obstacle at selected goal {} at grid loc {}".format(cfg['common_p']['goal'], drone_goal))
        print("Choose another goal")
        exit()

def run_sim(motion_p, cfg, drone_home_g):
    check_valid_goal(motion_p, cfg, drone_home_g)
    drone_run(motion_p, drone_home_g, cfg)

def run(mp_c, cfg, method, grid_goal, drone_home_g):
    np.random.seed(cfg['common_p']['seed'])
    start_t = timer()
    if method in cfg.keys():
        motion_p = mp_c(data, cfg['common_p']['target_altitude'], cfg['common_p']['safety_dist'], method, **cfg[method], save_plt=cfg['debug_p']['save_plt'], vis_dir=cfg['debug_p']['vis_dir'])
    else:
        motion_p = mp_c(data, cfg['common_p']['target_altitude'], cfg['common_p']['safety_dist'], method, save_plt=cfg['debug_p']['save_plt'], vis_dir=cfg['debug_p']['vis_dir'])
  
    print("Motion Planner Setup Time: {}s".format(timer() - start_t))
    
    if not cfg['debug_p']['plt_local']:
        motion_p.set_callback(get_global_pos)
    
    starts = []
    if cfg['common_p']['debug_algo'] and 'preset_starts' in cfg.keys():
        for key, val in cfg['preset_starts'].items():
            if 'start' not in key:
                continue

            starts.append((val['north'], val['east'], val['alt']))
    else:
        starts.append(motion_p.get_grid_coord([0, 0, 0], cfg['common_p']['target_altitude']))

    if 'preset_goals' in cfg.keys():
        for key, val in cfg['preset_goals'].items():
            if 'goal' not in key:
                continue

            if len(starts):
                cfg['common_p']['start'] = starts.pop(0)

            cfg['common_p']['goal'] = [val['lon'], val['lat'], val['alt']]
            run_sim(motion_p, cfg, drone_home_g)
    else:
        #Its easier to select grid offsets instead of lat/lon. But given requirements this is tedious route
        cfg['common_p']['goal'] = grid_goal
        run_sim(motion_p, cfg, drone_home_g)

if __name__ == "__main__":
    # TODO: read lat0, lon0 from colliders into floating point values
    obstacles_file = 'colliders.csv'
    drone_home_g = get_lon_lat(obstacles_file)
    drone_home_g.append(0)

    # Read in obstacle map
    data = np.loadtxt(obstacles_file, delimiter=',', dtype='Float64', skiprows=2)

    # TODO: Handle initialization better instead of creating grid object twice
    motion_p = GridPlanner(data, 5, 5, 'grid')
    lon_rng, lat_rng = motion_p.get_lat_lon_range(get_global_pos)
    default_goal = local_to_global([10, 10, 3], drone_home_g)
    
    helpstr = "[Longitude, Latitude, Altitude]" + ", Valid Range of Lon: " + str(lon_rng) + ", Valid Range of Lat: " + str(lat_rng)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="path to config")
    parser.add_argument('--method', type=str, default='grid', help="Motion Planning type: grid/med_axis/voronoi/prob_map/vox_map/reced_horz/rrt")
    parser.add_argument('--grid_goal', nargs=3, type=float, default=default_goal, help=helpstr)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    mp_class = {}
    mp_class['grid'] = GridPlanner
    mp_class['med_axis'] = MedialAxisPlanner
    mp_class['voronoi'] = VoronoiPlanner
    mp_class['vox_map'] = VoxMapPlanner
    mp_class['prob_map'] = ProbRoadMapPlanner
    mp_class['reced_horz'] = RecedingHorizonPlanner
    mp_class['rrt'] = RRTPlanner

    mp_c = mp_class[args.method]

    if args.method == 'rrt':
        algos = cfg[args.method]['algo']
        cfg[args.method]['seed'] = cfg['common_p']['seed']
        if type(algos) is not list:            
            algos = [algos]
        num_runs = len(algos)
    else:
        num_runs = 1

    for n in range(num_runs):
        if args.method == 'rrt':
            cfg[args.method]['algo'] = algos.pop(0)

        run(mp_c, cfg, args.method, args.grid_goal, drone_home_g)
