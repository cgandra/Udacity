import time
import msgpack
from enum import Enum, auto

import numpy as np
import collections

from srcs.planning_utils import prune_path

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

from timeit import default_timer as timer

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

class MotionPlanning(Drone):

    def __init__(self, connection, motion_p, grid_goal, drone_home_g, target_altitude, core_plan_path, local_path_plan, wps=None, wps_lr=None, inline_mp=True):
        super().__init__(connection)

        self.core_plan_path = core_plan_path
        self.local_path_plan = local_path_plan
        if inline_mp:
            self.plan_path = self.plan_path_inline
        else:
            self.plan_path = self.plan_path_offline

        if connection is not None:
            self.timeout = connection._timeout
        else:
            self.timeout = 0
        self.timeout_b = False

        self.wps = wps
        self.wps_lr = wps_lr
        self.drone_start = [0,0,0]
        self.drone_goal = [0,0,0]

        self.motion_p = motion_p
        self.grid_goal = grid_goal
        self.home_pos = drone_home_g
        self.home_g = self.home_pos
        self.target_altitude = target_altitude

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.waypoints_lr = []
        self.in_mission = True
        self.check_state = {}
        self.altitude = collections.deque(maxlen=3)

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state != States.TAKEOFF and self.flight_state != States.WAYPOINT:
            return

        # coordinate conversion 
        altitude = -1.0 * self.local_position[2]

        # check if local pos close to target
        if (self.flight_state == States.TAKEOFF):
            self.altitude.append(altitude)
            avg_altitude = sum(self.altitude)/len(self.altitude)
            print("LPC-TakeOff: ", avg_altitude, self.target_position[2])
            if (len(self.altitude) < self.altitude.maxlen) or (avg_altitude <= 0.95*self.target_position[2]) or (avg_altitude >= 1.05*self.target_position[2]):
                return

        if (self.flight_state == States.WAYPOINT):
            loc = np.array([self.local_position[0], self.local_position[1], altitude])
            dist = np.linalg.norm(loc-self.target_position[0:3])
            if dist >= 1.5:
                return
            
            if len(self.waypoints) == 0 and len(self.waypoints_lr) and self.motion_p.need_local_replan():
                self.replan_path()

        if len(self.waypoints):
            self.waypoint_transition()
        elif np.linalg.norm(self.local_velocity[0:2]) < 1.0:
            self.landing_transition()

    def velocity_callback(self):
        if self.flight_state != States.LANDING:
            return

        if abs(self.local_position[2]-self.grid_goal[2]) < 0.05:
            # Shud land within 1m of the goal. Handle cases where there's elevated pavement at heights > 0 but not identified as obstacle
            print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position, self.local_position))
            self.disarming_transition()

    def state_callback(self):
        if not self.in_mission:
            return
        if self.flight_state == States.MANUAL:
            self.arming_transition()
        elif self.flight_state == States.ARMING:
            if self.armed:
                self.plan_path()
        elif self.flight_state == States.PLANNING:
            self.takeoff_transition()
            # TODO: send waypoints to sim (this is just for visualization of waypoints)
            #if len(self.waypoints) > 0:
            #    self.send_waypoints()
        elif self.flight_state == States.DISARMING:
           if ~self.armed & ~self.guided:
               self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.take_control()
        self.arm()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        if len(self.waypoints):
            self.target_position[2] = self.waypoints[0][2]
            self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self, waypoints):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(waypoints)
        self.connection._master.write(data)

    def plan_path_inline(self):
        print("Searching for a path ...")

        self.timeout_start = timer()        
        if self.process_timeout():
            self.flight_state = States.PLANNING
            return

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(self.home_pos[0], self.home_pos[1], 0)

        # TODO: retrieve current global position
        global_position = self.global_position
 
        # TODO: convert to current local position using global_to_local()
        global_home = self.global_home
        grid_start_l = global_to_local(global_position, global_home)

        print('global home {0}, position {1}, local position {2}'.format(global_home, global_position, grid_start_l))
        # Define starting point on the grid (this is just grid center)
        #grid_start = self.motion_p.grid_center
        waypoints, self.drone_start, self.drone_goal = self.core_plan_path(self.motion_p, global_home, grid_start_l, self.grid_goal, self.target_altitude)
        self.set_waypoints(waypoints)

        elapsed_t = timer() - self.timeout_start
        if elapsed_t >= self.timeout:
            self.timeout_b = True

        if self.timeout_b or len(waypoints)==0:
            self.disarming_transition()
            self.manual_transition()
        else:
            self.flight_state = States.PLANNING

    def plan_path_offline(self):
        print("Searching for a path ...")

        self.timeout_start = timer()        
        if self.wps is not None:
            self.set_waypoints(self.wps)
            self.flight_state = States.PLANNING
            return

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(self.home_pos[0], self.home_pos[1], 0)

        # TODO: retrieve current global position
        global_position = self.global_position
 
        # TODO: convert to current local position using global_to_local()
        self.home_g = self.global_home
        self.grid_start_l = global_to_local(global_position, self.home_g)

        print('global home {0}, position {1}, local position {2}'.format(self.home_g, global_position, self.grid_start_l))
        self.timeout_b = True
        self.disarming_transition()
        self.manual_transition()

    def process_timeout(self):
        ret = False
        # Set self.waypoints
        if self.wps_lr is not None:
            self.waypoints_lr = self.wps_lr.copy()
            self.send_waypoints(self.waypoints_lr)
            ret = True
        if self.wps is not None:
            self.waypoints = self.wps.copy()
            self.send_waypoints(self.waypoints)
            ret = True

        return ret

    def set_waypoints(self, waypoints):
        # Set self.waypoints
        if len(waypoints) and self.motion_p.need_local_replan():
            self.send_waypoints(waypoints)
            self.wps_lr = waypoints.copy()
            self.waypoints_lr = waypoints
            wp = self.waypoints_lr.pop(0)
            waypoints = self.local_path_plan(self.motion_p, wp, self.waypoints_lr)

        if len(waypoints):
            self.send_waypoints(waypoints)
            self.waypoints = waypoints
            self.wps = waypoints.copy()

    def replan_path(self):
        # Replanning from target_position is smoother, however local_position is more accurate
        #wp = self.target_position
        wp = self.local_position
        wp = (wp[0], wp[1], -wp[2])
        waypoints = self.local_path_plan(self.motion_p, wp, self.waypoints_lr)
        if len(waypoints):
            self.send_waypoints(waypoints)

        waypoints.pop(0)
        while (len(waypoints)):
            wp = waypoints.pop(0)
            self.waypoints.append(wp)
            self.wps.append(wp)

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()

def local_path_plan(motion_p, wp, waypoints_lr):
    wp2 = waypoints_lr[0]
    path_o, wp2 = motion_p.local_path_plan(wp, wp2)
    path = prune_path(path_o)
    print('Path Len: {}, Pruned Path Len: {}'.format(len(path_o), len(path)))
    gc = motion_p.grid_center
    waypoints = [[int(p[0]-gc[0]), int(p[1]-gc[1]), int(p[2]), 0] for p in path]
    
    if wp2 is None:
        waypoints_lr.pop(0)
        
    return waypoints

def core_plan_path(motion_p, global_home, grid_start_l, grid_goal, target_altitude):
    # TODO: convert start position to current position rather than map center
    grid_start = motion_p.get_grid_coord(grid_start_l, target_altitude)

    # Set goal as some arbitrary position on the grid
    # grid_goal = motion_p.get_grid_coord([10, 10, 0], target_altitude)

    # TODO: adapt to set goal as latitude / longitude position and convert
    grid_goal_l = global_to_local(grid_goal, global_home)
    grid_goal = motion_p.get_grid_coord(grid_goal_l, target_altitude)

    # Run A* to find a path from start to goal
    # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
    # or move to a different search space such as a graph (not done here)
    # TODO (if you're feeling ambitious): Try a different approach altogether!
    print('Local Start and Goal: ', grid_start, grid_goal)
    start_t = timer()
    path_o = motion_p.a_star(grid_start, grid_goal)
    print("Path Search Time: {}s".format(timer() - start_t))
    if path_o is not None:
        
        # TODO: prune path to minimize number of waypoints
        path = prune_path(path_o)
        print('Path Len: {}, Pruned Path Len: {}'.format(len(path_o), len(path)))
        
        # Convert path to waypoints
        gc = motion_p.grid_center
        waypoints = [[int(p[0]-gc[0]), int(p[1]-gc[1]), int(p[2]), 0] for p in path]
        print(waypoints)
    else:
        print("Path not found")
        waypoints = []

    drone_start = np.round(grid_start_l).tolist()
    drone_goal = np.round(grid_goal_l).tolist()

    return waypoints, drone_start, drone_goal

def debug_algo(motion_p, cfg, home_g):
    grid_start = cfg['common_p']['start']
    grid_start_l = [grid_start[0]-motion_p.grid_center[0], grid_start[1]-motion_p.grid_center[1], grid_start[2]]
    
    waypoints, drone_start, drone_goal = core_plan_path(motion_p, home_g, grid_start_l, cfg['common_p']['goal'], cfg['common_p']['target_altitude'])
    if len(waypoints) == 0:
        waypoints = None
    
    waypoints_lr = None
    if len(waypoints) and motion_p.need_local_replan():
        wps_lr = waypoints.copy()
        waypoints_lr = waypoints
        waypoints = []
        wp = wps_lr.pop(0)
        while len(wps_lr):
            wps = local_path_plan(motion_p, wp, wps_lr)

            while (len(wps)):
                wp = wps.pop(0)
                waypoints.append(wp)

    return waypoints, waypoints_lr, drone_start, drone_goal

def get_drone(motion_p, home_g, cfg, wps, wps_lr, inline_mp=True):
    conn = MavlinkConnection('tcp:{0}:{1}'.format( cfg['common_p']['host'],  cfg['common_p']['port']), timeout=60)
    drone = MotionPlanning(conn, motion_p, cfg['common_p']['goal'], home_g, cfg['common_p']['target_altitude'], core_plan_path, local_path_plan, wps=wps, wps_lr=wps_lr, inline_mp=inline_mp)

    return drone

def drone_run_offline_mp(motion_p, home_g, cfg):
    print("Offline Mode planning")
    drone = get_drone(motion_p, home_g, cfg, None, None, inline_mp=False)
    time.sleep(1)
    start_t = timer()
    drone.start()

    wps, drone_start, drone_goal = core_plan_path(motion_p, drone.home_g, drone.grid_start_l, drone.grid_goal, drone.target_altitude)    
    if len(wps):
        drone = get_drone(motion_p, home_g, cfg, wps, None, inline_mp=False)
        time.sleep(1)
        start_t = timer()
        drone.start()
    
    print("Drone Start To Close Elapsed Time: {}s".format(timer() - start_t))

    return drone.wps, drone.wps_lr, drone_start, drone_goal

def drone_run_inline_mp(motion_p, home_g, cfg):
    print("Inline Mode planning")
    wps = None
    wps_lr = None
    drone_start = None
    drone_goal = None
    timeout_b = True
    while timeout_b:
        drone = get_drone(motion_p, home_g, cfg, wps, wps_lr)
        time.sleep(1)
        start_t = timer()
        drone.start()
        timeout_b = drone.timeout_b
        if timeout_b:
            print("Drone timed out, Start To Close Elapsed Time: {}s".format(timer() - start_t))
            drone_start = drone.drone_start
            drone_goal = drone.drone_goal
        else:
            print("Drone Start To Close Elapsed Time: {}s".format(timer() - start_t))
        
        wps = drone.wps
        wps_lr = drone.wps_lr

    if drone_start is None:
        drone_start = drone.drone_start
        drone_goal = drone.drone_goal

    return wps, wps_lr, drone_start, drone_goal

def drone_run(motion_p, home_g, cfg):
    if cfg['common_p']['debug_algo']:
        wps, wps_lr, drone_start, drone_goal = debug_algo(motion_p, cfg, home_g)
    elif cfg['common_p']['offline_mp']:
        wps, wps_lr, drone_start, drone_goal = drone_run_offline_mp(motion_p, home_g, cfg)
    else:
        wps, wps_lr, drone_start, drone_goal = drone_run_inline_mp(motion_p, home_g, cfg)

    if cfg['debug_p']['plt']:
        motion_p.plot_grid([0,0,0], drone_start=drone_start, drone_goal=drone_goal, path=wps, path_lr=wps_lr)

    if cfg['debug_p']['plt_voxm']:
        motion_p.plot_voxmap([0,0,0], drone_start=drone_start, drone_goal=drone_goal, path=wps, path_lr=wps_lr)
