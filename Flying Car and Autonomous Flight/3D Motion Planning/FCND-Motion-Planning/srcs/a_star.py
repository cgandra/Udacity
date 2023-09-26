import numpy as np
from enum import Enum
import numpy.linalg as LA
from queue import PriorityQueue

def heuristic_norm(position, goal_position):
    return LA.norm(np.array(position) - np.array(goal_position))

def heuristic_manh(position, goal_position):
    d = np.abs(np.array(position) - np.array(goal_position))
    return np.sum(d) #Manhattan

class AStarGraph():
    def search(self, graph, start, goal):
        h = heuristic_norm
        path = []
        path_cost = 0
        queue = PriorityQueue()
        h_dist = h(start, goal)
        queue.put((h_dist, start))
        visited = set()
        visited.add(start)
    
        branch = {}
        branch[start] = (0, start, 0)
        found = False
        
        while not queue.empty():
            item = queue.get()
            current_node = item[1]
            current_cost = branch[current_node][0]
            
            if current_node == goal:        
                print('Found a path.')
                found = True
                break
            else:
                for next_node in graph[current_node]:
                    # get the tuple representation
                    cost = graph.edges[current_node, next_node]['weight']
                    branch_cost = current_cost + cost
                    queue_cost = branch_cost + h(next_node, goal)
                    
                    if next_node not in visited or branch_cost < branch[next_node][0]:
                        visited.add(next_node)               
                        branch[next_node] = (branch_cost, current_node)
                        queue.put((queue_cost, next_node))

        if found and len(branch) > 0:
            # retrace steps
            n = goal
            path_cost = branch[n][0]
            path.append(goal)
            while branch[n][1] != start:
                path.append(branch[n][1])
                n = branch[n][1]
            path.append(branch[n][1])
        else:
            print('**********************')
            print('Failed to find a path!')
            print('**********************') 

        return path[::-1]

class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    NORTH_WEST = (-1, -1, np.sqrt(2))
    NORTH_EAST = (-1, 1, np.sqrt(2))
    SOUTH_WEST = (1, -1, np.sqrt(2))
    SOUTH_EAST = (1, 1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])

class AStarGrid():
    def valid_actions(self, grid, current_node, step):
        """
        Returns a list of valid actions given a grid and current node.
        """
        valid_actions = list(Action)
        n, m = grid.shape[0] - 1, grid.shape[1] - 1
        x, y, _ = current_node

        # check if the node is off the grid or it's an obstacle
        remove_actions = []
        for action in valid_actions:
            da = action.delta
            x_n = x + da[0]*step
            y_n = y + da[1]*step

            if x_n < 0 or x_n > n or y_n < 0 or y_n > m or grid[x_n, y_n] == 1 :
                remove_actions.append(action)

        for i in range(len(remove_actions)):
            valid_actions.remove(remove_actions[i])

        return valid_actions


    def search(self, grid, start, goal, h_type, step=1):
        if h_type == 'norm':
            h = heuristic_norm
        elif h_type == 'manh':
            h = heuristic_manh
        path = []
        path_cost = 0
        queue = PriorityQueue()
        h_dist = h(start, goal)
        queue.put((h_dist, start))
        visited = set()
        visited.add(start)

        branch = {}
        branch[start] = (0, start, 0)
        found = False
        h_dist_max = h(goal, (goal[0]+step, goal[1]+step, goal[2]))
    
        while not queue.empty():
            item = queue.get()
            current_node = item[1]
            current_cost = branch[current_node][0]
               
            if current_node != start:
                h_dist = item[0] - current_cost

            if h_dist <= h_dist_max:
                step = 1

            if current_node == goal:
                print('Found a path.')
                found = True
                break
            else:
                for action in self.valid_actions(grid, current_node, step):
                    # get the tuple representation
                    da = action.delta
                    next_node = (current_node[0] + da[0]*step, current_node[1] + da[1]*step, current_node[2])
                    branch_cost = current_cost + action.cost*step
                    queue_cost = branch_cost + h(next_node, goal)

                    if next_node not in visited or branch_cost < branch[next_node][0]:
                        visited.add(next_node)               
                        branch[next_node] = (branch_cost, current_node, action)
                        queue.put((queue_cost, next_node))


        if found and len(branch) > 0:
            # retrace steps
            n = goal
            path_cost = branch[n][0]
            path.append(goal)
            while branch[n][1] != start:
                path.append(branch[n][1])
                n = branch[n][1]
            path.append(branch[n][1])
        else:
            print('**********************')
            print('Failed to find a path!')
            print('**********************') 
    
        return path[::-1]

class Action3D(Enum):
    """
    An action is represented by a 4 element tuple.

    The first 3 values are the delta of the action relative
    to the current grid position. The final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 0, 1)
    EAST = (0, 1, 0, 1)
    NORTH = (-1, 0, 0, 1)
    SOUTH = (1, 0, 0, 1)
    UP = (0, 0, 1, 1)
    DOWN = (0, 0, -1, 1)


    NORTH_WEST = (-1, -1, 0, np.sqrt(2))
    NORTH_EAST = (-1, 1, 0, np.sqrt(2))
    NORTH_UP = (-1, 0, 1, np.sqrt(2))
    NORTH_DOWN = (-1, 0, -1, np.sqrt(2))

    SOUTH_WEST = (1, -1, 0, np.sqrt(2))
    SOUTH_EAST = (1, 1, 0, np.sqrt(2))
    SOUTH_UP = (1, 0, 1, np.sqrt(2))
    SOUTH_DOWN = (1, 0, -1, np.sqrt(2))

    WEST_UP = (0, -1, 1, np.sqrt(2))
    WEST_DOWN = (0, -1, -1, np.sqrt(2))
    EAST_UP = (0, 1, 1, np.sqrt(2))
    EAST_DOWN = (0, 1, -1, np.sqrt(2))

    NORTH_WEST_UP = (-1, -1, 1, np.sqrt(3))
    NORTH_WEST_DOWN = (-1, -1, -1, np.sqrt(3))
    NORTH_EAST_UP = (-1, 1, 1, np.sqrt(3))
    NORTH_EAST_DOWN = (-1, 1, -1, np.sqrt(3))

    SOUTH_WEST_UP = (1, -1, 1, np.sqrt(3))
    SOUTH_WEST_DOWN = (1, -1, -1, np.sqrt(3))
    SOUTH_EAST_UP = (1, 1, 1, np.sqrt(3))
    SOUTH_EAST_DOWN = (1, 1, -1, np.sqrt(3))

    @property
    def cost(self):
        return self.value[3]

    @property
    def delta(self):
        return (self.value[0], self.value[1], self.value[2])

class AStar3D():
    def valid_actions(self, voxmap, current_node, step):
        """
        Returns a list of valid actions given a voxmap and current node.
        """
        valid_actions = list(Action3D)
        n, m, h = voxmap.shape[0] - 1, voxmap.shape[1] - 1, voxmap.shape[2] - 1
        x, y, z = current_node

        # check if the node is off the grid or it's an obstacle
        remove_actions = []
        for action in valid_actions:
            da = action.delta
            x_n = x + da[0]*step
            y_n = y + da[1]*step
            z_n = z + da[2]*step

            if x_n < 0 or x_n > n or y_n < 0 or y_n > m or z_n < 0 or z_n > h or voxmap[x_n, y_n, z_n] == 1 :
                remove_actions.append(action)

        for i in range(len(remove_actions)):
            valid_actions.remove(remove_actions[i])

        return valid_actions


    def search(self, grid, start, goal, h_type, step=1):
        if h_type == 'norm':
            h = heuristic_norm
        elif h_type == 'manh':
            h = heuristic_manh
        path = []
        path_cost = 0
        queue = PriorityQueue()
        h_dist = h(start, goal)
        queue.put((h_dist, start))
        visited = set()
        visited.add(start)

        branch = {}
        branch[start] = (0, start, 0)
        found = False
        h_dist_max = h(goal, (goal[0]+step, goal[1]+step, goal[2]))
    
        while not queue.empty():
            item = queue.get()
            current_node = item[1]
            current_cost = branch[current_node][0]

            if current_node != start:
                h_dist = item[0] - current_cost

            if h_dist <= h_dist_max:
                step = 1
            
            if current_node == goal:
                print('Found a path.')
                found = True
                break
            else:
                for action in self.valid_actions(grid, current_node, step):
                    # get the tuple representation
                    da = action.delta
                    next_node = (current_node[0] + da[0]*step, current_node[1] + da[1]*step, current_node[2] + da[2]*step)
                    branch_cost = current_cost + action.cost*step
                    queue_cost = branch_cost + h(next_node, goal)

                    if next_node not in visited or branch_cost < branch[next_node][0]:
                        visited.add(next_node)            
                        branch[next_node] = (branch_cost, current_node, action)
                        queue.put((queue_cost, next_node))

        if found and len(branch) > 0:
            # retrace steps
            n = goal
            path_cost = branch[n][0]
            path.append(goal)
            while branch[n][1] != start:
                path.append(branch[n][1])
                n = branch[n][1]
            path.append(branch[n][1])
        else:
            print('**********************')
            print('Failed to find a path!')
            print('**********************') 
    
        return path[::-1]