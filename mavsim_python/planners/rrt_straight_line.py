# rrt straight line path planner for mavsim_python
import numpy as np
import matplotlib.pyplot as plt
from message_types.msg_waypoints import MsgWaypoints

class RRTStraightLine:
    def __init__(self):
        self.segment_length = 300 # standard length of path segments

    def update(self, start_pose, end_pose, Va, world_map, radius):
        tree = MsgWaypoints()
        waypoints = MsgWaypoints()
        waypoints_not_smoothed = MsgWaypoints()
        #tree.type = 'straight_line'
        tree.type = 'fillet'

        ###### TODO ######
        # add the start pose to the tree

        tree.add(start_pose, Va)
        
        # check to see if start_pose connects directly to end_pose
        if not collision(start_pose, end_pose, world_map):
            if self.close_to_end(start_pose,end_pose): # as far as I know, arbitrary threshold for when to stop
                dist = distance(start_pose, end_pose)
                tree.add(end_pose, Va, cost=dist, parent = 0, connect_to_goal= True)
                find_minimum_path(tree, end_pose) # TODO maybe don't automatically do this? IDK
                # return # I guess be done, maybe this shouldn't just be return though
        else:
            num_paths = 0
            while num_paths < 5: # find 5 different paths from start pose to end pose
                flag = self.extend_tree(tree,end_pose,Va,world_map)
                num_paths += flag
        
                # return # TODO I don't think I want a return here, but maybe I do?
        # if NOT feasible, I guess just ditch it?
        # update everything
        # check to see if you can connect to the end along path
        # if yes add the end to your tree and update everything
            
        # find path with minimum cost to end_node
        waypoints_not_smoothed = find_minimum_path(tree, end_pose)
        waypoints = smooth_path(waypoints_not_smoothed, world_map)
        self.waypoints_not_smoothed = waypoints_not_smoothed
        self.tree = tree
        return waypoints

    def extend_tree(self, tree, end_pose, Va, world_map):
        # extend tree by randomly selecting pose and extending tree toward that pose
        # generate a new point
        flag = False
        new_pose = random_pose(world_map, pd = end_pose.item(2)) # TODO maybe change what pd is, IDk
        # look for closest existing node by position (tree.ned)
        dists = np.linalg.norm(tree.ned - new_pose, axis=0)
        parent_node_index = int(np.argmin(dists))
        min_dists = float(dists[parent_node_index])

        initCost = min(min_dists, self.segment_length)
        if initCost == min_dists:
            new_node = new_pose
        else: # if the new pose wasn't close enough, make your new node the point distance linesegment along line betweeen new pose and start pose
            
            parent = tree.ned[:, parent_node_index].reshape(3,1)

            direction = new_pose - parent
            direction = direction / np.linalg.norm(direction)  # unit vector

            new_node = parent + initCost * direction
        # ---- DEBUG PLOT ----
        # plt.clf()
        
        # plt.xlim(0, world_map.city_width)
        # plt.ylim(0, world_map.city_width)

        # # plot all existing nodes (blue)
        # plt.scatter(tree.ned[0, :], tree.ned[1, :], s=10)

        # # plot new node (red)
        # plt.scatter(new_node[0], new_node[1], marker='x')
        # plt.scatter(end_pose[0],end_pose[1], marker = '*')
        # # optional: plot the sampled random point
        # plt.scatter(new_pose[0], new_pose[1], marker='o')
        # # set axis limits

        # # keep proportions correct
        # plt.axis('equal')
        # # optional: draw line from parent to new node
        # # parent = tree.ned[:, parent_node_index]
        # # plt.plot([parent[0].item(), new_node[0].item()],
        # #  [parent[1].item(), new_node[1].item()])

        # plt.pause(0.01)
        # get the point as either the generated point or the point along the line 
        # the cost is a running total, so you need to get the cost of the node it connects to and add the distance to the new nod
        totCost=tree.cost[parent_node_index]+initCost

        # generate path there
        # check to make sure path is feasible
        if not collision(tree.ned[:,parent_node_index].reshape(3,1), new_node, world_map):
            tree.add(new_node, Va, cost=totCost,parent=parent_node_index)

            new_node_index = tree.num_waypoints - 1
            if self.close_to_end(new_node, end_pose):
                dist = distance(new_node, end_pose)
                goal_cost = totCost + dist
                tree.add(end_pose, Va, cost=goal_cost, parent=new_node_index, connect_to_goal=True)
                flag = True
        ###### TODO ######
        
        return flag
        
    def process_app(self):
        self.planner_viewer.process_app()
    
    def close_to_end(self, start_pose, end_pose):
        if distance(start_pose, end_pose) < self.segment_length:
            return True
        else:
            return False

def smooth_path(waypoints, world_map):

    ##### TODO #####
    # smooth the waypoint path
    smooth = [0]  # add the first waypoint
    i = 0
    j = 1
    while j < waypoints.num_waypoints-1:
        ws = column(waypoints.ned, smooth[i])
        w_plus = column(waypoints.ned, j+1)
        if collision(ws,w_plus, world_map):
            smooth.append(j)
            i += 1
        j += 1
    smooth.append(waypoints.num_waypoints - 1)
    # construct smooth waypoint path
    
    smooth_waypoints = MsgWaypoints()

    # smooth_waypoints
    smooth_waypoints = MsgWaypoints()
    for idx in smooth:
        smooth_waypoints.add(column(waypoints.ned, idx),
                            waypoints.airspeed[idx],
                            waypoints.course[idx],
                            waypoints.cost[idx],
                            waypoints.parent[idx],
                            waypoints.connect_to_goal[idx])
    smooth_waypoints.type = waypoints.type

    return smooth_waypoints


def find_minimum_path(tree, end_pose):
    # find the lowest cost path to the end node
    # find nodes that connect to end_node
    connecting_nodes = []
    for i in range(tree.num_waypoints):
        if tree.connect_to_goal.item(i) == 1:
            connecting_nodes.append(i)
    # find minimum cost last node
    idx = np.argmin(tree.cost[connecting_nodes])
    # construct lowest cost path order
    path = [connecting_nodes[idx]]  # last node that connects to end node
    parent_node = tree.parent.item(connecting_nodes[idx])
    while parent_node >= 1:
        path.insert(0, int(parent_node))
        parent_node = tree.parent.item(int(parent_node))
    path.insert(0, 0)
    # construct waypoint path
    waypoints = MsgWaypoints()
    for i in path:
        waypoints.add(column(tree.ned, i),
                      tree.airspeed.item(i),
                      np.inf,
                      np.inf,
                      np.inf,
                      np.inf)
    waypoints.add(end_pose,
                  tree.airspeed[-1],
                  np.inf,
                  np.inf,
                  np.inf,
                  np.inf)
    waypoints.type = tree.type
    return waypoints
    


def random_pose(world_map, pd):
    # generate a random pose
    pn = world_map.city_width * np.random.rand()
    pe = world_map.city_width * np.random.rand()
    pose = np.array([[pn], [pe], [pd]])
    return pose


def distance(start_pose, end_pose):
    # compute distance between start and end pose
    d = np.linalg.norm(start_pose - end_pose)
    return d


def collision(start_pose, end_pose, world_map):
    # check to see of path from start_pose to end_pose colliding with map
    collision_flag = False
    points = points_along_path(start_pose, end_pose, 100)
    for i in range(points.shape[1]):
        if height_above_ground(world_map, column(points, i)) <= 0:
            collision_flag = True
            # print("collided at point: ", i)
            return collision_flag # return as soon as you collide
    return collision_flag


def height_above_ground(world_map, point):
    # find the altitude of point above ground level
    point_height = -point.item(2)
    tmp = np.abs(point.item(0)-world_map.building_north)
    d_n = np.min(tmp)
    idx_n = np.argmin(tmp)
    tmp = np.abs(point.item(1)-world_map.building_east)
    d_e = np.min(tmp)
    idx_e = np.argmin(tmp)
    if (d_n<world_map.building_width) and (d_e<world_map.building_width):
        map_height = world_map.building_height[idx_n, idx_e]
    else:
        map_height = 0
    h_agl = point_height - map_height
    return h_agl

def points_along_path(start_pose, end_pose, N):
    # returns points along path separated by Del
    points = start_pose
    q = (end_pose - start_pose)
    L = np.linalg.norm(q)
    q = q / L
    w = start_pose
    for i in range(1, N):
        w = w + (L / N) * q
        points = np.append(points, w, axis=1)
    return points


def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    # print("altitude value: ", col[2])
    return col

