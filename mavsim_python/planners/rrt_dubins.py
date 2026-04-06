"""THE COURSE IS DEFINITELY WRONG< IT SEEMS TO ALWAYS BE ZERO"""
# rrt dubins path planner for mavsim_python
import numpy as np
from message_types.msg_waypoints import MsgWaypoints
import matplotlib.pyplot as plt
from planners.dubins_parameters import DubinsParameters


# np.random.seed(1000) # for debugging

class RRTDubins:
    def __init__(self):
        self.segment_length = 450  # standard length of path segments
        self.dubins_path = DubinsParameters()

    def update(self, start_pose, end_pose, Va, world_map, radius):
        self.segment_length = 4 * radius
        tree = MsgWaypoints()
        tree.type = 'dubins'
        waypoints_not_smooth = MsgWaypoints()
        waypoints = MsgWaypoints()        

        ##### TODO #####
        # add the start pose to the tree
        tree.add(start_pose[0:3], Va, float(start_pose[3]))
        
        # check to see if start_pose connects directly to end_pose
        if not self.collision(start_pose, end_pose, world_map, radius):
            if self.close_to_end(start_pose[0:3],end_pose[0:3]): # as far as I know, arbitrary threshold for when to stop
                dist = distance(start_pose, end_pose)
                tree.add(end_pose[0:3], Va, float(end_pose[3]), cost=dist, parent = 0, connect_to_goal= True)
        else:
            num_paths = 0
            while num_paths < 1: # find 5 different paths from start pose to end pose
                flag = self.extendTree(tree,end_pose,Va,world_map, radius)
                num_paths += flag
        
                # return # TODO I don't think I want a return here, but maybe I do?
        # if NOT feasible, I guess just ditch it?
        # update everything
        # check to see if you can connect to the end along path
        # if yes add the end to your tree and update everything
            
        # find path with minimum cost to end_node
        waypoints_not_smooth = findMinimumPath(tree, end_pose)
        waypoints = self.smoothPath(waypoints_not_smooth, world_map, radius)
        self.waypoints_not_smoothed = waypoints_not_smooth
        self.tree = tree
        return waypoints
        
        # # check to see if start_pose connects directly to end_pose
       
        # # find path with minimum cost to end_node
        # # waypoints_not_smooth = findMinimumPath()
        # # waypoints = self.smoothPath()
        # self.waypoint_not_smooth = waypoints_not_smooth
        # self.tree = tree
        # return waypoints

    def extendTree(self, tree, end_pose, Va, world_map, radius):
        # extend tree by randomly selecting pose and extending tree toward that pose
        
        ##### TODO #####
        # flag = None

        # extend tree by randomly selecting pose and extending tree toward that pose
        # generate a new point
        flag = False
        new_pose = randomPose(world_map, pd = end_pose.item(2)) # TODO maybe change what pd is, IDk
        # look for closest existing node by position (tree.ned)
        dists = np.linalg.norm(tree.ned - new_pose[0:3], axis=0)
        parent_node_index = int(np.argmin(dists))
        min_dists = float(dists[parent_node_index])

        # in dubins case we always want the point to be segment lengthaway bc dubins can't work if its within 2R
        initCost = min(min_dists, self.segment_length)
        new_node = new_pose
        if initCost == min_dists:
            if min_dists < 3*radius:
                min_dists = 3*radius
                parent = tree.ned[:, parent_node_index].reshape(3,1)
                direction = new_pose[0:3] - parent
                direction = direction / np.linalg.norm(direction)  # unit vector
                angle = np.arccos(direction[0])
                new_node[0:3] = parent + radius*3 * direction
                new_node[3] = angle
            else:
                parent = tree.ned[:, parent_node_index].reshape(3,1)
                direction = new_pose[0:3] - parent
                direction = direction / np.linalg.norm(direction)  # unit vector
                angle = np.arccos(direction[0])
                new_node[3] = angle
        else: # if the new pose wasn't close enough, make your new node the point distance linesegment along line betweeen new pose and start pose
            parent = tree.ned[:, parent_node_index].reshape(3,1)
            direction = new_pose[0:3] - parent
            direction = direction / np.linalg.norm(direction)  # unit vector
            angle = np.arccos(direction[0])
            new_node[0:3] = parent + self.segment_length * direction
            new_node[3] = angle

        
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
        totCost=tree.cost[parent_node_index]+radius*2

        # generate path there
        # check to make sure path is feasible
        connector_node = np.vstack((tree.ned[:, parent_node_index].reshape(3,1),
                                    [[tree.course[parent_node_index]]]))
        
        if not self.collision(connector_node, new_node, world_map, radius):
            tree.add(new_node[0:3], Va, course = new_node[3] ,cost=totCost,parent=parent_node_index)

            new_node_index = tree.num_waypoints - 1
            if self.close_to_end(new_node, end_pose):
                if not self.collision(new_node, end_pose, world_map, radius):
                    dist = distance(new_node, end_pose)
                    goal_cost = totCost + dist
                    tree.add(end_pose[0:3], Va,course = end_pose[3] , cost=goal_cost, parent=new_node_index, connect_to_goal=True)
                    flag = True
        ###### TODO ######
        
        # return flag
        return flag

    def collision(self, start_pose, end_pose, world_map, radius):
        # debug stuff: 
        ell = np.linalg.norm(start_pose[0:2] - end_pose[0:2])
        if ell < 3 * radius:
            # print('About to get an error in dubins parameters')
            collision_flag = True
            # print("too short! REJECTED: ")
            return collision_flag




        collision_flag = False
        self.dubins_path.update(start_pose[0:3], float(start_pose[3]),end_pose[0:3],float(end_pose[3]),radius)
        points = self.dubins_path.compute_points()
        # plt.clf()
        # plt.xlim(0, world_map.city_width)
        # plt.ylim(0, world_map.city_width)

        # # plot all existing nodes (blue)
        # plt.scatter(points[:,0], points[:,1], s=10)
        # # plt.axis('equal')
        # plt.pause(0.01)

        # check to see of path from start_pose to end_pose colliding with world_map
        for i in range(points.shape[0]):
            if heightAboveGround(world_map, points[i,:]) <= 0:
                collision_flag = True
                # print("collided at point: ", i)
                return collision_flag # return as soon as you collide
                
        return collision_flag

    def process_app(self):
        self.planner_viewer.process_app()

    def smoothPath(self, waypoints, world_map, radius):
        smooth = [0]  # add the first waypoint
        i = 0
        j = 1
        while j < waypoints.num_waypoints-1:
            ws = np.vstack((column(waypoints.ned, smooth[i]),
                            [[waypoints.course[smooth[i]]]]))
            w_plus = np.vstack((column(waypoints.ned, j+1), [[waypoints.course[j+1]]]))
            if self.collision(ws,w_plus, world_map, radius):
                smooth.append(j)
                i += 1
            j += 1
        smooth.append(waypoints.num_waypoints - 1)
        # construct smooth waypoint path
        
        # smooth_waypoints = MsgWaypoints()

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
    
    def close_to_end(self, start_pose, end_pose):
        if distance(start_pose, end_pose) < self.segment_length:
            return True
        else:
            return False


def findMinimumPath(tree, end_pose):
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
                      tree.course.item(i),
                      np.inf,
                      np.inf,
                      np.inf)
    waypoints.add(end_pose[0:3],
                  tree.airspeed[-1],
                  end_pose.item(3),
                  np.inf,
                  np.inf,
                  np.inf)
    waypoints.type = tree.type
    return waypoints


def distance(start_pose, end_pose):
    # compute distance between start and end pose
    d = np.linalg.norm(start_pose[0:3] - end_pose[0:3])
    return d


def heightAboveGround(world_map, point):
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


def randomPose(world_map, pd):
    # generate a random pose
    pn =  world_map.city_width * np.random.rand() # 333
    pe =  world_map.city_width * np.random.rand() # 0
    chi = 0
    pose = np.array([[pn], [pe], [pd], [chi]])
    return pose


def mod(x):
    # force x to be between 0 and 2*pi
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x


def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col
