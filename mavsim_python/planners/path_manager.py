"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - RWB
        3/30/2022 - RWB
        7/13/2023 - RWB
        3/25/2024 - RWB
"""

import numpy as np
from planners.dubins_parameters import DubinsParameters
from message_types.msg_state import MsgState
from message_types.msg_path import MsgPath
from message_types.msg_waypoints import MsgWaypoints


class PathManager:
    '''
        Path manager

        Attributes
        ----------
        path : MsgPath
            path message sent to path follower
        num_waypoints : int
            number of waypoints
        ptr_previous : int
            pointer to previous waypoint
            MAV is traveling from previous to current waypoint
        ptr_current : int
            pointer to current waypoint
        ptr_next : int
            pointer to next waypoint
        halfspace_n : np.nparray (3x1)
            the normal vector that defines the current halfspace plane
        halfspace_r : np.nparray (3x1)
            the inertial vector that defines a point on the current halfspace plane
        manager_state : int
            state of the manager state machine
        manager_requests_waypoints : bool
            a flag for handshaking with the path planner
            True when new waypoints are needed, i.e., at the end of waypoint list.
        dubins_path : DubinsParameters
            A class that defines a dubins path      

        Methods
        -------
        update(waypoints, radius, state)

        _initialize_pointers() :
            initialize the points to 0(previous), 1(current), 2(next)  
        _increment_pointers() :  
            add one to every pointer - currently does it modulo num_waypoints          
        _inHalfSpace(pos):
            checks to see if the position pos is in the halfspace define

        _line_manager(waypoints, state):
            Assumes straight-line paths.  Transition is from one line to the next
            _construct_line(waypoints): 
                used by line manager to construct the next line path

        _fillet_manager(waypoints, radius, state):
            Assumes straight-line waypoints.  Constructs a fillet turn between lines.
            _construct_fillet_line(waypoints, radius):
                used by _fillet_manager to construct the next line path
            _construct_fillet_circle(waypoints, radius):
                used by _fillet_manager to construct the fillet orbit
            
        _dubins_manager(waypoints, radius, state):
            Assumes dubins waypoints.  Constructs Dubin's path between waypoints
            _construct_dubins_circle_start(waypoints, dubins_path):
                used by _dubins_manager to construct the start orbit
            _construct_dubins_line(waypoints, dubins_path):
                used by _dubins_manager to construct the middle line
            _construct_dubins_circle_end(waypoints, dubins_path):
                used by _dubins_manager to construct the end orbit
    '''
    def __init__(self):
        self._path = MsgPath()
        self._num_waypoints = 0
        self._ptr_previous = 0
        self._ptr_current = 1
        self._ptr_next = 2
        self._halfspace_n = np.inf * np.ones((3,1))
        self._halfspace_r = np.inf * np.ones((3,1))
        self._manager_state = 1
        self.manager_requests_waypoints = True
        self.dubins_path = DubinsParameters()


    def update(self, 
               waypoints: MsgWaypoints, 
               radius: float, 
               state: MsgState) -> MsgPath:
        if waypoints.num_waypoints == 0:
            self.manager_requests_waypoints = True
        if self.manager_requests_waypoints is True \
                and waypoints.flag_waypoints_changed is True:
            self.manager_requests_waypoints = False
            self._num_waypoints = waypoints.ned.shape[1]
        if waypoints.type == 'straight_line':
            self._line_manager(waypoints, state)
        elif waypoints.type == 'fillet':
            self._fillet_manager(waypoints, radius, state)
        elif waypoints.type == 'dubins':
            self._dubins_manager(waypoints, radius, state)
        else:
            print('Error in Path Manager: Undefined waypoint type.')
        return self._path

    def _line_manager(self,  
                      waypoints: MsgWaypoints, 
                      state: MsgState):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer

        ##### TODO ######
        if waypoints.flag_waypoints_changed:
            waypoints.flag_waypoints_changed = False
            self._initialize_pointers()
            self.manager_requests_waypoints = False
            self._construct_line(waypoints)
        if self._inHalfSpace(mav_pos) == True:
            self._increment_pointers()
            self._construct_line(waypoints)
        # Use functions - self._initialize_pointers(), self._construct_line()
        # self._inHalfSpace(mav_pos), self._increment_pointers(), self._construct_line()

        # Use variables - self._ptr_current, self.manager_requests_waypoints,
        # waypoints.__, radius
        

    def _fillet_manager(self,  
                        waypoints: MsgWaypoints, 
                        radius: float, 
                        state: MsgState):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer

        ##### TODO ######'
        if waypoints.flag_waypoints_changed:
            waypoints.flag_waypoints_changed = False
            self._initialize_pointers()
            self.manager_requests_waypoints = False
            self._construct_fillet_line(waypoints, radius)
            self._manager_state = 1
            # print(self._manager_state)
        if self._manager_state == 1:
            if self._inHalfSpace(mav_pos):
                self._manager_state = 2
                # print(self._manager_state)
                self._construct_fillet_circle(waypoints, radius)
        elif self._manager_state == 2:
            if self._inHalfSpace(mav_pos):
                self._increment_pointers()
                self._construct_fillet_line(waypoints, radius)
                self._manager_state = 1
                # print(self._manager_state)
        


        # Use functions - self._initialize_pointers(), self._construct_fillet_line(),
        # self._inHalfSpace(), self._construct_fillet_circle(), self._increment_pointers()

        # Use variables self._num_waypoints, self._manager_state, self._ptr_current
        # self.manager_requests_waypoints, waypoints.__, radius
      

    def _dubins_manager(self,  
                        waypoints: MsgWaypoints, 
                        radius: float, 
                        state: MsgState):
        mav_pos = np.array([[state.north, state.east, -state.altitude]]).T
        # if the waypoints have changed, update the waypoint pointer
        # waypoints.
        ##### TODO #####
        if waypoints.flag_waypoints_changed:
            waypoints.flag_waypoints_changed = False
            self._initialize_pointers()
            self.manager_requests_waypoints = False
            self.dubins_path.update(waypoints.ned[:,self._ptr_previous:self._ptr_previous+1], waypoints.course[self._ptr_previous], waypoints.ned[:,self._ptr_current:self._ptr_current+1],waypoints.course[self._ptr_current],radius)
            self._manager_state = 1
            print("state:, ", self._manager_state)
            self._construct_dubins_circle_start(waypoints, self.dubins_path) 
        if self._manager_state == 1: # START CIRCLE
            if not self._inHalfSpace(mav_pos):### MAKE SURE -q1 IS WHATS BEING LOOKED AT BY inHalfPlane
                self._manager_state = 2
                print("state:, ", self._manager_state)
        elif self._manager_state == 2: # CHECK TO ENTER STRAIGHT LINE
            if self._inHalfSpace(mav_pos):### MAKE SURE +q1 IS WHATS BEING LOOKED AT BY inHalfPlane
                self._manager_state = 3
                print("state:, ", self._manager_state)
                self._construct_dubins_line(waypoints,self.dubins_path)
        elif self._manager_state == 3: # STRAIGHT LINE
            if self._inHalfSpace(mav_pos):### MAKE SURE +q1 IS WHATS BEING LOOKED AT BY inHalfPlane 
                self._manager_state = 4
                print("state:, ", self._manager_state)
                self._construct_dubins_circle_end(waypoints,self.dubins_path)
        elif self._manager_state == 4: # END CIRCLE
            if not self._inHalfSpace(mav_pos):### MAKE SURE -q3 IS WHATS BEING LOOKED AT BY inHalfPlane
                self._manager_state = 5
                print("state:, ", self._manager_state)
        elif self._manager_state == 5:
            if self._inHalfSpace(mav_pos) :### MAKE SURE +q3 IS WHATS BEING LOOKED AT BY inHalfPlane
                self._manager_state = 1
                print("state:, ", self._manager_state)
                self._increment_pointers()
                self.dubins_path.update(waypoints.ned[:,self._ptr_previous], waypoints.course[self._ptr_previous], waypoints.ned[:,self._ptr_current],waypoints.course[self._ptr_current],radius)
                self._construct_dubins_circle_start(waypoints, self.dubins_path) 


            
            # print(self._manager_state)

        # Use functions - self._initialize_pointers(), self._dubins_path.update(),
        # self._construct_dubins_circle_start(), self._construct_dubins_line(),
        # self._inHalfSpace(), self._construct_dubins_circle_end(), self._increment_pointers(),

        # Use variables - self._num_waypoints, self._dubins_path, self._ptr_current,
        # self._ptr_previous, self._manager_state, self.manager_requests_waypoints,
        # waypoints.__, radius


    def _initialize_pointers(self):
        if self._num_waypoints >= 3:
            ##### TODO #####
            self._ptr_previous = 0
            self._ptr_current = 1
            self._ptr_next = 2
        else:
            print('Error Path Manager: need at least three waypoints')

    def _increment_pointers(self):
        ##### TODO #####
        # modulo wraps me back around to the first waypoint once I reach the end
        self._ptr_previous = (self._ptr_previous + 1) % self._num_waypoints
        self._ptr_current = (self._ptr_current + 1) % self._num_waypoints
        self._ptr_next = (self._ptr_next + 1) % self._num_waypoints

    def _construct_line(self, 
                        waypoints: MsgWaypoints):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous+1]
        ##### TODO #####
        current = waypoints.ned[:, self._ptr_current:self._ptr_current+1]
        next = waypoints.ned[:, self._ptr_next:self._ptr_next+1]

        q_prev = (current-previous)/np.linalg.norm(current-previous)
        q_current = (next-current)/np.linalg.norm(next-current)
        # update halfspace variables
        self._halfspace_n = (q_prev+q_current)/np.linalg.norm(q_prev+q_current)
        self._halfspace_r = current # TODO see if this wrong
        
        # Update path variables 
        # self._path.__ = ##### TODO #####
        self._path.plot_updated = False
        self._path.type = "line"
        self._path.line_origin = previous
        self._path.line_direction = q_prev
        

    def _construct_fillet_line(self, 
                               waypoints: MsgWaypoints, 
                               radius: float):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous+1]
        ##### TODO #####
        current = waypoints.ned[:, self._ptr_current:self._ptr_current+1]
        next = waypoints.ned[:, self._ptr_next:self._ptr_next+1]

        q_prev = (current-previous)/np.linalg.norm(current-previous)
        q_current = (next-current)/np.linalg.norm(next-current)

        var_rho = np.acos(-q_prev.T @ q_current)

        # update halfspace variables
        self._halfspace_n = q_prev
        self._halfspace_r = current - (radius/np.tan(var_rho/2))*q_prev
        
        # Update path variables
        # self._path.__ =
        self._path.plot_updated = False
        self._path.type = "line" # or orbit
        self._path.line_origin = previous
        self._path.line_direction = q_prev

    def _construct_fillet_circle(self, 
                                 waypoints: MsgWaypoints, 
                                 radius: float):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous+1]
        ##### TODO #####
        current = waypoints.ned[:, self._ptr_current:self._ptr_current+1]
        next = waypoints.ned[:, self._ptr_next:self._ptr_next+1]

        q_prev = (current-previous)/np.linalg.norm(current-previous)
        q_current = (next-current)/np.linalg.norm(next-current)

        var_rho = np.acos(-q_prev.T @ q_current)
        self._path.orbit_center = current - (radius/np.sin(var_rho/2))*(q_prev-q_current)/np.linalg.norm(q_prev-q_current)
        self._path.orbit_radius = radius
        lmbda = np.sign(q_prev[0]*q_current[1]-q_prev[1]*q_current[0])
        if lmbda == -1:
            self._path.orbit_direction = "CCW"
        else:
            self._path.orbit_direction = "CW"
        # update halfspace variables
        self._halfspace_n = q_current
        self._halfspace_r = current + (radius/np.atan(var_rho/2))*q_current
        
        # Update path variables
        # self._path.__ =
        self._path.plot_updated = False
        self._path.type = "orbit" # or orbit
        self._path.line_origin = previous
        self._path.line_direction = q_prev

    def _construct_dubins_circle_start(self, 
                                       waypoints: MsgWaypoints, 
                                       dubins_path: DubinsParameters):
        ##### TODO #####
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous+1]
        ##### TODO #####
        


        self._path.orbit_center = dubins_path.center_s
        self._path.orbit_radius = dubins_path.radius
        lmbda = dubins_path.dir_s
        if lmbda == -1:
            self._path.orbit_direction = "CCW"
        else:
            self._path.orbit_direction = "CW"
        # update halfspace variables
        self._halfspace_n = dubins_path.n1
        self._halfspace_r = dubins_path.r1
        self._path.plot_updated = False
        self._path.type = "orbit" # or orbit
        self._path.line_origin = previous
        self._path.line_direction = dubins_path.n1
        # Update path variables
        # self._path.__ =
        

    def _construct_dubins_line(self, 
                               waypoints: MsgWaypoints, 
                               dubins_path: DubinsParameters):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous+1]
        
        ##### TODO #####
        # update halfspace variables
        self._halfspace_n =dubins_path.n1
        self._halfspace_r = dubins_path.r2
        
        # Update path variables
        # self._path.__ =
        self._path.plot_updated = False
        self._path.type = "line" # or orbit
        self._path.line_origin = dubins_path.r1.reshape(3,1)
        self._path.line_direction = dubins_path.n1
        

    def _construct_dubins_circle_end(self, 
                                     waypoints: MsgWaypoints, 
                                     dubins_path: DubinsParameters):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous+1]
        ##### TODO #####
        self._path.orbit_center = dubins_path.center_e
        self._path.orbit_radius = dubins_path.radius
        lmbda = dubins_path.dir_e
        if lmbda == -1:
            self._path.orbit_direction = "CCW"
        else:
            self._path.orbit_direction = "CW"
        # update halfspace variables
        self._halfspace_n = dubins_path.n3
        self._halfspace_r = dubins_path.r3
        self._path.line_origin = dubins_path.p_e.reshape(3,1)
        self._path.line_direction = dubins_path.n3
        # update halfspace variables
        # self._halfspace_n =
        # self._halfspace_r = 
        
        # Update path variables
        # self._path.__ =
        
        self._path.plot_updated = False
        self._path.type = "orbit" # or orbit
        

    def _inHalfSpace(self, 
                     pos: np.ndarray)->bool:
        # '''Is pos in the half space defined by r and n?'''):
        if (pos-self._halfspace_r).T @ self._halfspace_n >= 0:
            # print("crossed half plane")
            return True
        else:
            return False

