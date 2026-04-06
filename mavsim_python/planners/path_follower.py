import numpy as np
from math import sin, cos
from message_types.msg_state import MsgState
from message_types.msg_path import MsgPath
from message_types.msg_autopilot import MsgAutopilot
from tools.wrap import wrap
from tools.rotations import inertial_to_path


class PathFollower:
    def __init__(self):
        ##### TODO #####
        self.chi_inf = np.pi/8  # approach angle for large distance from straight-line path
        self.k_path = .1  # path gain for straight-line path following
        self.k_orbit = 1.0  # increased from 0.01 for better convergence
        self.gravity = 9.81
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, 
               path: MsgPath, 
               state: MsgState)->MsgAutopilot:
        if path.type == 'line':
            self._follow_straight_line(path, state)
        elif path.type == 'orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, 
                              path: MsgPath, 
                              state: MsgState):
        ##### TODO #####
        r = path.line_origin
        q = path.line_direction
        p = np.array([state.north, state.east,state.altitude]).reshape(3,1)
        e_p_i= p - r
        
        k_i = np.array([0,0,1]).reshape(3,1)# unit vector in the down direction
        n = np.cross(k_i.flatten(), q.flatten())/(np.linalg.norm(np.cross(k_i.flatten(), q.flatten())))
        n = n.reshape(3,1)
        s_i = e_p_i - np.dot(e_p_i.flatten(), n.flatten())*n
        s_n = s_i[0]
        s_e = s_i[1]
        s_d = s_i[2]
        r_n = r[0]
        r_e =r[1]
        r_d = r[2] # down component
        q_n = q[0]
        q_e = q[1]
        q_d = q[2]

        #airspeed command
        
        self.autopilot_commands.airspeed_command = path.airspeed
        
        # course command
        chi_q = np.atan2(q_e,q_n) # from eq 10.2
        chi_q = wrap(chi_q,state.chi)
        ep = inertial_to_path(float(chi_q))@ e_p_i
        e_py = ep[1] # np.sin(chi_q)*(p[0]-r_n) + np.cos(chi_q)*(p[1]-r_e) # inertial_to_path(p-r)
        
        self.autopilot_commands.course_command = float(chi_q - self.chi_inf*2/np.pi*np.atan(self.k_path*e_py))
        
        # altitude command eq 10.6
        self.autopilot_commands.altitude_command = float(-r_d-np.sqrt(s_n**2+s_e**2)*(q_d/(q_n**2+q_e**2)))

        # feedforward roll angle for straight line is zero
        self.autopilot_commands.phi_feedforward = 0

    def _follow_orbit(self, 
                      path: MsgPath, 
                      state: MsgState):
        ##### TODO #####
        c = path.orbit_center
        c_n = c[0]
        c_e = c[1]
        c_d = c[2]
        rho = path.orbit_radius
        d = np.sqrt((state.north-c_n)**2 + (state.east-c_e)**2) # polar coordinates here
        #orbit error difference in the radius normalized by the radius
        orbit_error = abs((d-path.orbit_radius))/path.orbit_radius
        phi = np.atan2((state.east-c_e),(state.north-c_n))
        phi = wrap(phi, state.chi)
        lmbda = path.orbit_direction

        if lmbda == "CCW":
            lmbda = -1
        elif lmbda == "CW":
            lmbda = 1
        # airspeed command
        
        self.autopilot_commands.airspeed_command = path.airspeed

        # course command eq 10.15
        chi_c = phi + lmbda*(np.pi/2 + np.atan(self.k_orbit*(d-rho)/rho))

        self.autopilot_commands.course_command = chi_c
        # altitude command
        self.autopilot_commands.altitude_command = -c_d # negative down
        
        # roll feedforward command
        # if close to the circle implement this, if you're far away then don't, like if you're within 10% implement this, otherwise keep 0
        # slide 23 chptr 10
        if orbit_error <0.1:
            self.autopilot_commands.phi_feedforward = lmbda*np.atan(state.Vg**2/(self.gravity*rho*np.cos(state.chi-state.psi))) # TODO is this the right phi
            # print(self.autopilot_commands.phi_feedforward)
        else:
            self.autopilot_commands.phi_feedforward = 0
            # print("disabled")
            




