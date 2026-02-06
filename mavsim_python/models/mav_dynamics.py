"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
        7/13/2023 - RWB
        1/17/2024 - RWB
"""
import numpy as np
# load message types
from message_types.msg_state import MsgState
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler

class MavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.north0],  # (0)
                               [MAV.east0],   # (1)
                               [MAV.down0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0],    # (12)
                               [0],   # (13)
                               [0],   # (14)
                               ])
        # initialize true_state message
        self.true_state = MsgState()

    ###################################
    # public functions
    def update(self, forces_moments):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        self._rk4_step(forces_moments)
        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _rk4_step(self, forces_moments):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._f(self._state[0:13], forces_moments)
        k2 = self._f(self._state[0:13] + time_step/2.*k1, forces_moments)
        k3 = self._f(self._state[0:13] + time_step/2.*k2, forces_moments)
        k4 = self._f(self._state[0:13] + time_step*k3, forces_moments)
        self._state[0:13] += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

    def _f(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        ##### TODO #####

        m = MAV.mass
        g = MAV.gravity
        # Extract the States
        north = state.item(0)
        east = state.item(1)
        down = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)

        e = np.array([e0,e1,e2,e3])
        R = quaternion_to_rotation(e)
        # phi = np.atan2(2*(e0*e1+e2*e3),(e0**2+e3**2-e1**2-e2**2))
        # theta = np.asin(2*(e0*e2-e1*e3))
        # psi = np.atan2(2*(e0*e3+e1*e2),(e0**2+e1**2-e2**2-e3**2))
        
        # Extract Forces/Moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        Mx = forces_moments.item(3)
        My = forces_moments.item(4)
        Mz = forces_moments.item(5)

        

        # # Position Kinematics
        # angles = np.array([[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
        #                     [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
        #                     [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]])


        # normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        # e0 = e0/normE
        # e1 = e1/normE
        # e2 = e2/normE
        # e3 = e3/normE

        angles = np.array([[e1**2+e0**2-e2**2-e3**2, 2*(e1*e2-e3*e0),2*(e1*e3+e2*e0)],
                           [2*(e1*e2+e3*e0),e2**2+e0**2-e1**2-e3**2,2*(e2*e3-e1*e0)],
                           [2*(e1*e3-e2*e0),2*(e2*e3+e1*e0),e3**2+e0**2-e1**2-e2**2]])
        pos_dot = R @ np.array([[u],[v],[w]])

        # Position Dynamics
        first = np.array([[r*v-q*w],
                          [p*w-r*u],
                          [q*u-p*v]])
        forces = 1/MAV.mass*np.array([[fx],
                               [fy],
                               [fz]])
        u_dot = first + forces


        # rotational kinematics
        # e0_dot = np.array([[1, np.sin()*np.tan(theta), np.cos(phi)*np.tan(theta)],
        #                    [0, np.cos(phi), -np.sin(phi)],
        #                    [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
        
        #quaternion method, must normalize after each rk4 step
        matrix = 0.5*np.array([[0,-p,-q,-r],
                               [p,0,r,-q],
                               [q,-r,0,p],
                               [r,q,-p,0]])
        es = np.array([[e0],[e1],[e2],[e3]])
        e0_dot =matrix @ es
        
        Jx = MAV.Jx
        Jy = MAV.Jy
        Jz = MAV.Jz
        Jxz =  MAV.Jxz
        
        Ro = Jx*Jz-Jxz**2
        Ro1 = Jxz*(Jx-Jy+Jz)/(Ro)
        Ro2 = (Jz*(Jz-Jy)+Jxz**2)/Ro
        Ro3 = Jz/Ro
        Ro4 = Jxz/Ro
        Ro5 = (Jz-Jx)/Jy
        Ro6 = Jxz/Jy
        Ro7 = ((Jx-Jy)*Jx+Jxz**2)/Ro
        Ro8 = Jx/Ro
        # rotatonal dynamics
        n = Mz
        m = My
        el = Mx
        prt1 = np.array([[Ro1*p*q-Ro2*q*r],
                          [Ro5*p*r-Ro6*(p**2-r**2)],
                          [Ro7*p*q-Ro1*q*r]])
        prt2 = np.array([[Ro3*el + Ro4*n],
                         [m/Jy],
                         [Ro4*el + Ro8*n]])
        p_dot = prt1 + prt2

        x_dot = np.array([
    pos_dot[0], pos_dot[1], pos_dot[2],
    u_dot[0], u_dot[1], u_dot[2],
    e0_dot[0], e0_dot[1], e0_dot[2], e0_dot[3],
    p_dot[0], p_dot[1], p_dot[2]
    ]).reshape(13, 1)
        return x_dot

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = 0
        self.true_state.alpha = 0
        self.true_state.beta = 0
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = 0
        self.true_state.gamma = 0
        self.true_state.chi = 0
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = 0
        self.true_state.we = 0
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0
    
    # def _derivatives(self, state, forces_moments):
        
    #     while True: #for _ in range():
    #         self._rk4_step(forces_moments)