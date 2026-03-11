"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler


class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]]).flatten()  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]]).flatten()
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta).flatten()
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]

        m = MAV.mass
        g = MAV.gravity
        # Extract the States
        north = self._state.item(0)
        east = self._state.item(1)
        down = self._state.item(2)
        u = self._state.item(3)
        v = self._state.item(4)
        w = self._state.item(5)
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)

        e = np.array([e0,e1,e2,e3])
        R = quaternion_to_rotation(e)
        phi = np.atan2(2*(e0*e1+e2*e3),(e0**2+e3**2-e1**2-e2**2))
        theta = np.asin(2*(e0*e2-e1*e3))
        psi = np.atan2(2*(e0*e3+e1*e2),(e0**2+e1**2-e2**2-e3**2))

        ##### TODO #####
        # convert steady-state wind vector from world to body frame
        wind_body = R @ steady_state
        # add the gust 
        wind_body += gust
        # convert total wind to world frame
        self._wind = (R.T @ wind_body).flatten()
        
       
        
        
        #velocity vector relative to the airmass ([ur , vr, wr]= ?)

        # compute airspeed (self._Va = ?)
        V_a_in_b = np.array([[u-wind_body[0]],
                             [v-wind_body[1]],
                             [w-wind_body[2]]]).flatten()    
        # compute angle of attack (self._alpha = ?)
        self._Va = np.linalg.norm(V_a_in_b)
        self._alpha = np.atan(V_a_in_b[2]/V_a_in_b[0])
       
        
        
        # compute sideslip angle (self._beta = ?)
        self._beta = np.asin(V_a_in_b[1]/np.linalg.norm(self._Va))




    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        ##### TODO ######
        # extract states (phi, theta, psi, p, q, r)
        m = MAV.mass
        g = MAV.gravity

        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)

        e = np.array([e0,e1,e2,e3])
        R = quaternion_to_rotation(e)
        phi = np.atan2(2*(e0*e1+e2*e3),(e0**2+e3**2-e1**2-e2**2))
        theta = np.asin(2*(e0*e2-e1*e3))
        psi = np.atan2(2*(e0*e3+e1*e2),(e0**2+e1**2-e2**2-e3**2))
        

                #input by user?
        delta_e = delta.elevator
        delta_a = delta.aileron#0.5*(delta_a_left - delta_a_right)
        delta_r = delta.rudder
        throttle = delta.throttle
        
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, throttle)



        ################ REDO ###############
        # code in force of lift and drag and apply the rotation to get them to fx and fy (see slide 22)
        # dont include abs(on Se) to get it to work
        # Calculate Cl(alpha) as a function of parameters, the others are constants from parameters (see slide17)
        # compute gravitational forces ([fg_x, fg_y, fg_z])
        # this should be more right
        # gravity_forces = np.array([[-m*g*np.sin(theta)],
        #                            [m*g*np.cos(theta)*np.sin(phi)],
        #                            [m*g*np.cos(theta)*np.cos(phi)]])
        
        M = MAV.M
        sigma_of_alpha = (1 + np.exp(-M*(self._alpha -MAV.alpha0))+np.exp(M*(self._alpha+MAV.alpha0)))/((1+np.exp(-M*(self._alpha-MAV.alpha0)))*(1+np.exp(M*(self._alpha+MAV.alpha0))))
        C_L_of_alpha = (1-sigma_of_alpha)*(MAV.C_L_0 + MAV.C_L_alpha*self._alpha) + sigma_of_alpha*(2*np.sign(self._alpha)*np.sin(self._alpha)**2*np.cos(self._alpha))

        C_D_of_alpha = (MAV.C_D_p + MAV.C_L_0**2/(np.pi*MAV.e*MAV.AR))+((2*MAV.C_L_0*MAV.C_L_alpha)/((np.pi*MAV.e*MAV.AR)))*self._alpha+(MAV.C_L_alpha**2/((np.pi*MAV.e*MAV.AR)))*self._alpha**2
        
        gravity_forces = R.T @ np.array([[0],[0],[m*g]])
        C_X_of_alpha = -C_D_of_alpha*np.cos(self._alpha)+C_L_of_alpha*np.sin(self._alpha)
        C_Xq_of_alpha = -MAV.C_D_q*np.cos(self._alpha)+MAV.C_L_q*np.sin(self._alpha)
        C_Z_of_alpha = -C_D_of_alpha*np.sin(self._alpha)-C_L_of_alpha*np.cos(self._alpha)
        C_Zq_of_alpha = -MAV.C_D_q*np.sin(self._alpha)-MAV.C_L_q*np.cos(self._alpha)

        c1 = 0.5*MAV.rho*self._Va**2*MAV.S_wing

        lateral_forces = c1*np.array([[C_X_of_alpha + C_Xq_of_alpha*MAV.c/(2*self._Va)*q],
                                        [MAV.C_Y_0+MAV.C_Y_beta*self._beta+MAV.C_Y_p*MAV.b/(2*self._Va)*p+MAV.C_Y_r*MAV.b/(2*self._Va)*r],
                                        [C_Z_of_alpha+C_Zq_of_alpha*MAV.c/(2*self._Va)*q]])


        C_Xdeltae_of_alpha = -MAV.C_D_delta_e*np.cos(self._alpha)+MAV.C_L_delta_e*np.sin(self._alpha)
        C_Zdeltae_of_alpha = -MAV.C_D_delta_e*np.sin(self._alpha) - MAV.C_L_delta_e*np.cos(self._alpha)
        longitudinal_forces = c1*np.array([[float(C_Xdeltae_of_alpha*delta_e)],
                                            [float(MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r)],
                                            [float(C_Zdeltae_of_alpha*delta_e)]])
        
        propeller_forces = np.array([[thrust_prop],[0],[0]])
        forces_xyz = gravity_forces.reshape(3,1) + lateral_forces.reshape(3,1) + longitudinal_forces.reshape(3,1) + propeller_forces.reshape(3,1)
        forces_xyz.reshape(3,1)
        self._forces = forces_xyz
        fx = forces_xyz[0]
        fy = forces_xyz[1]
        fz = forces_xyz[2]

        b_coeff = MAV.b/(2*self._Va)
        c_coeff = MAV.c/(2*self._Va)

        longitudinal_moms = c1*np.array([[MAV.b*(MAV.C_ell_0 + MAV.C_ell_beta*self._beta+MAV.C_ell_p*b_coeff*p + MAV.C_ell_r*b_coeff*r)],
                                        [MAV.c*(MAV.C_m_0 + MAV.C_m_alpha*self._alpha +MAV.C_m_q*c_coeff*q)],
                                        [MAV.b*(MAV.C_n_0 + MAV.C_n_beta*self._beta +MAV.C_n_p*b_coeff*p + MAV.C_n_r*b_coeff*r)]])
        
        lateral_moms = c1*np.array([[MAV.b*(MAV.C_ell_delta_a*delta_a+MAV.C_ell_delta_r*delta_r)],
                                    [MAV.c*(MAV.C_m_delta_e*delta_e)],
                                    [MAV.b*(MAV.C_n_delta_a*delta_a+MAV.C_n_delta_r*delta_r)]])
        propeller_moms = np.array([[torque_prop],[0],[0]])

        moments_xyz = longitudinal_moms.reshape(3,1)+lateral_moms.reshape(3,1)-propeller_moms.reshape(3,1)
        moments_xyz.reshape(3,1)
        Mx = moments_xyz[0]
        My = moments_xyz[1]
        Mz = moments_xyz[2]

        # compute Lift and Drag coefficients (CL, CD)
        
        # compute Lift and Drag Forces (F_lift, F_drag)

        # propeller thrust and torque
        

        # compute longitudinal forces in body frame (fx, fz)

        # compute lateral forces in body frame (fy)

        # compute logitudinal torque in body frame (My)

        # compute lateral torques in body frame (Mx, Mz)

        forces_moments = np.array([[fx,fy,fz,Mx,My,Mz]]).reshape(6,1)
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        ##### TODO #####
        # map delta_t throttle command(0 to 1) into motor input voltage
        v_in = MAV.V_max * delta_t
        a = MAV.C_Q0*MAV.rho*np.power(MAV.D_prop,5)/((2*np.pi)**2)
        b = (MAV.C_Q1*MAV.rho*np.power(MAV.D_prop,4)/(2*np.pi))*Va + MAV.KQ**2/MAV.R_motor
        c = MAV.C_Q2*MAV.rho*np.power(MAV.D_prop,3)*Va**2-(MAV.KQ/MAV.R_motor)*v_in+MAV.KQ*MAV.i0

        # Angular speed of propeller (omega_p = ?)
        Omega_op = (-b+np.sqrt(b**2-4*a*c))/(2*a)

        J_op = 2*np.pi*Va / (Omega_op*MAV.D_prop)
        C_T = MAV.C_T2*J_op**2+MAV.C_T1*J_op+MAV.C_T0
        C_Q = MAV.C_Q2*J_op**2 + MAV.C_Q1*J_op+MAV.C_Q0

        # thrust and torque due to propeller
        n = Omega_op/(2*np.pi)
        thrust_prop = MAV.rho*n**2*np.power(MAV.D_prop,4)*C_T
        torque_prop = MAV.rho*n**2*np.power(MAV.D_prop,5)*C_Q

        return thrust_prop, torque_prop

    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0
