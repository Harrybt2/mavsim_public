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
from message_types.msg_sensors import MsgSensors
import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from models.mav_dynamics_control import MavDynamics as MavDynamicsNoSensors
from tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_rotation

class MavDynamics(MavDynamicsNoSensors):
    def __init__(self, Ts):
        super().__init__(Ts)
        # initialize the sensors message
        self._sensors = MsgSensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.

    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"
        
        north = self._state.item(0)
        east = self._state.item(1)
        down = self._state.item(2)
        h = -down
        u = self._state.item(3)
        v = self._state.item(4)
        w = self._state.item(5)
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)
        fx = self._forces[0]
        fy =self._forces[1]
        fz = self._forces[2]

        phi = np.atan2(2*(e0*e1+e2*e3),(e0**2+e3**2-e1**2-e2**2))
        theta = np.asin(2*(e0*e2-e1*e3))
        psi = np.atan2(2*(e0*e3+e1*e2),(e0**2+e1**2-e2**2-e3**2))

        # u_dot = 
        # v_dot = 
        # w_dot = 
        # simulate rate gyros(units are rad / sec)
        self._sensors.gyro_x = p + SENSOR.gyro_x_bias + SENSOR.gyro_sigma * np.random.normal()
        self._sensors.gyro_y = q + SENSOR.gyro_y_bias + SENSOR.gyro_sigma * np.random.normal()
        self._sensors.gyro_z = r + SENSOR.gyro_z_bias + SENSOR.gyro_sigma * np.random.normal()

        # simulate accelerometers(units of g)
        self._sensors.accel_x = float(fx/MAV.mass + MAV.gravity*np.sin(theta)) + SENSOR.accel_sigma * np.random.normal() # u_dot + q*w - r*v + g*np.sin(theta)
        self._sensors.accel_y = float(fy/MAV.mass - MAV.gravity*np.cos(theta)*np.sin(phi)) + SENSOR.accel_sigma * np.random.normal() # v_dot + r*u - p*w - g*np.cos(theta)*np.sin(phi)
        self._sensors.accel_z = float(fz/MAV.mass - MAV.gravity*np.cos(theta)*np.cos(phi)) + SENSOR.accel_sigma * np.random.normal() # w_dot + p*v - q*u - g*np.cos(theta)*np.cos(phi)
        
        # simulate magnetometers
        # magnetic field in provo has magnetic declination of 12.5 degrees
        # e1 = 
        # and magnetic inclination of 66 degrees
        delta =np.deg2rad(12.5)
        i = np.deg2rad(66)
        e1 = np.array([[1,0,0]]).T
        Rm = euler_to_rotation(0, -i, delta)
        mi = Rm @ e1
        Rbi = euler_to_rotation(phi, theta, psi).T
        mb = Rbi @ mi
        # M = 1 # magnetic field strength
        
        # m_inertial = M*np.array([[np.cos(delta)*np.cos(i)],
        #                        [np.sin(delta)*np.cos(i)],
        #                        [np.sin(i)]])
        
        # R_i_to_v1 = np.array([[np.cos(psi),np.sin(psi),0],
        #                  [-np.sin(psi),np.cos(psi),0],
        #                  [0,0,1]])
        # m_v1 = R_i_to_v1 @ m_inertial + SENSOR.mag_beta 
        self._sensors.mag_x = mb[0] + SENSOR.mag_sigma * np.random.normal()
        self._sensors.mag_y = mb[1] + SENSOR.mag_sigma * np.random.normal()
        self._sensors.mag_z = mb[2] + SENSOR.mag_sigma * np.random.normal()

        # simulate pressure sensors
        self._sensors.abs_pressure = MAV.rho*MAV.gravity*h  + SENSOR.abs_pres_sigma * np.random.normal() # Beta_abs + noise_abs
        self._sensors.diff_pressure = MAV.rho*self._Va**2/2 + SENSOR.diff_pres_sigma * np.random.normal() # beta_diff + noise_diff
        
        # simulate GPS sensor
        if self._t_gps >= SENSOR.ts_gps:
            alpha = np.exp(-SENSOR.gps_k * SENSOR.ts_gps)
            # update the internal error states (north, east, alt)
            self._gps_eta_n = alpha * self._gps_eta_n + SENSOR.gps_n_sigma * np.random.normal()
            self._gps_eta_e = alpha * self._gps_eta_e + SENSOR.gps_e_sigma * np.random.normal()
            self._gps_eta_h = alpha * self._gps_eta_h + SENSOR.gps_h_sigma * np.random.normal()
            self._sensors.gps_n = north + self._gps_eta_n 
            self._sensors.gps_e = east + self._gps_eta_e 
            self._sensors.gps_h = h + self._gps_eta_h 
            self._sensors.gps_Vg = np.sqrt((self._Va*np.cos(psi) + self._wind.item(0))**2 + (self._Va*np.sin(psi) + self._wind.item(1))**2) + SENSOR.gps_Vg_sigma * np.random.normal()
            self._sensors.gps_course = np.atan2(self._Va*np.sin(psi) + self._wind.item(1), self._Va*np.cos(psi) + self._wind.item(0)) + SENSOR.gps_course_sigma * np.random.normal()
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    def external_set_state(self, new_state):
        self._state = new_state

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
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
        self.true_state.bx = SENSOR.gyro_x_bias
        self.true_state.by = SENSOR.gyro_y_bias
        self.true_state.bz = SENSOR.gyro_z_bias
        self.true_state.camera_az = self._state.item(13)
        self.true_state.camera_el = self._state.item(14)