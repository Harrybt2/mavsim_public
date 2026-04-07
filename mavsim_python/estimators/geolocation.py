"""
target geolocation algorithm
    - Beard & McLain, PUP, 2012
    - Updated:
        4/1/2022 - RWB
        4/6/2022 - RWB
        7/13/2023 - RWB
        4/7/2025 - TWM
"""
import numpy as np
import parameters.simulation_parameters as SIM
import parameters.camera_parameters as CAM
from tools.rotations import euler_to_rotation
from estimators.filters import ExtendedKalmanFilterContinuousDiscrete

# Note that state equations assume a constant-velocity model for the target
class Geolocation:
    def __init__(self, ts: float=0.01):
        self.ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f, 
            Q = 0.01 * np.diag([
                (1)**2,   # target north position
                (1)**2,   # target east position
                (1)**2,   # target down position
                (10)**2,  # target north velocity
                (10)**2,  # target east velocity
                (10)**2,  # target down velocity
                (3)**2,   # distance to target L
                ]),
            P0= 0.1*np.diag([
                10**2,  # target north position
                10**2,  # target east position
                10**2,  # target down position
                10**2,  # target north velocity
                10**2,  # target east velocity
                10**2,  # target down velocity
                10**2,  # distance to target L
                ]), 
            xhat0=np.array([[
                0.,  # target north position
                0.,  # target east position
                0.,  # target down position
                0.,  # target north velocity
                0.,  # target east velocity
                0.,  # target down velocity
                100.,  # distance to target L
                ]]).T, 
            Qu=0.01*np.diag([
                1**2, # mav north position
                1**2, # mav east position
                1**2, # mav down position
                1**2, # mav north velocity
                1**2, # mav east velocity
                1**2, # mav down velocity
                ]), 
            Ts = ts,
            N = 10
        )
        self.R = 0.1 * np.diag([1.0, 1.0, 1.0, 1.0])

    def update(self, mav, pixels):
        # system input is mav state
        u = np.array([
            
            ######## TODO ######## pull out what you need for your state

            ])    
        xhat, P = self.ekf.propagate_model(u)
        # update with pixel measurement
        y=self.process_measurements(mav, pixels)
        xhat, P = self.ekf.measurement_update(
            y=y, 
            u=u,
            h=self.h,
            R=self.R)
        return xhat[0:3, :]  # return estimated NED position

    def f(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        # system dynamics for propagation model: xdot = f(x, u)

            ######## TODO ########

        target_position_dot = 
        target_velocity_dot = 
        L_dot = 
        xdot = np.concatenate((target_position_dot, target_velocity_dot, L_dot), axis=0)
        return xdot

    def h(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        # measurement model y
            ######## TODO ########
        target_position = 
        L = 
        y = np.concatenate((target_position, L), axis=0)
        return y

    def process_measurements(self, mav, pixels):
        # calculate measurement (target position, L) from mav position and pixel coordinates of target
        # assume flat earth
        h = mav.altitude
        mav_position = np.array([[mav.north], [mav.east], [-h]])
        ell = np.array([[pixels.pixel_x], [pixels.pixel_y], [CAM.f]])
        ell_c = ell / np.linalg.norm(ell)
            ######## TODO ########
        R_b_i = 
        R_g_b = 
        R_c_g = 
        ell_i = 
        L = 
        target_position = 
        y = np.concatenate((target_position, np.array([[L]])), axis=0)
        return y
