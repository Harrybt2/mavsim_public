"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
from tools.transfer_function import TransferFunction
import parameters.aerosonde_parameters as MAV
import numpy as np


class WindSimulation:
    def __init__(self, Ts, gust_flag = True, steady_state = np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state
        ##### TODO #####
        if gust_flag == True:
            sigma_u = 1.06
            sigma_v = 1.06
            sigma_w = 0.7
        else:
            sigma_u = 3.0
            sigma_v = -3.0
            sigma_w = 0.0

        
        L_u = 200
        L_v = 200
        L_w = 50

        #   Dryden gust model parameters (pg 62 UAV book)
        au = sigma_u*np.sqrt(2*MAV.Va0/(np.pi*L_u))
        av = sigma_v*np.sqrt(3*MAV.Va0/(np.pi*L_v))
        aw = sigma_w*np.sqrt(3*MAV.Va0/(np.pi*L_w))
        # Dryden transfer functions (section 4.4 UAV book) - Fill in proper num and den
        self.u_w = TransferFunction(num=np.array([[au]]), den=np.array([[1, MAV.Va0/L_u]]),Ts=Ts)
        self.v_w = TransferFunction(num=np.array([[av,av*MAV.Va0/(np.sqrt(3)*L_v)]]), den=np.array([[1,2*MAV.Va0/L_v,((MAV.Va0/L_v))**2]]),Ts=Ts)
        self.w_w = TransferFunction(num=np.array([[aw,aw*MAV.Va0/(np.sqrt(3)*L_w)]]), den=np.array([[1,2*MAV.Va0/L_w,((MAV.Va0/L_w))**2]]),Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate(( self._steady_state, gust ))

