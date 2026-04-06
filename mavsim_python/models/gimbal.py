"""
point_gimbal
    - point gimbal at target
part of mavsim
    - Beard & McLain, PUP, 2012
    - Update history:  
        3/31/2022 - RWB
        7/13/2023 - RWB
"""
import numpy as np
from tools.rotations import euler_to_rotation
import parameters.camera_parameters as CAM


class Gimbal:
    def pointAtGround(self, mav):
        az_d = 0 # these are the command angles in this scenario, down
        el_d = np.radians(-90)
        # proportional control for gimbal
        u_az = CAM.k_az*(az_d- CAM.az0)    ####### TODO ######
        u_el = CAM.k_el*(el_d - CAM.el0)    ####### TODO ######
        return( np.array([[u_az], [u_el]]) )
            
    def pointAtPosition(self, mav, target_position):
        # line-of-sight vector in the inertial frame
        mav_position = np.array([[mav.north], [mav.east], [-mav.altitude]])
        ell_i = target_position-mav_position   ####### TODO ######
        # rotate line-of-sight vector into body frame and normalize
        ####### TODO ######
        Ri_to_b = 0
        ell_b = 1/np.linalg.norm(ell_i)@Ri_to_b@ell_i # pg 263   ####### TODO ######
        return( self.pointAlongVector(ell_b, mav.gimbal_az, mav.gimbal_el) )

    def pointAlongVector(self, ell, azimuth, elevation):
        # point gimbal so that optical axis aligns with unit vector ell
        # ell is assumed to be aligned in the body frame
        # given current azimuth and elevation angles of the gimbal
        # compute control inputs to align gimbal

        # eq 13.13-13.15
        az_d = 0   ####### TODO ######
        el_d = 0    ####### TODO ######
        # proportional control for gimbal
        u_az = 0    ####### TODO ######
        u_el = 0    ####### TODO ######
        return( np.array([[u_az], [u_el]]) )




